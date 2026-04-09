#!/usr/bin/env python
# Topology Mininet – 4 slices: URLLC (1:10), URLLC_eMBB_MIX (1:11), eMBB (1:12), mMTC (1:13)
# Dataset: cesnet_points_clustered_4slices.csv
# Columns used: n_bytes, n_packets, avg_pkt_size, bytes_per_flow, pkts_per_flow, burst_cv,
#               dir_ratio_bytes, tcp_udp_ratio_bytes, cluster, slice

from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info

import re
import time
import threading
import requests
import random
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Globals for delta-computation of tc counters
# ──────────────────────────────────────────────────────────────────────────────
previous_bytes_sent    = {}
previous_packets_dropped = {}

# ──────────────────────────────────────────────────────────────────────────────
# Slice meta-data
# ──────────────────────────────────────────────────────────────────────────────
# index  slice name      tc class  DSCP hex   iperf port  priority (lower = higher)
SLICE_META = {
    'URLLC':          {'cls': '1:10', 'dscp': '0x28', 'port': 5001, 'priority': 0},
    'URLLC_eMBB_MIX': {'cls': '1:11', 'dscp': '0x18', 'port': 5002, 'priority': 1},
    'eMBB':           {'cls': '1:12', 'dscp': '0x10', 'port': 5003, 'priority': 2},
    'mMTC':           {'cls': '1:13', 'dscp': '0x50', 'port': 5004, 'priority': 3},
}
# Ordered list used throughout
SLICE_NAMES   = ['URLLC', 'URLLC_eMBB_MIX', 'eMBB', 'mMTC']
TC_CLASSES    = ['1:10', '1:11', '1:12', '1:13']
NUM_SLICES    = 4

# Max total bandwidth (Mbps) per port — must match NetworkConfig.max_bandwidth_mbps
MAX_BW_MBPS   = 750

# ──────────────────────────────────────────────────────────────────────────────
# Ryu helpers
# ──────────────────────────────────────────────────────────────────────────────

def register_interfaces_to_ryu(switch, dpid, ryu_ip="172.18.0.10", ryu_port=8080):
    interfaces = [i.name for i in switch.intfList()
                  if i.name.startswith(switch.name + '-eth')]
    print(f"Interfaces détectées pour {switch.name}: {interfaces}")
    try:
        r = requests.post(
            f"http://{ryu_ip}:{ryu_port}/register_interfaces",
            json={"dpid": dpid, "interfaces": interfaces},
        )
        print("Réponse Ryu:", r.text)
    except Exception as e:
        print("Erreur register_interfaces:", e)


def send_stats_to_ryu(dpid, stats, ryu_ip="172.18.0.10", ryu_port=8080):
    """Envoie les statistiques tc à Ryu; retourne les nouveaux rates."""
    data = {"dpid": dpid, "timestamp": time.time(), "stats": stats}
    try:
        r = requests.post(
            f"http://{ryu_ip}:{ryu_port}/monitoring",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        print("Stats envoyées à Ryu")
        return {
            "status":    resp.get("status"),
            "new_rates": resp.get("new_rates", []),
            "message":   resp.get("message", ""),
        }
    except Exception as e:
        print("Erreur send_stats_to_ryu:", e)
        return {"status": "error", "new_rates": [], "message": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# TC configuration – 4 HTB classes per interface
# ──────────────────────────────────────────────────────────────────────────────

def configure_tc_queues_switch(switch):
    """
    HTB hierarchy per interface:
        root 1: htb default 13   (mMTC = default fallback)
          └─ 1:1   parent  750 Mbit
               ├─ 1:10  URLLC          50 Mbit (pfifo limit 2)
               ├─ 1:11  URLLC_eMBB_MIX 50 Mbit (pfifo limit 4)
               ├─ 1:12  eMBB           50 Mbit (pfifo limit 8)
               └─ 1:13  mMTC           50 Mbit (pfifo limit 16)

    Filters match on DSCP (ToS byte) and UDP destination port as fallback.
    """
    for intf in switch.intfList():
        if not intf.name.startswith(switch.name + '-eth'):
            continue
        n = intf.name

        # ── Clean slate ──────────────────────────────────────────────────────
        switch.cmd(f'tc qdisc del dev {n} root 2>/dev/null')

        # ── Root + parent class ───────────────────────────────────────────────
        switch.cmd(f'tc qdisc add dev {n} root handle 1: htb default 13')
        switch.cmd(f'tc class add dev {n} parent 1:  classid 1:1  htb '
                   f'rate {MAX_BW_MBPS}mbit ceil {MAX_BW_MBPS}mbit')

        # ── Child classes ─────────────────────────────────────────────────────
        # URLLC  – strict low-latency (small queue)
        switch.cmd(f'tc class add dev {n} parent 1:1 classid 1:10 htb '
                   f'rate 50mbit ceil {MAX_BW_MBPS}mbit prio 1')
        switch.cmd(f'tc qdisc add dev {n} parent 1:10 handle 10: pfifo limit 2')

        # URLLC_eMBB_MIX
        switch.cmd(f'tc class add dev {n} parent 1:1 classid 1:11 htb '
                   f'rate 50mbit ceil {MAX_BW_MBPS}mbit prio 2')
        switch.cmd(f'tc qdisc add dev {n} parent 1:11 handle 11: pfifo limit 4')

        # eMBB  – larger queue, more burst tolerance
        switch.cmd(f'tc class add dev {n} parent 1:1 classid 1:12 htb '
                   f'rate 50mbit ceil {MAX_BW_MBPS}mbit prio 3')
        switch.cmd(f'tc qdisc add dev {n} parent 1:12 handle 12: pfifo limit 8')

        # mMTC  – low-priority best-effort
        switch.cmd(f'tc class add dev {n} parent 1:1 classid 1:13 htb '
                   f'rate 50mbit ceil {MAX_BW_MBPS}mbit prio 4')
        switch.cmd(f'tc qdisc add dev {n} parent 1:13 handle 13: pfifo limit 16')

        # ── Filters (DSCP / ToS byte at offset 1) ────────────────────────────
        # URLLC   DSCP 0x28
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 1 '
                   f'u32 match u8 0x28 0xff at 1 flowid 1:10')
        # URLLC_eMBB_MIX DSCP 0x18
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 2 '
                   f'u32 match u8 0x18 0xff at 1 flowid 1:11')
        # eMBB   DSCP 0x10
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 3 '
                   f'u32 match u8 0x10 0xff at 1 flowid 1:12')
        # mMTC   DSCP 0x50
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 4 '
                   f'u32 match u8 0x50 0xff at 1 flowid 1:13')

        # ── UDP-port fallback filters ─────────────────────────────────────────
        # port 5001 → URLLC (0x1389)
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 5 '
                   f'u32 match ip protocol 17 0xff match u16 0x1389 0xffff at 22 flowid 1:10')
        # port 5002 → URLLC_eMBB_MIX (0x138a)
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 6 '
                   f'u32 match ip protocol 17 0xff match u16 0x138a 0xffff at 22 flowid 1:11')
        # port 5003 → eMBB (0x138b)
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 7 '
                   f'u32 match ip protocol 17 0xff match u16 0x138b 0xffff at 22 flowid 1:12')
        # port 5004 → mMTC (0x138c)
        switch.cmd(f'tc filter add dev {n} parent 1: protocol ip prio 8 '
                   f'u32 match ip protocol 17 0xff match u16 0x138c 0xffff at 22 flowid 1:13')

        out = switch.cmd(f'tc filter show dev {n} parent 1:')
        print(f"Filtres tc sur {n}:\n{out}")


# ──────────────────────────────────────────────────────────────────────────────
# TC stats collection – 4 classes
# ──────────────────────────────────────────────────────────────────────────────

def collect_tc_stats(switch, dpid, sla_latencies: dict):
    """
    Collecte les stats tc pour les 4 classes sur chaque interface du switch.

    sla_latencies: dict  {slice_name: latency_ms}  e.g. from the sampled CSV row.

    Envoie les stats à Ryu et applique les nouveaux rates reçus en retour.
    Returns the raw stats list (may be empty after being flushed to Ryu).
    """
    global previous_bytes_sent, previous_packets_dropped
    stats = []

    for intf in switch.intfList():
        if not intf.name.startswith(switch.name + '-eth'):
            continue
        intf_name = intf.name
        output = switch.cmd(f'tc -s class show dev {intf_name}')
        lines  = output.strip().split('\n')

        # ── Parse parent rate (1:1) ───────────────────────────────────────────
        parent_rate = 0
        cur = None
        for line in lines:
            line = line.strip()
            if line.startswith('class'):
                cur = line.split()[2]
            if cur == '1:1' and 'rate' in line:
                tok = line.split()
                idx = tok.index('rate')
                parent_rate = int(''.join(filter(str.isdigit, tok[idx + 1])))
                break

        # ── Parse per-class data ──────────────────────────────────────────────
        class_data = {}
        cur = None
        for line in lines:
            line = line.strip()
            if line.startswith('class'):
                cur = line.split()[2]
                if cur not in class_data:
                    class_data[cur] = {
                        'bytes_sent': 0, 'pkts_sent': 0,
                        'backlog_bytes': 0, 'dropped': 0, 'rate': 0,
                    }
            if cur and 'class htb' in line and cur in line:
                tok = line.split()
                if 'rate' in tok:
                    class_data[cur]['rate'] = int(
                        ''.join(filter(str.isdigit, tok[tok.index('rate') + 1]))
                    )
            if cur and line.startswith('Sent'):
                m = re.search(r'Sent\s+(\d+)\s+bytes\s+(\d+)\s+pkt', line)
                if m:
                    class_data[cur]['bytes_sent'] = int(m.group(1))
                    class_data[cur]['pkts_sent']  = int(m.group(2))
            if cur and 'backlog' in line:
                m = re.search(r'backlog\s+(\d+)b', line)
                if m:
                    class_data[cur]['backlog_bytes'] = int(m.group(1))
            if cur and 'dropped' in line:
                after = line.split('dropped')[1]
                class_data[cur]['dropped'] = int(after.split(',')[0].strip())

        # ── Build stat entries for each of the 4 slices ───────────────────────
        for slice_name, meta in SLICE_META.items():
            cls = meta['cls']
            if cls not in class_data:
                continue
            data = class_data[cls]
            key  = (intf_name, cls)

            prev_bytes   = previous_bytes_sent.get(key,    data['bytes_sent'])
            prev_dropped = previous_packets_dropped.get(key, data['dropped'])

            delta_bytes   = data['bytes_sent'] - prev_bytes
            delta_dropped = data['dropped']     - prev_dropped
            previous_bytes_sent[key]      = data['bytes_sent']
            previous_packets_dropped[key] = data['dropped']

            avg_pkt = data['bytes_sent'] / max(1, data['pkts_sent'])
            nbre_demands = (
                data['pkts_sent']
                + delta_dropped
                + (data['backlog_bytes'] / avg_pkt if avg_pkt > 0 else 1)
            )
            nbre_demands_bytes = (
                (delta_dropped * avg_pkt) + delta_bytes + data['backlog_bytes']
            ) * 8 / 1e6

            actual_rate = max(data['rate'], 1)
            throughput  = delta_bytes * 8 / 1e6          # Mbps
            latency_ms  = (delta_bytes * 8) / (actual_rate * 1e6) * 1000

            sla_lat = sla_latencies.get(slice_name, 100.0)

            # Random scaling factor (kept from original design)
            scale  = np.random.choice([0, 1, 2, 3, 4], p=[1/5] * 5)
            factor = 10 if scale == 1 else 1

            stats.append({
                'interface':         intf_name,
                'class':             cls,
                'slice':             slice_name,
                'dropped':           delta_dropped / max(1, nbre_demands),
                'rate':              actual_rate,
                'throughput':        throughput,
                'latency':           latency_ms,
                'parent_rate':       parent_rate,
                'nbre_demands':      nbre_demands * factor,
                'nbre_demands_bytes': nbre_demands_bytes * factor,
                'sla_latency':       sla_lat,
            })

            print(f"  {intf_name} {cls}({slice_name}): "
                  f"tx={delta_bytes}B bklog={data['backlog_bytes']}B "
                  f"rate={actual_rate}Mbit tp={throughput:.3f} lat={latency_ms:.3f}ms")

    # ── Forward to Ryu, apply returned rates ─────────────────────────────────
    new_rates = []
    if dpid == 1 and stats:
        new_rates = send_stats_to_ryu(dpid, stats).get('new_rates', [])
        stats = []

    if new_rates:
        _apply_new_rates(switch, new_rates)

    return stats


def _apply_new_rates(switch, new_rates):
    """
    Apply rate updates from Ryu to the 4 HTB child classes.

    Each entry in new_rates is expected as:
        {'id': 's1-eth1', 'rates': [r0, r1, r2, r3]}
    where rates[i] maps to TC_CLASSES[i] = ['1:10','1:11','1:12','1:13'].
    """
    classids = TC_CLASSES  # ['1:10', '1:11', '1:12', '1:13']
    for stat in new_rates:
        intf  = stat['id']
        rates = stat.get('rates', [])
        for i, cls in enumerate(classids):
            if i >= len(rates):
                break
            r_mbit = max(1, int(rates[i]))
            switch.cmd(f'tc class change dev {intf} classid {cls} '
                       f'htb rate {r_mbit}mbit ceil {r_mbit}mbit')


# ──────────────────────────────────────────────────────────────────────────────
# Traffic generation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bw_from_row(row) -> float:
    """Convert n_bytes (bytes/sec) to Mbps, floor at 0.1 Mbps."""
    return max(0.1, float(row['n_bytes']) * 8 / 1e6)


def _launch_iperf_slice(hosts, bw_mbps, dscp_hex, udp_port, duration=1):
    """Send UDP traffic between all host pairs for one slice."""
    for src in hosts:
        for dst in hosts:
            if src is dst:
                continue
            src.cmd(
                f'iperf -c {dst.IP()} -u -b {bw_mbps:.3f}M '
                f'-t {duration} -p {udp_port} -S {dscp_hex} &'
            )


# ──────────────────────────────────────────────────────────────────────────────
# Topology + simulation loop
# ──────────────────────────────────────────────────────────────────────────────

def create_topology():
    info('*** Création de la topologie réseau – 4 slices\n')

    net = Mininet(controller=RemoteController, switch=OVSKernelSwitch, link=TCLink)

    info('*** Ajout du contrôleur\n')
    c0 = net.addController('c0', controller=RemoteController, ip='172.18.0.10')

    info('*** Ajout du switch\n')
    s1 = net.addSwitch('s1')

    info('*** Ajout des hôtes\n')
    h1 = net.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
    h2 = net.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
    h3 = net.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')

    info('*** Ajout des liens\n')
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)

    info('*** Démarrage du réseau\n')
    net.build()
    c0.start()
    s1.start([c0])

    register_interfaces_to_ryu(s1, dpid=1, ryu_ip='172.18.0.10')

    # Clean OVS QoS entries (avoid conflicts with tc)
    for eth in ['s1-eth1', 's1-eth2', 's1-eth3']:
        s1.cmd(f'ovs-vsctl clear port {eth} qos')

    # Configure 4-class HTB on all switch interfaces
    configure_tc_queues_switch(s1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    info('*** Chargement du dataset 4 slices\n')
    df = pd.read_csv('cesnet_points_clustered_4slices.csv')
    df = df.tail(1500).reset_index(drop=True)

    # Sub-DataFrames per slice (used for random row sampling)
    slice_dfs = {name: df[df['slice'] == name].reset_index(drop=True)
                 for name in SLICE_NAMES}
    for name, sub in slice_dfs.items():
        info(f'  {name}: {len(sub)} rows\n')

    step_interval = random.uniform(1, 5)   # seconds between iterations

    # ── Start iperf servers once ───────────────────────────────────────────────
    hosts = [net.get('h1'), net.get('h2'), net.get('h3')]
    info('*** Lancement des serveurs iperf\n')
    for h in hosts:
        for meta in SLICE_META.values():
            h.cmd(f'iperf -s -u -p {meta["port"]} &')
    time.sleep(2)

    # ── Main simulation loop ───────────────────────────────────────────────────
    num_steps = len(df)
    step = 0

    while step < num_steps:
        # Sample one row per slice
        rows = {}
        sla_latencies = {}
        for name in SLICE_NAMES:
            sub = slice_dfs[name]
            if len(sub) == 0:
                continue
            row = sub.sample(n=1).iloc[0]
            rows[name] = row
            # Use Latency Requirement if present, else derive from SLA defaults
            if 'Latency Requirement (ms)' in df.columns:
                sla_latencies[name] = float(row.get('Latency Requirement (ms)', 10.0))
            else:
                # Fallback SLA latencies per slice type
                sla_latencies[name] = {
                    'URLLC': 1.0,
                    'URLLC_eMBB_MIX': 2.0,
                    'eMBB': 10.0,
                    'mMTC': 50.0,
                }[name]

        # Launch iperf traffic for each slice
        for name, meta in SLICE_META.items():
            if name not in rows:
                continue
            bw = _bw_from_row(rows[name])
            _launch_iperf_slice(
                hosts,
                bw_mbps  = bw,
                dscp_hex = meta['dscp'],
                udp_port = meta['port'],
                duration = 1,
            )

        # Collect tc stats and forward to Ryu / PPO controller
        collect_tc_stats(s1, dpid=1, sla_latencies=sla_latencies)

        step += 1
        time.sleep(step_interval)

        # Kill iperf clients, keep servers alive
        for h in hosts:
            h.cmd('pkill -f "iperf -c"')

    # ── Cleanup ───────────────────────────────────────────────────────────────
    info('*** Nettoyage iperf\n')
    for h in hosts:
        h.cmd('pkill iperf')

    info('*** Fin de la simulation\n')


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    setLogLevel('info')
    create_topology()


if __name__ == '__main__':
    main()
