"""Multi-node serving: pick the machines, launch the ranks, watch them, reap them.

One file with five entry points, because they are one concern and GEAK keeps its
tooling in few large files (run_e2e.py is 5.3k lines, harness_lib.py 1.4k):

    mn.py preflight    pick + PIN the machines, prove the shared mount  (Setup)
    mn.py launch       render serve.sh.tmpl per rank, ship it via Ray   (per launch)
    mn.py supervise    the local pid bench_e2e.sh watches and kills
    mn.py reap         backstop: reap the ranks if the supervisor dies
    mn.py trace-gate   validate the per-rank trace set before parsing

Nothing here knows how to start a server. Each rank sources adapters/$BACKEND.sh
on its own node and asks it for the launch line and for adapter_dist_args -- how
that framework spells "I am rank i of n". This file only supplies node facts.

The adapter that calls all of this is adapters/launchers/multinode.sh, which has
to stay a separate shell file: bench_e2e.sh sources
adapters/launchers/$BENCH_LAUNCHER.sh and calls the adapter_* functions it
defines in its own shell.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import gzip
import ipaddress
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

# Optional on purpose. `mn.py trace-gate` re-validates an exported trace set and
# never touches a cluster, which is how a set gets re-gated after the fact -- on
# a laptop, from a copied profile dir, long after the nodes are gone. A top-level
# `import ray` would take that away, and connect() below refuses clearly instead.
try:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
except ImportError:                                     # pragma: no cover
    ray = None
    NodeAffinitySchedulingStrategy = None

# Set by the subcommand so supervise/reap keep the log prefixes they had.
_LOG_TAG = "mn"


def log(msg: str) -> None:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{_LOG_TAG} {stamp}] {msg}", flush=True)


# --------------------------------------------------------------------------
# FABRIC
# --------------------------------------------------------------------------
#
# Which network the ranks talk over, decided from what the nodes report.
#
# Pure functions, no ray import, so the decision is testable without a cluster.
# The fabric is inferred rather than declared: the caller hands over a Ray
# cluster, so which machines get offered -- and their addresses -- are unknown
# until Ray answers. What is knowable is a property of the set: the ranks must
# share a network, so the network the candidates have in common is the one that
# can carry collectives.
#
# "Share a network" is decided from the real masks, not from the address text.
# The MI300X boxes here make the difference concrete: each has eight 400Gb NICs
# addressed 192.168.15.64/31, .66/31 ... and the other node has .16/31, .18/31.
# Read as /24 they look like one fast shared fabric; they are in fact eight
# point-to-point links to a switch, they do not ping, and the host routes
# 192.168.15.0/24 back out the 25Gb management NIC. Choosing that "fabric" would
# hang every launch at rendezvous with no clue why.
#

# Shared by every node while able to carry nothing between them.
EXCLUDED_IP = ("127.", "169.254.")
EXCLUDED_IF = ("lo", "docker", "br-", "veth", "virbr", "cni", "flannel",
               "kube", "tun", "tap")


def usable(ifname: str, ip: str) -> bool:
    return not ip.startswith(EXCLUDED_IP) and not ifname.startswith(EXCLUDED_IF)


def subnet_of(ip: str) -> str:
    """The /24 prefix, trailing dot included so it works with startswith."""
    parts = ip.split(".")
    return ".".join(parts[:3]) + "." if len(parts) == 4 else ""


def _entries(addrs: list):
    """Probes report [if, ip, prefixlen]; a probe predating the mask reports
    [if, ip] and is read as /24, which is what this module assumed when it had
    no masks at all."""
    for e in addrs:
        yield (e[0], e[1], int(e[2])) if len(e) >= 3 else (e[0], e[1], 24)


def _network(ip: str, plen: int):
    try:
        return ipaddress.ip_network(f"{ip}/{plen}", strict=False)
    except ValueError:
        return None


def _glob(net) -> str:
    """The prefix serve.sh.tmpl matches with `case $ip in <prefix>*)`, and the
    one pick_addr uses. Byte-aligned part only: /24 -> `10.235.200.`. A mask
    that is not byte-aligned yields a SUPERSET of its network, which is safe
    here only because the driver has already proven each node holds an address
    on the real network before this prefix is ever matched."""
    octets = net.prefixlen // 8
    return ".".join(str(net.network_address).split(".")[:octets]) + "." if octets else ""


def decide(per_node_addrs: list, explicit: str = "") -> dict:
    """Choose the fabric across candidates: {subnet, how, nodes_covered}.

    An explicit GEAK_MN_CLUSTER_SUBNET wins, because an operator who names one
    knows which of two shared networks is the fast one. It is still reported
    with its coverage, so a value that matches nothing reads as such instead of
    as an empty cluster.
    """
    if explicit:
        # An operator names a prefix, not a mask, so this stays textual.
        covered = sum(
            1 for addrs in per_node_addrs
            if any(usable(i, ip) and ip.startswith(explicit)
                   for i, ip, _ in _entries(addrs)))
        return {"subnet": explicit, "how": "configured",
                "nodes_covered": covered, "candidates": {}}

    # One address per node per network: what that node would answer on it.
    per_node = []
    for addrs in per_node_addrs:
        got = {}
        for ifname, ip, plen in _entries(addrs):
            net = _network(ip, plen) if usable(ifname, ip) else None
            if net:
                got.setdefault(net, ip)
        per_node.append(got)

    counts, ranked = {}, []
    for net in {n for got in per_node for n in got}:
        ips = [got[net] for got in per_node if net in got]
        key = _glob(net) or str(net)
        counts[key] = max(counts.get(key, 0), len(ips))
        # Two nodes answering the SAME address are not on a shared network, they
        # are each on a node-local one: every box has docker's 172.17.0.1, which
        # by address alone looks like the most shared network in the cluster.
        # Unlike a name blacklist, this holds whatever the interface is called.
        if _glob(net) and len(ips) >= 2 and len(set(ips)) == len(ips):
            ranked.append((-len(ips), _glob(net)))
    # Under two nodes cannot carry a collective. Ties break on the lowest
    # prefix, so a re-pin decides the same way -- a fabric that moved between
    # A/B legs would be a silent variable.
    ranked.sort()
    if not ranked:
        return {"subnet": "", "how": "undecided", "nodes_covered": 0,
                "candidates": counts}
    return {"subnet": ranked[0][1], "how": "inferred",
            "nodes_covered": -ranked[0][0], "candidates": counts}


def pick_addr(addrs: list, subnet: str) -> tuple[str, str]:
    """This node's (interface, ip) on the chosen fabric, or ("", "")."""
    for ifname, ip, _ in _entries(addrs):
        if subnet and usable(ifname, ip) and ip.startswith(subnet):
            return ifname, ip
    return "", ""


def describe(per_node_addrs: list, names: list) -> str:
    """What each node had -- the evidence that is gone once the run ends.

    Masks included: with them the reader can see WHY a set of addresses that
    look alike was rejected."""
    return "\n".join(
        f"      {name}: "
        + (", ".join(f"{i}={a}/{p}" for i, a, p in _entries(addrs) if usable(i, a))
           or "none")
        for name, addrs in zip(names, per_node_addrs))


# --------------------------------------------------------------------------
# CLUSTER
# --------------------------------------------------------------------------
#
# Shared plumbing for the multi-node server launcher.
#
# Ray is used ONLY as a cluster manager here: it tells us which machines exist,
# runs a probe on each one, and ships a command to a chosen node. sglang still
# comes up as a plain torch-dist job -- Ray never owns the scheduler, so
# `--use-ray` is deliberately absent. That keeps the served stack identical to
# the one a human starts by hand on each box, which is the whole point: a
# benchmark that measures a different execution model than production is not a
# baseline.
#
# Two design choices carry most of the weight:
#
# * THE NODE SET IS PICKED ONCE AND PINNED. Picking idle machines afresh per
#   launch (what a standalone dispatcher does) is wrong for GEAK: the two legs of
#   an A/B must run on the same boxes and the same cards, or the delta measures
#   the hardware lottery rather than the change. Selection happens in preflight
#   and lands in mn_state.json; every later launch reuses it and fails loudly if a
#   pinned node is gone.
#
# * IDLENESS IS MEASURED ON THE MACHINE, NOT IN RAY'S LEDGER. Ray only accounts
#   for work it scheduled, so another tenant can have every card full while Ray
#   reports 0/16 GPU. And 0% utilisation alone is not enough either: a resident
#   but idling job -- including OUR OWN previous launch -- reads 0% while still
#   holding the VRAM and, fatally, the dist-init port. So the probe also counts
#   processes on the GPUs and running `sglang serve` instances.
#

# Empty means "infer it from the candidate nodes" (decide). A default
# baked in here would be wrong on every rig but the one it came from; set the
# env only to override -- e.g. two shared networks, one of them the fast one.
CLUSTER_SUBNET = os.environ.get("GEAK_MN_CLUSTER_SUBNET", "")

# bracket class so pgrep -f cannot match the shell that carries the pattern:
# run_on_node passes the probe as `bash -c <script>`, so the pattern text is in
# that process's own cmdline and a literal pattern counts itself (observed:
# "2 schedulers" on a completely idle node).
# Both sglang spellings: the CLI (`sglang serve`) and the module entry point
# (`python -m sglang.launch_server`) that adapters/sglang.sh uses. Matching only
# the first is the quiet failure here -- teardown would report success while
# sixteen ranks kept the GPUs, and the next launch would blame the dist port.
SERVE_PATTERN = "(sglang[ ]serve|sglang[.]launch_server|vllm[ ]serve)"
SCHED_PATTERN = "sglang::[s]cheduler"


def state_path() -> str:
    explicit = os.environ.get("GEAK_MN_STATE", "")
    if explicit:
        return explicit
    shared = os.environ.get("GEAK_MN_EXT_SHARED_FS", "")
    if not shared:
        raise SystemExit(
            "!!! multinode: neither GEAK_MN_STATE nor GEAK_MN_EXT_SHARED_FS is set,\n"
            "    so there is nowhere to record which machines this run is pinned to."
        )
    return os.path.join(shared, "mn_state.json")


def load_state() -> dict:
    path = state_path()
    if not os.path.isfile(path):
        raise SystemExit(
            f"!!! multinode: no pinned node set at {path}.\n"
            "    Run mn.py preflight first -- launching without it would pick a\n"
            "    different pair per leg and make every A/B delta meaningless."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> str:
    path = state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def connect(address: str = "") -> None:
    if ray is None:
        raise SystemExit(
            "!!! multinode: ray is not installed in this interpreter, so the\n"
            "    machines cannot be reached. Only 'mn.py trace-gate' runs\n"
            "    without it. Check that this is the container's python3.")
    if ray.is_initialized():
        return
    ray.init(address=address or os.environ.get("GEAK_MN_RAY_ADDRESS", "auto"),
             log_to_driver=False)


def pinned(node_id: str):
    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)


def _node_ip() -> str:
    try:
        return ray.util.get_node_ip_address()
    except Exception:
        return socket.gethostbyname(socket.gethostname())


class _Task:
    """A function that runs on a pinned node, wrapped by ray.remote on FIRST USE.

    `@ray.remote` evaluated at import time would make the import itself require
    Ray, and the trace gate does not. Deferring the wrapper keeps every call site
    unchanged -- `probe_node.options(...).remote()` still reads the same -- while
    leaving the module importable without a cluster.
    """

    def __init__(self, fn):
        self._fn = fn
        self._wrapped = None
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def _ray(self):
        if self._wrapped is None:
            connect_check = ray is None
            if connect_check:
                raise SystemExit(f"!!! multinode: {self.__name__} needs ray")
            self._wrapped = ray.remote(num_cpus=1, num_gpus=0)(self._fn)
        return self._wrapped

    def options(self, **kwargs):
        return self._ray().options(**kwargs)

    @property
    def remote(self):
        return self._ray().remote


# num_gpus=0 on purpose. Asking Ray for GPUs makes it slice ROCR_VISIBLE_DEVICES
# to the bundle, and sglang would then see only the cards Ray handed this task.
# The flip side is that some Ray versions set an EMPTY visible-device list for a
# 0-GPU task, which leaves sglang with no cards at all -- so the spawn below
# strips those names from the environment before exec.
@_Task
def probe_node(ports: list | None = None) -> dict:
    def sh(cmd, timeout=90):
        return subprocess.run(["bash", "-c", cmd], capture_output=True,
                              text=True, timeout=timeout).stdout

    use = sh("rocm-smi --showuse 2>/dev/null || true")
    uses = [float(m.group(1)) for m in re.finditer(r"GPU use \(%\):\s*([0-9.]+)", use)]
    # Utilisation is a phase-dependent sample: a neighbour that is loading
    # weights or running torch inductor reads 0% while holding most of the HBM,
    # and one that is between requests reads 0% too. Free VRAM is the quantity
    # that actually decides whether our ranks can allocate, so carry it as well
    # and report the TIGHTEST card -- sglang gives every rank the same
    # mem-fraction, so the fullest GPU sets the budget for all sixteen.
    mem = sh("rocm-smi --showmeminfo vram --csv 2>/dev/null || true")
    free_mib = []
    for line in mem.splitlines()[1:]:
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 3 and cols[1].isdigit() and cols[2].isdigit():
            free_mib.append((int(cols[1]) - int(cols[2])) // (1024 * 1024))
    # The dist-init port is the thing a resident neighbour actually takes from
    # us, so ask about it directly instead of inferring it from a process count.
    busy_ports = []
    for p in (ports or []):
        if sh(f"ss -ltn 'sport = :{int(p)}' 2>/dev/null | tail -n +2 || true").strip():
            busy_ports.append(int(p))
    pids = sh("rocm-smi --showpids 2>/dev/null || true")
    resident_pids = re.findall(r"^\s*(\d+)\s+\S+\s+\d+\s+\d+", pids, re.M)
    resident = len(resident_pids)
    # rocm-smi reports HOST pids, so a pid with no /proc entry in OUR namespace
    # belongs to another container: a CO-TENANT, not a leftover of ours. The
    # distinction decides what to do -- our own leftovers are reaped and the
    # launch proceeds, a co-tenant means this machine is shared and the pin has
    # to move. Reporting "the previous instance did not clear" for a neighbour's
    # job sends you hunting for orphans you do not have.
    foreign = sum(1 for p in resident_pids if not os.path.exists(f"/proc/{p}"))
    serve = sh(f"pgrep -c -f '{SERVE_PATTERN}' || true").strip()
    # Report every IPv4 interface and let the DRIVER choose: a node cannot know
    # which network the other ranks are on (decide).
    addrs = []
    for line in sh("ip -4 -o addr show 2>/dev/null || true").splitlines():
        cols = line.split()
        if len(cols) >= 4:
            # Carry the MASK, not just the address: eight 400Gb NICs on
            # 192.168.15.x read as one shared fabric and are in fact eight
            # /31 point-to-point links that cannot reach each other.
            ip, _, plen = cols[3].partition("/")
            addrs.append([cols[1], ip, int(plen) if plen.isdigit() else 24])
    return {
        "ip": _node_ip(),
        "hostname": socket.gethostname(),
        "ngpu": len(uses),
        "max_use": max(uses) if uses else None,
        "gpu_pids": resident,
        "foreign_gpu_pids": foreign,
        "serve_procs": int(serve or 0),
        "free_vram_mib": min(free_mib) if free_mib else None,
        "busy_ports": busy_ports,
        "addrs": addrs,
        # Filled in by the driver once the fabric is decided for the whole set.
        "cluster_if": "",
        "cluster_ip": "",
    }


@_Task
def run_on_node(script: str, timeout: int = 300) -> dict:
    """Run a shell snippet on one pinned node and hand back what it said."""
    env = os.environ.copy()
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env.pop(var, None)
    proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True,
                          text=True, timeout=timeout)
    return {
        "hostname": socket.gethostname(),
        "ip": _node_ip(),
        "rc": proc.returncode,
        "out": (proc.stdout or "").strip(),
        "err": (proc.stderr or "").strip(),
    }


@_Task
def spawn_on_node(script: str) -> dict:
    """Start the server and let it outlive this task.

    start_new_session detaches it from the Ray worker, so Ray is out of the
    picture the moment the pid exists -- which is also why a zero return code
    here says nothing about whether the server came up.
    """
    env = os.environ.copy()
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env.pop(var, None)
    proc = subprocess.Popen(["bash", "-c", script], env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate(timeout=300)
    return {
        "hostname": socket.gethostname(),
        "ip": _node_ip(),
        "rc": proc.returncode,
        "out": (out or b"").decode("utf-8", "replace").strip(),
    }


def alive_gpu_nodes(gpus_per_node: int) -> list[dict]:
    return [
        n for n in ray.nodes()
        if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) >= gpus_per_node
    ]


def resolve_pinned(state: dict) -> list[dict]:
    """Map the recorded node ids back onto the live cluster.

    A missing node is a hard stop rather than a re-pick: silently moving the run
    to a different machine mid-experiment is exactly the kind of "it still
    produced a number" failure that makes the number worthless.
    """
    live = {n["NodeID"]: n for n in ray.nodes() if n.get("Alive")}
    out = []
    for node in state["nodes"]:
        nid = node["node_id"]
        if nid not in live:
            raise SystemExit(
                f"!!! multinode: pinned rank{node['rank']} node {node['hostname']} "
                f"({node['ip']}) is no longer Alive in the cluster.\n"
                "    Refusing to substitute another machine -- the A/B legs would\n"
                "    no longer be comparable. Re-run `mn.py preflight` to re-pin and\n"
                "    restate the baseline."
            )
        out.append({**node, "_ray": live[nid]})
    return out


def teardown(state: dict, wait_idle_s: int = 180, verbose=print) -> bool:
    """Stop every rank of a previous launch and wait for the cards to clear.

    This runs before every launch, not only on failure. The documented symptom
    of skipping it is specific and misleading: the previous scheduler keeps the
    dist-init port, the new rank0 dies with "dist_init_port ... is not available
    in 30 seconds", the new rank1 hangs as an orphan, and /health keeps
    answering 200 from the old instance the whole time.
    """
    nodes = resolve_pinned(state)
    kill = (
        f"pkill -f '{SERVE_PATTERN}' 2>/dev/null || true; "
        "pkill -f 'sglang::' 2>/dev/null || true; "
        "sleep 3; "
        f"pkill -9 -f '{SERVE_PATTERN}' 2>/dev/null || true; "
        "pkill -9 -f 'sglang::' 2>/dev/null || true; "
        "echo killed"
    )
    ray.get([
        run_on_node.options(scheduling_strategy=pinned(n["node_id"])).remote(kill, 120)
        for n in nodes
    ])

    shared_ok = {h.upper() for h in state.get("allow_shared", [])}

    def still_ours(p) -> bool:
        """Is anything WE have to clear still on these cards?"""
        stem = p["hostname"].split(".")[0].upper()
        if shared_ok & {"*", stem, stem.split("-")[-1]}:
            # On a box the operator accepted as shared, the neighbour's pids and
            # their utilisation are not ours to reap and waiting them out just
            # burns the window. What must be gone is our OWN previous instance:
            # the pids visible in this container's namespace (rocm-smi reports
            # host pids, so a foreign one has no /proc entry here) and any
            # sglang serve this namespace can see.
            ours = (p["gpu_pids"] or 0) - (p.get("foreign_gpu_pids") or 0)
            return bool(p["serve_procs"]) or ours > 0
        return bool(p["serve_procs"] or p["gpu_pids"] or (p["max_use"] or 0) > 0)

    deadline = time.time() + wait_idle_s
    while True:
        probes = ray.get([
            probe_node.options(scheduling_strategy=pinned(n["node_id"])).remote()
            for n in nodes
        ])
        busy = [
            f"{p['hostname'].split('.')[0]}(use={p['max_use']} pids={p['gpu_pids']}"
            + (f" of which {p['foreign_gpu_pids']} from another container"
               if p.get("foreign_gpu_pids") else "")
            + f" serve={p['serve_procs']})"
            for p in probes
            if still_ours(p)
        ]
        if not busy:
            verbose(">>> multinode teardown: both nodes clear")
            return True
        if time.time() > deadline:
            # A co-tenant will never clear on our account, so say so instead of
            # letting the caller wait out the full window blaming itself.
            foreign = [p["hostname"].split(".")[0] for p in probes
                       if p.get("foreign_gpu_pids")]
            verbose(f"!!! multinode teardown: still busy after {wait_idle_s}s: "
                    + ", ".join(busy)
                    + (f"; {', '.join(foreign)} has a co-tenant we cannot reap"
                       if foreign else ""))
            return False
        verbose(">>> multinode teardown: waiting on " + ", ".join(busy))
        time.sleep(10)


def serve_counts(state: dict) -> list[int]:
    nodes = resolve_pinned(state)
    results = ray.get([
        run_on_node.options(scheduling_strategy=pinned(n["node_id"])).remote(
            f"pgrep -c -f '{SERVE_PATTERN}' || true", 90)
        for n in nodes
    ])
    return [int((r["out"] or "0").strip() or 0) for r in results]


def engine_health(state: dict) -> list[dict]:
    """Per rank: is the launcher there, and are its workers still alive?

    Counting `sglang serve` alone is not enough, and the gap is not theoretical
    -- it cost three hours on this rig. When the schedulers abort, the parent
    process and the HTTP layer survive: /health keeps answering 200, `pgrep`
    keeps finding the server, and a client sends requests that will never be
    answered until something times out. What actually changed was the workers:
    eight `sglang::scheduler_*` processes had become zombies.

    So the workers are counted too, and a zombie among them is reported as the
    engine being dead however healthy the front end looks.
    """
    nodes = resolve_pinned(state)
    probe = (
        f"echo serve=$(pgrep -c -f '{SERVE_PATTERN}' || true); "
        f"echo sched=$(pgrep -c -f '{SCHED_PATTERN}' || true); "
        "echo zombie=$(ps -eo stat=,args= | awk '/sglang::/ && $1 ~ /Z/' | wc -l)"
    )
    results = ray.get([
        run_on_node.options(scheduling_strategy=pinned(n["node_id"])).remote(probe, 90)
        for n in nodes
    ])
    out = []
    for node, res in zip(nodes, results):
        vals = dict(
            part.split("=", 1) for part in (res["out"] or "").split() if "=" in part
        )
        def num(key):
            try:
                return int(vals.get(key, "0") or 0)
            except ValueError:
                return 0
        out.append({
            "rank": node["rank"],
            "host": node["hostname"].split(".")[0],
            "serve": num("serve"),
            "schedulers": num("sched"),
            "zombies": num("zombie"),
        })
    return out


# --------------------------------------------------------------------------
# PREFLIGHT  (mn.py preflight)
# --------------------------------------------------------------------------
#
# Multi-node preflight: pin the machines ONCE and prove the handoff is real.
#
# Runs in Setup, before the baseline. Everything it checks is something that
# otherwise fails later as a plausible-looking number rather than as an error:
#
# * No pinned node set -> each leg picks its own pair -> the A/B measures which
#   machines were free, not the change under test.
# * A "shared" mount that is not actually shared -> the overlay reaches rank0
#   only, rank1 serves stock code, and the run reports a speedup for a kernel
#   that half the ranks never loaded. Proven with a marker written from each
#   node, because `df` showing the same size proves nothing.
# * A resident-but-idle neighbour -> 0% GPU use, so a naive idleness check picks
#   the box, and the launch then dies on the dist-init port.
#
# Writes mn_state.json (the pin) and mn_preflight.json (the evidence, for
# env_report.json).
#
# Usage:
#   python mn.py preflight [--nnodes 2] [--gpus-per-node 8] [--serve-port 30890]
#                          [--dist-init-port 30531] [--repin]
#

def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_handoff() -> dict:
    """What the caller must provide for a distributed run to be possible.

    The shared filesystem is the hard requirement: it is where the overlay, the
    logs and the traces live, and it is the reason no step of this launcher has
    to ship files between nodes.

    A service URL is NOT required. GEAK starts its own server, so the endpoint
    does not exist until after the launch; it is recorded only when the caller
    handed us an already-running service to attach to (the mode Hyperloom uses
    for a disaggregated deployment), and is informational here.
    """
    service_url = os.environ.get("GEAK_MN_EXT_SERVICE_URL", "").strip()
    shared = os.environ.get("GEAK_MN_EXT_SHARED_FS", "").strip()
    problems = []
    if not shared:
        problems.append(
            "GEAK_MN_EXT_SHARED_FS is unset -- there is no filesystem every rank can\n"
            "        read, so the overlay would reach one rank and the traces of the\n"
            "        others would be unreadable")
    elif not os.path.isdir(shared):
        problems.append(f"GEAK_MN_EXT_SHARED_FS={shared} is not a directory here")
    return {"service_url": service_url, "shared_fs": shared, "problems": problems}


def prove_shared(shared: str, nodes: list[dict]) -> dict:
    """A mount is shared only if a file written on one node is readable on the
    others. Do it in both directions: a read-only-from-here mount would pass a
    one-way check and then swallow every trace the server writes."""
    token = uuid.uuid4().hex[:12]
    probe_dir = os.path.join(shared, ".mn_probe")
    os.makedirs(probe_dir, exist_ok=True)
    here = socket.gethostname().split(".")[0]

    local_marker = os.path.join(probe_dir, f"from_bench_{token}")
    with open(local_marker, "w", encoding="utf-8") as f:
        f.write(token)

    results = {"token": token, "bench_host": here, "per_node": [], "shared": True}
    for node in nodes:
        script = (
            f"set -e; "
            f"test -r {local_marker} && echo READ_OK=$(cat {local_marker}) || echo READ_FAIL; "
            f"echo {token} > {probe_dir}/from_rank{node['rank']}_{token} && echo WRITE_OK || echo WRITE_FAIL"
        )
        out = ray.get(
            run_on_node.options(scheduling_strategy=pinned(node["node_id"])).remote(script, 120)
        )
        readable = f"READ_OK={token}" in out["out"]
        wrote = "WRITE_OK" in out["out"]
        seen_here = os.path.isfile(os.path.join(probe_dir, f"from_rank{node['rank']}_{token}"))
        ok = readable and wrote and seen_here
        results["per_node"].append({
            "rank": node["rank"],
            "hostname": node["hostname"],
            "reads_bench_marker": readable,
            "can_write": wrote,
            "its_marker_visible_to_bench": seen_here,
            "ok": ok,
        })
        results["shared"] = results["shared"] and ok

    for name in os.listdir(probe_dir):
        if token in name:
            try:
                os.remove(os.path.join(probe_dir, name))
            except OSError:
                pass
    return results


def pick(nnodes: int, gpus_per_node: int, ports: list | None = None,
         allow_shared: set | None = None,
         min_free_vram_mib: int = 0) -> tuple[list[dict], list[dict], dict]:
    candidates = alive_gpu_nodes(gpus_per_node)
    if len(candidates) < nnodes:
        raise SystemExit(
            f"!!! multinode: cluster has {len(candidates)} node(s) with >= "
            f"{gpus_per_node} GPUs, need {nnodes}."
        )
    probes = ray.get(
        [probe_node.options(scheduling_strategy=pinned(n["NodeID"])).remote(ports or [])
         for n in candidates],
        timeout=600,
    )
    # Decided across the whole candidate set, not per node: only the set knows
    # which network its members have in common, and until Ray answered above
    # none of these addresses existed as far as this process was concerned.
    per_node_addrs = [[tuple(a) for a in p.get("addrs", [])] for p in probes]
    fabric = decide(per_node_addrs, CLUSTER_SUBNET)
    for probe, addrs in zip(probes, per_node_addrs):
        probe["cluster_if"], probe["cluster_ip"] = pick_addr(
            addrs, fabric["subnet"])
    if fabric["subnet"]:
        how = ("GEAK_MN_CLUSTER_SUBNET" if fabric["how"] == "configured"
               else f"inferred, saw {sorted(fabric['candidates'])}")
        print(f">>> multinode: fabric {fabric['subnet']}x ({how}), on "
              f"{fabric['nodes_covered']}/{len(candidates)} candidate(s)")

    report, idle = [], []
    allow_shared = allow_shared or set()
    for node, probe in zip(candidates, probes):
        reasons = []
        # Match on the trailing segment too: these boxes report
        # "smc300x-ccs-aus-gpuf293", and an operator names them "GPUF293".
        stem = probe["hostname"].split(".")[0].upper()
        short = stem.split("-")[-1]
        # Opting a machine in here says "I accept a co-tenant on this box". It
        # does not waive the checks a co-tenant can still make us fail on: the
        # port they might hold and the HBM they have already taken are asked
        # about directly below, because those are what kill the launch. What it
        # waives is the coarse stand-in for both -- utilisation and pid count --
        # which on a 7-way shared rig is never zero for long.
        shared_ok = bool(allow_shared & {"*", short, stem})
        warn = []
        if probe["max_use"] is None:
            reasons.append("no rocm-smi reading")
        elif probe["max_use"] > 0:
            (warn if shared_ok else reasons).append(f"gpu busy {probe['max_use']}%")
        if probe["gpu_pids"]:
            (warn if shared_ok else reasons).append(
                f"{probe['gpu_pids']} process(es) resident on the GPUs"
                + (f" ({probe['foreign_gpu_pids']} of them another container's,"
                   " so this machine is shared)"
                   if probe.get("foreign_gpu_pids") else ""))
        if probe["serve_procs"]:
            (warn if shared_ok else reasons).append(
                f"{probe['serve_procs']} sglang serve already running")
        if probe.get("busy_ports"):
            # Never waived: two servers cannot bind the same port, and the rank
            # that loses reports it 20 minutes into a load-weight phase.
            reasons.append("port(s) already bound: "
                           + ",".join(str(p) for p in probe["busy_ports"]))
        free = probe.get("free_vram_mib")
        if min_free_vram_mib and free is not None and free < min_free_vram_mib:
            # Also never waived: every rank gets the same mem-fraction, so the
            # fullest card decides, and too little there is an OOM at load time.
            reasons.append(f"tightest GPU has {free} MiB free, need "
                           f"{min_free_vram_mib} MiB")
        if not probe["cluster_ip"]:
            reasons.append(
                f"no {fabric['subnet']}x address (not on the fabric)"
                if fabric["subnet"] else "no subnet shared with the other candidates")
        report.append({**probe, "node_id": node["NodeID"], "accepted_shared": bool(warn),
                       "waived": warn,
                       "verdict": "idle" if not reasons else "; ".join(reasons)})
        print(f"candidate {probe['ip']:<16} gpus={probe['ngpu']} max_use={probe['max_use']} "
              f"gpu_pids={probe['gpu_pids']} serve={probe['serve_procs']} "
              f"free_vram={free if free is not None else '?'}MiB "
              f"if={probe['cluster_if'] or '-'} -> "
              + ("idle" if not reasons else "skipped: " + "; ".join(reasons)))
        if warn and not reasons:
            # Say it out loud on every pin: a number measured next to someone
            # else's job is not a baseline, and six weeks later the log is the
            # only thing that remembers the box was shared.
            print(f"    !!! {short} ACCEPTED AS SHARED, waiving: " + "; ".join(warn)
                  + "\n        Measurements from this pin are contended -- usable as a"
                    " smoke test, not as a baseline.")
        if not reasons:
            idle.append((node, probe))

    if len(idle) < nnodes:
        # Name the reason that did the excluding: "0 idle node(s)" on a free
        # cluster sends the reader to look at GPU usage, when what happened was
        # that no candidate had an address on the network they must share.
        if all(not r["cluster_ip"] for r in report):
            why = (f"none has an address on {fabric['subnet']}x"
                   if fabric["subnet"] else
                   "no subnet is shared by two or more of them")
            detail = (
                f"    Every candidate was excluded for the same reason: {why}.\n"
                "    What they have:\n"
                + describe(per_node_addrs,
                                  [r["hostname"].split(".")[0] for r in report])
                + "\n    Set GEAK_MN_CLUSTER_SUBNET to the fabric prefix (e.g.\n"
                  "    10.0.1.) if they share a network this could not see."
            )
        else:
            detail = (
                "    A resident-but-idle job reads 0% GPU while still holding the\n"
                "    dist-init port, so this refuses rather than launching into a\n"
                "    rank0 that will die with 'dist_init_port not available'."
            )
        raise SystemExit(
            f"!!! multinode: only {len(idle)} usable node(s), need {nnodes}.\n"
            + detail)

    # Rank order follows the cluster IP so that a re-pin on the same machines
    # yields the same rank assignment, which keeps logs and traces comparable
    # across runs.
    idle.sort(key=lambda t: [int(o) for o in t[1]["cluster_ip"].split(".")])
    picked = [
        {"rank": i, "node_id": node["NodeID"], "ip": probe["cluster_ip"],
         "hostname": probe["hostname"], "cluster_if": probe["cluster_if"],
         "ngpu": probe["ngpu"]}
        for i, (node, probe) in enumerate(idle[:nnodes])
    ]
    return picked, report, fabric


def cmd_preflight(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nnodes", type=int, default=int(os.environ.get("GEAK_MN_NODES", 2)))
    ap.add_argument("--gpus-per-node", type=int,
                    default=int(os.environ.get("GEAK_MN_GPUS_PER_NODE", 8)))
    # 30890: inside bench_e2e.sh's required 30000..30999 window (the rig policy
    # "every port starts with 30"). A port outside it is clamped and re-allocated
    # by the caller, which would leave the client talking to a port no rank binds.
    ap.add_argument("--serve-port", type=int, default=int(os.environ.get("GEAK_MN_SERVE_PORT", 30890)))
    ap.add_argument("--dist-init-port", type=int,
                    default=int(os.environ.get("GEAK_MN_DIST_INIT_PORT", 30531)))
    ap.add_argument("--repin", action="store_true",
                    help="re-select machines even if a pin exists; invalidates the baseline")
    ap.add_argument("--allow-shared", default=os.environ.get("GEAK_MN_ALLOW_SHARED", ""),
                    help="comma-separated short hostnames (or *) to accept WITH a "
                         "co-tenant: waives the utilisation and resident-pid checks "
                         "on those boxes, never the port or free-VRAM ones. The "
                         "resulting numbers are contended.")
    ap.add_argument("--min-free-vram-mib", type=int,
                    default=int(os.environ.get("GEAK_MN_MIN_FREE_VRAM_MIB", 0)),
                    help="refuse a node whose fullest GPU has less than this free; "
                         "the fullest card sets the mem-fraction for all ranks")
    args = ap.parse_args(argv)

    handoff = check_handoff()
    if handoff["problems"]:
        sys.stderr.write("!!! multinode preflight: the cluster handoff is incomplete:\n")
        for p in handoff["problems"]:
            sys.stderr.write(f"      - {p}\n")
        return 2

    path = state_path()
    if os.path.isfile(path) and not args.repin:
        state = load_state()
        print(f">>> multinode: reusing the existing pin at {path}")
        for n in state["nodes"]:
            print(f"    rank{n['rank']} {n['hostname'].split('.')[0]} ({n['ip']})")
        connect(state.get("ray_address", "auto"))
        resolve_pinned(state)          # hard-fails if a pinned node vanished
        print(">>> multinode: every pinned node is still Alive")
        return 0

    connect()
    allow_shared = {h.strip().upper() for h in args.allow_shared.split(",") if h.strip()}
    picked, report, fabric = pick(
        args.nnodes, args.gpus_per_node,
        ports=[args.serve_port, args.dist_init_port],
        allow_shared=allow_shared, min_free_vram_mib=args.min_free_vram_mib)
    shared_proof = prove_shared(handoff["shared_fs"], picked)

    state = {
        "nnodes": args.nnodes,
        "gpus_per_node": args.gpus_per_node,
        "nodes": picked,
        "master_ip": picked[0]["ip"],
        "serve_port": args.serve_port,
        "dist_init_port": args.dist_init_port,
        # Carried in the pin because the LAUNCH path has to know it too: the
        # pre-launch teardown waits for the cards to come clear, and a co-tenant
        # never will. Without this the pin succeeds and every launch times out.
        "allow_shared": sorted(allow_shared),
        "ray_address": os.environ.get("GEAK_MN_RAY_ADDRESS", "auto"),
        "shared_fs": handoff["shared_fs"],
        "service_url": handoff["service_url"],
        # The DECIDED fabric, not the (usually empty) override: every rank
        # re-derives its own NIC from this prefix at launch, so the pin has to
        # carry the answer rather than the question.
        "cluster_subnet": fabric["subnet"],
        "pinned_at": _utc(),
    }

    evidence = {
        "handoff": {k: v for k, v in handoff.items() if k != "problems"},
        "candidates": report,
        "picked": picked,
        # Part of the evidence, not a footnote: whether a co-tenant was accepted
        # is the first thing to check when a later A/B refuses to reproduce.
        "shared_accepted": sorted(allow_shared),
        "min_free_vram_mib": args.min_free_vram_mib,
        "master_ip": state["master_ip"],
        "shared_fs_proof": shared_proof,
        "topology": {"nnodes": args.nnodes, "gpus_per_node": args.gpus_per_node,
                     "world_size": args.nnodes * args.gpus_per_node},
        # How the fabric was decided, so a run on a new rig can be audited for
        # having used the intended network rather than the first one that
        # happened to match.
        "fabric": fabric,
        "checked_at": _utc(),
    }
    with open(os.path.join(handoff["shared_fs"], "mn_preflight.json"), "w",
              encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)

    if not shared_proof["shared"]:
        sys.stderr.write(
            f"!!! multinode preflight: {handoff['shared_fs']} is NOT shared across the\n"
            "    picked nodes (marker exchange failed). Without it the overlay reaches\n"
            "    rank0 only and the profile traces from the other ranks are invisible,\n"
            "    which shows up as a speedup rather than as an error.\n")
        for n in shared_proof["per_node"]:
            sys.stderr.write(f"      rank{n['rank']} {n['hostname'].split('.')[0]}: {n}\n")
        return 2

    saved = save_state(state)
    print(f">>> multinode: pinned {args.nnodes} node(s) -> {saved}")
    for n in picked:
        print(f"    rank{n['rank']} {n['hostname'].split('.')[0]} ({n['ip']}) if={n['cluster_if']}")
    print(f">>> master={state['master_ip']} serve_port={state['serve_port']} "
          f"dist_init_port={state['dist_init_port']}")
    print(f">>> shared fs proven across every rank: {handoff['shared_fs']}")
    return 0


# --------------------------------------------------------------------------
# LAUNCH  (mn.py launch)
# --------------------------------------------------------------------------
#
# Start one server instance whose TP group spans the pinned nodes.
#
# Reads the same environment the single-host adapters read (MODEL / TP /
# MEM_FRACTION / EXTRA_SERVER_ARGS / EXTRA_ENV / OVERLAY_PYTHONPATH / PROFILE_DIR
# / LOG), renders serve.sh.tmpl once per rank, and ships each one to its pinned
# node through Ray. Returning 0 means the processes were SPAWNED -- nothing more.
# Readiness is bench_e2e.sh's health loop, which is the only place that can tell
# a slow cold start from a wedged one.
#
# Nothing here knows how to start a server. The rendered script sources
# adapters/$BACKEND.sh on the far node and asks it for the launch line and for the
# way that framework spells "I am rank i of n" (adapter_dist_args). This file only
# supplies node facts. That split is why adding vllm is a function in vllm.sh
# rather than a second launcher.
#
# What this refuses to do, and why each refusal exists:
#
# * Launch without tearing the previous instance down. The old scheduler keeps the
#   dist-init port; the new rank0 then dies with "dist_init_port ... not available
#   in 30 seconds" while /health still answers 200 from the corpse.
# * Put $LOG or $PROFILE_DIR anywhere but the proven-shared mount. bench_e2e.sh
#   greps $LOG for OOM/fatal markers and watches its size for stalls, and the
#   sglang profiler writes traces on whichever node the rank lives on. Off the
#   shared mount, both silently read an empty local file of the same name.
# * Let a swept environment drop SYNC_TOKEN_IDS_ACROSS_TP. Without it the
#   attention-TP group has no pynccl, the token-id sync falls back to
#   torch.distributed, and the NCCL watchdog queries an event recorded inside a
#   decode CUDA graph -- every rank dies with hipErrorCapturedEvent during
#   capture. That reads as a kernel bug, not as a missing env var.
#

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.environ.get("GEAK_MN_SERVE_TEMPLATE", os.path.join(HERE, "serve.sh.tmpl"))

# Run-scoped names the launcher owns. A GPU mask arriving through EXTRA_ENV
# would pin every rank to one card; the rest are set from this run's config and
# must not be shadowed by a swept value.
_OWNED = {
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "MASTER_ADDR", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    "PYTHONPATH", "SGLANG_TORCH_PROFILER_DIR",
}




def accepted_env(raw: str) -> tuple[list[str], list[str]]:
    """Keep only strict IDENTIFIER=VALUE tokens, and drop the ones this launcher
    owns. Same filter as adapters/launchers/magpie.sh: these tokens become
    operands of `env`, so a bare word would be read as the command to run."""
    kept, dropped = [], []
    for tok in (raw or "").split():
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            dropped.append(f"{tok} (not IDENTIFIER=VALUE; would be an env option/command)")
            continue
        name = tok.split("=", 1)[0]
        if name in _OWNED:
            dropped.append(f"{tok} (run-scoped name owned by the multinode launcher)")
            continue
        kept.append(tok)
    return kept, dropped


def resolve_adapter(backend: str, shared: str) -> str:
    """Locate the backend adapter and make sure it can actually speak multi-node.

    Both refusals below are things that would otherwise surface as sixteen ranks
    dying at once on the far side of a Ray task, where the real cause (a missing
    shell function, a file the node cannot see) is invisible."""
    path = os.path.realpath(
        os.environ.get("GEAK_MN_BACKEND_ADAPTER")
        or os.path.join(HERE, os.pardir, "adapters", f"{backend}.sh"))
    if not os.path.isfile(path):
        raise SystemExit(f"!!! multinode: no backend adapter at {path} (BACKEND={backend})")
    if not under(path, shared):
        raise SystemExit(
            f"!!! multinode: the backend adapter {path} is not under the shared\n"
            f"    mount {shared}, but every rank sources it on its own node.\n"
            "    Put the GEAK tree on the shared mount, or point\n"
            "    GEAK_MN_BACKEND_ADAPTER at a copy that is.")
    probe = subprocess.run(
        ["bash", "-c",
         f'. {shlex.quote(path)} >/dev/null 2>&1 || exit 3\n'
         'declare -F adapter_dist_args >/dev/null || { echo none; exit 0; }\n'
         'if declare -F adapter_dist_mode >/dev/null; then adapter_dist_mode; '
         'else echo per_rank; fi'],
        capture_output=True, text=True)
    mode = probe.stdout.strip()
    if probe.returncode != 0 or not mode:
        raise SystemExit(f"!!! multinode: could not read {path}:\n{probe.stderr.strip()}")
    if mode == "none":
        raise SystemExit(
            f"!!! multinode: {path} defines no adapter_dist_args, so it cannot say\n"
            f"    how {backend} expresses rank/world/rendezvous. Add it there --\n"
            "    the launcher is framework-agnostic by design and will not guess.")
    if mode != "per_rank":
        raise SystemExit(
            f"!!! multinode: {backend} declares adapter_dist_mode={mode}; this\n"
            "    launcher implements per_rank only (one server process per node).\n"
            f"    {mode} means the framework fans out to the other nodes itself\n"
            "    from a single process, which needs a different spawn path and\n"
            "    has not been validated on a real rig. Refusing rather than\n"
            "    launching sglang's shape and calling it that framework's.")
    return path


def under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == \
            os.path.realpath(root)
    except ValueError:
        return False


def render(template: str, subs: dict) -> str:
    out = template
    for key, val in subs.items():
        out = out.replace(f"__{key}__", val)
    left = sorted(set(re.findall(r"__[A-Z_]+__", out)))
    if left:
        raise SystemExit(
            f"!!! multinode: serve template still has unfilled placeholders {left}.\n"
            "    Refusing to launch: an unrendered placeholder means the served\n"
            "    config is not the one under test."
        )
    return out


def check_script(rank: int, script: str) -> None:
    """Catch a broken render HERE, not on the far side of a Ray task.

    A rank that dies on a malformed launch line leaves the same evidence as a
    model that failed to load -- an empty log and a health loop that runs to its
    ceiling -- so the cheap checks are worth running before the spawn. `\\ ` is
    called out by name because it is the failure this render already had once:
    an escaped space becomes a single-space argument, which `env` reads as the
    command to run.
    """
    if "\\ \\" in script:
        raise SystemExit(
            f"!!! multinode: rank{rank}'s rendered script contains an escaped space"
            " (`\\ \\`).\n    `env` would read that as the command to run and the"
            " rank would die instantly."
        )
    proc = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"!!! multinode: rank{rank}'s rendered script is not valid bash:\n"
            f"{proc.stderr.strip()}"
        )


def cmd_launch(argv: list) -> int:
    state = load_state()
    shared = state["shared_fs"]

    model = os.environ.get("MODEL", "")
    if not model:
        raise SystemExit("!!! multinode: MODEL is required")
    tp = int(os.environ.get("TP", state["nnodes"] * state["gpus_per_node"]))
    # Per-node device list: each box names its own cards 0..7 no matter how wide
    # the TP group is.
    gpu = os.environ.get("GPU") or ",".join(str(i) for i in range(state["gpus_per_node"]))
    mem_fraction = os.environ.get("MEM_FRACTION", "0.90")
    extra_args = (os.environ.get("EXTRA_SERVER_ARGS", "") or "").strip()
    # The model recipe (attention backend, kv dtype, speculative decoding, dp
    # width...). It is the caller's, not the launcher's: baking a model's flags
    # into the launcher is what makes a launcher single-model.
    recipe_args = (os.environ.get("GEAK_MN_RECIPE_ARGS", "") or "").strip()
    adapter = resolve_adapter(os.environ.get("BACKEND", "sglang"), shared)
    overlay = (os.environ.get("OVERLAY_PYTHONPATH", "") or "").strip()
    profile_dir = os.environ.get("PROFILE_DIR", "")
    log = os.environ.get("LOG", "")
    out_dir = os.environ.get("OUT_DIR", os.path.dirname(log) or shared)

    if tp != state["nnodes"] * state["gpus_per_node"]:
        print(f">>> multinode: TP={tp} does not fill the pinned topology "
              f"({state['nnodes']}x{state['gpus_per_node']}); serving it anyway "
              "because the caller may be sweeping TP.", file=sys.stderr)

    for name, path in (("LOG", log), ("PROFILE_DIR", profile_dir)):
        if not path:
            raise SystemExit(f"!!! multinode: {name} is required")
        if not under(path, shared):
            raise SystemExit(
                f"!!! multinode: {name}={path} is not under the shared mount {shared}.\n"
                "    Every rank writes there and the benchmark host reads it back;\n"
                "    off the shared mount each node would get its own empty copy and\n"
                "    the stall detector / trace collection would read nothing.\n"
                f"    Put OUT_DIR under {shared}."
            )

    # Recipe env first, GEAK's swept env after it: `env A=1 A=2` is last-wins, so
    # this is the same "recipe is the BASE layer" ordering used for the args.
    kept_env, dropped_env = accepted_env(
        f'{os.environ.get("GEAK_MN_RECIPE_ENV", "")} {os.environ.get("EXTRA_ENV", "")}')
    for tok in dropped_env:
        print(f">>> multinode: dropping EXTRA_ENV token: {tok}", file=sys.stderr)
    # Whether to SET this is the recipe's business (it is an sglang name, and this
    # launcher is framework-agnostic). Whether a sweep may turn it OFF is not --
    # hence last-wins here too, matching what `env` will actually do.
    _sync = [t.split("=", 1)[1] for t in kept_env if t.startswith("SYNC_TOKEN_IDS_ACROSS_TP=")]
    if _sync and _sync[-1] != "1":
        raise SystemExit(
            "!!! multinode: EXTRA_ENV turns SYNC_TOKEN_IDS_ACROSS_TP off.\n"
            "    With TP spanning nodes and dp != tp, the attention-TP group only\n"
            "    gets pynccl when this is set; without it the NCCL watchdog queries\n"
            "    an event captured inside a decode CUDA graph and every rank aborts\n"
            "    with hipErrorCapturedEvent during capture. Refusing: the crash\n"
            "    would be attributed to whatever kernel was being tested."
        )

    # OVERLAY_PYTHONPATH and PYTHONPATH travel separately: composing them is the
    # adapter's job (sglang.sh also has to slot its own source tree in between,
    # because its editable install can miss sglang.benchmark).
    pythonpath = os.environ.get("PYTHONPATH", "")

    launch_id = uuid.uuid4().hex[:10]
    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()

    scripts = {}
    for node in state["nodes"]:
        rank = node["rank"]
        # rank0's log IS $LOG: bench_e2e.sh already treats that path as "the
        # server's log", and on a shared mount there is nothing to copy back.
        # A fresh path per launch also sidesteps the truncation trap -- reusing
        # one filename leaves a NUL-padded sparse file when an old fd keeps
        # writing at its former offset.
        rank_log = log if rank == 0 else os.path.join(out_dir, f"server.rank{rank}.log")
        scripts[rank] = render(template, {
            "LAUNCH_ID": launch_id,
            "RANK": str(rank),
            "NNODES": str(state["nnodes"]),
            "MASTER": state["master_ip"],
            "PORT": str(state["serve_port"]),
            "DIST_PORT": str(state["dist_init_port"]),
            "MODEL": shlex.quote(model),
            "TP": str(tp),
            "GPU": gpu,
            "MEM_FRACTION": str(mem_fraction),
            "PROFILE_DIR": profile_dir,
            "LOG": rank_log,
            "ADAPTER": adapter,
            "OVERLAY": overlay,
            "PYTHONPATH": pythonpath,
            # One line, plain spaces: the adapter takes EXTRA_ENV as a single
            # string and word-splits it into operands of `env`.
            "EXTRA_ENV": " ".join(kept_env),
            "RECIPE_ARGS": recipe_args,
            "EXTRA_ARGS": extra_args,
            "CLUSTER_SUBNET": state.get("cluster_subnet", CLUSTER_SUBNET),
        })

    for rank, script in scripts.items():
        check_script(rank, script)

    # Keep the exact scripts that were run next to the results. "What was served"
    # is otherwise unanswerable after the fact, and it is the first question any
    # surprising number raises.
    for rank, script in scripts.items():
        with open(os.path.join(out_dir, f"mn_rank{rank}.sh"), "w", encoding="utf-8") as f:
            f.write(script)

    # The dry run is gated BEFORE anything touches the cluster: rendering is the
    # part worth checking on a live rig, and a dry run that tore down the running
    # server to prove its templating would destroy the measurement in progress.
    if os.environ.get("GEAK_MN_DRY_RUN") == "1":
        # Direct CLI use only: it sets no SERVER_PID, so the launcher adapter
        # must not use it.
        print(">>> multinode: dry run -- rendered scripts written, nothing launched")
        for rank in sorted(scripts):
            print(f"--- rank{rank} ---\n{scripts[rank]}")
        return 0

    connect(state.get("ray_address", "auto"))
    # Before every launch, not only after a failure: the previous scheduler holds
    # the dist-init port while /health still answers 200 from it.
    if not teardown(state):
        raise SystemExit(
            "!!! multinode: the pinned nodes did not come clear. Launching now would\n"
            "    hit the held dist-init port and produce an orphaned rank1.\n"
            "    If the lines above name a co-tenant, the GPUs belong to another\n"
            "    container and no amount of reaping here will free them: re-pin with\n"
            "    'mn.py preflight --repin' to move onto an idle pair."
        )

    nodes = resolve_pinned(state)
    results = ray.get([
        spawn_on_node.options(scheduling_strategy=pinned(n["node_id"])).remote(
            scripts[n["rank"]])
        for n in nodes
    ])

    summary = {
        "launch_id": launch_id,
        "launched_at": _utc(),
        "master": state["master_ip"],
        "serve_port": state["serve_port"],
        "dist_init_port": state["dist_init_port"],
        "base_url": f"http://{state['master_ip']}:{state['serve_port']}",
        "tp": tp, "gpu": gpu, "nnodes": state["nnodes"],
        "world_size": state["nnodes"] * state["gpus_per_node"],
        "log": log,
        "profile_dir": profile_dir,
        "overlay_pythonpath": overlay,
        # Both layers, separately: "what was served" is unanswerable later if the
        # recipe and the swept flags are recorded as one merged string.
        "backend_adapter": adapter,
        "recipe_args": recipe_args,
        "extra_server_args": extra_args,
        "extra_env_accepted": kept_env,
        "extra_env_dropped": dropped_env,
        "ranks": [
            {"rank": n["rank"], "hostname": r["hostname"], "ip": r["ip"],
             "rc": r["rc"], "out": r["out"]}
            for n, r in zip(nodes, results)
        ],
    }
    with open(os.path.join(out_dir, "mn_launch.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    # Which launch currently owns the cluster. `mn.py reap` reads it before
    # acting, so a reaper left over from a previous generation cannot tear down
    # the instance that replaced the one it was watching.
    with open(os.path.join(shared, "mn_live_launch"), "w", encoding="utf-8") as f:
        f.write(launch_id)
    for r in summary["ranks"]:
        print(f">>> multinode rank{r['rank']} on {r['hostname'].split('.')[0]} "
              f"rc={r['rc']} :: {r['out'].splitlines()[-1] if r['out'] else ''}")
    print("MN_LAUNCH_RESULT=" + json.dumps(summary))

    if any(r["rc"] != 0 for r in summary["ranks"]):
        return 1

    # A spawn that produced no process is a launch failure we can catch now,
    # rather than letting the health loop spend its whole stall window on it.
    counts = serve_counts(state)
    print(f">>> multinode: serve processes per rank = {counts}")
    if any(c < 1 for c in counts):
        print("!!! multinode: at least one rank has no sglang serve process after "
              "the spawn; see the rank logs.", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------
# TRACE GATE  (mn.py trace-gate)
# --------------------------------------------------------------------------
#
# Make a 16-rank trace set usable by a parser that reads ONE trace.
#
# GEAK's profiler role selects its input with
#
#     TRACE=$(ls -t "$PDIR"/*.json.gz "$PDIR"/*.json | head -1)
#
# On a single host with TP>1 that is harmless: the ranks are symmetric, so any of
# them tells the same story. Across two nodes it breaks in three ways at once,
# and all three produce a plausible-looking Top-N rather than an error:
#
#   * WHICH rank wins is a race. Sixteen ranks finish their export within seconds
#     of each other onto one NFS mount, so `ls -t` is decided by flush order. The
#     baseline leg and the candidate leg can therefore profile DIFFERENT ranks,
#     and the two Top-N tables are then not comparable -- which is the whole
#     point of the A/B.
#   * With dp_size=2 the two DP groups serve DIFFERENT requests. A trace from
#     DP1/TP-11 is a sample of the other node's traffic, not a view of the server.
#   * A rank set that is missing the remote node entirely looks exactly like a
#     healthy single-node capture. Nothing downstream can tell that half the
#     world was never recorded.
#
# So this gate does three things after a capture:
#
#   1. VERIFIES the set: every expected TP rank present, both nodes represented,
#      every file complete, and collective kernels actually in there (a trace with
#      no RCCL kernels did not capture the cross-node traffic the run exists to
#      measure).
#   2. ISOLATES the non-selected ranks into ranks/, so the role's existing `ls -t
#      | head -1` deterministically returns rank0's trace -- the same convention
#      GEAK already uses for TP>1, and the same trick the role already applies to
#      TraceLens's capture_traces/ subdir. No prompt change, no race.
#   3. REPORTS what the single parsed trace cannot show: per-rank work volume and
#      the imbalance between the two nodes.
#
# Traces are scanned by STREAMING the decompressed bytes. A full json.load of
# sixteen multi-hundred-MB traces would need more memory than the box has, and
# this gate must not be the reason a profiling round dies.
#
# Usage:
#   python mn.py trace-gate --profile-dir DIR [--expect-ranks 16] [--select-rank 0]
#                           [--no-isolate]
#

# RCCL/NCCL kernels as they appear in a ROCm torch trace. Matching a FAMILY of
# names rather than one spelling: the device-kernel symbol differs between
# rccl builds, and a gate that silently matches nothing would report "no
# collectives captured" for every healthy trace and be turned off within a day.
COLLECTIVE_PATTERNS = {
    "nccl_dev_kernel": rb"ncclDevKernel",
    "nccl_kernel": rb"nccl[A-Za-z]*Kernel",
    "rccl": rb"rccl",
    "all_reduce": rb"[Aa]ll[_]?[Rr]educe",
    "all_gather": rb"[Aa]ll[_]?[Gg]ather",
    "reduce_scatter": rb"[Rr]educe[_]?[Ss]catter",
}
KERNEL_EVENT = rb'"cat": ?"kernel"'

TRACE_RE = re.compile(r"TP-(\d+)")
DP_RE = re.compile(r"DP-(\d+)")




def discover(profile_dir: str) -> list[str]:
    """Rank traces at the top level AND in ranks/.

    Including ranks/ makes the gate idempotent: after a first run has isolated
    fifteen of the sixteen traces, re-running it (to re-check a set, or after a
    stray file appeared) must still see the whole set instead of reporting
    fifteen ranks missing.
    """
    files = []
    for directory in (profile_dir, os.path.join(profile_dir, "ranks")):
        for pattern in ("*.trace.json.gz", "*.trace.json", "*.json.gz"):
            files.extend(glob.glob(os.path.join(directory, pattern)))
    # merged-* is a combined trace, not a rank's; it carries no TP rank.
    return sorted(set(f for f in files if not os.path.basename(f).startswith("merged-")))


def parser_selection(profile_dir: str) -> str:
    """What roles/profiler.md's `ls -t "$PDIR"/*.json.gz "$PDIR"/*.json | head -1`
    returns. Kept identical to the role on purpose: this is the selection the
    gate exists to make deterministic, so it is checked, not assumed."""
    candidates = (glob.glob(os.path.join(profile_dir, "*.json.gz"))
                  + glob.glob(os.path.join(profile_dir, "*.json")))
    if not candidates:
        return ""
    return os.path.basename(max(candidates, key=os.path.getmtime))


def scan(path: str) -> dict:
    """Stream one trace: integrity, kernel-event count, collective evidence.

    Reading to EOF is the integrity check -- a truncated gzip (the server was
    torn down mid-flush) raises here instead of surfacing later as a trace with
    mysteriously few kernels.
    """
    info = {
        "file": os.path.basename(path),
        "path": path,
        "bytes": os.path.getsize(path),
        "kernel_events": 0,
        "collectives": {},
        "readable": False,
        "error": "",
    }
    opener = gzip.open if path.endswith(".gz") else open
    tail = b""
    try:
        with opener(path, "rb") as handle:
            while True:
                chunk = handle.read(8 << 20)
                if not chunk:
                    break
                # Carry a small overlap so a pattern split across the chunk
                # boundary is still counted.
                window = tail + chunk
                info["kernel_events"] += len(re.findall(KERNEL_EVENT, window))
                for name, pattern in COLLECTIVE_PATTERNS.items():
                    hits = len(re.findall(pattern, window))
                    if hits:
                        info["collectives"][name] = info["collectives"].get(name, 0) + hits
                tail = window[-64:]
        info["readable"] = True
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def cmd_trace_gate(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", default=os.environ.get("PROFILE_DIR", ""))
    ap.add_argument("--expect-ranks", type=int,
                    default=int(os.environ.get("TP", 0) or 0))
    ap.add_argument("--gpus-per-node", type=int,
                    default=int(os.environ.get("GEAK_MN_GPUS_PER_NODE", 8)))
    ap.add_argument("--select-rank", type=int, default=0,
                    help="the TP rank whose trace stays at the top level")
    ap.add_argument("--no-isolate", action="store_true",
                    help="verify and report only; leave every file in place")
    ap.add_argument("--report-path", default="",
                    help="where to write the report (default: <profile-dir>/ranks/)")
    args = ap.parse_args(argv)

    if not args.profile_dir or not os.path.isdir(args.profile_dir):
        sys.stderr.write(f"!!! mn trace-gate: no profile dir at '{args.profile_dir}'\n")
        return 2

    files = discover(args.profile_dir)
    if not files:
        sys.stderr.write(
            f"!!! mn trace-gate: no traces in {args.profile_dir}.\n"
            "    The capture produced nothing; a Top-N built on this would be\n"
            "    invented rather than measured.\n")
        return 2

    per_rank, unranked = {}, []
    for path in files:
        match = TRACE_RE.search(os.path.basename(path))
        scanned = scan(path)
        dp = DP_RE.search(os.path.basename(path))
        scanned["dp_rank"] = int(dp.group(1)) if dp else None
        if not match:
            unranked.append(scanned)
            continue
        rank = int(match.group(1))
        scanned["tp_rank"] = rank
        scanned["node_index"] = rank // max(args.gpus_per_node, 1)
        per_rank[rank] = scanned

    ranks = sorted(per_rank)
    expected = args.expect_ranks or (len(ranks) or 1)
    missing = [r for r in range(expected) if r not in per_rank]
    nodes = sorted({v["node_index"] for v in per_rank.values()})
    expected_nodes = sorted(range(-(-expected // max(args.gpus_per_node, 1))))
    unreadable = [v["file"] for v in per_rank.values() if not v["readable"]]
    no_kernels = [v["file"] for v in per_rank.values()
                  if v["readable"] and v["kernel_events"] == 0]
    with_collectives = [r for r, v in per_rank.items() if v["collectives"]]

    kernel_by_node = {}
    for value in per_rank.values():
        kernel_by_node.setdefault(value["node_index"], 0)
        kernel_by_node[value["node_index"]] += value["kernel_events"]
    imbalance_pct = None
    if len(kernel_by_node) > 1 and max(kernel_by_node.values()) > 0:
        low, high = min(kernel_by_node.values()), max(kernel_by_node.values())
        imbalance_pct = round((high - low) / high * 100.0, 2)

    problems = []
    if missing:
        problems.append(
            f"missing TP rank(s) {missing}: the capture covers "
            f"{len(ranks)}/{expected} ranks, so the set is not the server")
    if sorted(nodes) != expected_nodes:
        problems.append(
            f"only node(s) {nodes} produced traces (expected {expected_nodes}) -- "
            "a set missing a whole node is indistinguishable downstream from a "
            "healthy single-node capture")
    if unreadable:
        problems.append(
            f"unreadable/truncated trace(s) {unreadable}: most likely the server "
            "was torn down before the export flushed")
    if no_kernels:
        problems.append(f"trace(s) with zero kernel events {no_kernels}")
    if not with_collectives:
        problems.append(
            "no RCCL/NCCL kernels in ANY rank: the cross-node traffic this run "
            "exists to measure was not captured, so the Top-N would describe "
            "local compute only")

    report = {
        "profile_dir": args.profile_dir,
        "checked_at": _utc(),
        "expected_ranks": expected,
        "found_ranks": ranks,
        "nodes_represented": nodes,
        "gpus_per_node": args.gpus_per_node,
        "selected_rank": args.select_rank,
        "ranks_with_collectives": with_collectives,
        "kernel_events_by_node": kernel_by_node,
        "node_kernel_imbalance_pct": imbalance_pct,
        "per_rank": [per_rank[r] for r in ranks],
        "unranked_files": unranked,
        "problems": problems,
        "verdict": "pass" if not problems else "fail",
    }

    # Isolate BEFORE reporting failure: even a failed set is easier to diagnose
    # with one trace at the top level, and leaving sixteen there guarantees the
    # next parse picks a random one.
    isolated = []
    if not args.no_isolate:
        ranks_dir = os.path.join(args.profile_dir, "ranks")
        os.makedirs(ranks_dir, exist_ok=True)
        for rank in ranks:
            if rank == args.select_rank:
                continue
            src = per_rank[rank]["path"]
            # Already isolated by an earlier run: nothing to move.
            if os.path.dirname(src) != args.profile_dir.rstrip(os.sep):
                continue
            if os.path.isfile(src):
                shutil.move(src, os.path.join(ranks_dir, per_rank[rank]["file"]))
                isolated.append(per_rank[rank]["file"])
        report["isolated_to_ranks_dir"] = isolated
        if args.select_rank in per_rank:
            report["parsed_trace"] = per_rank[args.select_rank]["path"]
            # Don't assume the isolation worked -- reproduce the profiler
            # role's selection and check it lands on the selected rank. Any
            # stray .json/.json.gz left in this directory by any tool silently
            # becomes the parsed "trace", and the resulting empty Top-N looks
            # like a capture problem rather than a selection problem.
            actual = parser_selection(args.profile_dir)
            if actual and actual != per_rank[args.select_rank]["file"]:
                report["problems"].append(
                    f"the parser's `ls -t | head -1` returns '{actual}', not the "
                    f"selected rank's trace: that file shadows the trace and "
                    "would be parsed in its place")
                report["verdict"] = "fail"
            report["parser_selection"] = actual
        else:
            report["parsed_trace"] = ""
            report["problems"].append(
                f"selected rank {args.select_rank} has no trace; the parser will "
                "fall back to whatever ls -t returns")
            report["verdict"] = "fail"

    # The report goes in ranks/, NOT next to the selected trace. The profiler
    # role selects its input with `ls -t "$PDIR"/*.json.gz "$PDIR"/*.json |
    # head -1`, and that glob matches plain .json too -- so a report written
    # beside the traces is the NEWEST match and gets parsed AS the trace. That
    # is not hypothetical: it happened here, and it yields a Top-N of zero
    # kernels and 0.00 ms of GPU time, which reads like a bad capture rather
    # than a misselected file. A gate that shadows the artifact it just
    # approved is worse than no gate, so nothing this script writes may land in
    # the directory the parser globs.
    report_path = args.report_path or os.path.join(
        args.profile_dir, "ranks", "mn_trace_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report["report_path"] = report_path

    print(f">>> mn trace-gate: {len(ranks)}/{expected} rank trace(s), "
          f"node(s) {nodes}, collectives in {len(with_collectives)} rank(s)")
    for rank in ranks:
        value = per_rank[rank]
        print(f"    TP-{rank:<2} node{value['node_index']} "
              f"{value['bytes']/2**20:8.1f} MiB  kernels={value['kernel_events']:<8} "
              f"collectives={sum(value['collectives'].values())}")
    if imbalance_pct is not None:
        print(f">>> kernel-event imbalance between nodes: {imbalance_pct}% "
              f"({kernel_by_node})")
    if isolated:
        print(f">>> kept TP-{args.select_rank} at the top level; moved "
              f"{len(isolated)} other rank trace(s) into ranks/ so the parser's "
              "`ls -t | head -1` is deterministic")
    print(f">>> mn trace-gate: report at {report_path}")
    for problem in report["problems"]:
        sys.stderr.write(f"!!! mn trace-gate: {problem}\n")
    return 0 if report["verdict"] == "pass" else 1


# --------------------------------------------------------------------------
# SUPERVISE  (mn.py supervise)
# --------------------------------------------------------------------------
#
# Local stand-in for a remote server: the pid bench_e2e.sh can watch and kill.
#
# bench_e2e.sh's lifecycle is built on one local pid -- `kill -0 $SERVER_PID` is how
# it tells "still starting" from "died", and `server_teardown` signals that pid to
# stop the server. With the ranks living on other machines there is no such pid, so
# this process stands in for them:
#
#   * It polls every rank and EXITS as soon as one stops serving -- either no
#     process at all, or workers that have become zombies while the HTTP front end
#     keeps answering 200. That turns a dead rank into bench_e2e.sh's `died_early`
#     within one poll. Without it, a rank1 crash only surfaces as rank0 going
#     quiet, i.e. `stalled` after the full 600s stall window; and an engine that
#     aborts behind a live front end does not surface at all -- the client simply
#     waits forever for requests nothing will ever answer.
#   * On SIGTERM it reaps the remote ranks. It is the ONLY thing that does: a local
#     signal, group or not, cannot reach a process on another host, so without this
#     handler every leg would leave 16 GPUs' worth of resident VRAM and a held
#     dist-init port behind it.
#
# Two deliberate choices about the poll:
#
#   * A rank is declared dead only after MISSES consecutive empty observations, and
#     a probe that ERRORS (Ray hiccup, NFS stall) is counted separately and
#     tolerated for longer. A single failed `pgrep` is not evidence that a
#     20-minute cold start has died, and killing a healthy run on one bad sample is
#     worse than noticing a real death a poll later.
#   * It never writes to $LOG. bench_e2e.sh greps that file for OOM/fatal markers
#     and watches its size to detect stalls, and it is the fingerprint of the
#     served stack in result.json -- injecting supervisor lines would both forge
#     those signals and corrupt the fingerprint. Diagnosis goes to this process's
#     own log and to mn_supervisor.json.
#

SUP_POLL_S = int(os.environ.get("GEAK_MN_POLL_S", 20))
MISSES = int(os.environ.get("GEAK_MN_MISSES", 2))       # empty observations before "dead"
ERR_MISSES = int(os.environ.get("GEAK_MN_ERR_MISSES", 6))  # probe errors before giving up

_state = None
_out_dir = os.environ.get("OUT_DIR", ".")
_stopping = False




def record(verdict: str, detail: dict) -> None:
    payload = {"verdict": verdict, "at": datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **detail}
    try:
        with open(os.path.join(_out_dir, "mn_supervisor.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except OSError as exc:
        log(f"could not write mn_supervisor.json: {exc}")


def on_term(signum, _frame):
    """Reap the remote ranks, then leave. Exit code 0: being asked to stop is not
    a failure, and bench_e2e.sh must not read a normal teardown as a crash."""
    global _stopping
    if _stopping:
        return
    _stopping = True
    log(f"signal {signum}: tearing down the remote ranks")
    try:
        # Sized to finish inside server_teardown.sh's TERM->KILL grace (the
        # launcher sets 90s). The kill itself goes out first and the rest is
        # confirmation, so a SIGKILL landing mid-wait still leaves the ranks
        # dead -- but a wait longer than the grace would report nothing.
        cleared = teardown(_state, wait_idle_s=int(os.environ.get("GEAK_MN_TEARDOWN_WAIT_S", 60)))
        log("remote ranks reaped" if cleared else
            "remote teardown did not fully clear; the next launch will wait for the cards")
        record("torn_down", {"signal": int(signum), "cleared": bool(cleared)})
    except Exception as exc:                        # never block the shutdown path
        log(f"remote teardown raised: {exc}")
        record("torn_down", {"signal": int(signum), "error": str(exc)})
    sys.exit(0)


def spawn_reaper() -> None:
    """Start the outlives-us reaper, unless asked not to.

    The TERM handler above covers an orderly stop. It cannot cover the ways this
    process can die without running Python: SIGKILL when the grace expires, an
    unhandled crash, an OOM kill. In every one of those the ranks keep 16 GPUs
    and the dist-init port, which then reads as "the machine is idle at 0%" to
    anything that only looks at utilisation.

    The reaper is a detached process whose only job is to notice that this pid
    is gone and reap what it was supervising.
    """
    if os.environ.get("GEAK_MN_NO_REAPER") == "1":
        log("reaper disabled by GEAK_MN_NO_REAPER")
        return
    reaper = os.path.abspath(__file__)
    launch_id = ""
    try:
        with open(os.path.join(_out_dir, "mn_launch.json"), encoding="utf-8") as f:
            launch_id = json.load(f).get("launch_id", "")
    except (OSError, ValueError):
        pass
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(os.path.join(_out_dir, "mn_reaper.log"), "ab") as sink:
        subprocess.Popen(
            [sys.executable, reaper, "reap", "--watch-pid", str(os.getpid()),
             "--launch-id", launch_id],
            env=env, start_new_session=True, stdout=sink, stderr=subprocess.STDOUT,
        )
    log(f"reaper started, watching pid {os.getpid()} (launch {launch_id or 'unknown'})")


def cmd_supervise(argv: list) -> int:
    global _state
    _state = load_state()
    connect(_state.get("ray_address", "auto"))
    # AFTER ray.init, never before. Ray installs its own C-level failure signal
    # handler, which logs the SIGTERM and lets the default action kill the
    # process -- so a handler registered first is silently replaced and never
    # runs. That exact ordering bug left a full 16-GPU instance resident after
    # the benchmark had finished and reported success, reading 0% utilisation
    # while holding every card.
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    if signal.getsignal(signal.SIGTERM) is not on_term:
        log("!!! SIGTERM is not ours; the ranks would outlive this process")
        record("handler_lost", {})
        return 1
    spawn_reaper()
    ranks = [f"rank{n['rank']}@{n['hostname'].split('.')[0]}" for n in _state["nodes"]]
    log(f"watching {', '.join(ranks)} every {SUP_POLL_S}s "
        f"(dead after {MISSES} empty polls, {ERR_MISSES} failed probes)")
    record("watching", {"ranks": ranks, "poll_s": SUP_POLL_S})

    empty = [0] * len(_state["nodes"])
    errors = 0
    while True:
        time.sleep(SUP_POLL_S)
        try:
            health = engine_health(_state)
            errors = 0
        except Exception as exc:
            errors += 1
            log(f"probe failed ({errors}/{ERR_MISSES}): {exc}")
            if errors >= ERR_MISSES:
                # Persistently unable to see the ranks. Exiting hands bench_e2e.sh
                # a `died_early` it can report, which beats supervising a cluster
                # we have lost contact with.
                record("unobservable", {"error": str(exc), "consecutive_failures": errors})
                log("giving up: the ranks have been unobservable for too long")
                return 1
            continue

        for i, rank in enumerate(health):
            gone = rank["serve"] < 1 or rank["zombies"] > 0
            empty[i] = empty[i] + 1 if gone else 0

        dead = [i for i, misses in enumerate(empty) if misses >= MISSES]
        if dead:
            names = [f"rank{health[i]['rank']}@{health[i]['host']}" for i in dead]
            why = "; ".join(
                f"rank{health[i]['rank']}: " + (
                    "no sglang process" if health[i]["serve"] < 1
                    else f"{health[i]['zombies']} zombie worker(s) -- the engine aborted "
                         "while the HTTP front end kept answering"
                )
                for i in dead
            )
            log(f"{why} -- exiting so the benchmark sees a dead server now instead of "
                "hanging until the stall window expires")
            record("rank_died", {"dead_ranks": names, "why": why, "health": health})
            # The survivors are blocked in a collective that will never complete;
            # leave nothing resident.
            try:
                teardown(_state, wait_idle_s=60)
            except Exception as exc:
                log(f"post-mortem teardown raised: {exc}")
            return 1


# --------------------------------------------------------------------------
# REAP  (mn.py reap)
# --------------------------------------------------------------------------
#
# Last line of defence: reap the remote ranks when their supervisor dies.
#
# The supervisor stops the ranks on SIGTERM. That covers an orderly teardown and
# nothing else -- SIGKILL after the grace expires, a crash, an OOM kill, and
# `kill -9` by a frustrated operator all end the supervisor without running any
# Python. What is left behind is a full 16-GPU instance that holds its VRAM and
# the dist-init port while reporting 0% utilisation, so every "is this machine
# free?" check that looks at utilisation says yes and the next launch dies on the
# port.
#
# The trigger is narrow on purpose -- the supervisor's pid disappearing, verified
# against pid reuse and against the launch it belonged to:
#
#   * /proc start time is captured up front. A pid that comes back with a
#     different start time is a different process, not our supervisor.
#   * The launch id is re-read before acting. If a NEW launch has already taken
#     over (its own launch id in mn_live_launch), this reaper belongs to a dead
#     generation and must exit quietly -- tearing down then would kill the server
#     that replaced ours, which is a far worse failure than the leak.
#
# Usage (spawned by `mn.py supervise`; not called by hand):
#   python mn.py reap --watch-pid <pid> [--launch-id <id>]
#

REAP_POLL_S = int(os.environ.get("GEAK_MN_REAPER_POLL_S", 10))




def start_ticks(pid: int) -> str:
    """/proc/<pid>/stat field 22. comm can contain spaces and parens, so the
    parse cuts after the LAST ') '."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            stat = f.read()
    except OSError:
        return ""
    return stat.rsplit(") ", 1)[-1].split()[19]


def alive(pid: int, ticks: str) -> bool:
    if not os.path.isdir(f"/proc/{pid}"):
        return False
    now = start_ticks(pid)
    return not (now and ticks and now != ticks)


def current_launch(state: dict) -> str:
    try:
        with open(os.path.join(state["shared_fs"], "mn_live_launch"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def cmd_reap(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch-pid", type=int, required=True)
    ap.add_argument("--launch-id", default="")
    args = ap.parse_args(argv)

    state = load_state()
    ticks = start_ticks(args.watch_pid)
    log(f"watching supervisor pid {args.watch_pid} (start_ticks={ticks or '?'}) "
        f"for launch {args.launch_id or 'unknown'}")

    while alive(args.watch_pid, ticks):
        time.sleep(REAP_POLL_S)

    live = current_launch(state)
    if args.launch_id and live and live != args.launch_id:
        log(f"supervisor gone, but launch {live} has taken over from {args.launch_id};"
            " exiting without touching it")
        return 0

    log(f"supervisor pid {args.watch_pid} is gone -- reaping the ranks it was watching")
    # Only connect now. A reaper that held a Ray connection for the whole run
    # would keep a worker slot for hours to do ten seconds of work.
    connect(state.get("ray_address", "auto"))
    counts_before = serve_counts(state)
    if not any(counts_before):
        log("nothing left to reap")
        return 0
    cleared = teardown(state, wait_idle_s=120, verbose=log)
    log(f"serve procs before={counts_before} cleared={cleared}")
    try:
        with open(os.path.join(os.environ.get("OUT_DIR", state["shared_fs"]),
                               "mn_reaper.json"), "w", encoding="utf-8") as f:
            json.dump({
                "reaped_launch": args.launch_id,
                "supervisor_pid": args.watch_pid,
                "serve_procs_found": counts_before,
                "cleared": bool(cleared),
                "at": datetime.datetime.now(datetime.timezone.utc)
                      .strftime("%Y-%m-%dT%H:%M:%SZ"),
            }, f, indent=2, sort_keys=True)
    except OSError as exc:
        log(f"could not write mn_reaper.json: {exc}")
    return 0 if cleared else 1


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

COMMANDS = {
    "preflight": cmd_preflight,
    "launch": cmd_launch,
    "supervise": cmd_supervise,
    "reap": cmd_reap,
    "trace-gate": cmd_trace_gate,
}
# supervise and reap set their own tag; the rest print with the plain one.
TAGS = {"supervise": "mn-sup", "reap": "mn-reaper"}


def main(argv: list) -> int:
    if not argv or argv[0] not in COMMANDS:
        sys.stderr.write(
            "usage: mn.py {%s} [options]\n" % "|".join(COMMANDS))
        return 2
    global _LOG_TAG
    _LOG_TAG = TAGS.get(argv[0], _LOG_TAG)
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

