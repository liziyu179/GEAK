"""The multi-node suite: fabric inference, per-rank render, profile window, gate.

Run:  python3 -m unittest test_mn            (from e2e_workflow/scripts/tests)
      python3 e2e_workflow/scripts/tests/test_mn.py

WHY THESE EXIST. Every failure covered here is one that otherwise produces a
NUMBER rather than an error -- a benchmark that finishes, reports a throughput
and is wrong:

  * fabric inference picking a /31 point-to-point link that cannot reach the
    other node, so the ranks hang at rendezvous instead of saying why;
  * a render that lets a swept flag overwrite the rank triple, so two processes
    both believe they are rank 0;
  * a profile window that returns while ranks are still streaming their trace,
    so Top-N is parsed from a truncated file;
  * a gate that lets a stray file shadow the parser's selection, so the A and
    the B of a comparison are parsed from different ranks.

mn.py imports without Ray installed (only the subcommands that reach a cluster
need it), so everything here except the profile window -- which needs bash --
runs on a laptop as well as on a node.
"""
import contextlib
import glob
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
MN = os.path.join(SCRIPTS_DIR, "multi_node")
sys.path.insert(0, MN)

# The kernel name both trace suites write into their fixtures. A collective is
# what proves a trace came from a multi-rank run at all; without one, a trace is
# indistinguishable from a single-host profile.
COLLECTIVE = "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage)"

import mn  # noqa: E402


# ==========================================================================
# FABRIC INFERENCE  (mn.py preflight -> decide / pick_addr / describe)
# ==========================================================================
#
# Fabric-subnet decision tests.
#
# GEAK is handed a Ray cluster and nothing else, so the candidate machines and
# their addresses are unknown until Ray answers -- the fabric has to come from the
# candidate set. Every case below is one where getting it wrong still produces a
# working launch, just over the wrong link, which reads as a slow cluster.
#



def node(*pairs):
    return list(pairs)


class DecideFabric(unittest.TestCase):
    def test_the_subnet_the_candidates_share_is_the_fabric(self):
        d = mn.decide([
            node(("ens50f0", "10.235.200.166"), ("eno1", "192.168.1.10")),
            node(("ens50f0", "10.235.200.182"), ("eno1", "192.168.7.22")),
        ])
        self.assertEqual((d["subnet"], d["how"], d["nodes_covered"]),
                         ("10.235.200.", "inferred", 2))

    def test_it_works_on_a_rig_nobody_configured(self):
        d = mn.decide([node(("ib0", "172.31.9.4")), node(("ib0", "172.31.9.11")),
                           node(("ib0", "172.31.9.30"))])
        self.assertEqual((d["subnet"], d["nodes_covered"]), ("172.31.9.", 3))

    def test_a_network_every_node_shares_one_address_on_is_node_local(self):
        # Docker's bridge: same address everywhere, so by address alone it is the
        # most "shared" network in the cluster and carries nothing between boxes.
        d = mn.decide([
            node(("eth0", "172.17.0.2"), ("ens50f0", "10.235.200.166")),
            node(("eth0", "172.17.0.2"), ("ens50f0", "10.235.200.182")),
        ])
        self.assertEqual(d["subnet"], "10.235.200.")

    def test_the_docker_bridge_is_excluded_by_name_too(self):
        d = mn.decide([
            node(("docker0", "172.17.0.1"), ("ens50f0", "10.235.200.166")),
            node(("docker0", "172.17.0.1"), ("ens50f0", "10.235.200.182")),
        ])
        self.assertEqual(d["subnet"], "10.235.200.")

    def test_loopback_and_link_local_are_never_the_fabric(self):
        d = mn.decide([
            node(("lo", "127.0.0.1"), ("ens1", "169.254.3.1"), ("ens2", "10.1.1.5")),
            node(("lo", "127.0.0.1"), ("ens1", "169.254.7.2"), ("ens2", "10.1.1.6")),
        ])
        self.assertEqual(d["subnet"], "10.1.1.")

    def test_a_network_only_one_node_is_on_is_not_a_fabric(self):
        d = mn.decide([node(("ens50f0", "10.1.1.5")), node(("ens50f0", "10.2.2.6"))])
        self.assertEqual((d["subnet"], d["how"]), ("", "undecided"))

    def test_an_explicit_subnet_wins_over_inference(self):
        addrs = [node(("eno1", "10.0.9.1"), ("ens50f0", "10.235.200.166")),
                 node(("eno1", "10.0.9.2"), ("ens50f0", "10.235.200.182"))]
        self.assertEqual(mn.decide(addrs)["subnet"], "10.0.9.")
        forced = mn.decide(addrs, explicit="10.235.200.")
        self.assertEqual((forced["subnet"], forced["how"], forced["nodes_covered"]),
                         ("10.235.200.", "configured", 2))

    def test_a_configured_subnet_that_matches_nothing_reports_zero_coverage(self):
        # So the caller says "this value excluded every node", not "no idle machines".
        d = mn.decide([node(("ens50f0", "10.1.1.5")), node(("ens50f0", "10.1.1.6"))],
                          explicit="192.168.0.")
        self.assertEqual((d["how"], d["nodes_covered"]), ("configured", 0))

    def test_the_decision_is_stable_across_runs(self):
        # A fabric that moved between A/B legs would be a hidden variable.
        addrs = [node(("a", "10.5.5.1"), ("b", "10.6.6.1")),
                 node(("a", "10.5.5.2"), ("b", "10.6.6.2"))]
        self.assertEqual({mn.decide(addrs)["subnet"] for _ in range(5)},
                         {"10.5.5."})

    def test_more_coverage_beats_a_lower_prefix(self):
        d = mn.decide([node(("a", "10.5.5.1")), node(("b", "10.9.9.1")),
                           node(("b", "10.9.9.2")), node(("b", "10.9.9.3"))])
        self.assertEqual(d["subnet"], "10.9.9.")


class Masks(unittest.TestCase):
    """The mask decides whether two addresses are on the same network.

    Measured on the MI300X pair: each node has eight 400Gb NICs on 192.168.15.x
    and they are eight /31 point-to-point links to a switch -- FC54 holds
    .64/31, FC58 holds .16/31. They do not ping, and the host routes
    192.168.15.0/24 back out the 25Gb management NIC. Read as /24 they are the
    most attractive fabric in the cluster.
    """

    def test_point_to_point_links_are_not_a_shared_fabric(self):
        # The management NICs are on DIFFERENT /24s here, so nothing else can be
        # chosen: without masks this returns 192.168.15. and every launch hangs
        # at rendezvous over a link that cannot carry a packet.
        d = mn.decide([
            node(("ens50f0", "10.235.200.166", 24), ("benic1p1", "192.168.15.64", 31),
                 ("benic2p1", "192.168.15.66", 31)),
            node(("ens50f0", "10.235.201.182", 24), ("benic1p1", "192.168.15.16", 31),
                 ("benic2p1", "192.168.15.18", 31)),
        ])
        self.assertEqual((d["subnet"], d["how"]), ("", "undecided"))

    def test_the_live_pair_still_decides_on_the_management_network(self):
        d = mn.decide([
            node(("ens50f0", "10.235.200.166", 24), ("benic1p1", "192.168.15.64", 31)),
            node(("ens50f0", "10.235.200.182", 24), ("benic1p1", "192.168.15.16", 31)),
        ])
        self.assertEqual((d["subnet"], d["nodes_covered"]), ("10.235.200.", 2))

    def test_a_point_to_point_link_between_the_two_nodes_is_usable(self):
        # Same /31, one address each: that pair really can carry a collective.
        d = mn.decide([
            node(("benic1p1", "192.168.15.64", 31)),
            node(("benic1p1", "192.168.15.65", 31)),
        ])
        self.assertEqual((d["subnet"], d["nodes_covered"]), ("192.168.15.", 2))

    def test_the_same_three_octets_can_be_two_networks(self):
        # 10.0.0.10/25 is 10.0.0.0/25; 10.0.0.200/25 is 10.0.0.128/25.
        d = mn.decide([node(("a", "10.0.0.10", 25)), node(("a", "10.0.0.200", 25))])
        self.assertEqual(d["how"], "undecided")

    def test_a_probe_without_a_mask_is_read_as_a_slash_24(self):
        # Old pins and old probes predate the mask; they must keep deciding as
        # they did rather than failing to decide at all.
        d = mn.decide([node(("ens50f0", "10.235.200.166")),
                           node(("ens50f0", "10.235.200.182", 24))])
        self.assertEqual((d["subnet"], d["nodes_covered"]), ("10.235.200.", 2))


class PickAddr(unittest.TestCase):
    def test_it_reads_an_entry_that_carries_a_mask(self):
        self.assertEqual(
            mn.pick_addr(node(("lo", "127.0.0.1", 8), ("ens50f0", "10.1.1.7", 24)),
                             "10.1.1."),
            ("ens50f0", "10.1.1.7"))

    def test_the_nodes_own_interface_name_is_returned(self):
        # Names differ between machines, so each rank needs ITS name for
        # NCCL_SOCKET_IFNAME; the driver cannot supply one for all.
        self.assertEqual(
            mn.pick_addr(node(("lo", "127.0.0.1"), ("enp1s0f1", "10.1.1.7")), "10.1.1."),
            ("enp1s0f1", "10.1.1.7"))

    def test_a_node_off_the_fabric_returns_nothing(self):
        self.assertEqual(mn.pick_addr(node(("eno1", "192.168.0.4")), "10.1.1."), ("", ""))

    def test_an_undecided_fabric_matches_no_node(self):
        # Must not degrade into "any address will do": that picks a management NIC.
        self.assertEqual(mn.pick_addr(node(("ens50f0", "10.1.1.5")), ""), ("", ""))


class PickHosts(unittest.TestCase):
    """Selection must be nameable, or two runs on one fleet collide.

    Without --hosts, pick() answers "the first nnodes idle boxes", which is the
    same answer for every run on the cluster. Observed: a PD validation run and
    an unrelated EP16 run, started minutes apart, both selected the same pair --
    the second re-pinned on top of the first and its legs launched onto
    machines already holding another run's engines.
    """

    HOSTS = ("smc300x-ccs-aus-gpuc489", "smc300x-ccs-aus-gpufc54",
             "smc300x-ccs-aus-gpuf293", "smc300x-ccs-aus-gpuf2a0")

    def probe(self, i, hostname):
        ip = f"10.235.200.{10 + i}"
        return {"hostname": hostname, "ip": ip, "ngpu": 8, "max_use": 0,
                "gpu_pids": 0, "serve_procs": 0, "busy_ports": [],
                "free_vram_mib": 191000, "foreign_gpu_pids": 0,
                "addrs": [["lo", "127.0.0.1", 8], ["ens50f0", ip, 24]]}

    def pick(self, nnodes=2, **kw):
        probes = [self.probe(i, h) for i, h in enumerate(self.HOSTS)]
        nodes = [{"NodeID": f"n{i}"} for i, _ in enumerate(self.HOSTS)]
        task = mock.Mock()
        task.options.return_value.remote.return_value = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(mn, "alive_gpu_nodes",
                                                  lambda _g: nodes))
            stack.enter_context(mock.patch.object(mn, "probe_node", task))
            stack.enter_context(mock.patch.object(mn, "pinned", lambda _n: None))
            stack.enter_context(mock.patch.object(
                mn, "ray", mock.Mock(get=lambda _xs, timeout=None: probes)))
            return mn.pick(nnodes, 8, **kw)

    def shorts(self, picked):
        return [p["hostname"].split("-")[-1].upper() for p in picked]

    def test_without_hosts_it_takes_whatever_is_idle(self):
        picked, _report, _fabric = self.pick()
        self.assertEqual(self.shorts(picked), ["GPUC489", "GPUFC54"])

    def test_hosts_confines_the_pin_to_the_named_pair(self):
        picked, _report, _fabric = self.pick(hosts={"GPUF293", "GPUF2A0"})
        self.assertEqual(sorted(self.shorts(picked)), ["GPUF293", "GPUF2A0"])

    def test_the_two_runs_no_longer_land_on_each_other(self):
        """The whole point, stated as the two calls that used to collide."""
        mine, _r, _f = self.pick(hosts={"GPUC489", "GPUFC54"})
        theirs, _r, _f = self.pick(hosts={"GPUF293", "GPUF2A0"})
        self.assertFalse(set(self.shorts(mine)) & set(self.shorts(theirs)))

    def test_a_skipped_box_says_it_was_not_asked_for(self):
        """Not 'gpu busy': a machine the operator excluded is not a fleet
        problem, and the wrong reason sends the reader to rocm-smi."""
        _picked, report, _fabric = self.pick(hosts={"GPUF293", "GPUF2A0"})
        verdicts = {r["hostname"].split("-")[-1].upper(): r["verdict"]
                    for r in report}
        self.assertIn("not among the requested --hosts", verdicts["GPUC489"])
        self.assertEqual(verdicts["GPUF293"], "idle")

    def test_rank_order_stays_ip_ordered_not_list_ordered(self):
        """A re-pin on the same machines must keep rank0 rank0: under pd_disagg
        rank0 is the prefill engine, and a set literal has no order anyway."""
        picked, _r, _f = self.pick(hosts={"GPUFC54", "GPUC489"})
        self.assertEqual([p["ip"] for p in picked],
                         sorted((p["ip"] for p in picked),
                                key=lambda s: [int(o) for o in s.split(".")]))
        self.assertEqual(picked[0]["rank"], 0)

    def test_a_hostname_the_cluster_does_not_have_is_named(self):
        """A typo must not read as 'not enough idle nodes'."""
        with self.assertRaises(SystemExit) as e:
            self.pick(hosts={"GPUC489", "GPUTYPO"})
        self.assertIn("GPUTYPO", str(e.exception))
        self.assertIn("Ray sees", str(e.exception))

    def test_fewer_hosts_than_nodes_is_refused_up_front(self):
        with self.assertRaises(SystemExit) as e:
            self.pick(nnodes=2, hosts={"GPUC489"})
        self.assertIn("--nnodes is 2", str(e.exception))


class Describe(unittest.TestCase):
    def test_it_lists_what_each_node_had(self):
        text = mn.describe(
            [node(("lo", "127.0.0.1"), ("ens50f0", "10.1.1.5")), node(("eno1", "192.168.0.4"))],
            ["gpu-a", "gpu-b"])
        self.assertIn("gpu-a: ens50f0=10.1.1.5", text)
        self.assertIn("gpu-b: eno1=192.168.0.4", text)
        self.assertNotIn("127.0.0.1", text)

    def test_a_node_with_nothing_usable_says_so(self):
        self.assertIn("none", mn.describe([node(("lo", "127.0.0.1"))], ["gpu-a"]))


# ==========================================================================
# PER-RANK RENDER  (mn.py launch -> render / resolve_adapter)
# ==========================================================================
#
# Per-rank render tests: the launcher must stay framework-agnostic, and the
# flag layering must survive the split.
#
# The refactor these guard moved the launch line out of serve.sh.tmpl and into the
# backend adapter. Two things can now break silently:
#
# * Layering. The recipe is the BASE layer, GEAK's swept flags come later and win
#   by argparse last-wins, and the rank/world triple is appended LAST because it is
#   structural. Get the order wrong and a sweep either loses its flag or -- worse --
#   changes which rank a process thinks it is, which reads as a hang at rendezvous.
# * The seam itself. If the template ever restates a framework's launch line again,
#   a second backend needs a second template and the two drift. The end-to-end case
#   below runs the rendered script against a FAKE adapter: it can only pass if the
#   script really does delegate.
#

HAVE_SH = bool(shutil.which("bash") and shutil.which("ip"))

FAKE_ADAPTER = """\
adapter_dist_mode() { echo per_rank; }
adapter_dist_args() { echo "--nnodes $2 --node-rank $1 --dist-init-addr $3:$4"; }
adapter_launch() {
  echo "FINAL_ARGS=[$EXTRA_SERVER_ARGS]"
  echo "FINAL_ENV=[$EXTRA_ENV]"
  echo "GPU=$GPU HOST=$HOST PORT=$PORT TP=$TP MODEL=$MODEL"
  SERVER_PID=4242
}
"""


def render(tmp, **over):
    with open(os.path.join(MN, "serve.sh.tmpl"), encoding="utf-8") as f:
        tmpl = f.read()
    subs = {
        "LAUNCH_ID": "abc123", "RANK": "1", "NNODES": "2",
        "MASTER": "10.235.200.166", "PORT": "30890", "DIST_PORT": "30531",
        # Both are rendered whole by cmd_launch because the disaggregated
        # topology needs different text here. The values below are the ones it
        # produces for tp_span, so every assertion in this class doubles as a
        # check that adding PD did not move the TP-spanning render.
        "BASE_URL": "http://10.235.200.166:30890",
        "STRUCTURAL_CALL": 'adapter_dist_args 1 2 "$MASTER_ADDR" 30531',
        "MODEL": "/models/m", "TP": "16", "GPU": "0,1,2,3,4,5,6,7",
        "MEM_FRACTION": "0.90", "PROFILE_DIR": os.path.join(tmp, "prof"),
        "LOG": os.path.join(tmp, "server.rank1.log"),
        "ADAPTER": os.path.join(tmp, "fake.sh"), "OVERLAY": "/ovl",
        "PYTHONPATH": "", "EXTRA_ENV": "RECIPE_V=1 SWEPT_V=2",
        "RECIPE_ARGS": "--attention-backend dsv4 --page-size 256",
        "EXTRA_ARGS": "--page-size 64", "CLUSTER_SUBNET": "10.235.200.",
    }
    subs.update(over)
    return mn.render(tmpl, subs)


class Layering(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "fake.sh"), "w", encoding="utf-8") as f:
            f.write(FAKE_ADAPTER)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_render_leaves_no_placeholder(self):
        # render() raises on a survivor; the assert documents what that protects.
        self.assertNotIn("__", render(self.tmp).replace("__RANK__", ""))

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_the_script_delegates_to_the_adapter_for_the_launch_line(self):
        out = subprocess.run(["bash"], input=render(self.tmp),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        # Only the fake adapter can print these, so the script cannot have
        # carried a launch line of its own.
        self.assertIn("STARTED_PID=4242", out.stdout)
        self.assertIn("GPU=0,1,2,3,4,5,6,7 HOST=0.0.0.0 PORT=30890", out.stdout)

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_recipe_first_swept_flag_next_rank_triple_last(self):
        out = subprocess.run(["bash"], input=render(self.tmp),
                             capture_output=True, text=True)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        # The recipe asks for page-size 256 and the sweep for 64. argparse is
        # last-wins, so the sweep has to come second or GEAK's own choice loses.
        self.assertLess(args.index("--page-size 256"), args.index("--page-size 64"))
        # The rank triple is structural: last, so no swept flag can displace it.
        self.assertLess(args.index("--page-size 64"), args.index("--node-rank 1"))
        self.assertIn("--nnodes 2 --node-rank 1 --dist-init-addr 10.235.200.166:30531",
                      args)

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_recipe_env_stays_ahead_of_the_swept_env(self):
        out = subprocess.run(["bash"], input=render(self.tmp),
                             capture_output=True, text=True)
        env = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ENV=")][0]
        self.assertEqual(env, "FINAL_ENV=[RECIPE_V=1 SWEPT_V=2]")


class AdapterSeam(unittest.TestCase):
    """resolve_adapter's refusals. Each one would otherwise surface as sixteen
    ranks dying at once on the far side of a Ray task."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, body):
        p = os.path.join(self.tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        return p

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_a_backend_that_cannot_say_how_it_spells_rank_is_refused(self):
        p = self.write("quiet.sh", "adapter_launch() { :; }\n")
        os.environ["GEAK_MN_BACKEND_ADAPTER"] = p
        try:
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("quiet", self.tmp)
            self.assertIn("adapter_dist_args", str(e.exception))
        finally:
            del os.environ["GEAK_MN_BACKEND_ADAPTER"]

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_a_head_only_backend_is_refused_rather_than_launched_as_per_rank(self):
        p = self.write("fan.sh", "adapter_dist_mode() { echo head_only; }\n"
                                 "adapter_dist_args() { echo x; }\n")
        os.environ["GEAK_MN_BACKEND_ADAPTER"] = p
        try:
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("vllm", self.tmp)
            self.assertIn("head_only", str(e.exception))
        finally:
            del os.environ["GEAK_MN_BACKEND_ADAPTER"]

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_an_adapter_off_the_shared_mount_is_refused(self):
        # Every rank sources it on its own node; off the shared mount the other
        # nodes have no such file and the whole set dies at once.
        p = self.write("sglang.sh", "adapter_dist_args() { echo x; }\n")
        os.environ["GEAK_MN_BACKEND_ADAPTER"] = p
        try:
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("sglang", os.path.join(self.tmp, "elsewhere"))
            self.assertIn("shared", str(e.exception))
        finally:
            del os.environ["GEAK_MN_BACKEND_ADAPTER"]

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_real_sglang_adapter_declares_per_rank(self):
        root = os.path.realpath(os.path.join(HERE, "..", ".."))
        self.assertTrue(
            mn.resolve_adapter("sglang", root).endswith("adapters/sglang.sh"))


# ==========================================================================
# PROFILE WINDOW  (launchers/multinode.sh -> adapter_profile_window)
# ==========================================================================
#
# Shell tests for the multi-node adapter_profile_window override.
#
# Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
#   or just this section: python3 -m unittest test_mn.MultinodeProfileWindowTest
# (skipped where bash is unavailable, i.e. when authoring on Windows -- these are
# meant to run in the container that actually serves.)
#
# WHY THESE EXIST: the backend's window returns as soon as ONE trace lands. On two
# nodes that means bench_e2e.sh proceeds -- and the teardown may fire -- while
# fifteen ranks are still flushing over NFS, and whichever file happened to land
# first becomes the one the parser reads. The override waits for the set and then
# gates it, so the behaviour under test is entirely in the SHELL: whether the wait
# actually blocks, whether the gate is invoked, and whether an incomplete set
# still lets the round finish (it must -- failing here would discard the bench
# that came with the trace).
#
# The native window is replaced by a stub that drops trace files at a controlled
# pace. No server, no GPU, no Ray: sourcing the launcher only needs a pinned-state
# file, which is written here.
#

LAUNCHER = os.path.join(SCRIPTS_DIR, "adapters", "launchers", "multinode.sh")
BASH = shutil.which("bash")


# Sourcing multinode.sh runs its source-time block (HOST/PORT from the pin, the
# cluster lock, the stop grace). HERE is what it uses to find scripts/multi_node.
DRIVER = """set -uo pipefail
HERE="{scripts}"
source "{launcher}"
{stub}
adapter_profile_window
echo "WINDOW_RC=$?"
"""

# Writes N per-rank traces, named as sglang names them, then returns like the
# real native window does once the first file exists.
STUB_NATIVE = """
adapter_profile_window_native() {{
  python3 - "$PROFILE_DIR" {count} <<'PY'
import gzip, json, sys
d, n = sys.argv[1], int(sys.argv[2])
for tp in range(n):
    events = [{{"cat": "kernel", "name": "{collective}", "ts": 1, "dur": 9}}]
    with gzip.open(f"{{d}}/prof-TP-{{tp}}-DP-{{tp//8}}.trace.json.gz", "wt") as fh:
        fh.write(json.dumps({{"traceEvents": events}}))
PY
  return 0
}}
"""

# How the real export behaves, and the shape that broke the first version of
# this wait: every file is CREATED (empty) up front, then streamed into over
# seconds. A wait that counts files sees 16/16 immediately and hands the gate a
# set of truncated and zero-byte traces. The writer keeps running after the
# native window returns, exactly as the ranks do.
STUB_STREAMING = """
adapter_profile_window_native() {{
  python3 - "$PROFILE_DIR" {count} <<'PY' &
import gzip, io, json, sys, time
d, n = sys.argv[1], int(sys.argv[2])
paths = [f"{{d}}/prof-TP-{{tp}}-DP-{{tp//8}}.trace.json.gz" for tp in range(n)]
for path in paths:
    open(path, "wb").close()          # created, still zero bytes
blob = io.BytesIO()
with gzip.GzipFile(fileobj=blob, mode="wb") as gz:
    events = [{{"cat": "kernel", "name": "{collective}", "ts": 1, "dur": 9}}] * 200
    gz.write(json.dumps({{"traceEvents": events}}).encode())
raw = blob.getvalue()
for piece in range(4):
    time.sleep(2)
    lo, hi = len(raw) * piece // 4, len(raw) * (piece + 1) // 4
    for path in paths:
        with open(path, "ab") as fh:
            fh.write(raw[lo:hi])
PY
  sleep 1
  return 0
}}
"""

NO_NATIVE_STUB = ""   # exercise the fallback path (no preserved native window)


# ==========================================================================
# GPU OCCUPANCY  (mn.py preflight: who actually holds the cards)
# ==========================================================================
#
# Both payloads below are verbatim `rocm-smi --showpids` output captured on
# 2026-09-20, when the fleet had exactly one node the pin would accept and the
# two-node preflight failed. GPUF2B9 was the false positive that motivated this.

SHOWPIDS_HOST_DAEMON_ONLY = """
============================ ROCm System Management Interface ============================
===================================== KFD Processes ======================================
KFD process information:
PID    \tPROCESS NAME\tGPU(s)\tVRAM USED\tSDMA USED\tCU OCCUPANCY
2727680\tgpuagent    \t0     \t0        \t0        \t0
==========================================================================================
"""

SHOWPIDS_REAL_CO_TENANT = """
============================ ROCm System Management Interface ============================
===================================== KFD Processes ======================================
KFD process information:
PID    \tPROCESS NAME\tGPU(s)\tVRAM USED \tSDMA USED\tCU OCCUPANCY
3979771\tpython3     \t0     \t0         \t0        \t0
1792981\tpython3     \t1     \t1323409408\t0        \t0
==========================================================================================
"""


class ResidentGpuPidsTest(unittest.TestCase):
    def test_a_zero_vram_host_daemon_is_not_occupancy(self):
        """gpuagent runs on every host and would fail every pin if counted."""
        holders, parked = mn._resident_gpu_pids(SHOWPIDS_HOST_DAEMON_ONLY)
        self.assertEqual(holders, [])
        self.assertEqual(parked, 1)

    def test_a_neighbour_holding_memory_still_counts(self):
        """The guard's real case: only the pid with VRAM makes the box shared."""
        holders, parked = mn._resident_gpu_pids(SHOWPIDS_REAL_CO_TENANT)
        self.assertEqual(holders, ["1792981"])
        self.assertEqual(parked, 1)

    def test_no_kfd_processes_at_all(self):
        holders, parked = mn._resident_gpu_pids("no KFD processes\n")
        self.assertEqual((holders, parked), ([], 0))


class UtilisationIsOccupancyTest(unittest.TestCase):
    """A utilisation reading may only reject a node if something backs it up.

    Same shape as the gpuagent false positive above: a coarse signal must not
    overrule the precise ones. This one is the more expensive of the two --
    there is no process to reap, so a caller that trusts it waits out its
    whole window and then loses the machine to another team.
    """

    def test_residue_from_an_exited_job_is_not_occupancy(self):
        """GPUF293, minutes after a neighbour's training run finished."""
        self.assertFalse(mn._utilisation_is_occupancy(
            {"max_use": 6.0, "gpu_pids": 0, "serve_procs": 0,
             "free_vram_mib": 196283}))

    def test_a_resident_process_makes_it_real(self):
        self.assertTrue(mn._utilisation_is_occupancy(
            {"max_use": 76.0, "gpu_pids": 8, "serve_procs": 0,
             "free_vram_mib": 46937}))

    def test_an_sglang_serve_makes_it_real_with_no_kfd_pids(self):
        """rocm-smi's pid table can be empty while a server holds the port."""
        self.assertTrue(mn._utilisation_is_occupancy(
            {"max_use": 3.0, "gpu_pids": 0, "serve_procs": 1,
             "free_vram_mib": 20000}))


# ==========================================================================
# SINGLE-CARD GUARD  (launchers/multinode.sh, source time)
# ==========================================================================
#
# BENCH_LAUNCHER is inherited from the run env, so a multi-node run hands this
# launcher to its ISOLATED single-GPU legs as well (op_bench bake-offs, shape
# capture, the recursive kernel layer). Those legs have no pin and cannot fill
# the pinned nodes, so the launcher must step ASIDE rather than abort them. It
# aborted them before this guard existed, and because the abort came from the
# multi-node launcher it read as a cluster fault rather than as a leg that was
# never distributed in the first place.
#
# The guard therefore has to run BEFORE the pin and directory checks, and the
# tests below pin exactly that ordering by pointing GEAK_MN_STATE at a file that
# does not exist: a one-card leg must not need one.

GUARD_DRIVER = """set -uo pipefail
HERE="{scripts}"
adapter_launch() {{ echo "NATIVE_LAUNCH"; }}
source "{launcher}"
echo "SOURCED_OK"
adapter_launch
"""


@unittest.skipIf(BASH is None, "bash is required to exercise the launcher adapter")
class MultinodeSingleCardGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_guard_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _source(self, tp=None):
        driver = os.path.join(self.tmp, "guard.sh")
        with open(driver, "w", encoding="utf-8") as handle:
            handle.write(GUARD_DRIVER.format(
                scripts=SCRIPTS_DIR.replace("\\", "/"),
                launcher=LAUNCHER.replace("\\", "/")))
        env = {k: v for k, v in os.environ.items() if k != "TP"}
        if tp is not None:
            env["TP"] = tp
        env["GEAK_MN_STATE"] = os.path.join(self.tmp, "no_such_pin.json")
        return subprocess.run([BASH, driver], env=env, cwd=self.tmp,
                              capture_output=True, text=True, timeout=120)

    def test_a_one_card_leg_keeps_the_backends_own_launcher(self):
        proc = self._source(tp="1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("SOURCED_OK", proc.stdout)
        # The backend's adapter_launch survived: bench_e2e.sh cloned it, and this
        # launcher returned without replacing it.
        self.assertIn("NATIVE_LAUNCH", proc.stdout)
        self.assertIn("single-card launch", proc.stderr)

    def test_an_unset_tp_is_the_same_case(self):
        """bench_e2e.sh defaults TP to 1, so an absent TP is also one card."""
        proc = self._source(tp=None)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NATIVE_LAUNCH", proc.stdout)

    def test_a_distributed_leg_still_demands_its_pin(self):
        """The guard must not become a way to serve TP>1 without a node set."""
        proc = self._source(tp="16")
        self.assertNotEqual(proc.returncode, 0)
        self.assertNotIn("SOURCED_OK", proc.stdout)
        self.assertIn("no pinned node set", proc.stderr)


@unittest.skipIf(BASH is None, "bash is required to exercise the launcher adapter")
class MultinodeProfileWindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_window_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.profile = os.path.join(self.tmp, "profile")
        os.makedirs(self.profile)
        self.state = os.path.join(self.tmp, "mn_state.json")
        with open(self.state, "w", encoding="utf-8") as handle:
            json.dump({
                "nnodes": 2, "gpus_per_node": 8,
                "nodes": [
                    {"rank": 0, "node_id": "n0", "ip": "10.235.200.32",
                     "hostname": "h0", "cluster_if": "ens50f0", "ngpu": 8},
                    {"rank": 1, "node_id": "n1", "ip": "10.235.200.166",
                     "hostname": "h1", "cluster_if": "ens50f0", "ngpu": 8},
                ],
                "master_ip": "10.235.200.32", "serve_port": 30890,
                "dist_init_port": 30531, "ray_address": "auto",
                "shared_fs": self.tmp, "cluster_subnet": "10.235.200.",
            }, handle)

    def _run(self, *, traces=16, expect=16, stub=STUB_NATIVE, timeout=120,
             quiet_s=5, settle_s=5, base_url="http://127.0.0.1:1"):
        driver = os.path.join(self.tmp, "driver.sh")
        with open(driver, "w", encoding="utf-8") as handle:
            handle.write(DRIVER.format(
                scripts=SCRIPTS_DIR.replace("\\", "/"),
                launcher=LAUNCHER.replace("\\", "/"),
                stub=stub.format(count=traces, collective=COLLECTIVE)
                if stub else ""))
        env = dict(os.environ)
        env.update(
            GEAK_MN_STATE=self.state,
            GEAK_MN_EXT_SHARED_FS=self.tmp,
            GEAK_MN_DIR=os.path.join(SCRIPTS_DIR, "multi_node"),
            PROFILE_DIR=self.profile,
            PROFILE_NUM_STEPS="40",
            TP=str(expect),
            BASE_URL=base_url,
            GEAK_MN_TRACE_TIMEOUT=str(timeout),
            GEAK_MN_TRACE_QUIET_S=str(quiet_s),
            GEAK_MN_TRACE_SETTLE_S=str(settle_s),
            GEAK_MN_GPUS_PER_NODE="8",
        )
        proc = subprocess.run([BASH, driver], env=env, cwd=self.tmp,
                              capture_output=True, text=True, timeout=300)
        # ranks/, not beside the traces: a report in the parsed directory is
        # the newest match for the role's `ls -t *.json.gz *.json | head -1`
        # and gets parsed AS the trace.
        report_path = os.path.join(self.profile, "ranks", "mn_trace_report.json")
        report = {}
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        return proc, report

    def _rc(self, proc):
        for line in proc.stdout.splitlines():
            if line.startswith("WINDOW_RC="):
                return int(line.split("=", 1)[1])
        self.fail(f"the driver never reported a return code:\n{proc.stdout}\n{proc.stderr}")

    def test_complete_set_passes_and_is_gated(self):
        proc, report = self._run(traces=16, expect=16)
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertEqual(report.get("verdict"), "pass", report)
        self.assertIn("16/16 rank trace(s) landed", proc.stdout)
        # The gate ran: only the selected rank is left where the parser looks,
        # and the parser's own selection resolves to it.
        self.assertEqual(len(glob.glob(os.path.join(self.profile, "*.json.gz"))), 1)
        self.assertIn("TP-0-DP-0", report["parser_selection"])
        self.assertEqual(
            len(glob.glob(os.path.join(self.profile, "ranks", "*.json.gz"))), 15)

    def test_incomplete_set_is_reported_but_does_not_fail_the_round(self):
        """Half the world is a bad trace, not a reason to discard the bench."""
        proc, report = self._run(traces=8, expect=16, quiet_s=5, timeout=40)
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertEqual(report.get("verdict"), "fail", report)
        self.assertIn("did NOT pass the gate", proc.stderr)
        self.assertIn("8/16 rank trace(s) landed", proc.stdout)

    def test_no_traces_at_all_fails_the_window(self):
        """Nothing captured: there is no ranking to build, so say so."""
        proc, _ = self._run(traces=0, expect=16, quiet_s=5, timeout=30)
        self.assertEqual(self._rc(proc), 1, proc.stdout + proc.stderr)

    def test_waits_for_streaming_writers_to_finish(self):
        """The regression: files appear instantly, bytes arrive for seconds.

        Observed on the cluster when the wait counted files -- the gate read
        three traces mid-write and six at zero bytes, and the teardown then
        killed the ranks that had not started flushing. The wait must hold
        until the byte total goes quiet, which is proved here by every trace
        being COMPLETE (a gzip truncated by an early return does not read).
        """
        import time
        start = time.time()
        proc, report = self._run(traces=16, expect=16, stub=STUB_STREAMING,
                                 settle_s=5, quiet_s=60, timeout=300)
        elapsed = time.time() - start
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertGreater(elapsed, 8, "returned before the writers finished")
        self.assertEqual(report.get("verdict"), "pass", report.get("problems"))
        self.assertTrue(all(r["readable"] for r in report["per_rank"]),
                        [r for r in report["per_rank"] if not r["readable"]])
        self.assertTrue(all(r["kernel_events"] > 0 for r in report["per_rank"]))

    def test_empty_files_do_not_count_as_landed_traces(self):
        """A created-but-empty trace is a rank that has not started writing."""
        # Braces are doubled: the stub is passed through str.format like the
        # module-level ones.
        stub = """
adapter_profile_window_native() {{
  python3 - "$PROFILE_DIR" <<'PY'
import sys
for tp in range(16):
    open(f"{{sys.argv[1]}}/prof-TP-{{tp}}-DP-{{tp//8}}.trace.json.gz", "wb").close()
PY
  return 0
}}
"""
        proc, _ = self._run(traces=0, expect=16, stub=stub, quiet_s=5,
                            settle_s=5, timeout=40)
        # Zero non-empty traces: the window reports nothing landed and fails.
        self.assertEqual(self._rc(proc), 1, proc.stdout + proc.stderr)
        self.assertIn("0/16 rank trace(s) landed", proc.stdout)

    def test_stuck_export_stops_waiting_instead_of_burning_the_timeout(self):
        """The quiet window, not the total timeout, is what ends a stuck export.

        With 8 of 16 files and no further writes, the wait must end shortly
        after GEAK_MN_TRACE_QUIET_S rather than holding the server (and the
        cluster lock) for the full GEAK_MN_TRACE_TIMEOUT.
        """
        import time
        start = time.time()
        proc, _ = self._run(traces=8, expect=16, quiet_s=5, timeout=600)
        elapsed = time.time() - start
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertLess(elapsed, 120, f"waited {elapsed:.0f}s; the quiet-window "
                                      "check did not end the wait")
        self.assertIn("no new bytes", proc.stderr)

    def test_falls_back_to_a_direct_request_without_a_native_window(self):
        """An older bench_e2e.sh does not preserve the native window.

        The override must still issue a capture request rather than skip it; a
        dead BASE_URL makes the request fail, which is the observable that the
        fallback ran at all.
        """
        proc, _ = self._run(traces=0, expect=16, stub=NO_NATIVE_STUB,
                            quiet_s=5, timeout=30)
        self.assertEqual(self._rc(proc), 1, proc.stdout + proc.stderr)
        self.assertIn("no adapter_profile_window_native", proc.stderr)

    def test_expected_count_can_be_overridden(self):
        """TP need not equal the trace count on every build, so the expectation
        is settable without editing the launcher."""
        driver_env = {"GEAK_MN_EXPECT_TRACES": "8"}
        os.environ.update(driver_env)
        self.addCleanup(os.environ.pop, "GEAK_MN_EXPECT_TRACES", None)
        proc, report = self._run(traces=8, expect=16)
        self.assertEqual(self._rc(proc), 0, proc.stdout + proc.stderr)
        self.assertEqual(report.get("expected_ranks"), 8)
        self.assertEqual(report.get("verdict"), "pass", report)


# ==========================================================================
# TRACE GATE  (mn.py trace-gate)
# ==========================================================================
#
# Tests for multi_node/mn.py trace-gate.
#
# Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
#   or just this section: python3 -m unittest test_mn.TraceGateTest
#
# WHY THESE EXIST: every failure this gate catches is one that otherwise produces
# a PLAUSIBLE Top-N instead of an error, and none of them can be reproduced on
# demand on the cluster -- you cannot ask sixteen ranks to drop rank 11, or to
# truncate one export, on the run where you happen to be watching. So the shapes
# are built here as synthetic traces:
#
#   * the whole remote node missing, which downstream is indistinguishable from a
#     healthy single-host capture;
#   * a truncated export, the signature of a teardown that raced the flush;
#   * a set with no RCCL kernels at all, i.e. a capture that missed the very
#     cross-node traffic a two-node run exists to measure;
#   * and the isolation step, which is what makes the profiler role's
#     `ls -t | head -1` return the SAME rank on the baseline leg and the
#     candidate leg. Without it that selection is decided by NFS flush order and
#     the two legs of an A/B can describe different processes.
#
# The gate must also survive traces too large to hold in memory, so it streams;
# these fixtures stay small on purpose and the streaming path is exercised by
# reading them through the same code.
#

GATE = os.path.join(SCRIPTS_DIR, "multi_node", "mn.py")

COMPUTE = "Cijk_Alik_Bljk_HHS_BH_MT128x128x16"


def write_trace(directory, tp_rank, dp_rank, *, collectives=2, kernels=5,
                truncate=False):
    """One per-rank sglang trace, named exactly as the server names them."""
    events = []
    for i in range(kernels):
        events.append({"cat": "kernel", "name": COMPUTE, "ts": 1000 + i,
                       "dur": 42, "pid": tp_rank, "tid": 1})
    for i in range(collectives):
        events.append({"cat": "kernel", "name": COLLECTIVE, "ts": 2000 + i,
                       "dur": 137, "pid": tp_rank, "tid": 2})
    payload = json.dumps({"schemaVersion": 1, "traceEvents": events})
    path = os.path.join(
        directory, f"prof-2026-09-15-TP-{tp_rank}-DP-{dp_rank}.trace.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    if truncate:
        # A teardown that raced the export: the gzip stream ends mid-member.
        with open(path, "rb") as handle:
            raw = handle.read()
        with open(path, "wb") as handle:
            handle.write(raw[: len(raw) // 2])
    return path


def full_set(directory, ranks=range(16), gpus_per_node=8, **kwargs):
    for rank in ranks:
        write_trace(directory, rank, rank // gpus_per_node, **kwargs)


def run_gate(directory, *args):
    proc = subprocess.run(
        [sys.executable, GATE, "trace-gate", "--profile-dir", directory, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    # The report lives in ranks/, deliberately outside the glob the profiler
    # role uses to pick its trace.
    report = {}
    for candidate in (os.path.join(directory, "ranks", "mn_trace_report.json"),
                      os.path.join(directory, "mn_trace_report.json")):
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                report = json.load(handle)
            break
    return proc, report


def newest_top_level(directory):
    """What the profiler role's `ls -t "$PDIR"/*.json.gz | head -1` returns."""
    candidates = glob.glob(os.path.join(directory, "*.json.gz"))
    candidates.sort(key=os.path.getmtime, reverse=True)
    return os.path.basename(candidates[0]) if candidates else ""


class TraceGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_gate_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    # ---- the healthy shape -------------------------------------------------
    def test_complete_set_passes_and_reports_both_nodes(self):
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["verdict"], "pass", report["problems"])
        self.assertEqual(report["found_ranks"], list(range(16)))
        self.assertEqual(report["nodes_represented"], [0, 1])
        self.assertEqual(len(report["ranks_with_collectives"]), 16)

    def test_isolation_makes_the_parser_selection_deterministic(self):
        """The point of the gate: both A/B legs must parse the SAME rank.

        Before isolation the selection is whichever rank's write landed last;
        after it, only the selected rank is visible at the top level.
        """
        full_set(self.tmp)
        self.assertNotEqual(newest_top_level(self.tmp), "")
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(report["isolated_to_ranks_dir"]), 15)
        self.assertIn("TP-0-DP-0", newest_top_level(self.tmp))
        self.assertIn("TP-0-DP-0", os.path.basename(report["parsed_trace"]))
        # The other fifteen are kept, not deleted: they are the only evidence
        # of what the remote ranks were doing.
        self.assertEqual(
            len(glob.glob(os.path.join(self.tmp, "ranks", "*.json.gz"))), 15)

    def test_the_report_does_not_shadow_the_trace_it_approves(self):
        """The bug this caught in production, on the gate itself.

        The role selects with `ls -t *.json.gz *.json | head -1`, which matches
        plain .json as well -- so a report written beside the traces is the
        newest match and gets parsed AS the trace, yielding a Top-N of zero
        kernels that reads like a bad capture. Nothing the gate writes may land
        in the parsed directory.
        """
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(
            os.path.isfile(os.path.join(self.tmp, "mn_trace_report.json")),
            "the report is in the directory the parser globs")
        self.assertTrue(os.path.isfile(
            os.path.join(self.tmp, "ranks", "mn_trace_report.json")))
        # And the role's own selection resolves to the trace, not the report.
        self.assertIn("TP-0-DP-0", newest_top_level(self.tmp))
        self.assertIn("TP-0-DP-0", report["parser_selection"])

    def test_a_stray_json_that_shadows_the_trace_is_a_failure(self):
        """Any tool's leftover .json breaks the selection, not just the gate's,
        so the check reproduces the role's glob instead of trusting isolation."""
        full_set(self.tmp)
        run_gate(self.tmp, "--expect-ranks", "16")
        stray = os.path.join(self.tmp, "zz_someone_elses_notes.json")
        with open(stray, "w", encoding="utf-8") as handle:
            json.dump({"not": "a trace"}, handle)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["parser_selection"],
                         "zz_someone_elses_notes.json")
        self.assertTrue(any("shadows the trace" in p for p in report["problems"]),
                        report["problems"])

    def test_selected_rank_is_configurable(self):
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16",
                                "--select-rank", "8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("TP-8-DP-1", newest_top_level(self.tmp))

    def test_no_isolate_leaves_every_trace_in_place(self):
        full_set(self.tmp)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16", "--no-isolate")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("isolated_to_ranks_dir", report)
        self.assertEqual(len(glob.glob(os.path.join(self.tmp, "*.json.gz"))), 16)
        # ranks/ holds the report even here: writing it beside the traces would
        # shadow whichever one the parser picks. No TRACE is moved.
        self.assertEqual(os.listdir(os.path.join(self.tmp, "ranks")),
                         ["mn_trace_report.json"])

    # ---- the shapes that must NOT pass silently ----------------------------
    def test_missing_ranks_fail(self):
        full_set(self.tmp, ranks=[r for r in range(16) if r != 11])
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["verdict"], "fail")
        self.assertTrue(any("11" in p for p in report["problems"]),
                        report["problems"])

    def test_whole_remote_node_missing_fails(self):
        """Ranks 0-7 only: a complete-looking capture of half the world."""
        full_set(self.tmp, ranks=range(8))
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["nodes_represented"], [0])
        self.assertTrue(any("node" in p for p in report["problems"]),
                        report["problems"])

    def test_truncated_export_fails(self):
        full_set(self.tmp, ranks=range(15))
        write_trace(self.tmp, 15, 1, truncate=True)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(any("unreadable" in p or "truncated" in p
                            for p in report["problems"]), report["problems"])
        bad = [r for r in report["per_rank"] if r["tp_rank"] == 15][0]
        self.assertFalse(bad["readable"])
        self.assertTrue(bad["error"])

    def test_set_without_collectives_fails(self):
        """No RCCL kernels => the cross-node traffic was never captured."""
        full_set(self.tmp, collectives=0)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["ranks_with_collectives"], [])
        self.assertTrue(any("RCCL" in p for p in report["problems"]),
                        report["problems"])

    def test_empty_dir_is_a_hard_failure(self):
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(report, {})
        self.assertIn("no traces", proc.stderr)

    def test_missing_dir_is_a_hard_failure(self):
        proc, _ = run_gate(os.path.join(self.tmp, "nope"), "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 2)

    def test_selected_rank_absent_is_a_failure(self):
        """Isolating around a rank that never wrote would hand the parser a
        random trace again, so it is called out rather than tolerated."""
        full_set(self.tmp, ranks=range(1, 16))
        proc, report = run_gate(self.tmp, "--expect-ranks", "16",
                                "--select-rank", "0")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["parsed_trace"], "")

    # ---- what the single parsed trace cannot show ---------------------------
    def test_node_imbalance_is_quantified(self):
        full_set(self.tmp, ranks=range(8), kernels=10)          # node0 busy
        for rank in range(8, 16):                               # node1 idle-ish
            write_trace(self.tmp, rank, 1, kernels=2)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        by_node = {int(k): v for k, v in report["kernel_events_by_node"].items()}
        self.assertGreater(by_node[0], by_node[1])
        self.assertGreater(report["node_kernel_imbalance_pct"], 0)

    def test_a_merged_trace_is_not_a_rank_but_does_shadow_the_selection(self):
        """A merged trace carries no TP rank, so it is not counted as one --
        but it still matches the parser's glob, so leaving it beside the
        per-rank traces silently redirects the Top-N to it."""
        full_set(self.tmp)
        with gzip.open(os.path.join(self.tmp, "merged-prof.trace.json.gz"),
                       "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"traceEvents": []}))
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        # Not mistaken for a rank: the set is still 16/16 with no stray entry.
        self.assertEqual(report["found_ranks"], list(range(16)))
        self.assertEqual(report["unranked_files"], [])
        # But the shadowing is called out rather than left to be discovered as
        # an empty Top-N.
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(report["parser_selection"], "merged-prof.trace.json.gz")

    def test_the_gate_is_idempotent_after_isolating(self):
        """Re-running on an already-isolated set must still see 16 ranks.

        The traces it moved into ranks/ are still the set; a second run that
        globbed only the top level would report fifteen ranks missing and turn
        a healthy capture into a failure.
        """
        full_set(self.tmp)
        run_gate(self.tmp, "--expect-ranks", "16")
        proc, report = run_gate(self.tmp, "--expect-ranks", "16")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["found_ranks"], list(range(16)))
        self.assertEqual(report["isolated_to_ranks_dir"], [])
        self.assertIn("TP-0-DP-0", os.path.basename(report["parsed_trace"]))

    def test_gpus_per_node_drives_the_node_mapping(self):
        """TP16 over four 4-GPU nodes maps ranks to nodes differently."""
        full_set(self.tmp, gpus_per_node=4)
        proc, report = run_gate(self.tmp, "--expect-ranks", "16",
                                "--gpus-per-node", "4")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["nodes_represented"], [0, 1, 2, 3])


# ==========================================================================
# TRACE GATE under pd_disagg  (mn.py trace-gate --launch-json)
# ==========================================================================
#
# The PD capture is TWO profiles, and everything the gate exists to prevent gets
# a second, quieter form here:
#
# * The two engines are separate TP worlds, so both write TP-0..TP-7. In one
#   directory that is sixteen files under eight rank numbers, and a gate keyed on
#   the rank number keeps whichever landed second -- a full-looking eight-rank
#   set assembled from both engines at random. The subdirectory the launcher
#   gives each engine is the fix, and this is what checks the gate reads it.
# * `rank // gpus_per_node` files every one of those ranks under node 0, so the
#   two-node check that catches a missing remote node in tp_span fires on a
#   PERFECTLY HEALTHY PD capture. The node has to come from the engine.
# * An engine that produced nothing must be named. "8/16 traces" is the symptom;
#   "the prefill engine produced none" is the fact that tells you the phase you
#   were about to tune has no evidence behind it.


def pd_launch_json(directory, roles=("prefill", "decode"), engine_tp=8):
    path = os.path.join(directory, "mn_launch.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"topology": "pd_disagg", "pd": {
            "engine_tp": engine_tp,
            "engines": [{"role": r, "rank": i, "url": f"http://10.0.0.{i}:3089{i}"}
                        for i, r in enumerate(roles)],
        }}, handle)
    return path


class PdTraceGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_pd_gate_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.profile = os.path.join(self.tmp, "profile")
        os.makedirs(self.profile)

    def engine(self, role, ranks=range(8), **kwargs):
        directory = os.path.join(self.profile, role)
        os.makedirs(directory, exist_ok=True)
        for rank in ranks:
            write_trace(directory, rank, 0, **kwargs)
        return directory

    def gate(self, *args):
        return run_gate(self.profile, "--launch-json",
                        pd_launch_json(self.tmp), *args)

    def test_both_engines_present_passes_with_per_role_accounting(self):
        self.engine("prefill")
        self.engine("decode")
        proc, report = self.gate()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["verdict"], "pass", report["problems"])
        self.assertEqual(report["topology"], "pd_disagg")
        by_role = {r["role"]: r for r in report["per_role"]}
        self.assertEqual(by_role["prefill"]["found_ranks"], list(range(8)))
        self.assertEqual(by_role["decode"]["found_ranks"], list(range(8)))
        # The engine, not `rank // gpus_per_node`, decides the node -- both
        # engines' ranks are 0..7 and the tp_span arithmetic would say node 0.
        self.assertEqual(by_role["prefill"]["node_index"], 0)
        self.assertEqual(by_role["decode"]["node_index"], 1)
        self.assertEqual(report["nodes_represented"], [0, 1])

    def test_each_engine_keeps_its_own_rank0_for_the_parser(self):
        """One trace per engine at its subdirectory's top level: that is what
        makes two per-role Top-Ns possible at all."""
        self.engine("prefill")
        self.engine("decode")
        proc, _report = self.gate()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for role in ("prefill", "decode"):
            directory = os.path.join(self.profile, role)
            self.assertIn("TP-0-DP-0", newest_top_level(directory))
            self.assertEqual(
                len(glob.glob(os.path.join(directory, "*.json.gz"))), 1)
            self.assertEqual(
                len(glob.glob(os.path.join(directory, "ranks", "*.json.gz"))), 7)

    def test_the_roles_index_names_one_trace_per_engine(self):
        self.engine("prefill")
        self.engine("decode")
        self.gate()
        with open(os.path.join(self.profile, "ranks", "mn_trace_roles.json"),
                  encoding="utf-8") as handle:
            index = json.load(handle)
        self.assertEqual([r["role"] for r in index["roles"]],
                         ["prefill", "decode"])
        for entry in index["roles"]:
            self.assertTrue(os.path.isfile(entry["trace"]), entry)
            self.assertIn(f"{os.sep}{entry['role']}{os.sep}", entry["trace"])
            self.assertEqual(entry["ranks"], 8)

    def test_a_silent_engine_is_named_not_counted_as_a_missing_node(self):
        """The round_0 failure: decode saved, prefill's window never closed.

        The old gate called this "only node [0] produced traces", which reads as
        a network/pin problem. The engine that has no profile is the fact.
        """
        self.engine("decode")
        proc, report = self.gate()
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(any("prefill" in p and "NO traces" in p
                            for p in report["problems"]), report["problems"])
        by_role = {r["role"]: r for r in report["per_role"]}
        self.assertEqual(by_role["prefill"]["found_ranks"], [])
        self.assertEqual(by_role["decode"]["found_ranks"], list(range(8)))
        # Still indexed, with an empty trace for the engine that has none: a
        # round with one engine profiled should tune that engine and say so.
        with open(os.path.join(self.profile, "ranks", "mn_trace_roles.json"),
                  encoding="utf-8") as handle:
            index = json.load(handle)
        self.assertEqual({r["role"]: bool(r["trace"]) for r in index["roles"]},
                         {"prefill": False, "decode": True})

    def test_a_rank_missing_inside_one_engine_names_the_engine(self):
        self.engine("prefill", ranks=[r for r in range(8) if r != 5])
        self.engine("decode")
        proc, report = self.gate()
        self.assertEqual(proc.returncode, 1)
        self.assertTrue(any("prefill:5" in p for p in report["problems"]),
                        report["problems"])

    def test_the_shared_directory_collision_is_what_this_avoids(self):
        """Both engines in ONE directory: eight rank numbers, sixteen files.

        Gated WITHOUT --launch-json (the pre-fix behaviour) the set reads as a
        complete single-node capture. This is the control for the tests above.
        """
        for rank in range(8):
            write_trace(self.profile, rank, 0)
        proc, report = run_gate(self.profile, "--expect-ranks", "8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["nodes_represented"], [0])
        self.assertNotIn("per_role", report)


class PdTopnMergeTest(unittest.TestCase):
    """mn.py topn-merge: the two engine tables folded into the one table the
    Architect routes on, weighted by each engine's share of GPU time."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mn_topn_")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def topn(self, role, total_ms, kernels):
        path = os.path.join(self.tmp, f"profile_topN_{role}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "source": "torch-trace", "total_gpu_time_ms": total_ms,
                "num_kernel_launches": 100, "num_distinct_kernels": len(kernels),
                "top_kernels": [
                    {"rank": i + 1, "short_name": name, "calls": 10,
                     "total_ms": total_ms * pct / 100.0, "avg_us": 1.0,
                     "pct_gpu_time": pct, "shapes": [], "dtypes": [],
                     "classification": "library_gemm", "backend_guess": "hipblaslt",
                     "editable": False, "opt_hint": "hint"}
                    for i, (name, pct) in enumerate(kernels)],
            }, handle)
        return f"{role}={path}"

    def merge(self, *roles):
        out = os.path.join(self.tmp, "profile_topN")
        proc = subprocess.run(
            [sys.executable, GATE, "topn-merge", "--out", out,
             *[a for r in roles for a in ("--role", r)]],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        doc = {}
        if os.path.isfile(out + ".json"):
            with open(out + ".json", encoding="utf-8") as handle:
                doc = json.load(handle)
        return proc, doc

    def test_rows_are_weighted_by_each_engine_share_of_gpu_time(self):
        """A kernel at 60% of an engine that is a quarter of the deployment is
        15% of the deployment -- NOT 60%, which is what reading one engine's
        table as the whole picture would say."""
        proc, doc = self.merge(self.topn("prefill", 100.0, [("gemm_p", 60.0)]),
                               self.topn("decode", 300.0, [("attn_d", 40.0)]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(doc["total_gpu_time_ms"], 400.0)
        rows = {r["short_name"]: r for r in doc["top_kernels"]}
        self.assertAlmostEqual(rows["gemm_p"]["pct_gpu_time"], 15.0, places=3)
        self.assertAlmostEqual(rows["attn_d"]["pct_gpu_time"], 30.0, places=3)
        # Decode's kernel outranks prefill's despite the smaller in-engine share.
        self.assertEqual(doc["top_kernels"][0]["short_name"], "attn_d")
        # The in-engine share survives: it is what that engine's own knobs move.
        self.assertAlmostEqual(rows["gemm_p"]["pct_gpu_time_in_role"], 60.0)

    def test_every_row_carries_its_engine(self):
        _proc, doc = self.merge(self.topn("prefill", 100.0, [("shared", 50.0)]),
                                self.topn("decode", 100.0, [("shared", 50.0)]))
        # One name on both engines stays TWO rows: same kernel, different
        # shapes, different fix.
        self.assertEqual(sorted(r["role"] for r in doc["top_kernels"]),
                         ["decode", "prefill"])

    def test_one_unreadable_engine_still_yields_the_other(self):
        """A round with one engine profiled is a real result; failing the merge
        would throw it away along with the bench that came with it."""
        proc, doc = self.merge(self.topn("decode", 100.0, [("attn_d", 40.0)]),
                               f"prefill={os.path.join(self.tmp, 'nope.json')}")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual([r["role"] for r in doc["per_role"]], ["decode"])
        self.assertAlmostEqual(doc["top_kernels"][0]["pct_gpu_time"], 40.0)
        self.assertIn("skipping prefill", proc.stderr)

    def bench(self, ttft_ms, tpot_ms):
        path = os.path.join(self.tmp, "bench_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"ttft_ms_median": ttft_ms, "tpot_ms_median": tpot_ms}, handle)
        return path

    def merge_with_bench(self, bench, osl, *roles):
        out = os.path.join(self.tmp, "profile_topN")
        proc = subprocess.run(
            [sys.executable, GATE, "topn-merge", "--out", out,
             "--bench-summary", bench, "--osl", str(osl),
             *[a for r in roles for a in ("--role", r)]],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        with open(out + ".json", encoding="utf-8") as handle:
            return proc, json.load(handle)

    def test_the_bench_latency_split_names_the_engine_throughput_waits_on(self):
        """GPU time says prefill; the bench says the requests wait on decode.

        Prefill holds 75% of GPU time, but TTFT 200 ms against 32 x 25 ms of
        decode puts 80% of every request's life on the decode engine. Under a
        closed-loop bench that 80% is the Amdahl weight on throughput."""
        proc, doc = self.merge_with_bench(
            self.bench(200.0, 25.0), 33,
            self.topn("prefill", 300.0, [("gemm_p", 50.0)]),
            self.topn("decode", 100.0, [("attn_d", 50.0)]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(doc["bottleneck_engine"], "decode")
        self.assertAlmostEqual(doc["latency_split"]["share"]["decode"], 0.8, places=4)
        roles = {r["role"]: r for r in doc["per_role"]}
        self.assertAlmostEqual(roles["prefill"]["e2e_share_pct"], 20.0)
        rows = {r["short_name"]: r for r in doc["top_kernels"]}
        # The ranking is still GPU time...
        self.assertEqual(doc["top_kernels"][0]["short_name"], "gemm_p")
        # ...and each row now also says what it can buy.
        self.assertAlmostEqual(rows["gemm_p"]["pct_e2e_est"], 10.0, places=3)
        self.assertAlmostEqual(rows["attn_d"]["pct_e2e_est"], 40.0, places=3)

    def test_a_bench_without_latencies_leaves_the_gpu_time_merge_alone(self):
        path = os.path.join(self.tmp, "bench_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"throughput_tok_s_median": 100.0}, handle)
        proc, doc = self.merge_with_bench(
            path, 33, self.topn("prefill", 100.0, [("gemm_p", 60.0)]),
            self.topn("decode", 300.0, [("attn_d", 40.0)]))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("no TTFT/TPOT", proc.stderr)
        self.assertNotIn("latency_split", doc)
        self.assertNotIn("bottleneck_engine", doc)
        self.assertIsNone(doc["top_kernels"][0]["pct_e2e_est"])
        self.assertAlmostEqual(doc["top_kernels"][0]["pct_gpu_time"], 30.0, places=3)


class PdSweepLeakTest(unittest.TestCase):
    """A one-engine knob that is ALSO in the shared string runs on both engines.

    Argparse last-wins still gives the named engine its own value, so nothing
    fails at launch; the leak is only visible if it is looked for."""

    NAMES = ("GEAK_MN_PD_SWEEP_ARGS_PREFILL", "GEAK_MN_PD_SWEEP_ARGS_DECODE",
             "GEAK_MN_PD_SWEEP_ENV_PREFILL", "GEAK_MN_PD_SWEEP_ENV_DECODE")

    def setUp(self):
        self._saved = {n: os.environ.pop(n, None) for n in self.NAMES}

    def tearDown(self):
        for name, value in self._saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    def test_a_decode_flag_also_in_the_shared_args_is_reported_for_decode(self):
        os.environ["GEAK_MN_PD_SWEEP_ARGS_DECODE"] = "--cuda-graph-max-bs 256"
        self.assertEqual(
            mn.pd_sweep_leaks("--mem-fraction-static 0.9 --cuda-graph-max-bs=128", "",
                              ["prefill", "decode"]),
            {"decode": ["--cuda-graph-max-bs"]})

    def test_env_leaks_are_found_by_key(self):
        os.environ["GEAK_MN_PD_SWEEP_ENV_PREFILL"] = "SGLANG_USE_AITER=1"
        self.assertEqual(mn.pd_sweep_leaks("", "SGLANG_USE_AITER=0", ["prefill", "decode"]),
                         {"prefill": ["SGLANG_USE_AITER"]})

    def test_disjoint_layers_are_clean(self):
        os.environ["GEAK_MN_PD_SWEEP_ARGS_PREFILL"] = "--chunked-prefill-size 8192"
        self.assertEqual(mn.pd_sweep_leaks("--mem-fraction-static 0.9", "",
                                           ["prefill", "decode"]), {})

    def test_an_unbalanced_quote_does_not_crash_the_launch(self):
        os.environ["GEAK_MN_PD_SWEEP_ARGS_DECODE"] = "--x 'oops"
        self.assertEqual(mn.pd_sweep_leaks("--x 1", "", ["decode"]), {"decode": ["--x"]})


class PdRankOverlayTest(unittest.TestCase):
    """A kernel accepted for one engine reaches that engine only.

    It was measured on that engine's shapes; the other engine would run the
    same patched function at an M nobody measured."""

    NAMES = ("GEAK_MN_PD_OVERLAY_PREFILL", "GEAK_MN_PD_OVERLAY_DECODE")

    def setUp(self):
        self._saved = {n: os.environ.pop(n, None) for n in self.NAMES}

    def tearDown(self):
        for name, value in self._saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value

    def test_the_engine_overlay_goes_ahead_of_the_shared_one_on_that_engine_only(self):
        os.environ["GEAK_MN_PD_OVERLAY_DECODE"] = "/geak/overlay/cand_attn"
        self.assertEqual(mn.pd_rank_overlay("decode", "/geak/overlay/shared"),
                         "/geak/overlay/cand_attn:/geak/overlay/shared")
        self.assertEqual(mn.pd_rank_overlay("prefill", "/geak/overlay/shared"),
                         "/geak/overlay/shared")

    def test_unset_leaves_the_shared_overlay_byte_identical(self):
        self.assertEqual(mn.pd_rank_overlay("prefill", "/ovl"), "/ovl")
        self.assertEqual(mn.pd_rank_overlay("decode", ""), "")

    def test_an_engine_overlay_with_no_shared_layer_stands_alone(self):
        os.environ["GEAK_MN_PD_OVERLAY_PREFILL"] = "/geak/overlay/cand_gemm"
        self.assertEqual(mn.pd_rank_overlay("prefill", ""), "/geak/overlay/cand_gemm")

    def test_the_rendered_rank_script_exports_the_composed_overlay(self):
        with open(os.path.join(MN, "serve.sh.tmpl"), encoding="utf-8") as f:
            tmpl = f.read()
        os.environ["GEAK_MN_PD_OVERLAY_DECODE"] = "/dec"
        values = {"LAUNCH_ID": "o1", "RANK": "1", "NNODES": "2", "MASTER": "10.0.0.1",
                  "PORT": "30892", "BASE_URL": "http://10.0.0.2:30892",
                  "STRUCTURAL_CALL": "adapter_pd_args decode 30898 mooncake_tcp",
                  "DIST_PORT": "30531", "MODEL": "/m", "TP": "8", "GPU": "0",
                  "MEM_FRACTION": "0.9", "PROFILE_DIR": "/p", "LOG": "/l",
                  "ADAPTER": "/a.sh", "PYTHONPATH": "", "EXTRA_ENV": "",
                  "RECIPE_ARGS": "", "EXTRA_ARGS": "", "CLUSTER_SUBNET": "10.0.0."}
        decode = mn.render(tmpl, {**values, "OVERLAY": mn.pd_rank_overlay("decode", "/ovl")})
        prefill = mn.render(tmpl, {**values, "RANK": "0",
                                   "OVERLAY": mn.pd_rank_overlay("prefill", "/ovl")})
        self.assertIn("OVERLAY_PYTHONPATH=/dec:/ovl ", decode)
        self.assertIn("OVERLAY_PYTHONPATH=/ovl ", prefill)
        self.assertNotIn("/dec:", prefill)


# ==========================================================================
# PREFILL/DECODE DISAGGREGATION  (topology=pd_disagg)
# ==========================================================================
#
# The pd_disagg topology puts a COMPLETE server on each pinned node -- prefill on
# rank0, decode on rank1, each TP-closed inside its own box -- and a load balancer
# on the master. Every case below is a way that can go wrong while still
# producing a number, which is the only kind of failure worth a test here:
#
# * A pin of the wrong topology gets reused. The launch succeeds, serves the
#   other shape, and the throughput is filed under this one.
# * The rank/world triple leaks into a PD engine. Each engine would then wait at
#   rendezvous for a peer that does not exist, which reads as a slow cold start.
# * The bootstrap flag is set on decode as well. It names the PREFILL side's
#   registry there, so a second and possibly disagreeing source of truth.
# * The engines are reachable where the client dials. A bench that hit an engine
#   directly would measure prefill alone and call it end to end.

PD_STATE = {
    "nnodes": 2, "gpus_per_node": 8,
    "topology": "pd_disagg",
    "master_ip": "10.235.200.166",
    "serve_port": 30890, "dist_init_port": 30531,
    "shared_fs": "/workhome/geak",
    "cluster_subnet": "10.235.200.",
    "nodes": [
        {"rank": 0, "node_id": "n0", "ip": "10.235.200.166",
         "hostname": "smc300x-ccs-aus-gpufc54", "cluster_if": "ens50f0", "ngpu": 8},
        {"rank": 1, "node_id": "n1", "ip": "10.235.200.183",
         "hostname": "smc300x-ccs-aus-gpuf2a0", "cluster_if": "ens50f0", "ngpu": 8},
    ],
}

FAKE_PD_ADAPTER = """\
adapter_pd_mode() { echo pd_roles; }
adapter_pd_args() {
  local role="$1" boot="$2" backend="${3:-mooncake_tcp}" ib="${4:-}"
  local out="--disaggregation-mode $role --disaggregation-transfer-backend $backend"
  [ "$role" = "prefill" ] && out="$out --disaggregation-bootstrap-port $boot"
  [ -n "$ib" ] && out="$out --disaggregation-ib-device $ib"
  echo "$out"
}
adapter_pd_lb_launch() {
  echo "LB_CALL=[$1 $2 $3 $4 $5]"
  echo "LB_MODEL=$MODEL LB_EXTRA=[$LB_EXTRA_ARGS]"
  SERVER_PID=777
}
adapter_launch() {
  echo "FINAL_ARGS=[$EXTRA_SERVER_ARGS]"
  echo "GPU=$GPU HOST=$HOST PORT=$PORT TP=$TP BASE_URL=$BASE_URL"
  SERVER_PID=4242
}
"""


class PdConfigTest(unittest.TestCase):
    """The plan derived from a pin, and the one property that must not move."""

    def test_the_client_port_is_the_load_balancers(self):
        # Everything downstream of the launcher -- BASE_URL, the health loop,
        # the bench client -- keeps working across topologies precisely because
        # the port the client dials is unchanged. The engines move instead.
        pd = mn.pd_config(PD_STATE)
        self.assertEqual(pd["lb_port"], PD_STATE["serve_port"])
        self.assertNotIn(PD_STATE["serve_port"], pd["role_ports"])

    def test_defaults_are_derived_from_the_pin(self):
        pd = mn.pd_config(PD_STATE)
        self.assertEqual(pd["roles"], ["prefill", "decode"])
        self.assertEqual(pd["role_ports"], [30891, 30892])
        self.assertEqual(pd["bootstrap_port"], 30898)

    def test_the_transfer_backend_defaults_to_tcp(self):
        # This fleet has no verbs provider: the RDMA backends have nothing to
        # bind to, and mooncake_tcp is the one that resolves the IB device to
        # None instead of hunting for an HCA that is not there.
        self.assertEqual(mn.pd_config(PD_STATE)["transfer_backend"], "mooncake_tcp")

    def test_an_explicit_plan_in_the_pin_wins_over_the_defaults(self):
        state = dict(PD_STATE, pd={"roles": ["prefill", "decode"],
                                   "role_ports": [31001, 31002],
                                   "bootstrap_port": 31008,
                                   "transfer_backend": "mooncake",
                                   "lb": "minilb"})
        pd = mn.pd_config(state)
        self.assertEqual(pd["role_ports"], [31001, 31002])
        self.assertEqual(pd["lb"], "minilb")


class TopologyOfTest(unittest.TestCase):
    def test_the_pin_decides_not_the_environment(self):
        """BENCH_LAUNCHER is inherited by every leg; the rest of the env is not.

        A leg that read the topology from its own environment would get the
        default and launch the other shape onto these machines.
        """
        self.assertEqual(mn.topology_of(PD_STATE), "pd_disagg")

    def test_an_older_pin_without_the_field_is_tp_span(self):
        self.assertEqual(mn.topology_of({"serve_port": 30890}), "tp_span")

    def test_an_unknown_topology_is_refused_by_name(self):
        os.environ["GEAK_MN_TOPOLOGY"] = "half_and_half"
        try:
            with self.assertRaises(SystemExit) as e:
                mn.topology_of(PD_STATE)
            self.assertIn("half_and_half", str(e.exception))
        finally:
            del os.environ["GEAK_MN_TOPOLOGY"]


class PdRenderTest(unittest.TestCase):
    """What each engine is actually told, rendered through the real template."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "fake.sh"), "w", encoding="utf-8") as f:
            f.write(FAKE_PD_ADAPTER)
        with open(os.path.join(MN, "serve.sh.tmpl"), encoding="utf-8") as f:
            self.tmpl = f.read()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def engine(self, rank):
        """Render one engine the way cmd_launch does for pd_disagg."""
        pd = mn.pd_config(PD_STATE)
        node = PD_STATE["nodes"][rank]
        port = pd["role_ports"][rank]
        return mn.render(self.tmpl, {
            "LAUNCH_ID": "pd01", "RANK": str(rank), "NNODES": "2",
            "MASTER": PD_STATE["master_ip"], "PORT": str(port),
            "BASE_URL": f"http://{node['ip']}:{port}",
            "STRUCTURAL_CALL": (f'adapter_pd_args {pd["roles"][rank]} '
                                f'{pd["bootstrap_port"]} '
                                f'{pd["transfer_backend"]} ').rstrip(),
            "DIST_PORT": str(PD_STATE["dist_init_port"]),
            "MODEL": "/models/m", "TP": "8", "GPU": "0,1,2,3,4,5,6,7",
            "MEM_FRACTION": "0.90", "PROFILE_DIR": os.path.join(self.tmp, "prof"),
            "LOG": os.path.join(self.tmp, f"server.rank{rank}.log"),
            "ADAPTER": os.path.join(self.tmp, "fake.sh"), "OVERLAY": "/ovl",
            "PYTHONPATH": "", "EXTRA_ENV": "RECIPE_V=1",
            "RECIPE_ARGS": "--attention-backend aiter",
            "EXTRA_ARGS": "--chunked-prefill-size 8192",
            "CLUSTER_SUBNET": "10.235.200.",
        })

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_the_prefill_engine_gets_the_role_and_the_bootstrap_registry(self):
        out = subprocess.run(["bash"], input=self.engine(0),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertIn("--disaggregation-mode prefill", args)
        self.assertIn("--disaggregation-bootstrap-port 30898", args)
        self.assertIn("--disaggregation-transfer-backend mooncake_tcp", args)

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_the_decode_engine_is_not_given_a_bootstrap_port(self):
        # There the flag names the PREFILL side's registry, which decode already
        # receives per request from the load balancer.
        out = subprocess.run(["bash"], input=self.engine(1),
                             capture_output=True, text=True)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertIn("--disaggregation-mode decode", args)
        self.assertNotIn("--disaggregation-bootstrap-port", args)

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_no_rank_or_world_reaches_a_disaggregated_engine(self):
        """The two engines never form a collective.

        A --nnodes/--node-rank/--dist-init-addr triple here would make each one
        wait at rendezvous for a peer that is not coming, and the symptom is a
        cold start that never finishes rather than a bad flag.
        """
        for rank in (0, 1):
            out = subprocess.run(["bash"], input=self.engine(rank),
                                 capture_output=True, text=True)
            args = [l for l in out.stdout.splitlines()
                    if l.startswith("FINAL_ARGS=")][0]
            for leak in ("--nnodes", "--node-rank", "--dist-init-addr"):
                self.assertNotIn(leak, args, f"rank{rank} was given {leak}")

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_each_engine_answers_on_its_own_endpoint_not_the_clients(self):
        for rank, port in ((0, "30891"), (1, "30892")):
            out = subprocess.run(["bash"], input=self.engine(rank),
                                 capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("GPU=")][0]
            self.assertIn(f"PORT={port}", line)
            # TP is per engine: the caller's TP names the whole pin, and handing
            # it to one box would ask eight cards to be sixteen.
            self.assertIn("TP=8", line)
            self.assertNotIn(":30890", line)

    @unittest.skipUnless(HAVE_SH, "needs bash and ip")
    def test_the_recipe_still_comes_before_the_swept_flags(self):
        # Same layering contract as tp_span, and for the same reason: argparse
        # is last-wins, so GEAK's accepted flags must land after the recipe.
        out = subprocess.run(["bash"], input=self.engine(0),
                             capture_output=True, text=True)
        args = [l for l in out.stdout.splitlines() if l.startswith("FINAL_ARGS=")][0]
        self.assertLess(args.index("--attention-backend aiter"),
                        args.index("--chunked-prefill-size 8192"))
        # ...and the structural half is last, so no sweep can change which role
        # this process is.
        self.assertLess(args.index("--chunked-prefill-size 8192"),
                        args.index("--disaggregation-mode"))


class PdRankSweepTest(unittest.TestCase):
    """A swept flag aimed at one engine must not land on the other.

    EXTRA_SERVER_ARGS is one string and both engines receive it. The per-role
    sweep is a later layer, so last-wins can still override the shared string
    on the engine that owns the knob without retuning its partner.
    """

    def setUp(self):
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "GEAK_MN_PD_SWEEP_ARGS_PREFILL", "GEAK_MN_PD_SWEEP_ARGS_DECODE",
                "GEAK_MN_PD_SWEEP_ENV_PREFILL", "GEAK_MN_PD_SWEEP_ENV_DECODE",
            )
        }
        for name in self._saved:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_each_engine_keeps_only_its_own_sweep(self):
        os.environ["GEAK_MN_PD_SWEEP_ARGS_PREFILL"] = "--chunked-prefill-size 8192"
        os.environ["GEAK_MN_PD_SWEEP_ARGS_DECODE"] = "--max-running-requests 64"
        os.environ["GEAK_MN_PD_SWEEP_ENV_PREFILL"] = "PREFILL_ONLY=1"
        os.environ["GEAK_MN_PD_SWEEP_ENV_DECODE"] = "DECODE_ONLY=1"
        shared = ["SHARED=1"]
        prefill_args, prefill_env, prefill_dropped = mn.pd_rank_swept(
            "prefill", "--attention-backend aiter", shared)
        decode_args, decode_env, decode_dropped = mn.pd_rank_swept(
            "decode", "--attention-backend aiter", shared)
        self.assertEqual(
            prefill_args, "--attention-backend aiter --chunked-prefill-size 8192")
        self.assertNotIn("max-running-requests", prefill_args)
        self.assertEqual(
            decode_args, "--attention-backend aiter --max-running-requests 64")
        self.assertNotIn("chunked-prefill-size", decode_args)
        # Shared sweep stays ahead of the role sweep, so a role value wins a
        # collision on that engine only.
        self.assertLess(prefill_args.index("--attention-backend"),
                        prefill_args.index("--chunked-prefill-size"))
        self.assertEqual(prefill_env, ["SHARED=1", "PREFILL_ONLY=1"])
        self.assertEqual(decode_env, ["SHARED=1", "DECODE_ONLY=1"])
        self.assertEqual(prefill_dropped, [])
        self.assertEqual(decode_dropped, [])
        # The shared list the caller handed in is not mutated.
        self.assertEqual(shared, ["SHARED=1"])

    def test_unset_role_sweep_leaves_the_shared_layer(self):
        args, env, dropped = mn.pd_rank_swept("prefill", "--page-size 64", ["A=1"])
        self.assertEqual(args, "--page-size 64")
        self.assertEqual(env, ["A=1"])
        self.assertEqual(dropped, [])


class PdLoadBalancerRenderTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "fake.sh"), "w", encoding="utf-8") as f:
            f.write(FAKE_PD_ADAPTER)
        with open(os.path.join(MN, "lb.sh.tmpl"), encoding="utf-8") as f:
            self.tmpl = f.read()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def script(self, lb="router"):
        pd = mn.pd_config(PD_STATE)
        return mn.render(self.tmpl, {
            "LAUNCH_ID": "pd01", "LB_HOST": "0.0.0.0",
            "LB_PORT": str(pd["lb_port"]),
            "PREFILL_URL": "http://10.235.200.166:30891",
            "DECODE_URL": "http://10.235.200.183:30892",
            "BOOTSTRAP_PORT": str(pd["bootstrap_port"]),
            "MODEL": "/models/m", "LOG": os.path.join(self.tmp, "mn_lb.log"),
            "PYTHONPATH": "", "EXTRA_ENV": "",
            "ADAPTER": os.path.join(self.tmp, "fake.sh"),
            "LB_EXTRA_ARGS": "--mini-lb" if lb == "minilb" else "",
        })

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_lb_script_delegates_and_is_handed_both_engines(self):
        out = subprocess.run(["bash"], input=self.script(),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        call = [l for l in out.stdout.splitlines() if l.startswith("LB_CALL=")][0]
        # Only the fake adapter can print this, so the template cannot have
        # carried a launch line of its own.
        self.assertEqual(
            call,
            "LB_CALL=[0.0.0.0 30890 http://10.235.200.166:30891 30898 "
            "http://10.235.200.183:30892]")
        self.assertIn("STARTED_PID=777", out.stdout)

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_lb_choice_travels_as_an_argument_not_a_second_template(self):
        out = subprocess.run(["bash"], input=self.script(lb="minilb"),
                             capture_output=True, text=True)
        self.assertIn("LB_EXTRA=[--mini-lb]", out.stdout)


class PdAdapterSeamTest(unittest.TestCase):
    """A backend that cannot describe PD must fail by name, here."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, name, body):
        p = os.path.join(self.tmp, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
        return p

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_an_adapter_with_no_pd_functions_is_refused(self):
        p = self.write("tponly.sh", "adapter_dist_mode() { echo per_rank; }\n"
                                    "adapter_dist_args() { echo x; }\n")
        os.environ["GEAK_MN_BACKEND_ADAPTER"] = p
        try:
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("tponly", self.tmp, "pd_disagg")
            self.assertIn("adapter_pd_args", str(e.exception))
        finally:
            del os.environ["GEAK_MN_BACKEND_ADAPTER"]

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_an_adapter_that_can_split_but_not_balance_is_refused(self):
        # Half a contract is the worse case: the engines would come up and
        # nothing would serve the client port.
        p = self.write("half.sh", "adapter_pd_mode() { echo pd_roles; }\n"
                                  "adapter_pd_args() { echo x; }\n")
        os.environ["GEAK_MN_BACKEND_ADAPTER"] = p
        try:
            with self.assertRaises(SystemExit) as e:
                mn.resolve_adapter("half", self.tmp, "pd_disagg")
            self.assertIn("adapter_pd_lb_launch", str(e.exception))
        finally:
            del os.environ["GEAK_MN_BACKEND_ADAPTER"]

    @unittest.skipUnless(HAVE_SH, "needs bash")
    def test_the_real_sglang_adapter_declares_pd_roles(self):
        root = os.path.realpath(os.path.join(HERE, "..", ".."))
        self.assertTrue(mn.resolve_adapter("sglang", root, "pd_disagg")
                        .endswith("adapters/sglang.sh"))


class PdPinReuseTest(unittest.TestCase):
    """Reusing a pin of the other topology is refused rather than served.

    This is the quietest failure of the set: the machines are right, the mount
    is right, the launch succeeds -- and what gets served is the other
    configuration, reported as this one.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.env = {k: os.environ.get(k) for k in
                    ("GEAK_MN_STATE", "GEAK_MN_EXT_SHARED_FS", "GEAK_MN_TOPOLOGY")}
        os.environ["GEAK_MN_STATE"] = os.path.join(self.tmp, "mn_state.json")
        os.environ["GEAK_MN_EXT_SHARED_FS"] = self.tmp
        os.environ.pop("GEAK_MN_TOPOLOGY", None)

    def tearDown(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def pin(self, topology):
        state = dict(PD_STATE, topology=topology, shared_fs=self.tmp)
        with open(os.environ["GEAK_MN_STATE"], "w", encoding="utf-8") as f:
            json.dump(state, f)

    def test_a_tp_span_pin_is_not_reused_for_a_pd_run(self):
        self.pin("tp_span")
        self.assertEqual(mn.cmd_preflight(["--topology", "pd_disagg"]), 2)

    def test_a_pd_pin_is_not_reused_for_a_tp_span_run(self):
        self.pin("pd_disagg")
        self.assertEqual(mn.cmd_preflight(["--topology", "tp_span"]), 2)

    def test_pd_disagg_pins_exactly_one_prefill_and_one_decode(self):
        # Three nodes have no meaning yet: which is the second decode, and does
        # the LB then split traffic? Refuse instead of inventing an answer.
        with self.assertRaises(SystemExit) as e:
            mn.cmd_preflight(["--repin", "--topology", "pd_disagg", "--nnodes", "3"])
        self.assertIn("one prefill and one decode", str(e.exception))

    def test_a_repin_keeps_the_pd_topology_when_none_is_stated(self):
        """The failure this exists to prevent is silent, and it really happened.

        A co-tenant takes the pinned pair mid-run; the teardown refusal tells the
        caller to `mn.py preflight --repin`; that used to default to tp_span, so a
        disaggregated run turned into a TP-spanning one -- different processes,
        different ports, no load balancer -- and carried on reporting throughput
        against a PD baseline. Nothing failed, so nothing was noticed.

        Asserted through the nnodes guard because it is the one branch that fires
        before the cluster is contacted: it triggers ONLY for pd_disagg, so
        reaching it from a bare --repin proves pd_disagg was what got inherited.
        """
        self.pin("pd_disagg")
        with self.assertRaises(SystemExit) as e:
            mn.cmd_preflight(["--repin", "--nnodes", "3"])
        self.assertIn("one prefill and one decode", str(e.exception))

    def reached_selection(self, argv):
        """Run a --repin as far as node selection, then stop.

        tp_span has no pre-cluster guard, so "it got to connect()" is the only
        signal available that the inherit did NOT resolve to pd_disagg.
        """
        sentinel = RuntimeError("reached node selection")
        with mock.patch.object(mn, "connect", side_effect=sentinel):
            with self.assertRaises(RuntimeError) as e:
                mn.cmd_preflight(argv)
        return str(e.exception)

    def test_a_repin_with_no_pin_at_all_is_still_tp_span(self):
        if os.path.isfile(os.environ["GEAK_MN_STATE"]):
            os.remove(os.environ["GEAK_MN_STATE"])
        self.assertIn("reached node selection",
                      self.reached_selection(["--repin", "--nnodes", "3"]))

    def test_an_explicit_topology_still_overrides_the_pin(self):
        """Inheriting must not become a cage: a deliberate switch still works."""
        self.pin("pd_disagg")
        self.assertIn("reached node selection", self.reached_selection(
            ["--repin", "--topology", "tp_span", "--nnodes", "3"]))

    def test_the_co_tenant_refusal_names_the_topology_to_keep(self):
        """That message is read by an agent as an instruction to run verbatim."""
        self.assertIn("--topology pd_disagg",
                      mn.cotenant_refusal(dict(PD_STATE, topology="pd_disagg")))
        self.assertIn("--topology tp_span",
                      mn.cotenant_refusal(dict(PD_STATE, topology="tp_span")))


class PdTeardownTest(unittest.TestCase):
    def test_the_load_balancer_is_in_the_kill_set(self):
        """It holds no GPU, so the idleness wait would never notice it.

        What it does hold is the client port, and a survivor there answers
        /health from a front end whose engines are gone: the next launch looks
        instantly healthy and every request hangs.
        """
        self.assertIn("mini_lb", mn.LB_PATTERN)
        # ...and it is NOT one of the serve patterns, or a node with only a
        # stranded front end would read as still holding an engine.
        self.assertNotIn("router", mn.SERVE_PATTERN)

    def test_the_pattern_matches_the_name_the_router_gives_itself(self):
        """Not the one on our launch line -- they are different strings.

        `python -m sglang_router.launch_router` overwrites its own argv with
        "sglang::router" almost immediately, so by the time anything greps for
        it, the module path we spawned it with is no longer in the process
        table. A pattern copied from the launch line matches nothing, and the
        LB then survives every teardown that believes it was killed.
        """
        seen = "sglang::router"          # what ps/pgrep actually report
        self.assertRegex(seen, mn.LB_PATTERN)
        launched_as = "python -m sglang_router.launch_router --pd-disaggregation"
        self.assertNotRegex(launched_as, mn.LB_PATTERN)

    def test_killing_the_load_balancer_does_not_match_the_engines(self):
        """`sglang::router` and `sglang::scheduler` differ by one word.

        A pattern loose enough to cover both would make the LB teardown reap
        every engine worker on the node -- including, on a shared box, someone
        else's.
        """
        self.assertNotRegex("sglang::scheduler", mn.LB_PATTERN)
        self.assertNotRegex("sglang::scheduler_TP0", mn.LB_PATTERN)


class PdLoadBalancerLivenessTest(unittest.TestCase):
    """The supervisor must ASK the LB, not look it up by name.

    This is a regression test with a cost attached. The first version probed
    `pgrep -f <the module path on our launch line>`, which the router had
    already renamed away from; the probe returned zero, the supervisor
    declared the front end gone and tore the pair down -- and the LB's own log
    showed it answering /health with 200 on a five-second cadence throughout,
    including the second it was killed. A name is owned by the process and can
    be rewritten at will. The client port cannot: serving it is the entire job.
    """

    def setUp(self):
        self.state = dict(PD_STATE, pd=mn.pd_config(PD_STATE))
        self.sent = []
        replies = self.replies = {}

        class _Task:
            def options(_self, **_kw):
                return _self

            def remote(_self, cmd, _timeout):
                self.sent.append(cmd)
                return {"out": replies.get(len(self.sent) - 1, "serve=1 sched=8 zombie=0")}

        self.patches = [
            mock.patch.object(mn, "run_on_node", _Task()),
            mock.patch.object(mn, "resolve_pinned", lambda s: s["nodes"]),
            mock.patch.object(mn, "pinned", lambda _nid: None),
            mock.patch.object(mn, "ray", mock.Mock(get=lambda xs: xs)),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()

    def test_the_probe_queries_the_client_port(self):
        mn.engine_health(self.state)
        master = self.sent[0]
        self.assertIn(str(self.state["pd"]["lb_port"]), master)
        self.assertIn("/health", master)

    def test_a_router_that_answers_is_not_reported_dead(self):
        """The exact shape of the run this test exists because of."""
        self.replies[0] = "serve=1 sched=8 zombie=0 lb=200"
        master = mn.engine_health(self.state)[0]
        self.assertEqual(master["lb"], 1)
        self.assertEqual(master["lb_http"], 200)

    def test_nothing_listening_is_reported_dead(self):
        self.replies[0] = "serve=1 sched=8 zombie=0 lb=000"
        self.assertEqual(mn.engine_health(self.state)[0]["lb"], 0)

    def test_listening_but_refusing_is_reported_dead_and_distinguishable(self):
        """503 and "no socket" are both fatal, but they are not the same bug."""
        self.replies[0] = "serve=1 sched=8 zombie=0 lb=503"
        master = mn.engine_health(self.state)[0]
        self.assertEqual(master["lb"], 0)
        self.assertEqual(master["lb_http"], 503)

    def test_only_the_master_is_asked_about_the_load_balancer(self):
        mn.engine_health(self.state)
        self.assertNotIn("/health", self.sent[1])

    def test_a_tp_span_pin_is_not_asked_at_all(self):
        """There is no front end to lose there; a number would only mislead."""
        span = dict(PD_STATE, topology="tp_span")
        span.pop("pd", None)
        health = mn.engine_health(span)
        self.assertNotIn("/health", self.sent[0])
        self.assertNotIn("lb", health[0])


class SharedRankPreambleTest(unittest.TestCase):
    """`launch` and `fanout` must decide the NIC with the SAME text.

    Two copies of the fabric decision is a second chance to pick the wrong link,
    and picking the wrong link does not fail — it completes, over the slow one.
    A serving A/B and a collective microbenchmark measured on different networks
    are both "successful" and not comparable, which is the expensive version of
    this bug because nothing ever reports it.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _preamble(self):
        with open(mn.PREAMBLE, encoding="utf-8") as f:
            return f.read().rstrip("\n")

    def test_the_serve_template_does_not_carry_its_own_copy(self):
        with open(os.path.join(MN, "serve.sh.tmpl"), encoding="utf-8") as f:
            tmpl = f.read()
        self.assertIn("__RANK_PREAMBLE__", tmpl)
        self.assertNotIn("NCCL_SOCKET_IFNAME", tmpl,
                         "the NIC decision belongs to rank_preamble.sh.inc alone")

    def test_launch_and_fanout_render_the_same_preamble_bytes(self):
        served = render(self.tmp)
        state = {
            "nnodes": 2, "gpus_per_node": 2, "master_ip": "10.235.200.166",
            "cluster_subnet": "10.235.200.",
            "nodes": [{"rank": 0, "ip": "10.235.200.166"},
                      {"rank": 1, "ip": "10.235.200.183"}],
        }
        cfg = {"fanout_id": "abc123", "cmd": "python3 leg_runner.py", "gpu": "0,1",
               "out_dir": self.tmp, "profile_dir": os.path.join(self.tmp, "prof"),
               "master_port": 30541, "overlay": "/ovl", "pythonpath": "",
               "kept_env": ["RECIPE_V=1"]}
        jobs = mn.fanout_scripts(state, cfg)
        self.assertEqual(sorted(jobs), [0, 1, 2, 3], "one script per card, keyed by global rank")

        # The node-fact block is everything up to the marker each side appends
        # after it, compared as bytes rather than as "both mention ens50f0".
        head = self._preamble().split("__")[0]
        self.assertTrue(head)
        for rank, script in jobs.items():
            shared = [ln for ln in script.splitlines() if "NCCL_SOCKET_IFNAME" in ln]
            self.assertEqual(shared, [ln for ln in served.splitlines()
                                      if "NCCL_SOCKET_IFNAME" in ln],
                             f"rank{rank} decides the NIC differently from a served rank")

    def test_fanout_owns_the_rank_triple(self):
        """A swept RANK is two processes that both believe they are rank 0.

        That presents as a rendezvous that never completes, i.e. as a hang, not
        as a bad flag — so it has to be refused where the flag arrives.
        """
        kept, dropped = mn.accepted_env("RANK=7 TUNABLE=1", owned=mn._FANOUT_OWNED)
        self.assertEqual(kept, ["TUNABLE=1"])
        self.assertTrue(any(tok.startswith("RANK=7") for tok in dropped))
        # A server launch does NOT own it: there the rank is a rendered
        # placeholder, not an env token, so widening _OWNED would drop a name
        # the launcher never sets.
        kept_serve, _ = mn.accepted_env("RANK=7 TUNABLE=1")
        self.assertIn("RANK=7", kept_serve)


if __name__ == "__main__":
    unittest.main(verbosity=2)
