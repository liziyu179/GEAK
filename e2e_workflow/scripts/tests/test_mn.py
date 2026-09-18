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
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

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

#!/usr/bin/env python3
"""Fabric-subnet decision tests.

GEAK is handed a Ray cluster and nothing else, so the candidate machines and
their addresses are unknown until Ray answers -- the fabric has to come from the
candidate set. Every case below is one where getting it wrong still produces a
working launch, just over the wrong link, which reads as a slow cluster.
"""




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

#!/usr/bin/env python3
"""Per-rank render tests: the launcher must stay framework-agnostic, and the
flag layering must survive the split.

The refactor these guard moved the launch line out of serve.sh.tmpl and into the
backend adapter. Two things can now break silently:

* Layering. The recipe is the BASE layer, GEAK's swept flags come later and win
  by argparse last-wins, and the rank/world triple is appended LAST because it is
  structural. Get the order wrong and a sweep either loses its flag or -- worse --
  changes which rank a process thinks it is, which reads as a hang at rendezvous.
* The seam itself. If the template ever restates a framework's launch line again,
  a second backend needs a second template and the two drift. The end-to-end case
  below runs the rendered script against a FAKE adapter: it can only pass if the
  script really does delegate.
"""



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

#!/usr/bin/env python3
"""Shell tests for the multi-node adapter_profile_window override.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or just this section: python3 -m unittest test_mn.MultinodeProfileWindowTest
(skipped where bash is unavailable, i.e. when authoring on Windows -- these are
meant to run in the container that actually serves.)

WHY THESE EXIST: the backend's window returns as soon as ONE trace lands. On two
nodes that means bench_e2e.sh proceeds -- and the teardown may fire -- while
fifteen ranks are still flushing over NFS, and whichever file happened to land
first becomes the one the parser reads. The override waits for the set and then
gates it, so the behaviour under test is entirely in the SHELL: whether the wait
actually blocks, whether the gate is invoked, and whether an incomplete set
still lets the round finish (it must -- failing here would discard the bench
that came with the trace).

The native window is replaced by a stub that drops trace files at a controlled
pace. No server, no GPU, no Ray: sourcing the launcher only needs a pinned-state
file, which is written here.
"""

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

#!/usr/bin/env python3
"""Tests for multi_node/mn.py trace-gate.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or just this section: python3 -m unittest test_mn.TraceGateTest

WHY THESE EXIST: every failure this gate catches is one that otherwise produces
a PLAUSIBLE Top-N instead of an error, and none of them can be reproduced on
demand on the cluster -- you cannot ask sixteen ranks to drop rank 11, or to
truncate one export, on the run where you happen to be watching. So the shapes
are built here as synthetic traces:

  * the whole remote node missing, which downstream is indistinguishable from a
    healthy single-host capture;
  * a truncated export, the signature of a teardown that raced the flush;
  * a set with no RCCL kernels at all, i.e. a capture that missed the very
    cross-node traffic a two-node run exists to measure;
  * and the isolation step, which is what makes the profiler role's
    `ls -t | head -1` return the SAME rank on the baseline leg and the
    candidate leg. Without it that selection is decided by NFS flush order and
    the two legs of an A/B can describe different processes.

The gate must also survive traces too large to hold in memory, so it streams;
these fixtures stay small on purpose and the streaming path is exercised by
reading them through the same code.
"""

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

if __name__ == "__main__":
    unittest.main(verbosity=2)
