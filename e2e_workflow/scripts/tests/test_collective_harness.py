#!/usr/bin/env python3
"""The distributed-collective measurement path: dispatch, rank plumbing, per-rank correctness.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_collective_harness.py

WHY THESE EXIST. Every case here is a failure that still produces a NUMBER instead of an error, and the
number is the thing a candidate gets accepted or rejected on:

  * a dispatch that sends `time` down the LOCAL path for a collective task -- one rank, no peers, so the
    measurement is of an all-reduce that had nothing to reduce. It does not raise; it reports a time.
  * a dispatch that sends `resolve`/`list` down the FANOUT path -- correct numbers, 16x the cost, and the
    identity guard suddenly needs a live cluster to answer "which file did this leg import".
  * a rank triple that leaks in from the environment instead of arriving from `mn.py fanout` -- two
    processes believe they are rank 0 and the run hangs at rendezvous.
  * a per-rank time reported as THE time. A collective is finished when its slowest rank is; reporting
    this rank's own number makes a 1.4x out of skew.
  * an oracle compared on rank0 only -- the characteristic collective bug (right on the root, wrong on a
    non-root rank) is then invisible.
  * sixteen ranks writing one oracle path on a shared mount: whoever loses the race supplies the golden.

No GPU, no Ray, no cluster. torch is a fake injected into sys.modules (harness_lib and leg_runner both
import it lazily), and `mn.py` is replaced by a stub that records the environment it was handed -- which
is precisely the contract under test, since everything real about a fanout happens on the far side of it.
"""
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import types
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MISSING = object()


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hl = _load("harness_lib_collective", "harness_lib.py")
lr = _load("leg_runner_collective", "leg_runner.py")

COLLECTIVE = {"target_callable": "live_stack:all_reduce", "regime": {},
              "execution_model": "distributed_collective"}
SINGLE = {"target_callable": "live_stack:gemm", "regime": {}}

# What `mn.py fanout` prints for a successful 4-rank job. The harness reads exactly one line of it.
_MN_STUB = textwrap.dedent('''
    """Stand-in for mn.py: records the env it was handed, answers like a real fanout."""
    import json, os, sys

    ranks = int(os.environ.get("STUB_WORLD", "4"))
    fail = int(os.environ.get("STUB_FAIL_RANK", "-1"))
    with open(os.environ["STUB_ENV_DUMP"], "w") as fh:
        json.dump({"argv": sys.argv[1:], "env": {k: os.environ.get(k, "") for k in (
            "CMD", "OUT_DIR", "PYTHONPATH", "OVERLAY_PYTHONPATH", "FANOUT_KILL_PATTERN")}}, fh)
    if os.environ.get("STUB_SILENT") == "1":
        sys.exit(9)
    payload = {"fanout_id": "stub", "world_size": ranks, "ranks": [
        {"rank": r, "hostname": f"node{r // 2}.dc", "ip": "10.0.0.1",
         "rc": (1 if r == fail else 0),
         "out": (f"mn-launch id=stub rank={r}\\n"
                 + ("boom: something on this rank\\n" if r == fail
                    else json.dumps({"cases": [{"sig": "b1", "ms": 0.5}], "rank": r}) + "\\n")),
         "err": ""}
        for r in range(ranks)]}
    print("noise before the result line")
    print("MN_FANOUT_RESULT=" + json.dumps(payload))
    sys.exit(1 if fail >= 0 else 0)
''')


class _Fanout(unittest.TestCase):
    """A task dir plus a stub mn.py on GEAK_MN_PY."""

    def setUp(self):
        self.task = tempfile.mkdtemp(prefix="collective_task_")
        self.addCleanup(shutil.rmtree, self.task, True)
        self.mn = os.path.join(self.task, "_stub", "mn.py")
        os.makedirs(os.path.dirname(self.mn))
        with open(self.mn, "w") as fh:
            fh.write(_MN_STUB)
        self.dump = os.path.join(self.task, "stub_env.json")
        for key, val in (("GEAK_MN_PY", self.mn), ("STUB_ENV_DUMP", self.dump)):
            self._setenv(key, val)

    def _setenv(self, key, val):
        prev = os.environ.get(key, _MISSING)
        os.environ[key] = val
        self.addCleanup(lambda k=key, p=prev: os.environ.pop(k)
                        if p is _MISSING else os.environ.__setitem__(k, p))

    def handed(self):
        with open(self.dump) as fh:
            return json.load(fh)


# ==========================================================================
# DISPATCH  (harness_lib._run_leg)
# ==========================================================================
class Dispatch(_Fanout):
    def test_a_collective_times_across_the_pin_not_locally(self):
        got = hl._run_leg(self.task, "/ovl", "time", meta=COLLECTIVE, bucket="b1")
        # The stub can only have been reached through the fanout branch; the local branch would have
        # tried to run leg_runner.py out of a task dir that has none.
        self.assertEqual(got["cases"][0]["sig"], "b1")
        self.assertEqual(got["rank"], 0, "the harness must read RANK0's reduced result")

    def test_the_command_handed_over_is_the_same_leg_runner_invocation(self):
        hl._run_leg(self.task, "/ovl", "time", meta=COLLECTIVE, bucket="b1", seed=3)
        cmd = self.handed()["env"]["CMD"]
        for fragment in ("leg_runner.py", "--task", self.task, "--mode time", "--seed 3",
                         "--bucket b1"):
            self.assertIn(fragment, cmd.replace("'", ""))

    def test_the_overlay_still_decides_which_leg_this_is(self):
        # The whole two-leg contract: identity comes from PYTHONPATH and nothing else. If the overlay
        # failed to reach the ranks, both legs would import the same code and every "speedup" is noise.
        hl._run_leg(self.task, "/task/_cand_overlay", "time", meta=COLLECTIVE)
        env = self.handed()["env"]
        self.assertEqual(env["OVERLAY_PYTHONPATH"], "/task/_cand_overlay")
        self.assertTrue(env["PYTHONPATH"].startswith("/task/_cand_overlay"))

    def test_the_caller_supplies_what_a_leftover_rank_looks_like(self):
        # mn.py cannot know; a timed-out leg that keeps a card would make the NEXT leg read as a slowdown.
        hl._run_leg(self.task, "/ovl", "time", meta=COLLECTIVE)
        self.assertEqual(self.handed()["env"]["FANOUT_KILL_PATTERN"],
                         os.path.join(self.task, "leg_runner.py"))

    def test_the_timeout_travels_to_the_ranks(self):
        hl._run_leg(self.task, "/ovl", "time", meta=COLLECTIVE, timeout=77)
        self.assertEqual(self.handed()["argv"], ["fanout", "--timeout", "77"])

    def test_identity_and_bucket_labels_do_not_pay_for_a_rendezvous(self):
        # resolve/list are properties of the overlay, not of the topology. Fanning them out would make
        # assert_legs_differ depend on a live cluster -- and the stub would answer with timing rows.
        for mode in ("resolve", "list"):
            with self.assertRaises(RuntimeError) as e:
                hl._run_leg(self.task, "/ovl", mode, meta=COLLECTIVE)
            self.assertIn(f"leg({mode})", str(e.exception))
            self.assertFalse(os.path.exists(self.dump),
                             f"{mode} reached mn.py fanout; it must stay local")

    def test_a_single_rank_task_never_reaches_the_launcher(self):
        with self.assertRaises(RuntimeError):
            hl._run_leg(self.task, "/ovl", "time", meta=SINGLE)
        self.assertFalse(os.path.exists(self.dump))

    def test_a_task_with_no_execution_model_is_single_rank(self):
        with self.assertRaises(RuntimeError):
            hl._run_leg(self.task, "/ovl", "time", meta={"target_callable": "x:y"})
        self.assertFalse(os.path.exists(self.dump))

    def test_a_failed_rank_is_named_with_its_box(self):
        self._setenv("STUB_FAIL_RANK", "2")
        with self.assertRaises(RuntimeError) as e:
            hl._run_leg(self.task, "/ovl", "time", meta=COLLECTIVE)
        msg = str(e.exception)
        self.assertIn("rank2", msg)
        self.assertIn("node1", msg)          # which BOX behaved differently is the whole question
        self.assertIn("1/4 rank(s)", msg)
        self.assertIn("boom", msg)

    def test_a_launcher_that_says_nothing_is_not_read_as_an_empty_measurement(self):
        self._setenv("STUB_SILENT", "1")
        with self.assertRaises(RuntimeError) as e:
            hl._run_leg(self.task, "/ovl", "time", meta=COLLECTIVE)
        self.assertIn("launcher/pin failure", str(e.exception))

    def test_the_launcher_must_be_findable_by_name(self):
        os.environ["GEAK_MN_PY"] = os.path.join(self.task, "_stub", "not_mn.py")
        with self.assertRaises(RuntimeError) as e:
            hl._run_leg(self.task, "/ovl", "time", meta=COLLECTIVE)
        self.assertIn("GEAK_MN_PY", str(e.exception))


# ==========================================================================
# RANK PLUMBING  (leg_runner's distributed preamble)
# ==========================================================================
_HARNESS_STUB = textwrap.dedent('''
    import sys
    def deployment_compile_mode(regime): return ""
    def deployment_graph_mode(regime): return False
    def time_op(fn, graph=False, detail=False):
        fn()
        return {"ms": sys.modules["torch"].scripted_ms, "wall_ms": 2.0, "timer": "event"}
    def _torch(): return sys.modules["torch"]
''')

_CASES_STUB = textwrap.dedent('''
    warm = []
    class _Out:
        def detach(self): return self
        def clone(self): return self
        def cpu(self): return self
    def timing_cases(h, meta):
        return [{"sig": "8MiB", "args": {"n": 8}, "regime": "decode", "m": 1}]
    def random_shapes(h, meta):
        return [{"sig": "8MiB", "make_inputs": lambda rng: {"seed": rng.seed}}]
    def call(args):
        warm.append(args)
        return _Out()
''')


def _fake_torch(*, reduced=9.0):
    torch = types.ModuleType("torch")
    torch.saved, torch.scripted_ms, torch.log = [], 1.0, []

    class _T(list):
        def tolist(self):
            return list(self)

    def tensor(vals, device=None):
        torch.log.append(("tensor", list(vals), device))
        return _T(vals)

    def all_reduce(t, op=None):
        torch.log.append(("all_reduce", list(t), op))
        t[:] = [reduced] * len(t)

    class Generator:
        def __init__(self, device=None):
            self.device, self.seed = device, None

        def manual_seed(self, s):
            self.seed = s
            return self

    torch.tensor = tensor
    torch.Generator = Generator
    torch.save = lambda obj, path: torch.saved.append((obj, path))
    torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                       set_device=lambda i: torch.log.append(("set_device", i)))
    torch.distributed = types.SimpleNamespace(
        is_initialized=lambda: False,
        init_process_group=lambda **kw: torch.log.append(("init", kw)),
        barrier=lambda: torch.log.append(("barrier",)),
        destroy_process_group=lambda: torch.log.append(("destroy",)),
        all_reduce=all_reduce,
        ReduceOp=types.SimpleNamespace(MAX="MAX"),
    )
    return torch


class RankPlumbing(unittest.TestCase):
    def setUp(self):
        self.task = tempfile.mkdtemp(prefix="collective_leg_")
        self.addCleanup(shutil.rmtree, self.task, True)
        self.torch = _fake_torch()
        prev = sys.modules.get("torch", _MISSING)
        sys.modules["torch"] = self.torch
        self.addCleanup(lambda: sys.modules.__setitem__("torch", prev)
                        if prev is not _MISSING else sys.modules.pop("torch", None))
        for name in ("harness_lib", "cases"):
            self.addCleanup(sys.modules.pop, name, None)
        self.addCleanup(setattr, sys, "argv", sys.argv)
        self._write("meta.json", json.dumps(COLLECTIVE))
        self._write("harness_lib.py", _HARNESS_STUB)
        self._write("cases.py", _CASES_STUB)

    def _write(self, name, text):
        with open(os.path.join(self.task, name), "w") as fh:
            fh.write(text)

    def _env(self, **over):
        env = {"RANK": "0", "WORLD_SIZE": "4", "MASTER_ADDR": "10.0.0.1", "MASTER_PORT": "30541"}
        env.update(over)
        for key, val in env.items():
            prev = os.environ.get(key, _MISSING)
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
            self.addCleanup(lambda k=key, p=prev: os.environ.pop(k, None)
                            if p is _MISSING else os.environ.__setitem__(k, p))

    def _main(self, *argv):
        import contextlib
        sys.argv = ["leg_runner.py", *argv]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lr.main()
        return out.getvalue()

    def test_a_leg_run_by_hand_refuses_instead_of_timing_one_rank(self):
        self._env(RANK=None, WORLD_SIZE=None)
        with self.assertRaises(SystemExit) as e:
            self._main("--task", self.task, "--mode", "time")
        self.assertIn("mn.py fanout", str(e.exception))
        self.assertIn("RANK", str(e.exception))

    def test_the_reported_time_is_the_slowest_rank_and_this_rank_is_kept(self):
        self._env()
        self.torch.scripted_ms = 1.0
        row = json.loads(self._main("--task", self.task, "--mode", "time"))["cases"][0]
        self.assertEqual(row["ms"], 9.0, "the operator's time is the max over ranks")
        self.assertEqual(row["ms_this_rank"], 1.0, "this rank's own number stays auditable")
        self.assertIn(("all_reduce", [1.0], "MAX"), self.torch.log)

    def test_the_candidate_is_warmed_before_the_start_line_is_held(self):
        self._env()
        self._main("--task", self.task, "--mode", "time")
        cases = sys.modules["cases"]
        self.assertEqual(len(cases.warm), 2, "one warm call, then the timed one")
        # A barrier AFTER the warm call: a candidate that JITs on first touch would otherwise have its
        # compile charged to fifteen peers as wire time.
        kinds = [entry[0] for entry in self.torch.log]
        self.assertIn("barrier", kinds)
        self.assertLess(kinds.index("init"), kinds.index("barrier"))

    def test_only_rank0_answers_on_stdout_but_every_rank_records(self):
        mine = os.path.join(self.task, "rank3.json")
        self._env(RANK="3", GEAK_COLLECTIVE_RANK_OUT=mine)
        self.assertEqual(self._main("--task", self.task, "--mode", "time").strip(), "",
                         "a non-root rank must not answer for the job")
        with open(mine) as fh:
            self.assertEqual(json.load(fh)["rank"], 3)

    def test_each_rank_writes_its_own_oracle_blob(self):
        self._env(RANK="3")
        out = os.path.join(self.task, "_baseline_random.pt")
        got = json.loads(self._main("--task", self.task, "--mode", "oracle", "--out", out) or "{}")
        self.assertEqual(self.torch.saved[0][1], out + ".rank3",
                         "one path for sixteen writers is a race whose loser supplies the golden")
        self.assertEqual(got, {}, "rank3 does not answer on stdout")

    def test_the_process_group_is_left_cleanly(self):
        self._env()
        self._main("--task", self.task, "--mode", "time")
        # Without this the peers wait out the pg timeout, which reads as a hang rather than an exit.
        self.assertEqual([e[0] for e in self.torch.log][-1], "destroy")

    def test_a_single_rank_task_never_touches_the_process_group(self):
        self._write("meta.json", json.dumps(SINGLE))
        self._env()
        row = json.loads(self._main("--task", self.task, "--mode", "time"))["cases"][0]
        self.assertEqual(row["ms"], 1.0)
        self.assertNotIn("ms_this_rank", row)
        self.assertEqual(self.torch.log, [], "no group, no barrier, no reduction")


# ==========================================================================
# PER-RANK CORRECTNESS  (harness_lib.check_random_vs_baseline_per_rank)
# ==========================================================================
class PerRankCorrectness(unittest.TestCase):
    def setUp(self):
        # `correct` itself is pinned in test_harness_lib; what is under test here is the rank/draw
        # pairing, so the comparison is stubbed down to equality.
        prev = hl.correct
        hl.correct = lambda out, ref, tol: (out == ref, 0.0 if out == ref else 1.0)
        self.addCleanup(setattr, hl, "correct", prev)

    def test_every_rank_is_compared_not_just_the_root(self):
        base = {r: {"8MiB|0": "gold"} for r in range(4)}
        cand = dict(base)
        cand[3] = {"8MiB|0": "wrong"}        # right on the root, wrong on a non-root rank
        ok, per = hl.check_random_vs_baseline_per_rank(base, cand, 1e-2)
        self.assertFalse(ok)
        self.assertEqual(len(per), 4)
        self.assertEqual([row["correct"] for row in per], [True, True, True, False])
        self.assertTrue(per[3]["case"].startswith("rank3:"))

    def test_a_clean_run_reports_every_pair(self):
        blobs = {r: {"8MiB|0": "gold", "8MiB|1": "gold2"} for r in range(2)}
        ok, per = hl.check_random_vs_baseline_per_rank(blobs, dict(blobs), 1e-2)
        self.assertTrue(ok)
        self.assertEqual(len(per), 4)
        self.assertTrue(all(row["speedup"] is None for row in per),
                        "timing belongs to measure_legs, never to the correctness gate")

    def test_a_rank_that_recorded_nothing_fails_rather_than_being_skipped(self):
        base = {r: {"8MiB|0": "gold"} for r in range(4)}
        cand = {r: {"8MiB|0": "gold"} for r in range(3)}
        ok, per = hl.check_random_vs_baseline_per_rank(base, cand, 1e-2)
        self.assertFalse(ok, "half the ranks silently producing nothing must not pass")
        self.assertIn("baseline leg only", per[-1]["note"])

    def test_a_draw_recorded_on_one_side_only_fails(self):
        ok, per = hl.check_random_vs_baseline_per_rank(
            {0: {"a|0": "g", "a|1": "g"}}, {0: {"a|0": "g"}}, 1e-2)
        self.assertFalse(ok)
        self.assertIn("one side only", per[-1]["note"])

    def test_comparing_nothing_is_an_error_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            hl.check_random_vs_baseline_per_rank({}, {}, 1e-2)


class DistributedOracleCollection(unittest.TestCase):
    """distributed_random_outputs: one fanout, sixteen files, keyed by rank."""

    def setUp(self):
        self.task = tempfile.mkdtemp(prefix="collective_oracle_")
        self.addCleanup(shutil.rmtree, self.task, True)
        torch = types.ModuleType("torch")
        torch.load = lambda path, map_location=None: {"from": os.path.basename(path)}
        prev = sys.modules.get("torch", _MISSING)
        sys.modules["torch"] = torch
        self.addCleanup(lambda: sys.modules.__setitem__("torch", prev)
                        if prev is not _MISSING else sys.modules.pop("torch", None))
        self.written = []
        prev_run = hl._run_leg

        def fake_run_leg(task_dir, overlay, mode, **kw):
            self.written.append((overlay, mode, kw.get("out")))
            for rank in (0, 3):
                with open(f"{kw['out']}.rank{rank}", "w") as fh:
                    fh.write("blob")
            return {"out": kw["out"], "n": 2}

        hl._run_leg = fake_run_leg
        self.addCleanup(setattr, hl, "_run_leg", prev_run)

    def test_blobs_come_back_keyed_by_rank(self):
        got = hl.distributed_random_outputs(self.task, COLLECTIVE, "candidate", draws=2)
        self.assertEqual(sorted(got), [0, 3])
        self.assertEqual(got[3]["from"], "_candidate_random.pt.rank3")
        overlay, mode, _out = self.written[0]
        self.assertTrue(overlay.endswith("_cand_overlay"))
        self.assertEqual(mode, "oracle")

    def test_the_baseline_leg_is_the_serving_stack_overlay(self):
        hl.distributed_random_outputs(self.task, COLLECTIVE, "baseline")
        self.assertTrue(self.written[0][0].endswith("baseline_overlay"))

    def test_a_blob_from_a_previous_round_cannot_be_read_as_this_round(self):
        stale = os.path.join(self.task, "_candidate_random.pt.rank9")
        with open(stale, "w") as fh:
            fh.write("old")
        got = hl.distributed_random_outputs(self.task, COLLECTIVE, "candidate")
        self.assertNotIn(9, got)
        self.assertFalse(os.path.exists(stale))

    def test_no_files_at_all_is_an_error_not_an_empty_pass(self):
        hl._run_leg = lambda *a, **kw: {"out": kw.get("out"), "n": 0}
        with self.assertRaises(RuntimeError) as e:
            hl.distributed_random_outputs(self.task, COLLECTIVE, "baseline")
        self.assertIn("shared mount", str(e.exception))

    def test_an_unknown_side_is_refused(self):
        with self.assertRaises(ValueError):
            hl.distributed_random_outputs(self.task, COLLECTIVE, "whatever")


if __name__ == "__main__":
    unittest.main(verbosity=2)
