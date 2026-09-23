#!/usr/bin/env python3
"""Terminal-marker + completion-grace tests for run_e2e.

These guard the two ways an UNFINISHED run gets permanently mistaken for a
finished one -- both of which end in a plausible-looking result.json rather
than an error, which is the only reason they are worth tests:

  * ``workflow_return.json`` is the terminal marker AND the file the disk
    recovery path writes when the workflow did NOT finish. Counting our own
    reconstruction as terminal pins the run to the phase it died in: the resume
    short-circuit re-emits the reconstruction forever and the completion gate
    reports done the moment recovery has ever run. A run recovered at, say, the
    tuning checkpoint could then never be continued into the kernel phase.
  * The completion grace window keeps the SDK client (and therefore the CLI
    process tree) alive while a detached leg lands. A FLAT deadline SIGKILLs
    whatever is running when it expires -- including a workflow the driving
    agent relaunched after a failed phase, which we can no longer see because
    we have already left the message loop. Extending the window while the
    eval_dir is still being written is what makes the wait track real work.

Run: python3 -m pytest GEAK/interface/test_run_e2e_resume.py -v
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()

# A workflow return carries these regardless of how it was produced; the
# recovered_* keys are what distinguish a reconstruction from the real thing.
_RETURN = {
    "eval_dir": "/exp/geak/e2e_M_1Z",
    "baseline_throughput_tok_s": 28.696,
    "final_throughput_tok_s": 33.789,
    "throughput_speedup": 1.1775,
}


def _write(eval_dir: Path, name: str, obj: dict) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / name).write_text(json.dumps(obj), encoding="utf-8")


# ---------------------------------------------------------------------------
# Which workflow_return.json counts as "the workflow finished"
# ---------------------------------------------------------------------------

def test_a_real_workflow_return_is_terminal(tmp_path) -> None:
    """The workflow's own return is the authoritative done signal."""
    _write(tmp_path, "workflow_return.json", _RETURN)
    assert rx._workflow_done_on_disk(str(tmp_path)) is True


def test_our_own_disk_recovery_is_not_terminal(tmp_path) -> None:
    """A reconstruction means the run did NOT finish -- it must stay resumable.

    This is the exact shape the tuning-checkpoint recovery writes after the
    workflow handoff fails.
    """
    _write(tmp_path, "workflow_return.json", {
        **_RETURN,
        "recovered_from_disk": True,
        "recovered_intermediate": True,
        "recovered_tuning_skillset": True,
        "recovered_e2e_checkpoint_level": "tuning_skillset",
    })
    assert rx._workflow_done_on_disk(str(tmp_path)) is False


def test_a_future_recovery_tier_is_recognized_without_registering_it(tmp_path) -> None:
    """Matching the recovered_* PREFIX is what keeps new tiers self-identifying.

    A tier added later must not have to be listed here (and in the recovery
    path) to be excluded -- that split is how the original bug survived.
    """
    _write(tmp_path, "workflow_return.json",
           {**_RETURN, "recovered_some_tier_added_later": True})
    assert rx._workflow_done_on_disk(str(tmp_path)) is False


def test_a_falsey_recovery_marker_does_not_disqualify(tmp_path) -> None:
    """``recovered_from_disk: false`` is a real return stating it is not one."""
    _write(tmp_path, "workflow_return.json",
           {**_RETURN, "recovered_from_disk": False})
    assert rx._workflow_done_on_disk(str(tmp_path)) is True


def test_the_director_marker_still_stands_alone(tmp_path) -> None:
    """Kept for when the canonical persist (an agent Write) failed."""
    _write(tmp_path, "director_e2e_validation.json", {"status": "ok"})
    assert rx._workflow_done_on_disk(str(tmp_path)) is True


def test_a_recovered_return_beside_the_director_marker_is_terminal(tmp_path) -> None:
    """Validate DID run; only the canonical persist was reconstructed."""
    _write(tmp_path, "workflow_return.json", {**_RETURN, "recovered_from_disk": True})
    _write(tmp_path, "director_e2e_validation.json", {"status": "ok"})
    assert rx._workflow_done_on_disk(str(tmp_path)) is True


def test_an_unfinished_eval_dir_is_not_terminal(tmp_path) -> None:
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "final_launch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    assert rx._workflow_done_on_disk(str(tmp_path)) is False


def test_no_eval_dir_is_not_terminal() -> None:
    assert rx._workflow_done_on_disk(None) is False
    assert rx._workflow_done_on_disk("") is False


def test_recovery_refuses_to_trust_its_own_earlier_reconstruction(tmp_path) -> None:
    """_recover_workflow_return and _workflow_done_on_disk share ONE rule.

    They disagreed before: recovery re-derived past a reconstruction while the
    done-gate accepted it.
    """
    recovered = {**_RETURN, "recovered_from_disk": True}
    assert rx._is_recovered_return(recovered) is True
    assert rx._is_recovered_return(_RETURN) is False


# ---------------------------------------------------------------------------
# Progress detection behind the grace window
# ---------------------------------------------------------------------------

def test_newest_mtime_sees_a_write_anywhere_in_the_tree(tmp_path) -> None:
    """A resumed run writes deep under the eval_dir, not at its root."""
    deep = tmp_path / "kernels" / "fused_moe_kernel_task" / "kernel_src"
    deep.mkdir(parents=True)
    old = rx._newest_mtime(str(tmp_path))

    f = deep / "fused_moe.py"
    f.write_text("# authored\n", encoding="utf-8")
    os.utime(f, (old + 600, old + 600))

    assert rx._newest_mtime(str(tmp_path)) > old


def test_newest_mtime_is_bounded(tmp_path) -> None:
    """The poll runs every DONE_POLL_S; it may not walk an unbounded tree."""
    for i in range(40):
        (tmp_path / f"f{i}").write_text("x", encoding="utf-8")
    assert rx._newest_mtime(str(tmp_path), max_entries=5) > 0


def test_newest_mtime_tolerates_an_unreadable_tree(tmp_path) -> None:
    """0.0 means "no progress observed", which falls back to the flat window."""
    assert rx._newest_mtime(str(tmp_path / "does-not-exist")) == 0.0
    assert rx._newest_mtime(None) == 0.0


def test_an_idle_eval_dir_reports_no_new_progress(tmp_path) -> None:
    """Idle work must NOT extend the window, or the budget is the only stop."""
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "bench_summary.json").write_text("{}", encoding="utf-8")
    first = rx._newest_mtime(str(tmp_path))
    assert rx._newest_mtime(str(tmp_path)) == first
