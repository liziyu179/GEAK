#!/usr/bin/env python3
"""Multi-node handoff tests for run_e2e.

Every case here guards a failure that produces a NUMBER rather than an error,
which is the only reason these are worth tests instead of a code comment:

  * A distributed handoff that silently uses the single-host launcher measures
    a TP group that fits on one box, then reports it against a two-node
    baseline.
  * A device list derived from tp rather than from gpus_per_node names eight
    cards that do not exist on either machine.
  * A distributed run with no shared filesystem applies the authored overlay to
    rank0 only -- the other ranks serve stock code and the run still reports a
    speedup for a kernel half of them never loaded.
  * result.json that records tok/s without recording how many GPUs produced it
    is not comparable to anything.

Run: python3 -m pytest GEAK/interface/test_run_e2e_multinode.py -v
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent

_MN_ENV = (
    "GEAK_MN_EXT_SHARED_FS", "GEAK_MN_EXT_SERVICE_URL", "GEAK_MN_NODES",
    "GEAK_MN_GPUS_PER_NODE", "GEAK_MN_STATE", "GEAK_MN_RAY_ADDRESS",
    "GEAK_MN_SERVE_PORT", "GEAK_MN_DIST_INIT_PORT", "GEAK_MN_CLUSTER_SUBNET",
    "BENCH_LAUNCHER",
)


def _load():
    spec = importlib.util.spec_from_file_location("run_e2e", _HERE / "run_e2e.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The module reads and writes process env; leaking it across cases would
    make a later test pass for the wrong reason."""
    for name in _MN_ENV:
        monkeypatch.delenv(name, raising=False)


def _handoff(**over) -> dict:
    h = {
        "model_path": "/apps/DeepSeek-V4-Flash-FP8",
        "exp_root": "/workhome/geak/exp",
        "framework": "sglang",
        "tp": 16,
        "workload": {"isl": 1024, "osl": 1024, "conc": 64},
    }
    h.update(over)
    return h


# --- topology ---------------------------------------------------------------

def test_single_node_handoff_is_untouched() -> None:
    """nodes absent => nothing about the single-host path changes."""
    args = rx.map_args(_handoff(tp=8), timeout_s=3600)
    assert "nodes" not in args and "gpus_per_node" not in args
    assert args["gpu_ids"] == "0,1,2,3,4,5,6,7"


def test_device_list_is_per_node_not_world() -> None:
    """TP=16 over two 8-GPU boxes must ask for 8 devices, not 16.

    $GPU is what the allow-list is checked against and what the serving lock is
    keyed on; a world-sized list names cards that exist on neither machine.
    """
    args = rx.map_args(_handoff(nodes=2), timeout_s=3600)
    assert args["gpu_ids"] == "0,1,2,3,4,5,6,7"
    assert args["nodes"] == 2
    assert args["gpus_per_node"] == 8


def test_explicit_gpus_per_node_wins_over_the_split() -> None:
    args = rx.map_args(_handoff(nodes=2, tp=8, gpus_per_node=4), timeout_s=3600)
    assert args["gpu_ids"] == "0,1,2,3"
    assert args["gpus_per_node"] == 4


def test_explicit_gpu_ids_are_never_overridden() -> None:
    args = rx.map_args(_handoff(nodes=2, gpu_ids="2,3,4,5"), timeout_s=3600)
    assert args["gpu_ids"] == "2,3,4,5"


def test_tp_alone_never_implies_multiple_nodes() -> None:
    """A wrong tp on a 16-GPU host must stay a wrong tp, not become a silent
    distributed launch."""
    nodes, gpus = rx._multinode_topology(_handoff(tp=16), 16)
    assert nodes == 1 and gpus == 16


def test_topology_mismatch_is_reported_not_swallowed(capsys) -> None:
    rx._multinode_topology(_handoff(nodes=2, tp=16, gpus_per_node=4), 16)
    assert "nodes=2 x gpus_per_node=4" in capsys.readouterr().err


# --- launcher selection -----------------------------------------------------

def test_multinode_handoff_selects_the_multinode_launcher(monkeypatch) -> None:
    monkeypatch.setenv("GEAK_MN_EXT_SHARED_FS", "/workhome/geak")
    h = _handoff(nodes=2)
    rx.apply_multinode(h)
    assert rx.apply_bench_launcher(h) == "multinode"


def test_magpie_script_does_not_win_on_a_multinode_handoff(monkeypatch) -> None:
    """Magpie's script starts ONE local server, so discovering it must not
    downgrade a distributed run to a single-node one."""
    monkeypatch.setenv("GEAK_MN_EXT_SHARED_FS", "/workhome/geak")
    monkeypatch.setenv("MAGPIE_LAUNCH_SCRIPT", "/opt/magpie/run.sh")
    h = _handoff(nodes=2)
    rx.apply_multinode(h)
    assert rx.apply_bench_launcher(h) == "multinode"
    monkeypatch.delenv("MAGPIE_LAUNCH_SCRIPT", raising=False)


def test_single_node_still_prefers_magpie_when_a_script_exists(monkeypatch) -> None:
    monkeypatch.setenv("MAGPIE_LAUNCH_SCRIPT", "/opt/magpie/run.sh")
    assert rx.apply_bench_launcher(_handoff(tp=8)) == "magpie"
    monkeypatch.delenv("MAGPIE_LAUNCH_SCRIPT", raising=False)


def test_explicit_single_node_launcher_on_a_multinode_handoff_warns(
    monkeypatch, capsys
) -> None:
    """The override is honoured -- the operator may know something we do not --
    but it cannot be silent: the run will measure a single-node config."""
    monkeypatch.setenv("GEAK_MN_EXT_SHARED_FS", "/workhome/geak")
    h = _handoff(nodes=2, bench_launcher="native")
    rx.apply_multinode(h)
    assert rx.apply_bench_launcher(h) == "native"
    assert "single-node config" in capsys.readouterr().err


# --- the cluster handoff ----------------------------------------------------

def test_distributed_without_a_shared_fs_refuses_to_run() -> None:
    with pytest.raises(SystemExit) as exc:
        rx.apply_multinode(_handoff(nodes=2))
    assert "shared filesystem" in str(exc.value)


def test_handoff_block_is_exported_for_the_launcher(monkeypatch, tmp_path) -> None:
    shared = tmp_path / "workhome"
    shared.mkdir()
    info = rx.apply_multinode(_handoff(nodes=2, multinode={
        "shared_fs": str(shared),
        "ray_address": "10.235.200.32:6779",
        "serve_port": 30890,
        "dist_init_port": 30531,
        "dp": 2,
    }))
    import os
    assert os.environ["GEAK_MN_EXT_SHARED_FS"] == str(shared)
    assert os.environ["GEAK_MN_RAY_ADDRESS"] == "10.235.200.32:6779"
    assert os.environ["GEAK_MN_SERVE_PORT"] == "30890"
    assert os.environ["GEAK_MN_DIST_INIT_PORT"] == "30531"
    assert os.environ["GEAK_MN_NODES"] == "2"
    # dp is deliberately NOT promoted: data-parallel width is an ordinary server
    # flag, and the launcher stays framework-agnostic by not knowing its spelling.
    assert "GEAK_MN_DP" not in os.environ
    # The pin lands on the shared mount so every process in the run -- driver,
    # each bench_e2e.sh, the supervisor -- reads the same one.
    assert info["state"] == str(shared / "mn_state.json")
    assert info["state_exists"] is False


def test_single_node_exports_nothing(monkeypatch) -> None:
    import os
    info = rx.apply_multinode(_handoff(tp=8))
    assert info == {"nodes": 1, "gpus_per_node": 8}
    assert "GEAK_MN_EXT_SHARED_FS" not in os.environ


# --- provenance -------------------------------------------------------------

def test_result_records_the_machines_that_served(monkeypatch, tmp_path) -> None:
    """tok/s from 16 GPUs across two boxes is not comparable to tok/s from 8,
    and the pin is the only record of which it was."""
    state = tmp_path / "mn_state.json"
    state.write_text(json.dumps({
        "nnodes": 2, "gpus_per_node": 8, "master_ip": "10.235.200.32",
        "shared_fs": str(tmp_path), "pinned_at": "2026-09-15T12:14:52Z",
        "nodes": [
            {"rank": 0, "hostname": "gpuc489.prov.aus.ccs.cpe.ice.amd.com"},
            {"rank": 1, "hostname": "gpufc54.prov.aus.ccs.cpe.ice.amd.com"},
        ],
    }), encoding="utf-8")
    monkeypatch.setenv("GEAK_MN_NODES", "2")
    monkeypatch.setenv("GEAK_MN_GPUS_PER_NODE", "8")
    monkeypatch.setenv("GEAK_MN_STATE", str(state))

    topology = rx._serving_topology()
    assert topology["nodes"] == 2
    assert topology["gpus_per_node"] == 8
    assert topology["pinned_hosts"] == ["gpuc489", "gpufc54"]
    assert topology["master_ip"] == "10.235.200.32"


def test_single_node_topology_stays_minimal() -> None:
    assert rx._serving_topology() == {"nodes": 1, "gpus_per_node": None}


def test_missing_pin_does_not_break_the_record(monkeypatch, tmp_path) -> None:
    """A result must still be written when the pin is unreadable; the topology
    block degrades instead of raising."""
    monkeypatch.setenv("GEAK_MN_NODES", "2")
    monkeypatch.setenv("GEAK_MN_STATE", str(tmp_path / "absent.json"))
    topology = rx._serving_topology()
    assert topology["nodes"] == 2
    assert "pinned_hosts" not in topology
