#!/usr/bin/env python3
"""Run ONE measurement leg of a kernel task. WHICH leg is decided ONLY by the overlay on PYTHONPATH.

  baseline leg   PYTHONPATH=<task>/baseline_overlay    -> the REAL online serving stack (install +
                 every already-accepted kernel). Nothing in the task dir can change what it imports.
  candidate leg  PYTHONPATH=<task>/_cand_overlay       -> that SAME stack + exactly ONE entry built
                 from kernel_src/ (meta.candidate_bind).

Both legs execute THIS file and the task's OWN cases.py, so `speedup = baseline_ms / candidate_ms` is
always measured against the live path, whatever LANGUAGE the candidate is written in.

Modes: list (bucket sigs) | resolve (leg identity, for the direction assert) | time | oracle.
Driven by harness_lib.measure_legs — a unittest.py should not invoke it directly.

A COLLECTIVE task runs this same file, unchanged in everything above: same overlay rule, same cases.py,
same modes. What it adds is that there are fifteen other copies of it, started by `mn.py fanout`, and
three consequences follow from that alone — the ranks must join one process group, the leg's time is
the SLOWEST rank's rather than this one's, and only rank0 may answer on stdout.
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# this file lives IN the task dir next to a file named unittest.py — drop it from sys.path before
# torch is imported, or that file shadows stdlib unittest.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve(dotted_attr):
    mod_name, _, attr = dotted_attr.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


def _identity(target_callable):
    """Where this leg's callable ACTUALLY came from. The two legs must not report the same tuple."""
    try:
        fn = _resolve(target_callable)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    mod = getattr(fn, "__module__", "") or ""
    f = ""
    try:
        f = getattr(sys.modules.get(mod), "__file__", "") or ""
    except Exception:
        pass
    if not f:
        try:
            f = getattr(importlib.import_module(target_callable.split(":")[0]), "__file__", "") or ""
        except Exception:
            pass
    return {"module": mod, "file": os.path.realpath(f) if f else "",
            "qualname": getattr(fn, "__qualname__", repr(fn))}


# The modes that actually put load on the operator. `list` and `resolve` only read labels and import
# paths, so making them pay for a rendezvous would mean the identity guard needs a live cluster.
_FANOUT_MODES = ("time", "oracle")


def _collective(meta, mode):
    """This rank's identity in the job, or None when the task is single-rank.

    The triple arrives from `mn.py fanout` and from nowhere else. Refusing when it is absent is the
    whole point: a collective started by hand has no peers, so it times an all-reduce that had nothing
    to reduce — and that does not raise, it reports a number.
    """
    if (meta or {}).get("execution_model") != "distributed_collective" or mode not in _FANOUT_MODES:
        return None
    rank, world = os.environ.get("RANK"), os.environ.get("WORLD_SIZE")
    if rank is None or world is None:
        raise SystemExit(
            "!!! this task is a distributed_collective: it must be started by `mn.py fanout`, which\n"
            "    supplies RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT. Run on its own this leg has\n"
            "    no peers, so it would time a collective with nothing to collect and report that as a\n"
            "    measurement.")
    return {"rank": int(rank), "world": int(world),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0") or 0),
            "rank_out": os.environ.get("GEAK_COLLECTIVE_RANK_OUT", "")}


def _join(torch, dist):
    """Join the job's process group. Its own port, so it can coexist with a served dist-init."""
    if torch.cuda.is_available():
        torch.cuda.set_device(dist["local_rank"])
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=os.environ.get("GEAK_COLLECTIVE_BACKEND", "nccl"),
            init_method="tcp://{}:{}".format(os.environ.get("MASTER_ADDR", "127.0.0.1"),
                                             os.environ.get("MASTER_PORT", "30541")),
            rank=dist["rank"], world_size=dist["world"])


def _slowest(torch, ms):
    """The operator's time is the max over ranks: a collective is finished when its slowest rank is.

    Reporting this rank's own number instead turns skew into speedup — and skew is exactly what a
    change to a collective moves."""
    t = torch.tensor([float(ms or 0.0)],
                     device="cuda" if torch.cuda.is_available() else "cpu")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return float(t.tolist()[0])


def _answer(payload, dist):
    """Rank0 answers for the job; every rank still records its own copy.

    Sixteen ranks printing sixteen results would leave the harness reading whichever line landed last
    as THE measurement. The non-root records have to exist all the same: a reduction that is right on
    the root and wrong on a non-root rank is the characteristic collective bug, and rank0's line is
    where it is invisible."""
    if dist:
        payload = dict(payload, rank=dist["rank"])
        if dist["rank_out"]:
            with open(dist["rank_out"], "w") as fh:
                json.dump(payload, fh)
        if dist["rank"] != 0:
            return
    print(json.dumps(payload))


def _snapshot(out):
    """Detach one op's output for the oracle blob.

    Attention entries commonly return `(out, lse)` and some return a dict, so a bare `out.detach()`
    raises AttributeError and — through `_run_leg` — surfaces as a generic non-zero-exit RuntimeError
    instead of naming the shape of the return value. Recurse over the container instead. Containers
    are matched BEFORE the tensor duck-check: a namedtuple carries both."""
    if isinstance(out, tuple) and hasattr(out, "_fields"):
        return type(out)(*(_snapshot(o) for o in out))
    if isinstance(out, (tuple, list)):
        return type(out)(_snapshot(o) for o in out)
    if isinstance(out, dict):
        return {k: _snapshot(v) for k, v in out.items()}
    if hasattr(out, "detach"):
        return out.detach().clone().cpu()
    if out is None or isinstance(out, (int, float, bool, str)):
        return out
    raise TypeError(
        f"oracle cannot record a {type(out).__name__} returned by the op — cases.call must return a "
        "tensor, or a tuple/list/dict of tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", required=True, choices=("list", "resolve", "time", "oracle"))
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=0)
    a = ap.parse_args()

    task = os.path.abspath(a.task)
    with open(os.path.join(task, "meta.json")) as fh:
        meta = json.load(fh)

    if a.mode == "resolve":
        print(json.dumps(_identity(meta["target_callable"])))
        return

    dist = _collective(meta, a.mode)

    h = _load("harness_lib", os.path.join(task, "harness_lib.py"))
    cases = _load("cases", os.path.join(task, "cases.py"))
    regime = meta.get("regime", {})

    if a.mode == "list":
        print(json.dumps({"sigs": [c["sig"] for c in cases.timing_cases(h, meta)]}))
        return

    call = cases.call
    if h.deployment_compile_mode(regime):
        call = h.compiled_op(call, regime)
    graph = h.deployment_graph_mode(regime)

    torch = h._torch() if dist else None
    if dist:
        _join(torch, dist)
    try:
        if a.mode == "time":
            sel = [c for c in cases.timing_cases(h, meta) if not a.bucket or c["sig"] == a.bucket]
            out = []
            for c in sel:
                if dist:
                    # Warm, THEN hold the start line. A candidate that JITs on first touch would
                    # otherwise have its compile charged to fifteen peers as wire time.
                    call(c["args"])
                    torch.distributed.barrier()
                r = h.time_op(lambda c=c: call(c["args"]), graph=graph, detail=True)
                row = {"sig": c["sig"], "regime": c.get("regime", ""), "m": c.get("m"),
                       "ms": (r or {}).get("ms"), "wall_ms": (r or {}).get("wall_ms"),
                       "timer": (r or {}).get("timer")}
                if dist:
                    row["ms_this_rank"] = row["ms"]
                    row["ms"] = _slowest(torch, row["ms"])
                out.append(row)
            _answer({"cases": out, "identity": _identity(meta["target_callable"])}, dist)
            return

        torch = torch or h._torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        draws = a.draws or int(meta.get("random_draws", 3))
        blob = {}
        for shape in cases.random_shapes(h, meta):
            for i in range(max(1, draws)):
                rng = torch.Generator(device=device).manual_seed(int(a.seed) + i)
                out = call(shape["make_inputs"](rng))
                blob[f"{shape['sig']}|{i}"] = _snapshot(out)
        # One path for sixteen writers is a race whose loser supplies the golden blob.
        out_path = f"{a.out}.rank{dist['rank']}" if dist else a.out
        torch.save(blob, out_path)
        _answer({"out": out_path, "n": len(blob)}, dist)
    finally:
        # Leaving the group open makes the peers wait out the pg timeout, which reads as a hang
        # rather than as an exit.
        if dist:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
