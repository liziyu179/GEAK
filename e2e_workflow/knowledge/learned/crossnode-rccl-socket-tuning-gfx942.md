# Cross-node RCCL socket tuning (gfx942 · CCS AUS fleet · TP16 over 2 nodes)

**Reuse key:** `(comm / all-reduce, gfx942, nnodes>1, socket transport)`
**Confidence:** ★★★ — 4 sweeps, 3 host pairs, median of 3 timed blocks per point, spread reported.
**One-line:** the cross-node wire is a 25 Gb/s management NIC and RCCL's **default socket settings use
about a tenth of it**; `NCCL_SOCKET_NTHREADS=2 NCCL_NSOCKS_PERTHREAD=2 NCCL_MIN_NCHANNELS=16` buys
**4–13× on the collective** and costs nothing but three env vars.

> **CORRECTION (q8e, 2026-09-21). The e2e number in the last section of this card was wrong.**
> It read "+93% (114.384 → 220.701 tok/s)". Re-measured on an exclusively held, idle pair with the
> same workload, **stock alone is 231.29 tok/s** — higher than the figure this card reported as
> *tuned*. The true e2e gain of these three env vars is **+8.9%** (231.29 → 251.86, two legs each,
> leg-to-leg 0.21%). The collective-level 4–13× above is reproducible and stands; what does not
> transfer is the claim that it is worth 1.93x end to end.
>
> The error was in the denominator. That baseline leg's own three rounds were 148.4 / 114.4 / 107.2,
> a **36% spread**, against 2.11% for the re-measured one — a leg that disperses by 36% measured the
> neighbours, not the configuration. **A leg's spread is its credibility: above ~10% it cannot serve
> as either end of an A/B.** And the noise band has to come from repeats of the same label, because
> RCCL re-chooses its algorithm at every communicator init, which the rounds inside one leg cannot see.
> Full leg table and method: `geak_runs/q8e/RESULT.md`.

This is the answer to the question the previous card
(`crossnode-tp-allreduce-gfx942.md`) left open: it established that no *operator* can help cross-node
and said "observe the transport first". This card is that observation, plus the numbers.

## The wire is 25 Gb/s, and that is not fixable from software

Every host carries eight 400 Gb/s backend NICs, all `state=up` with ACTIVE RDMA links. They are useless
for cross-node collectives, and this was tested rather than assumed:

| check | result |
|---|---|
| backend NIC addressing | `GPUC489` 192.168.11.16–30, `GPUF293` 192.168.11.48–62, `GPUF2c5` 192.168.13.112–126, `GPUFC58` 192.168.15.16–30 — all **/31 point-to-point** |
| `GPUF2B9`, `GPUFC54` | 400 Gb/s links up, **zero IPv4 addresses** — not reachable at all |
| ping matrix, every backend address from every other host | **192 attempts, 0 reachable** |

The /31s are point-to-point links to the fabric and nothing routes between them. So collectives fall to
`ens50f0`, 25 Gb/s, shared with NFS/ssh/image pulls for the whole cluster. **≈3.1 GB/s is the roof**, and
`ip -o -4 addr` plus one ping matrix re-derives all of this in under a minute when the pin moves.

## What the defaults actually deliver, and what the knobs recover

Qwen3-8B shapes, 16 ranks over 2 nodes, bf16, median of 3 blocks. Baseline is `NCCL_SOCKET_IFNAME` alone,
which is what `serve.sh.tmpl` already sets — i.e. this is the server's real default, not a strawman.

| message | baseline | **+sockets +channels** | gain |
|---|---|---|---|
| 128 KB (decode AR) | 0.938 ms | **0.499** | 1.9× |
| 512 KB | 1.602 | **0.604** | 2.7× |
| 2 MB | 2.647 | **1.805** | 1.5× |
| 8 MB | 60.485 | **4.649** | **13.0×** |
| 32 MB (prefill AR) | 150.620 | **13.902** | **10.8×** |
| 64 MB | 210.293 | **35.470** | 5.9× |
| 128 MB | 286.272 | **64.762** | 4.4× |

**Sockets do nearly all of the work; channels are a small-message top-up.** Varied one at a time against
the same baseline in one sweep:

| knob | 8 MB | 32 MB | 128 MB |
|---|---|---|---|
| `NCCL_MIN_NCHANNELS=16` alone | 1.6× | 1.4× | 1.3× |
| `NTHREADS=2 NSOCKS=2` alone | **11.8×** | **7.4×** | **4.1×** |
| both | 10.6× | 8.9× | 4.2× |

One TCP stream cannot fill 25 Gb/s; a few can. Channels add their bit at 512 KB (2.3× on top of sockets)
and are free, so keep both — but if only one knob survives review, keep the sockets.

## Settings that look plausible and are not

- **`NCCL_ALGO=Tree` buys nothing.** Forced Tree matched plain sockets (0.83–1.00×) and was *worse* at
  128 MB (67 vs 55 ms). Leave the algorithm to RCCL once the sockets are opened up.
- **`NCCL_ALGO=Ring` is a trap:** 9.0 ms at 128 KB against 0.9 ms auto — 10× worse on the decode path.
- **`NTHREADS=4 NSOCKS=4`, `8×2`, and `NCCL_BUFFSIZE=8 MiB` fail outright.** Rank exits 1 during the first
  collective. Reproduced on `GPUF2c5` (4 GiB free) *and* `GPUF293` (21.7 GiB free), so it is **not** VRAM.
  Curiously `4×4` *plus* `MIN_NCHANNELS=16` runs fine. Not worth chasing: 2×2 is already at the knee.

## Measuring this without lying to yourself

- **Run-to-run variance is ~5×; within-run variance is a few percent.** The same config measured 11 ms and
  52 ms at 8 MB minutes apart, while 3 repeats inside one run agreed closely. The choice is made **once at
  communicator init**, so a single timing is a coin flip — always repeat and print the spread. Tuned
  configs are also visibly *steadier*, which is a second reason to pin them.
- **A neighbour on the box distorts everything.** `GPUF2c5` runs ~95 GiB/card of another team's job and
  ten containers; its numbers disagreed with the clean `GPUF293`+`GPUF2B9` pair by 2–5×. Check
  `docker ps` and per-card free VRAM before trusting a pair.
- **Set `NCCL_SOCKET_IFNAME` in any standalone tool or it hangs.** These boxes expose calico, docker0 and
  veth interfaces; RCCL picks an unroutable one and blocks forever in the first collective. The server is
  safe (`serve.sh.tmpl` exports it from the discovered `CLUSTER_IF`) — bare `torchrun` is not.
- **Skip the model.** A 16-rank `torch.distributed` all-reduce probe reproduces the collective in seconds
  instead of a ~7 minute server boot, and its few hundred MB coexist with the VRAM holds that keep the
  machines reserved. `shell/ar_probe.py` + `shell/ar_sweep.py`.

## How to apply it

`EXTRA_ENV` reaches every rank: `mn.py`'s `_OWNED` filter drops only `NCCL_SOCKET_IFNAME`,
`GLOO_SOCKET_IFNAME`, `MASTER_ADDR`, the visible-device masks and `PYTHONPATH`, so the three knobs below
pass through untouched.

```
EXTRA_ENV="NCCL_SOCKET_NTHREADS=2 NCCL_NSOCKS_PERTHREAD=2 NCCL_MIN_NCHANNELS=16"
```

These are `op_select`-free pure env, so they are legal even when `CONFIG_TUNE_ENABLED=false`.

## e2e, measured

Paired A/B on the same pin (`GPUF293`+`GPUF2B9`), Qwen3-8B-FP8, TP16 DP2, ISL 512 / OSL 32 / conc 16,
32 prompts, warm_server, 3 rounds each. Only the three env vars differ, and they were confirmed present
in the tuned launch log and absent from the baseline's, with no `dropping EXTRA_ENV token` on either.

**Superseded — the row below is the 2026-09-21 re-measurement, on a pair held exclusively for the
whole A/B and with the labels interleaved rather than run in blocks.** Both legs of each label, same
pin, same recipe, only `EXTRA_ENV` differing.

| | stock | +3 env vars | |
|---|---|---|---|
| throughput median | 231.29 tok/s | **251.86 tok/s** | **+8.9% (1.089×)** |
| leg-to-leg spread | 0.33% | **0.21%** | |
| TPOT median | 64.0 ms | **56.9 ms** | −11% |

The original claim (114.384 → 220.701, "+93%") came from a baseline leg whose own rounds were
148.4 / 114.4 / 107.2 — a 36% spread. It was measuring the neighbours. **Take a leg with a double-digit
spread out of the comparison entirely rather than quoting its median**, and take the noise band from
repeats of the same label: RCCL picks its algorithm once per communicator init, so the rounds inside a
single leg are blind to the largest source of variance there is here.

Two further results from the same sitting, both of which bound how much this lever can ever be worth:

- **`NCCL_PROTO=LL` is 1.62× on the isolated 128 KB all-reduce and +2.0% e2e** — inside its own
  leg-to-leg spread, i.e. not demonstrated. Use the microbenchmark to pick candidates, never to size
  a result.
- **`--enable-two-batch-overlap` costs 27%** (TPOT 56.3 → 79.3 ms). It halves the bytes per collective
  and doubles the count, so a bandwidth-bound path would be roughly neutral. The regression says the
  cross-node cost here is **per-call latency**, which is also why these env vars saturate near +9%.

For scale: GEAK's own tuning round on this workload returned +1.33%.

## What this still does not fix

The previous card's closing advice stands and is now much better supported: TP8 inside one xGMI island
with PP/DP across nodes avoids this wire entirely. A single-node AR sustains ~316–330 GB/s against the
~3 GB/s ceiling measured here, so after a +8.9% the topology is not merely the larger lever, it is the
only one left — the full stack measured in q8e (these env vars + `NCCL_PROTO=LL` + capped
`max-running-requests`/`cuda-graph-max-bs`) tops out at **1.122×**, and TPOT bottoms out near 55 ms
against the 32 ms that a 2× would need.

And the fabric is dark for a more basic reason than "the /31s do not route": **there is no RDMA stack
at all.** `/sys/class/infiniband` does not exist, `rdma link` is empty, and `ibv_devices` is an empty
table **on the host**, not just inside the container — so this is not a `--device=/dev/infiniband`
passthrough that someone forgot. All eight backend NICs carry `carrier=1`, yet every /31 link peer
(`A xor 1`, the switch side) is silent. Note the earlier ping matrix could not have shown this: it
pinged the *far host's* backend addresses, for which no route exists, so the packets never left the
box — that test cannot separate "the fabric does not route" from "this host has no route to it".
RCCL agrees in one line: `NET/IB : No device found.` → `Using network Socket`.
