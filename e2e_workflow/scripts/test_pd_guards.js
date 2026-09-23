#!/usr/bin/env node
// Regression guard for the two PD invariants the orchestrator enforces instead of the role files
// (no GPU, no model, no agent needed).
//
//   A. DIRECTIONS — every profiled engine gets config directions, or is named as having none. The
//      merged Top-N keeps rows by DEPLOYMENT share, so the engine with less GPU time can drop out of
//      it; a sweep with no direction for decode then looks exactly like a sweep that found nothing.
//   B. NO LEAK — a knob a kept trial accepted for ONE engine must not come back in the shared string,
//      which is appended on both machines. Only keys NEW in the shared string move (the baseline's
//      own flags are not the sweep's), and a key both engines claim stays shared.
//
// The guards are EXTRACTED from the shipped source rather than reimplemented here, so this cannot
// pass while e2e_workflow.js drifts.
//
// Run:  node e2e_workflow/scripts/test_pd_guards.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..'); // .../GEAK
const src = fs.readFileSync(path.join(ROOT, 'e2e_workflow', 'e2e_workflow.js'), 'utf8');

let failures = 0;
const eq = (got, want, msg) => {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) { console.error('  FAIL:', msg, '\n    got: ', a, '\n    want:', b); failures++; }
  else console.log('  ok:', msg);
};

const mStart = src.indexOf('function normalizePd(');
const mEnd = src.indexOf('function pdConfigFields(', mStart);
if (mStart < 0 || mEnd < 0) { console.error('FAIL: PD helper block not found in e2e_workflow.js'); process.exit(1); }
const load = (topology, prof) => new Function('SERVING_TOPOLOGY', 'profile', 'log',
  `${src.slice(mStart, mEnd)}\nreturn { applyPdDirectionGuard, applyPdLeakGuard, argGroups };`)(
  topology, prof, () => {});

const BY_ROLE = { profile_topN_by_role: { prefill: '/p.json', decode: '/d.json' } };

// ------------------------------------------------------------------ A. directions
console.log('A. directions');
{
  const { applyPdDirectionGuard } = load('pd_disagg', BY_ROLE);
  const strat = { config_directions: [
    { axis: 'chunked-prefill-size', engine: 'Prefill' },
    { axis: 'attention-backend', role: 'decode' },
    { axis: 'kv-cache-dtype', engine: 'both' },
  ] };
  eq(applyPdDirectionGuard(strat, 't'), [], 'both engines covered -> no gap');
  eq(strat.config_directions.map((d) => d.engine), ['prefill', 'decode', ''],
    'engine is normalized: case, the row\'s `role`, and an unknown value becomes shared');
}
{
  const { applyPdDirectionGuard } = load('pd_disagg', BY_ROLE);
  eq(applyPdDirectionGuard({ config_directions: [{ axis: 'chunk', engine: 'prefill' }] }, 't'),
    ['decode'], 'a profiled engine with no direction is named');
}
{
  const { applyPdDirectionGuard } = load('pd_disagg', { profile_topN_by_role: { prefill: '/p.json' } });
  eq(applyPdDirectionGuard({ config_directions: [{ axis: 'chunk', engine: 'prefill' }] }, 't'),
    [], 'an engine that was never profiled is not a gap (the profiler already said so)');
}
{
  const { applyPdDirectionGuard } = load('tp_span', BY_ROLE);
  const strat = { config_directions: [{ axis: 'x', engine: 'Decode' }] };
  eq(applyPdDirectionGuard(strat, 't'), [], 'off PD the guard reports nothing');
  eq(strat.config_directions[0].engine, 'Decode', '...and changes nothing');
}

// ------------------------------------------------------------------ B. leak
console.log('B. leak');
{
  const { argGroups } = load('pd_disagg', BY_ROLE);
  eq(argGroups('--a 1 --b=2 --c', false).map((g) => [g.key, g.text]),
    [['--a', '--a 1'], ['--b', '--b=2'], ['--c', '--c']], 'a flag travels with its value');
  eq(argGroups('A=1 B=2', true).map((g) => g.key), ['A', 'B'], 'env groups by KEY');
}
{
  const { applyPdLeakGuard } = load('pd_disagg', BY_ROLE);
  const sweep = {
    accepted_flags: '--mem-fraction-static 0.9 --cuda-graph-max-bs 256',
    accepted_flags_by_role: { prefill: '', decode: '' },
    accepted_env: '', trials: [{ engine: 'decode', change: '--cuda-graph-max-bs 256', kept: true }],
  };
  const moved = applyPdLeakGuard(sweep, '--mem-fraction-static 0.9', '');
  eq(moved, ['--cuda-graph-max-bs 256 -> decode'], 'a kept decode-only flag in the shared string is moved');
  eq(sweep.accepted_flags, '--mem-fraction-static 0.9', 'the baseline flag stays shared');
  eq(sweep.accepted_flags_by_role, { prefill: '', decode: '--cuda-graph-max-bs 256' },
    '...and the decode engine keeps its value');
}
{
  const { applyPdLeakGuard } = load('pd_disagg', BY_ROLE);
  const sweep = {
    accepted_flags: '--chunked-prefill-size 8192',
    accepted_flags_by_role: { prefill: '--chunked-prefill-size 8192', decode: '' },
    accepted_env: 'SGLANG_USE_AITER=1',
    accepted_env_by_role: { prefill: '', decode: 'SGLANG_USE_AITER=1' },
    trials: [],
  };
  applyPdLeakGuard(sweep, '', '');
  eq(sweep.accepted_flags, '', 'a key already in that engine\'s by_role string is dropped from shared');
  eq(sweep.accepted_flags_by_role.prefill, '--chunked-prefill-size 8192', '...without being duplicated');
  eq(sweep.accepted_env, '', 'env leaks are moved the same way');
}
{
  const { applyPdLeakGuard } = load('pd_disagg', BY_ROLE);
  const sweep = {
    accepted_flags: '--attention-backend aiter',
    accepted_flags_by_role: { prefill: '--attention-backend aiter', decode: '--attention-backend aiter' },
    accepted_env: '', trials: [],
  };
  eq(applyPdLeakGuard(sweep, '', ''), [], 'a key BOTH engines claim is shared, not a leak');
  eq(sweep.accepted_flags, '--attention-backend aiter', '...and stays in the shared string');
}
{
  const { applyPdLeakGuard } = load('pd_disagg', BY_ROLE);
  const sweep = { accepted_flags: '--x 1', accepted_env: '', trials: [] };
  eq(applyPdLeakGuard(sweep, '', ''), [], 'no one-engine claim -> nothing moves');
  eq('accepted_flags_by_role' in sweep, false,
    'an untouched sweep gains no by_role field, so an earlier accepted per-engine config survives');
}
{
  const { applyPdLeakGuard } = load('tp_span', BY_ROLE);
  const sweep = { accepted_flags: '--x 1', trials: [{ engine: 'decode', change: '--x 1', kept: true }] };
  eq(applyPdLeakGuard(sweep, '', ''), [], 'off PD the guard never moves anything');
}

if (failures) { console.error(`\n${failures} failure(s)`); process.exit(1); }
console.log('\nall PD guard checks passed');
