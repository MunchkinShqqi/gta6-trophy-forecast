// GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).
//
// Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
// quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the
// case's target, from the case's history of reads of the game's trophy page.
//
// The page is not a live feed: Exophase refreshes a game's counts on its own
// schedule (hours apart), so a read shows the rates as of the last refresh
// before it, and the answer is the rates the page shows at the target, as of
// its last refresh before that. A trophy's rate is earners / players_tracked,
// and the two move differently: players_tracked jumps when a refresh brings a
// wave of newly synced players (who have earned almost nothing yet beyond the
// trophy every tracked player holds), while earners grow as the tracked
// cohort plays further into the game. So the forecaster decomposes the rate
// into earners and players and models both.
//
// Two paths, chosen by how much the history shows:
//   - Trend path (3+ distinct page states): fit, per trophy, logit(rate)
//     against ln(1 + hours since release) as a concave quadratic (the rise
//     saturates as new players dilute the tracked population), curvature
//     shrunk toward the pooled curvature, and extrapolate from the last
//     state with the fitted curve's local slope there, damped. Points at
//     0.00% are censored and left out. The spread combines the random-walk
//     spread, the fit residual, the refresh-staleness of the answer and the
//     extrapolation uncertainty.
//   - Refresh-mixture path (fewer than 3 distinct states, e.g. the practice
//     cases, whose history is one frozen launch-night page): the page has
//     been unchanged for hours, so a refresh before the target is likely.
//     The forecast is a mixture over "no refresh: rates stay" and "a refresh
//     comes: players multiply and rates dilute", with the refresh probability
//     from the page's staleness against a refresh-interval prior (lognormal,
//     median 4 h, fairly regular), the players multiplier lognormal around 5x,
//     and new players holding the universal trophy plus ~0.03 further
//     trophies each, spread over the other trophies in proportion to their
//     old earners. Quantiles are read off the mixture's atoms, so a likely
//     refresh puts the median on the diluted branch while the upper quantiles
//     still cover "no refresh".
//
// Everything is deterministic (no randomness, no clock, no network): the same
// case always writes the same bytes. Standard library only.
//
// Tuning knobs (TF_* environment variables) exist only so the build can sweep
// them against the practice reconstruction; unset, the documented defaults
// below apply, and those defaults are what the entry ships.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Refresh interval prior: median hours between Exophase refreshes. */
const INTERVAL_MEDIAN_H = Number(process.env.TF_INTERVAL_MEDIAN_H ?? 4);
/** Log-sigma of the refresh interval (a schedule, not a memoryless process). */
const INTERVAL_SIGMA = Number(process.env.TF_INTERVAL_SIGMA ?? 0.25);
/** Players multiplier at one early refresh: median. */
const MULT_MEDIAN = Number(process.env.TF_MULT_MEDIAN ?? 5);
/** Log-sigma of that multiplier. */
const MULT_SIGMA = Number(process.env.TF_MULT_SIGMA ?? 0.5);
/** Further trophies a newly tracked player holds, beyond the universal one. */
const EXTRA_PER_NEW = Number(process.env.TF_EXTRA_PER_NEW ?? 0.03);
/** Random-walk spread of a rate's logit, per square-root day (house baseline). */
const RW_SIGMA = 0.5;
/** Shrinkage strength (pseudo-count) for per-trophy trend curvature. */
const SHRINK_K = 3;
/** Soft cap (logit) on a trend extrapolation move. */
const MOVE_CAP = 2.5;
/** Damping applied to a trend extrapolation move (curves saturate). */
const MOVE_DAMP = Number(process.env.TF_MOVE_DAMP ?? 0.8);
/** Scale on the trend path's total spread (tune against synthetic finals). */
const SPREAD_SCALE = Number(process.env.TF_SPREAD_SCALE ?? 0.65);
/** Floor on a per-trophy fit residual (logit). */
const RESID_FLOOR = 0.15;

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const c = clamp(pct);
  return Math.log(c / (100 - c));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Standard normal CDF (Abramowitz-Stegun 7.1.26, |err| < 1.5e-7). */
const normCdf = (z: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = 0.3989422804014327 * Math.exp(-0.5 * z * z);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return z > 0 ? 1 - p : p;
};
/** Lognormal CDF with the given median and log-sigma. */
const lognormCdf = (x: number, median: number, sigma: number): number =>
  normCdf((Math.log(x) - Math.log(median)) / sigma);
/** Median of an array of numbers. */
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
/** Ordinary least squares: returns [intercept, slope]. */
const ols = (xs: number[], ys: number[]): [number, number] => {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) ** 2;
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 0 ? sxy / sxx : 0;
  return [my - slope * mx, slope];
};
/**
 * Least squares for y = a + b x + c x^2 (3x3 normal equations, Gaussian
 * elimination). Returns null when the system is degenerate.
 */
const olsQuad = (xs: number[], ys: number[]): [number, number, number] | null => {
  const n = xs.length;
  const sum = (f: (x: number, y: number) => number): number =>
    xs.reduce((a, x, i) => a + f(x, ys[i]), 0);
  // Normal-equation matrix rows: [n, Sx, Sx2 | Sy], [Sx, Sx2, Sx3 | Sxy],
  // [Sx2, Sx3, Sx4 | Sx2y].
  const m: number[][] = [
    [n, sum((x) => x), sum((x) => x * x), sum((_, y) => y)],
    [sum((x) => x), sum((x) => x * x), sum((x) => x * x * x), sum((x, y) => x * y)],
    [sum((x) => x * x), sum((x) => x * x * x), sum((x) => x ** 4), sum((x, y) => x * x * y)],
  ];
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let r = col + 1; r < 3; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    for (let r = 0; r < 3; r++) {
      if (r === col) continue;
      const f = m[r][col] / m[col][col];
      for (let k = col; k < 4; k++) m[r][k] -= f * m[col][k];
    }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
};

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes: string;
}

const ms = (iso: string): number => Date.parse(iso);
const hoursSince = (iso: string, baseMs: number): number => (ms(iso) - baseMs) / 3_600_000;

/** Distinct page states in the history: consecutive identical reads collapse. */
interface State {
  atMs: number;
  players: number;
  rates: Array<number | null>;
}
const distinctStates = (c: Case): State[] => {
  const out: State[] = [];
  for (const r of c.history) {
    const last = out[out.length - 1];
    if (last && last.players === r.players_tracked && last.rates.every((v, i) => v === r.rates[i])) continue;
    out.push({ atMs: ms(r.at), players: r.players_tracked, rates: r.rates });
  }
  return out;
};

/** Earners of trophy i in a state: the integer behind its printed 2dp rate. */
const earners = (st: State, i: number): number => {
  const r = st.rates[i];
  if (r === null || r === undefined) return 0;
  return Math.round((r / 100) * st.players);
};

/** Last non-null rate of trophy i over the history (and the state that had it). */
const lastKnown = (c: Case, i: number): { rate: number; players: number } => {
  for (let k = c.history.length - 1; k >= 0; k--) {
    const v = c.history[k].rates[i];
    if (v !== null && v !== undefined) return { rate: v, players: c.history[k].players_tracked };
  }
  return { rate: 0, players: c.history[c.history.length - 1].players_tracked };
};

/**
 * Refresh-mixture forecast for one trophy. Returns the five quantiles.
 * `pRef` is the probability a refresh lands before the target (plus one read
 * gap, since the answer is the first read in the hour from the target).
 * `multGrid` is [multiplier, weight] pairs for the players multiplier at a
 * refresh (a lognormal grid, or the compounded grid for a long horizon).
 */
const mixtureQuantiles = (
  lastRate: number,
  lastEarners: number,
  lastPlayers: number,
  universal: boolean,
  extraWeight: number,
  pRef: number,
  multGrid: Array<[number, number]>,
): number[] => {
  // Atoms of the predictive distribution: (value, weight).
  const atoms: Array<[number, number]> = [[clamp(lastRate), 1 - pRef]];
  for (const [m, w] of multGrid) {
    const p1 = lastPlayers * m;
    const dP = p1 - lastPlayers;
    const ne = universal ? dP : dP * EXTRA_PER_NEW * extraWeight;
    const y = ((lastEarners + ne) / p1) * 100;
    atoms.push([clamp(Math.round(y * 100) / 100), pRef * w]);
  }
  atoms.sort((a, b) => a[0] - b[0]);
  const qs: number[] = [];
  for (const tau of LEVELS) {
    let cum = 0;
    let q = atoms[atoms.length - 1][0];
    for (const [v, w] of atoms) {
      cum += w;
      if (cum >= tau) {
        q = v;
        break;
      }
    }
    qs.push(round4(q));
  }
  // Keep the row non-decreasing (atoms already sort it, but rounding can tie).
  for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
  return qs;
};

/** Lognormal grid: `n` points spanning median * exp(+-2.5 sigma), density weights. */
const lognormGrid = (medianV: number, sigma: number, n = 41): Array<[number, number]> => {
  const pts: Array<[number, number]> = [];
  let tot = 0;
  for (let i = 0; i < n; i++) {
    const z = -2.5 + (5 * i) / (n - 1);
    const m = medianV * Math.exp(sigma * z);
    const w = Math.exp(-0.5 * z * z);
    pts.push([m, w]);
    tot += w;
  }
  return pts.map(([m, w]) => [m, w / tot] as [number, number]);
};

const forecastCase = (c: Case): Array<{ id: number; q: number[] }> => {
  const states = distinctStates(c);
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const atMs = ms(c.at);
  const horizonH = c.horizon_minutes / 60;
  const days = c.horizon_minutes / 1440;

  // Median read gap (hours): the answer read lands within one gap past target.
  const gaps: number[] = [];
  for (let i = 1; i < c.history.length; i++) gaps.push((ms(c.history[i].at) - ms(c.history[i - 1].at)) / 3_600_000);
  const gapH = gaps.length ? median(gaps) : 0.5;
  const horizonEffH = horizonH + gapH;

  // ---- Trend path: 3+ distinct page states. ----
  // A trophy's rate rises as the tracked cohort plays further in, but the
  // rise saturates: new players keep joining (diluting it) and the cohort's
  // progress slows. In logit(rate) against ln(1 + hours since release) that
  // is a concave curve, so fit a quadratic (curvature <= 0, shrunk toward the
  // pooled curvature) and extrapolate from the last state with the fitted
  // curve's local slope there, damped: a straight line, like the house curve
  // baseline's, keeps rising and overshoots. Points at 0.00% are censored
  // (the page prints 2 decimals, so 0.00% only says < 0.005%) and are left
  // out of the fit: their logit would be an invented -9.21. The level is
  // anchored at the last observed rate (an exact read); the fit only predicts
  // the change from there, damped and soft-capped.
  if (states.length >= 3) {
    const xs = states.map((s) => Math.log1p(Math.max(0, (s.atMs - releaseMs) / 3_600_000)));
    const xTarget = Math.log1p(Math.max(0, hoursSince(c.target, releaseMs)));
    const xLast = xs[xs.length - 1];
    const rw = RW_SIGMA * Math.sqrt(days);
    // Typical refresh interval from the gaps between distinct states (hours).
    const stateGaps: number[] = [];
    for (let i = 1; i < states.length; i++) stateGaps.push((states[i].atMs - states[i - 1].atMs) / 3_600_000);
    const intervalH = stateGaps.length ? Math.max(0.5, median(stateGaps)) : INTERVAL_MEDIAN_H;

    // First pass: each trophy's usable points and quadratic curvature.
    type Fit = { xx: number[]; ys: number[]; quad: [number, number, number] | null; lin: [number, number] | null };
    const fits: Fit[] = c.trophies.map((_, i) => {
      const xx: number[] = [];
      const ys: number[] = [];
      for (let k = 0; k < states.length; k++) {
        const r = states[k].rates[i];
        if (r !== null && r !== undefined && r > 0 && r < 99.9) {
          xx.push(xs[k]);
          ys.push(logit(r));
        }
      }
      const quad = xx.length >= 4 ? olsQuad(xx, ys) : null;
      const lin = xx.length >= 2 ? ols(xx, ys) : null;
      return { xx, ys, quad, lin };
    });
    const curvatures = fits
      .map((f) => (f.quad && f.quad[2] < 0 ? f.quad[2] : null))
      .filter((v): v is number => v !== null);
    const pooledC = curvatures.length ? median(curvatures) : 0;

    return c.trophies.map((t, i) => {
      const { rate: lastRate } = lastKnown(c, i);
      const level = logit(lastRate);
      const f = fits[i];
      // Saturated trophy (or no usable points): stay where it is, tight spread.
      if (lastRate >= 99.9 || f.xx.length < 2 || !f.lin) {
        const s = Math.sqrt(rw * rw + 0.05 * 0.05);
        return { id: t.id, q: Z.map((z) => round4(clamp(invLogit(level + z * s)))) };
      }
      const n = f.xx.length;
      let a: number;
      let b: number;
      let cCurv: number;
      if (f.quad && f.quad[2] <= 0) {
        // Shrink the curvature toward the pooled curvature, refit a and b.
        cCurv = (n * f.quad[2] + SHRINK_K * pooledC) / (n + SHRINK_K);
        const adj = f.ys.map((y, k) => y - cCurv * f.xx[k] * f.xx[k]);
        [a, b] = ols(f.xx, adj);
      } else {
        [a, b] = f.lin;
        cCurv = 0;
      }
      const fitAt = (x: number): number => a + b * x + cCurv * x * x;
      // Extrapolate linearly from the last state with the fitted curve's
      // LOCAL slope there (the concavity has already bent it down from the
      // average slope); a parabola evaluated far out would turn down and
      // freeze the forecast, so the curve is used for its slope, not its
      // far value. Damped, then soft-capped.
      const slopeLast = b + 2 * cCurv * xLast;
      let move = slopeLast * (xTarget - xLast) * MOVE_DAMP;
      move = MOVE_CAP * Math.tanh(move / MOVE_CAP);
      const resid = f.ys.map((y, k) => y - fitAt(f.xx[k]));
      const dof = Math.max(1, n - (cCurv !== 0 ? 3 : 2));
      const sd = Math.max(RESID_FLOOR, Math.sqrt(resid.reduce((s, e) => s + e * e, 0) / dof));
      // Staleness: the answer reflects the page as of its last refresh before
      // the answer read, up to one interval before the target.
      const dXInterval = Math.max(0, xTarget - Math.log1p(Math.max(0, hoursSince(c.target, releaseMs) - intervalH)));
      const staleSd = (Math.abs(slopeLast) * dXInterval) / Math.sqrt(12);
      const slopeUnc = Math.min(0.8, 0.3 * Math.abs(move));
      const s =
        Math.sqrt(rw * rw + sd * sd + staleSd * staleSd + slopeUnc * slopeUnc) * SPREAD_SCALE;
      const medianLogit = level + move;
      return { id: t.id, q: Z.map((z) => round4(clamp(invLogit(medianLogit + z * s)))) };
    });
  }

  // ---- Refresh-mixture path: fewer than 3 distinct states. ----
  const st = states[states.length - 1];
  const ageH = Math.max(0, (atMs - states[0].atMs) / 3_600_000);
  // Probability a refresh lands in (at, target + one read gap], given the
  // current page state has already survived `ageH` hours.
  const fAge = lognormCdf(Math.max(ageH, 0.01), INTERVAL_MEDIAN_H, INTERVAL_SIGMA);
  const fEnd = lognormCdf(ageH + horizonEffH, INTERVAL_MEDIAN_H, INTERVAL_SIGMA);
  let pRef = (fEnd - fAge) / Math.max(1e-9, 1 - fAge);
  pRef = Math.min(0.98, Math.max(0.02, pRef));

  // Players multiplier grid: one refresh, or compounded over a long horizon.
  let multGrid: Array<[number, number]>;
  if (horizonEffH > 3 * INTERVAL_MEDIAN_H) {
    const nRef = horizonEffH / INTERVAL_MEDIAN_H;
    multGrid = lognormGrid(Math.pow(MULT_MEDIAN, nRef), Math.sqrt(nRef) * MULT_SIGMA);
  } else {
    multGrid = lognormGrid(MULT_MEDIAN, MULT_SIGMA);
  }

  // Earners of the last state, and the universal trophy (held by everyone).
  const lastEarners = c.trophies.map((_, i) => earners(st, i));
  const universalIdx = c.trophies.map((_, i) => (st.rates[i] ?? 0) >= 99.99);
  const extraDenom = lastEarners.reduce((s, e, i) => (universalIdx[i] ? s : s + e), 0);

  return c.trophies.map((t, i) => {
    const { rate } = lastKnown(c, i);
    const extraWeight = extraDenom > 0 && !universalIdx[i] ? lastEarners[i] / extraDenom : 0;
    return {
      id: t.id,
      q: mixtureQuantiles(rate, lastEarners[i], st.players, universalIdx[i], extraWeight, pRef, multGrid),
    };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history.length) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
