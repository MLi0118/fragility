# Raw result logs

## Top-1 / Top-3, n=209 (ds_train.py) — CURRENT HEADLINE

    Built 209 proofs | mean 13.9 lines, 7.4 blocks
    LINE level (i* granularity):
      Ranker Top-1                27.8%  +/- 3.1%   (random 10.1%, chance-corr  19.6%)
      Ranker Top-3                42.1%  +/- 3.4%   (random 30.0%, chance-corr  17.2%)
      Positional prior Top-1      11.5%  +/- 2.2%
      Positional prior Top-3      29.2%  +/- 3.1%
    BLOCK level:
      Ranker Top-1                67.9%  +/- 3.2%   (random 25.2%, chance-corr  57.2%)
      Ranker Top-3                82.3%  +/- 2.6%   (random 55.4%, chance-corr  60.3%)
    n=209: one flip = 0.5 points.
    constant baseline: Top-1 11.5%  Top-3 31.6%
    i*/n mean 0.478

## Proof-length distribution (labeled corpus, n=209)

    mean 13.9 | median 12 | min 2 | max 89
    p10 6  p25 8  p75 17  p90 23   -> implied random Top-1 10.1%
     1-5     19 ( 9.1%)
     6-10    65 (31.1%)
    11-20    95 (45.5%)
    21-40    26 (12.4%)
    41+       4 ( 1.9%)

## Harvests

    run 1:   60 problems -> 24 solvable (40%) ->  32 labels /  14 problems
    run 2:  200 problems -> 62 solvable (31%) ->  29 labels /  29 problems
    run 3: 1000 problems -> 286 solvable (29%) -> 148 labels / 148 problems
    combined: 209 labels / 191 problems, zero problem overlap
    run 3 cost: 12,652 generations, 4,690,360 tokens
    harvest proof length: 17.4 / 16.8 / 18.4 mean lines (runs 1/2/3)

## Earlier corpus sizes (same method)

    n= 12 / 12 problems: Top-1  8.3%  (below constant baseline)
    n= 32 / 14 problems: Top-1 31.2%  Top-3 37.5%  gain +12.4  SE +/-8.2
    n= 61 / 43 problems: Top-1 39.3%  Top-3 55.7%  gain +24.6  SE +/-6.3
    n=209 /191 problems: Top-1 27.8%  Top-3 42.1%  gain +16.3  SE +/-3.1

## Hybrids + timeout stability (short_boost.py, n=61)

    timeout 120s: ranker 39.3%; compiler-nearest 39.3%; switch 39.3%;
                  rank fusion 37.7%; weighted fusion 39.3%
    timeout 420s: all five variants 37.7%   (42/61 proofs carry a compiler error)
    at n=32 the same timeout change swung Top-1 by 12.4 points (31.2% -> 18.8%)
    NOT re-tested at n=209.
    largest-block baseline at n=61: 30/61 = 49.2% (true block = 38% of proof)

## Extrinsic: with vs without the model (short_pareto.py, n=32, 3 seeds)

    seed   without (restart)        with (predicted)     ratio
    0      30/32  1,009 tok/solve   20/32  1,482         0.68x
    1      29/32    975             20/32  1,363         0.72x
    2      27/32  1,556             21/32  1,318         1.18x
    mean 0.86x, sd 0.23, helps 1/3 seeds
    oracle ceiling: 83/96 @ 926 tok/solve (1.26x vs restart)
    PAIRED per-proof (96): with-model better 42 | without better 47 | tie 7
    RERUN AT n=209 IN PROGRESS.

## External model on our labels (peer_model_test.py)

    Top-1 accuracy              : 7/32 = 21.9%
    predicts the FIRST step     : 29/32 = 90.6%
    random                      : 26.9%
    mean max-probability        : 0.963
    prediction distribution     : step 0: 29, step 1: 3

## Long-proof corpus (Goedel-Prover-V2-8B) — where the method FAILS

    n=28 / 26 problems, mean 41.7 lines
    Top-1 17.9% vs constant baseline 28.6%   (worse than constant)
    on the i*>0 subset (genuine mid-proof failure): 0/20
    branch-vs-restart decision: 0/8 doomed proofs detected
    oracle vs restart, 3 seeds: 0.74x / 0.83x / 0.69x  (mean 0.75x, sd 0.06)
    paired per-proof: oracle better 19 | restart better 7 | tie 7
    perfect selective-branching ceiling: 1.39x
