# Raw result logs

## Top-1 / Top-3, n=61 (ds_train.py, 120s REPL timeout)

    LINE level (i* granularity):
      Ranker Top-1                39.3%  +/- 6.3%   (random 10.8%, chance-corr  32.0%)
      Ranker Top-3                55.7%  +/- 6.4%   (random 31.6%, chance-corr  35.3%)
      Positional prior Top-1      14.8%  +/- 4.5%
      Positional prior Top-3      36.1%  +/- 6.1%
    BLOCK level:
      Ranker Top-1                62.3%  +/- 6.2%   (random 24.7%, chance-corr  49.9%)
      Ranker Top-3                75.4%  +/- 5.5%   (random 56.8%, chance-corr  43.1%)
    n=61: one flip = 1.6 points.
    Largest-block baseline: 30/61 = 49.2%  (true block = 38% of proof)

## Hybrids + stability (short_boost.py)

    ### timeout 120s, n=61
      Learned ranker                          39.3%  (24/61)
      Compiler-nearest (model fallback)       39.3%  (24/61)
      Switch: compiler else model             39.3%  (24/61)
      Rank fusion (equal weight)              37.7%  (23/61)
      Rank fusion (weight from train fold)    39.3%  (24/61)
      Constant baseline                       14.8%
      Random                                  10.8%

    ### timeout 420s, n=61   (42/61 proofs carry a compiler error)
      all five variants                       37.7%  (23/61)
      Constant baseline                       14.8%

    ### timeout comparison at n=32 (for contrast)
      120s: 31.2%   420s: 18.8%   swing 12.4 pts

## Extrinsic: with vs without the model (short_pareto.py, n=32, 3 seeds)

    seed   without (restart)        with (predicted)     ratio
    0      30/32  1,009 tok/solve   20/32  1,482         0.68x
    1      29/32    975             20/32  1,363         0.72x
    2      27/32  1,556             21/32  1,318         1.18x
    mean 0.86x, sd 0.23, helps 1/3 seeds
    oracle ceiling: 83/96 @ 926 tok/solve (1.26x vs restart)
    PAIRED per-proof (96): with-model better 42 | without better 47 | tie 7

## External model on these labels (peer_model_test.py)

    Top-1 accuracy              : 7/32 = 21.9%
    predicts the FIRST step     : 29/32 = 90.6%
    our labels AT first step    : 6/32 = 18.8%
    random                      : 26.9%
    mean max-probability        : 0.963
    prediction distribution     : step 0: 29, step 1: 3

## Harvests

    run 1:  60 problems -> 24 solvable (40%) -> 32 labels / 14 problems
    run 2: 200 problems -> 62 solvable (31%) -> 29 labels / 29 problems
    combined: 61 labels / 43 problems, zero problem overlap
    proof length: 16.8 mean / 13 median (harvest), 14.3 mean (labeled)
