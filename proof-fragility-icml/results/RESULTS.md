# Raw results (from scores.json, written by scripts/score_arms.py)

## Direct counts: combined proof with pass and complete, 85 problems

    full85_feedback        solved 12   prover tokens  2,199,319 in   2,964,692 out   attempts   920  repairs   482
    full85_predict_cut     solved 16   prover tokens  4,724,519 in   5,464,066 out   attempts  2695  repairs  2163
    full85_random_cut      solved 11   prover tokens  5,246,879 in   4,666,364 out   attempts  2661  repairs  2157
    full85s_predict_cut    solved 13   prover tokens  4,882,551 in   5,504,630 out   attempts  2690  repairs  2163
    full85s_random_cut     solved 13   prover tokens  5,141,079 in   4,561,154 out   attempts  2599  repairs  2103

## Solved problems

    full85_feedback: 1967_b2, 1968_a2, 1977_a3, 1986_a1, 1986_a2, 1986_b1, 1988_b1, 1988_b2, 1990_a1, 1999_a1, 2001_a1, 2008_a1
    full85_predict_cut: 1965_a6, 1965_b3, 1967_b2, 1968_a2, 1977_a3, 1984_b2, 1986_a1, 1986_a2, 1986_b1, 1988_b1, 1988_b2, 1990_a1, 1999_a1, 2000_a2, 2001_a1, 2008_a1
    full85_random_cut: 1967_b2, 1977_a3, 1986_a1, 1986_a2, 1986_b1, 1988_b1, 1988_b2, 1990_a1, 1999_a1, 2001_a1, 2008_a1
    full85s_predict_cut: 1967_b2, 1968_a2, 1977_a3, 1986_a1, 1986_a2, 1986_b1, 1988_b1, 1988_b2, 1990_a1, 1998_b1, 1999_a1, 2001_a1, 2008_a1
    full85s_random_cut: 1965_a6, 1967_b2, 1977_a3, 1986_a1, 1986_a2, 1986_b1, 1988_b1, 1988_b2, 1990_a1, 1998_b1, 1999_a1, 2001_a1, 2008_a1

## Simulator replay (fixed waterfall, mean solved of 84 over 32 seeds)

    run                    b   k=1    k=2    k=3    k=4    k=5    k=6    k=7    k=8
    full85_feedback        1    5.78   6.31   6.66   6.66   6.66   6.66   6.66   6.66
    full85_feedback        2    8.34   8.72   9.47   9.47   9.47   9.47   9.47   9.47
    full85_feedback        3   10.03  10.03  10.84  10.84  10.84  10.84  10.84  10.84
    full85_feedback        4   11.00  11.00  12.00  12.00  12.00  12.00  12.00  12.00
    full85_predict_cut     1    6.81   7.94   8.22   8.22   8.22   8.22   8.22   8.22
    full85_predict_cut     2   10.47  11.59  12.03  12.03  12.03  12.03  12.03  12.03
    full85_predict_cut     3   12.72  13.59  14.28  14.28  14.28  14.28  14.28  14.28
    full85_predict_cut     4   14.00  15.00  16.00  16.00  16.00  16.00  16.00  16.00
    full85_random_cut      1    5.34   5.59   6.03   6.34   6.63   6.94   6.94   6.94
    full85_random_cut      2    8.53   8.72   9.13   9.28   9.28   9.44   9.44   9.44
    full85_random_cut      3   10.22  10.22  10.41  10.41  10.41  10.41  10.41  10.41
    full85_random_cut      4   11.00  11.00  11.00  11.00  11.00  11.00  11.00  11.00
    full85s_predict_cut    1    5.78   7.13   7.47   7.47   7.47   7.88   7.88   7.88
    full85s_predict_cut    2    8.34   9.88  10.31  10.31  10.31  11.00  11.00  11.00
    full85s_predict_cut    3   10.03  11.41  11.41  11.41  11.41  12.22  12.22  12.22
    full85s_predict_cut    4   11.00  12.00  12.00  12.00  12.00  13.00  13.00  13.00
    full85s_random_cut     1    5.78   6.22   7.09   7.22   7.22   7.34   7.34   7.34
    full85s_random_cut     2    8.34   8.66   9.81  10.00  10.00  10.19  10.19  10.19
    full85s_random_cut     3   10.03  10.22  11.69  11.69  11.69  11.69  11.69  11.69
    full85s_random_cut     4   11.00  11.00  13.00  13.00  13.00  13.00  13.00  13.00

## Repair prompts dropped by the 6144-token prompt filter (prover logs)

    full85 feedback      284/766 = 37%
    full85s predict_cut  11.7%
    full85s random_cut   12.0%

## Output-journal reuse in the controlled rerun (first attempts replayed)

    iteration          0        1        2      3
    predict_cut    300/300  112/191  21/32   3/4
    random_cut     300/300  112/164  18/28   3/4
