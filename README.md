# rsomics-normaltest

The D'Agostino-Pearson normality battery — a value-exact Rust port of SciPy's
`normaltest`, `skewtest` and `kurtosistest`. One cohesive family: `normaltest`
literally combines the skew and kurtosis z-scores into an omnibus K² statistic,
so the three live in one tool selected by `--test`.

## Usage

```sh
rsomics-normaltest sample.tsv                       # normaltest (default)
rsomics-normaltest sample.tsv --test skewtest
rsomics-normaltest sample.tsv --test kurtosistest --alternative greater
cat sample.tsv | rsomics-normaltest -               # '-' or omitted reads stdin
```

Input is one numeric value per line. Output is a single tab-separated line:

| `--test`       | output            | statistic                         |
|----------------|-------------------|-----------------------------------|
| `normaltest`   | `K2<TAB>p`        | K² = Z_skew² + Z_kurt²            |
| `skewtest`     | `Z<TAB>p`         | D'Agostino 1970 skew z-score      |
| `kurtosistest` | `Z<TAB>p`         | Anscombe-Glynn 1983 kurtosis z    |

`--alternative {two-sided,less,greater}` applies to `skewtest`/`kurtosistest`
only; `normaltest` is always a two-sided K² chi-squared test. `--json` emits the
same fields inside the rsomics JSON envelope.

Minimum sample sizes match SciPy: `skewtest` and `normaltest` require n ≥ 8,
`kurtosistest` requires n ≥ 5.

## Method

For a sample `x` of length `n`, with biased central moments
`m_i = (1/n)·Σ(x − x̄)^i` (Fisher-Pearson, `bias=True`):

```
b1 = m3 / m2^1.5          (skewness)
b2 = m4 / m2^2            (Pearson kurtosis)

Z_skew = δ·asinh(y / α)                          (D'Agostino 1970)
Z_kurt = ((1−2/(9A)) − cbrt((1−2/A)/|denom|)) / √(2/(9A))   (Anscombe-Glynn 1983)

K2 = Z_skew² + Z_kurt²    p = chi2.sf(K2, 2)     (normaltest)
```

The skew/kurtosis p-values use `2·ndtr(−|Z|)` (two-sided) or `ndtr(±Z)`
(one-sided) via a direct port of the Cephes `ndtr` normal CDF, matching SciPy's
`special.ndtr`. The K² p-value uses a port of the Cephes `igam`/`igamc`
incomplete-gamma routines underlying `special.chdtrc`. The central moments are
accumulated with NumPy's block-recursive pairwise summation, so the statistics
are bit-identical to SciPy rather than diverging by a last-ULP summation-order
difference on large samples.

## Origin

This crate is an independent Rust reimplementation of
`scipy.stats.normaltest` / `skewtest` / `kurtosistest` based on:

- D'Agostino, R. B. (1970/1971), "An omnibus test of normality for moderate and
  large sample size", *Biometrika* 58, 341-348. doi:10.1093/biomet/58.2.341
- Anscombe, F. J. and Glynn, W. J. (1983), "Distribution of the kurtosis
  statistic b2 for normal samples", *Biometrika* 70, 227-234.
  doi:10.1093/biomet/70.1.227
- D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from
  normality", *Biometrika* 60, 613-622. doi:10.1093/biomet/60.3.613
- SciPy's documented moment definitions and the exact transforms in
  `scipy.stats.{skew,kurtosis,skewtest,kurtosistest,normaltest}` (SciPy 1.17.1,
  BSD-3-Clause — reading and citing permitted)
- The Cephes `ndtr` normal CDF and `igam`/`igamc` incomplete-gamma routines
  underlying `scipy.special.ndtr` and `scipy.special.chdtrc`
- Black-box value testing against the SciPy binary (golden fixtures in
  `tests/golden/`, computed once with SciPy 1.17.1 and checked in)

License: MIT OR Apache-2.0.
Upstream credit: SciPy <https://scipy.org> (BSD-3-Clause); Cephes Mathematical
Library by Stephen L. Moshier (BSD-style, as vendored in SciPy).
