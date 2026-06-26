//! The D'Agostino-Pearson normality battery, matching `scipy.stats`.
//!
//! Three tests share one moment computation:
//!
//! - `skewtest` (D'Agostino 1970): standardize the sample skewness `b1` through
//!   the `y/α → δ·asinh` transform into a z-score.
//! - `kurtosistest` (Anscombe-Glynn 1983): standardize the Pearson kurtosis `b2`
//!   through the `(1−2/(9A)) − cbrt(...)` transform into a z-score.
//! - `normaltest` (D'Agostino-Pearson 1973): `K² = Z_skew² + Z_kurt²`, an omnibus
//!   chi-squared-with-2-df statistic.
//!
//! Skewness and kurtosis use SciPy's biased central moments `m_i = (1/n)·Σ(x−x̄)^i`
//! (Fisher-Pearson, `bias=True`). SciPy centers once around the sample mean, then
//! `skew`/`kurtosis` accumulate powers of the demeaned values — reproduced here so
//! the moments are bit-identical, not merely algebraically equal.

use serde::Serialize;

use crate::igamc::chi2_sf;
use crate::ndtr::ndtr;
use rsomics_common::{Result, RsomicsError};

/// Tail of the alternative hypothesis for `skewtest` / `kurtosistest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// A `statistic`/`p`-value pair, mirroring SciPy's `*testResult`.
#[derive(Debug, Clone, Serialize)]
pub struct TestResult {
    /// Test statistic: the z-score for skew/kurtosis, or K² for normaltest.
    pub statistic: f64,
    /// p-value for the hypothesis test.
    pub p: f64,
}

/// NumPy's pairwise summation of `f(0)..f(len)`. `np.mean`/`np.sum` reduce with
/// this block-recursive scheme, not a flat left fold, so reproducing it is what
/// keeps the moments — and therefore the statistics — bit-identical to SciPy on
/// large samples where summation order shows up in the last ULP.
fn pairwise_sum(len: usize, off: usize, f: &impl Fn(usize) -> f64) -> f64 {
    if len <= 8 {
        let mut acc = 0.0;
        for i in 0..len {
            acc += f(off + i);
        }
        acc
    } else if len <= 128 {
        let mut a = [0.0f64; 8];
        for (k, acc) in a.iter_mut().enumerate() {
            *acc = f(off + k);
        }
        let mut i = 8;
        while i + 8 <= len {
            for (k, acc) in a.iter_mut().enumerate() {
                *acc += f(off + i + k);
            }
            i += 8;
        }
        let mut res = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        while i < len {
            res += f(off + i);
            i += 1;
        }
        res
    } else {
        let mut n2 = len / 2;
        n2 -= n2 % 8;
        pairwise_sum(n2, off, f) + pairwise_sum(len - n2, off + n2, f)
    }
}

/// Biased central moments `m2, m3, m4` of `x` around its sample mean, matching
/// SciPy's `_moment`: demean once, then take the pairwise-summed mean of the
/// powers of the demeaned values.
fn central_moments(x: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mean = pairwise_sum(x.len(), 0, &|i| x[i]) / n;

    let m2 = pairwise_sum(x.len(), 0, &|i| {
        let d = x[i] - mean;
        d * d
    }) / n;
    let m3 = pairwise_sum(x.len(), 0, &|i| {
        let d = x[i] - mean;
        d * d * d
    }) / n;
    let m4 = pairwise_sum(x.len(), 0, &|i| {
        let d = x[i] - mean;
        let d2 = d * d;
        d2 * d2
    }) / n;
    (m2, m3, m4)
}

/// Fisher-Pearson sample skewness `b1 = m3 / m2^1.5` (`scipy.stats.skew`, bias=True).
fn skew(x: &[f64]) -> f64 {
    let (m2, m3, _) = central_moments(x);
    m3 / m2.powf(1.5)
}

/// Pearson kurtosis `b2 = m4 / m2^2` (`scipy.stats.kurtosis`, fisher=False).
fn pearson_kurtosis(x: &[f64]) -> f64 {
    let (m2, _, m4) = central_moments(x);
    m4 / (m2 * m2)
}

/// Two-sided / one-sided p-value from a normal z-score, via `scipy._SimpleNormal`:
/// `sf(x) = ndtr(−x)`, `cdf(x) = ndtr(x)`, two-sided `= 2·ndtr(−|z|)`.
fn normal_pvalue(z: f64, alt: Alternative) -> f64 {
    match alt {
        Alternative::Less => ndtr(z),
        Alternative::Greater => ndtr(-z),
        Alternative::TwoSided => 2.0 * ndtr(-z.abs()),
    }
}

/// `skewtest` z-score (D'Agostino 1970), as SciPy's `skewtest` computes it.
fn skewtest_z(b2: f64, n: f64) -> f64 {
    let y = b2 * (((n + 1.0) * (n + 3.0)) / (6.0 * (n - 2.0))).sqrt();
    let beta2 = 3.0 * (n * n + 27.0 * n - 70.0) * (n + 1.0) * (n + 3.0)
        / ((n - 2.0) * (n + 5.0) * (n + 7.0) * (n + 9.0));
    let w2 = -1.0 + (2.0 * (beta2 - 1.0)).sqrt();
    let delta = 1.0 / (0.5 * w2.ln()).sqrt();
    let alpha = (2.0 / (w2 - 1.0)).sqrt();
    let y = if y == 0.0 { 1.0 } else { y };
    delta * (y / alpha + ((y / alpha).powi(2) + 1.0).sqrt()).ln()
}

/// `kurtosistest` z-score (Anscombe-Glynn 1983), as SciPy's `kurtosistest` computes it.
fn kurtosistest_z(b2: f64, n: f64) -> f64 {
    let e = 3.0 * (n - 1.0) / (n + 1.0);
    let varb2 = 24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) * (n + 1.0) * (n + 3.0) * (n + 5.0));
    let x = (b2 - e) / varb2.sqrt();
    let sqrtbeta1 = 6.0 * (n * n - 5.0 * n + 2.0) / ((n + 7.0) * (n + 9.0))
        * (6.0 * (n + 3.0) * (n + 5.0) / (n * (n - 2.0) * (n - 3.0))).sqrt();
    let a =
        6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + (1.0 + 4.0 / (sqrtbeta1 * sqrtbeta1)).sqrt());
    let term1 = 1.0 - 2.0 / (9.0 * a);
    let denom = 1.0 + x * (2.0 / (a - 4.0)).sqrt();
    let term2 = denom.signum() * ((1.0 - 2.0 / a) / denom.abs()).powf(1.0 / 3.0);
    (term1 - term2) / (2.0 / (9.0 * a)).sqrt()
}

fn check_input(x: &[f64], min_n: usize, name: &str) -> Result<()> {
    if x.iter().any(|v| v.is_nan()) {
        return Err(RsomicsError::InvalidInput("input contains NaN".into()));
    }
    if x.len() < min_n {
        return Err(RsomicsError::InvalidInput(format!(
            "{name} requires at least {min_n} observations, got {}",
            x.len()
        )));
    }
    Ok(())
}

/// `scipy.stats.skewtest` — requires n ≥ 8.
pub fn skewtest(x: &[f64], alt: Alternative) -> Result<TestResult> {
    check_input(x, 8, "skewtest")?;
    let n = x.len() as f64;
    let z = skewtest_z(skew(x), n);
    Ok(TestResult {
        statistic: z,
        p: normal_pvalue(z, alt),
    })
}

/// `scipy.stats.kurtosistest` — requires n ≥ 5.
pub fn kurtosistest(x: &[f64], alt: Alternative) -> Result<TestResult> {
    check_input(x, 5, "kurtosistest")?;
    let n = x.len() as f64;
    let z = kurtosistest_z(pearson_kurtosis(x), n);
    Ok(TestResult {
        statistic: z,
        p: normal_pvalue(z, alt),
    })
}

/// `scipy.stats.normaltest` — the omnibus K² test, requires n ≥ 8 (skewtest's floor).
pub fn normaltest(x: &[f64]) -> Result<TestResult> {
    check_input(x, 8, "normaltest")?;
    let n = x.len() as f64;
    let s = skewtest_z(skew(x), n);
    let k = kurtosistest_z(pearson_kurtosis(x), n);
    let k2 = s * s + k * k;
    Ok(TestResult {
        statistic: k2,
        p: chi2_sf(2.0, k2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: f64, want: f64, rel: f64) {
        let d = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(d <= rel, "got {got:e} want {want:e} rel {d:e} > {rel:e}");
    }

    // scipy.stats.skewtest([1,2,3,4,5,6,7,8]) docstring example.
    #[test]
    fn skewtest_docstring() {
        let r = skewtest(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Alternative::TwoSided,
        )
        .unwrap();
        close(r.statistic, 1.010_804_860_917_778_7, 1e-12);
        close(r.p, 0.312_109_836_142_189_7, 1e-12);
    }

    #[test]
    fn skewtest_alternatives() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let less = skewtest(&x, Alternative::Less).unwrap();
        close(less.p, 0.843_945_081_928_905_2, 1e-12);
        let greater = skewtest(&x, Alternative::Greater).unwrap();
        close(greater.p, 0.156_054_918_071_094_84, 1e-12);
    }

    // scipy.stats.kurtosistest(range(20)) docstring example.
    #[test]
    fn kurtosistest_docstring() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = kurtosistest(&x, Alternative::TwoSided).unwrap();
        close(r.statistic, -1.705_810_415_212_206_2, 1e-12);
        close(r.p, 0.088_043_383_325_283_48, 1e-12);
        let less = kurtosistest(&x, Alternative::Less).unwrap();
        close(less.p, 0.044_021_691_662_641_74, 1e-12);
        let greater = kurtosistest(&x, Alternative::Greater).unwrap();
        close(greater.p, 0.955_978_308_337_358_3, 1e-12);
    }

    #[test]
    fn skewtest_requires_eight() {
        assert!(skewtest(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], Alternative::TwoSided).is_err());
    }

    #[test]
    fn kurtosistest_requires_five() {
        assert!(kurtosistest(&[1.0, 2.0, 3.0, 4.0], Alternative::TwoSided).is_err());
    }

    #[test]
    fn rejects_nan() {
        let x = [1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(normaltest(&x).is_err());
    }
}
