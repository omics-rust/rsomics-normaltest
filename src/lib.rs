//! D'Agostino-Pearson normality battery — `scipy.stats` equivalent.
//!
//! Input is one numeric value per line. One of three tests is selected:
//! `normaltest` (omnibus K²), `skewtest` (skew z-score) or `kurtosistest`
//! (kurtosis z-score).

mod dagostino;
mod igamc;
mod ndtr;

use std::io::{BufRead, Write};

use rsomics_common::{Result, RsomicsError};

pub use dagostino::{Alternative, TestResult, kurtosistest, normaltest, skewtest};

/// Which member of the D'Agostino-Pearson battery to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Test {
    Normaltest,
    Skewtest,
    Kurtosistest,
}

/// Parse one numeric value per line. Blank lines are skipped.
pub fn parse_values<R: BufRead>(reader: R) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line.map_err(RsomicsError::Io)?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: f64 = line.parse().map_err(|_| {
            RsomicsError::InvalidInput(format!("line {}: '{line}' is not a number", lineno + 1))
        })?;
        values.push(v);
    }
    Ok(values)
}

/// Run the selected test on a sample. `alt` is ignored for `normaltest`, which is
/// always a two-sided K² chi-squared test.
pub fn run_test(values: &[f64], test: Test, alt: Alternative) -> Result<TestResult> {
    match test {
        Test::Normaltest => normaltest(values),
        Test::Skewtest => skewtest(values, alt),
        Test::Kurtosistest => kurtosistest(values, alt),
    }
}

/// Parse a reader and write `statistic<TAB>p` (no JSON; the framework emits the
/// JSON envelope when `--json` is set).
pub fn run<R: BufRead, W: Write>(
    reader: R,
    out: &mut W,
    test: Test,
    alt: Alternative,
) -> Result<TestResult> {
    let values = parse_values(reader)?;
    let result = run_test(&values, test, alt)?;
    writeln!(out, "{}\t{}", result.statistic, result.p).map_err(RsomicsError::Io)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_one_per_line() {
        let v = parse_values("1\n2\n3\n".as_bytes()).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn parse_skips_blanks() {
        let v = parse_values("1\n\n2\n  \n3\n".as_bytes()).unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn parse_rejects_non_numeric() {
        assert!(parse_values("1\nfoo\n".as_bytes()).is_err());
    }

    #[test]
    fn run_normaltest_emits_two_fields() {
        let input = "1\n2\n3\n4\n5\n6\n7\n100\n";
        let mut out = Vec::new();
        let r = run(
            input.as_bytes(),
            &mut out,
            Test::Normaltest,
            Alternative::TwoSided,
        )
        .unwrap();
        let s = String::from_utf8(out).unwrap();
        let parts: Vec<&str> = s.trim().split('\t').collect();
        assert_eq!(parts.len(), 2);
        assert!(r.statistic > 0.0);
    }
}
