//! Value-exactness against `scipy.stats.{normaltest,skewtest,kurtosistest}`.
//!
//! `expected.tsv` holds `name<TAB>test<TAB>alternative<TAB>statistic<TAB>p`
//! computed by SciPy 1.17.1; each `name.tsv` is the corresponding
//! one-value-per-line sample. The statistic must match to 1e-13 relative and the
//! p-value to 1e-12 through the Cephes ndtr / igamc paths. SciPy is not invoked.
//!
//! `constant` (zero variance) and `infval` (a non-finite observation) are
//! degenerate samples where SciPy returns `nan`/`nan`; the expected `nan` cells
//! assert we produce NaN too instead of looping in the igamc tail.

use std::fs;
use std::path::PathBuf;

use rsomics_normaltest::{Alternative, Test, run_test};

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

fn rel_close(got: f64, want: f64, rel: f64, ctx: &str) {
    if want.is_nan() {
        assert!(got.is_nan(), "{ctx}: got {got:e} want nan");
        return;
    }
    if want == 0.0 {
        assert!(got.abs() <= rel, "{ctx}: got {got:e} want 0, abs > {rel:e}");
        return;
    }
    let d = (got - want).abs() / want.abs();
    assert!(
        d <= rel,
        "{ctx}: got {got:e} want {want:e} rel {d:e} > {rel:e}"
    );
}

fn read_sample(name: &str) -> Vec<f64> {
    fs::read_to_string(golden(&format!("{name}.tsv")))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

fn parse_test(s: &str) -> Test {
    match s {
        "normaltest" => Test::Normaltest,
        "skewtest" => Test::Skewtest,
        "kurtosistest" => Test::Kurtosistest,
        other => panic!("unknown test {other}"),
    }
}

fn parse_alt(s: &str) -> Alternative {
    match s {
        "two-sided" => Alternative::TwoSided,
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        other => panic!("unknown alternative {other}"),
    }
}

#[test]
fn matches_scipy_golden() {
    let expected = fs::read_to_string(golden("expected.tsv")).unwrap();
    let mut seen = 0;
    for line in expected.lines() {
        let mut f = line.split('\t');
        let name = f.next().unwrap();
        let test = parse_test(f.next().unwrap());
        let alt = parse_alt(f.next().unwrap());
        let stat: f64 = f.next().unwrap().parse().unwrap();
        let p: f64 = f.next().unwrap().parse().unwrap();

        let x = read_sample(name);
        let r = run_test(&x, test, alt).unwrap();
        let ctx = format!("{name}/{test:?}/{alt:?}");
        rel_close(r.statistic, stat, 1e-13, &ctx);
        rel_close(r.p, p, 1e-12, &ctx);
        seen += 1;
    }
    assert_eq!(seen, 49, "expected 49 golden cases");
}
