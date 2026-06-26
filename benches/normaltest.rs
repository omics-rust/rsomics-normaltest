use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use rsomics_normaltest::{Alternative, Test, run_test};

fn sample(n: usize) -> Vec<f64> {
    // Deterministic skewed sample via a simple LCG fed through a cube — no rng dep.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            let z = u - 0.5;
            z * z * z * 12.0
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let x = sample(1_000_000);
    c.bench_function("normaltest_1M", |b| {
        b.iter(|| run_test(black_box(&x), Test::Normaltest, Alternative::TwoSided).unwrap())
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
