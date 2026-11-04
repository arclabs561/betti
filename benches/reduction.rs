//! Reduction throughput against `lophat` on one boundary matrix.
//!
//! Each case is a seeded random point cloud in the unit square or cube, turned
//! into a Vietoris-Rips 2-skeleton truncated at a radius that keeps the
//! expected vertex degree near [`EXPECTED_DEGREE`], so the complex grows
//! roughly linearly with the point count. `betti` builds the boundary matrix
//! once; `lophat` receives the same columns in the same order. Matrix
//! construction is outside every timed region.
//!
//! Groups:
//!
//! - `vr2`: the 2-skeleton reduced by `betti` (plain and clearing) and by
//!   `lophat` serial and lock-free. Lock-free runs on rayon's global pool, so it
//!   uses every core unless `RAYON_NUM_THREADS` says otherwise; the others are
//!   single-threaded.
//! - `h0`: the 1-skeleton only. `betti`'s union-find path takes vertices and
//!   edges, not a matrix, so it is compared with both matrix reducers on the
//!   1-skeleton boundary matrix, which yields the same dimension-zero pairs.

#[path = "../tests/common/mod.rs"]
mod common;

use std::f64::consts::PI;
use std::hint::black_box;

use betti::{reduction, reduction_with, BoundaryMatrix, H0Filtration, ReductionOptions};
use common::{rips, Complex, XorShift64};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use lophat::{
    algorithms::{Decomposition, DecompositionAlgo, LockFreeAlgorithm, SerialAlgorithm},
    columns::VecColumn,
    options::LoPhatOptions,
};

/// Target mean vertex degree of the truncated Rips graph.
const EXPECTED_DEGREE: f64 = 10.0;

/// Point counts per ambient dimension.
const SIZES: [usize; 3] = [50, 200, 1000];

/// A uniform sample of the unit cube in `dim` dimensions.
fn points(count: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = XorShift64::new(seed);
    (0..count)
        .map(|_| {
            (0..dim)
                .map(|_| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
                .collect()
        })
        .collect()
}

/// The radius at which a uniform sample of `count` points has mean degree
/// [`EXPECTED_DEGREE`], ignoring boundary effects.
fn radius(count: usize, dim: usize) -> f64 {
    let n = count as f64;
    match dim {
        2 => (EXPECTED_DEGREE / (n * PI)).sqrt(),
        3 => (3.0 * EXPECTED_DEGREE / (4.0 * PI * n)).cbrt(),
        _ => unreachable!("only R^2 and R^3 are benchmarked"),
    }
}

fn boundary(complex: &Complex) -> BoundaryMatrix {
    BoundaryMatrix::new(&complex.filtration()).unwrap()
}

fn h0_filtration(complex: &Complex) -> H0Filtration {
    let (births, edges) = complex.one_skeleton();
    let mut filtration = H0Filtration::new();
    for birth in births {
        filtration.add_vertex(birth).unwrap();
    }
    for (i, j, value) in edges {
        filtration.add_edge(i, j, value).unwrap();
    }
    filtration
}

fn columns(matrix: &BoundaryMatrix) -> Vec<VecColumn> {
    (0..matrix.len())
        .map(|column| VecColumn::from((matrix.dim(column), matrix.column(column).to_vec())))
        .collect()
}

fn lophat_serial(columns: Vec<VecColumn>) -> usize {
    SerialAlgorithm::<VecColumn>::init(None)
        .add_cols(columns.into_iter())
        .decompose()
        .diagram()
        .len()
}

fn lophat_lock_free(columns: Vec<VecColumn>) -> usize {
    let options = LoPhatOptions {
        clearing: true,
        ..Default::default()
    };
    LockFreeAlgorithm::init(Some(options))
        .add_cols(columns.into_iter())
        .decompose()
        .diagram()
        .len()
}

fn label(dim: usize, count: usize, matrix: &BoundaryMatrix) -> String {
    format!("R{dim}/n{count}/cols{}", matrix.len())
}

fn vr2(c: &mut Criterion) {
    let mut group = c.benchmark_group("vr2");
    group.sample_size(20);
    for dim in [2, 3] {
        for count in SIZES {
            let complex = rips(
                &points(count, dim, 0xBE77 + count as u64),
                radius(count, dim),
                true,
            );
            let matrix = boundary(&complex);
            let cols = columns(&matrix);
            // Both reducers must agree on the number of classes before their
            // timings mean anything.
            let pairs = reduction(&matrix).len();
            assert_eq!(pairs, lophat_serial(cols.clone()));
            assert_eq!(pairs, lophat_lock_free(cols.clone()));

            let id = label(dim, count, &matrix);
            group.bench_function(BenchmarkId::new("betti", &id), |b| {
                b.iter(|| black_box(reduction(black_box(&matrix))).len())
            });
            group.bench_function(BenchmarkId::new("betti_clearing", &id), |b| {
                b.iter(|| {
                    black_box(reduction_with(
                        black_box(&matrix),
                        ReductionOptions::clearing(),
                    ))
                    .len()
                })
            });
            group.bench_function(BenchmarkId::new("lophat_serial", &id), |b| {
                b.iter_batched(|| cols.clone(), lophat_serial, BatchSize::LargeInput)
            });
            group.bench_function(BenchmarkId::new("lophat_lock_free", &id), |b| {
                b.iter_batched(|| cols.clone(), lophat_lock_free, BatchSize::LargeInput)
            });
        }
    }
    group.finish();
}

fn h0(c: &mut Criterion) {
    let mut group = c.benchmark_group("h0");
    group.sample_size(20);
    for dim in [2, 3] {
        for count in SIZES {
            let complex = rips(
                &points(count, dim, 0xBE77 + count as u64),
                radius(count, dim),
                false,
            );
            let filtration = h0_filtration(&complex);
            let matrix = boundary(&complex);
            let cols = columns(&matrix);
            let essential = filtration.decompose().essential().count();
            assert_eq!(essential, reduction(&matrix).betti_numbers(0)[0]);

            let id = label(dim, count, &matrix);
            group.bench_function(BenchmarkId::new("betti_union_find", &id), |b| {
                b.iter(|| black_box(black_box(&filtration).decompose()).len())
            });
            group.bench_function(BenchmarkId::new("betti_matrix", &id), |b| {
                b.iter(|| black_box(reduction(black_box(&matrix))).len())
            });
            group.bench_function(BenchmarkId::new("lophat_serial", &id), |b| {
                b.iter_batched(|| cols.clone(), lophat_serial, BatchSize::LargeInput)
            });
        }
    }
    group.finish();
}

criterion_group!(benches, vr2, h0);
criterion_main!(benches);
