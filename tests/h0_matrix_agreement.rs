//! The union-find `H0` path must agree with the matrix reducer.
//!
//! Both paths see the same one-dimensional filtration: vertices in id order
//! with their birth values, and edges with their death values. The pairing of
//! a filtration is unique, so the pairs, and every component count derived
//! from them, must match exactly.

mod common;

use std::collections::BTreeMap;

use betti::{
    reduction, reduction_with, BoundaryMatrix, Decomposition, H0Filtration, ReductionOptions,
};
use common::{random_one_dimensional, Complex, XorShift64};

/// Seeds compared: `1..=SEEDS`.
const SEEDS: u64 = 300;

/// Build the same filtration on both sides.
fn build(complex: &Complex) -> (BoundaryMatrix, H0Filtration) {
    let matrix = BoundaryMatrix::new(&complex.filtration()).expect("a valid filtration");
    let (births, edges) = complex.one_skeleton();
    let mut filtration = H0Filtration::new();
    for birth in births {
        filtration.add_vertex(birth).expect("a valid vertex");
    }
    for (left, right, death) in edges {
        filtration
            .add_edge(left, right, death)
            .expect("a valid edge");
    }
    (matrix, filtration)
}

/// The dimension-zero pairs of a decomposition, keyed by birth vertex.
fn betti_pairs(decomposition: &Decomposition) -> BTreeMap<usize, Option<f64>> {
    let mut pairs = BTreeMap::new();
    for (birth, death) in decomposition.pairs() {
        if birth.dim() != 0 {
            continue;
        }
        let vertex = birth.simplex().vertices()[0];
        assert!(
            pairs.insert(vertex, death).is_none(),
            "vertex {vertex} gave birth to two classes"
        );
    }
    pairs
}

/// Components alive at `threshold` according to the matrix reducer.
fn betti_count(decomposition: &Decomposition, threshold: f64) -> usize {
    decomposition
        .pairs()
        .into_iter()
        .filter(|(birth, death)| {
            birth.dim() == 0
                && birth.value() <= threshold
                && death.is_none_or(|death| death > threshold)
        })
        .count()
}

/// Thresholds that land between, on, and beyond the quarter-grid values.
fn thresholds() -> impl Iterator<Item = f64> {
    (-1..=12).map(|step| 0.25 * step as f64)
}

#[test]
fn h0_pairs_agree_with_the_matrix_reducer() {
    for seed in 1..=SEEDS {
        let mut rng = XorShift64::new(seed);
        let complex = random_one_dimensional(&mut rng);
        let (matrix, filtration) = build(&complex);

        let plain = reduction(&matrix);
        let cleared = reduction_with(&matrix, ReductionOptions::clearing());
        let expected = betti_pairs(&plain);
        assert_eq!(
            expected,
            betti_pairs(&cleared),
            "seed {seed}: clearing changed the dimension-zero pairs"
        );

        let union_find = filtration.decompose();
        let actual: BTreeMap<usize, Option<f64>> = union_find
            .pairs()
            .iter()
            .map(|pair| (pair.birth_vertex(), pair.death()))
            .collect();

        assert_eq!(
            actual,
            expected,
            "seed {seed}: union-find disagrees with the matrix reducer on {} simplices",
            complex.simplices.len()
        );
        let vertex_count = matrix.simplices().iter().filter(|s| s.dim() == 0).count();
        assert_eq!(
            union_find.len(),
            vertex_count,
            "seed {seed}: one pair per vertex"
        );
        assert_eq!(
            union_find.essential().count(),
            plain.betti_numbers(0)[0],
            "seed {seed}: essential classes are the Betti number"
        );
    }
}

#[test]
fn component_counts_agree_at_every_threshold() {
    for seed in 1..=SEEDS {
        let mut rng = XorShift64::new(seed);
        let complex = random_one_dimensional(&mut rng);
        let (matrix, filtration) = build(&complex);
        let decomposition = reduction(&matrix);
        let union_find = filtration.decompose();

        for threshold in thresholds() {
            assert_eq!(
                union_find.count_components_above(threshold),
                betti_count(&decomposition, threshold),
                "seed {seed}: component count at {threshold} differs"
            );
        }
    }
}
