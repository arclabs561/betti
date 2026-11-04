//! Persistence diagrams checked against GUDHI.
//!
//! `tests/fixtures/oracle/` holds small point clouds and the H0 and H1
//! Vietoris-Rips diagrams GUDHI computed for them over GF(2), cross-checked
//! against Ripser when generated. `scripts/oracle_fixtures.py` regenerates
//! them. Each test builds the full Rips 2-skeleton here, reduces it, and
//! compares the diagrams.
//!
//! Comparison: per dimension, both diagrams are sorted by `(birth, death)` and
//! matched in order. The counts must be equal and every matched coordinate
//! must agree within [`TOLERANCE`]; essential classes must match essential
//! classes. A passing match bounds the bottleneck distance by the tolerance.
//! Zero-persistence pairs are dropped on the `betti` side because GUDHI and
//! Ripser never report them.

mod common;

use std::fs;
use std::path::PathBuf;

use betti::{reduction, BoundaryMatrix};
use common::rips;

/// Absolute tolerance on every birth and death value.
///
/// The fixtures carry GUDHI's double-precision values to 17 significant
/// digits, and both sides take square roots of the same sums of squares, so
/// any honest disagreement sits near 1e-16; 1e-9 leaves room for a different
/// summation order without hiding a wrong pairing, whose error is the gap
/// between two distinct distances.
const TOLERANCE: f64 = 1e-9;

/// The dimensions the fixtures record.
const DIMENSIONS: [usize; 2] = [0, 1];

type Diagram = Vec<(f64, f64)>;

fn fixture(name: &str, extension: &str) -> String {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "fixtures",
        "oracle",
        &format!("{name}.{extension}"),
    ]
    .iter()
    .collect();
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn data_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
}

fn parse_value(token: &str) -> f64 {
    if token == "inf" {
        f64::INFINITY
    } else {
        token
            .parse()
            .unwrap_or_else(|error| panic!("{token:?}: {error}"))
    }
}

fn points(name: &str) -> Vec<Vec<f64>> {
    data_lines(&fixture(name, "points"))
        .map(|line| line.split_whitespace().map(parse_value).collect())
        .collect()
}

fn expected(name: &str) -> [Diagram; 2] {
    let mut diagrams: [Diagram; 2] = Default::default();
    for line in data_lines(&fixture(name, "diagram")) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 3, "malformed diagram line {line:?}");
        let dimension: usize = fields[0].parse().unwrap();
        diagrams[dimension].push((parse_value(fields[1]), parse_value(fields[2])));
    }
    for diagram in &mut diagrams {
        diagram.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    }
    diagrams
}

fn computed(points: &[Vec<f64>]) -> [Diagram; 2] {
    let complex = rips(points, f64::INFINITY, true);
    let matrix = BoundaryMatrix::new(&complex.filtration()).unwrap();
    let mut diagrams: [Diagram; 2] = Default::default();
    for (birth, death) in reduction(&matrix).pairs() {
        let Some(diagram) = diagrams.get_mut(birth.dim()) else {
            // The 2-skeleton's own essential H2 classes are an artifact of
            // truncation; the fixtures stop at H1.
            continue;
        };
        let death = death.unwrap_or(f64::INFINITY);
        if death > birth.value() {
            diagram.push((birth.value(), death));
        }
    }
    for diagram in &mut diagrams {
        diagram.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    }
    diagrams
}

fn close(left: f64, right: f64) -> bool {
    if left.is_infinite() || right.is_infinite() {
        left == right
    } else {
        (left - right).abs() <= TOLERANCE
    }
}

fn check(name: &str) {
    let points = points(name);
    let ours = computed(&points);
    let theirs = expected(name);
    for dimension in DIMENSIONS {
        let (ours, theirs) = (&ours[dimension], &theirs[dimension]);
        assert_eq!(
            ours.len(),
            theirs.len(),
            "{name}: H{dimension} has {} pairs, GUDHI has {}\n  betti: {ours:?}\n  gudhi: {theirs:?}",
            ours.len(),
            theirs.len()
        );
        for (a, b) in ours.iter().zip(theirs) {
            assert!(
                close(a.0, b.0) && close(a.1, b.1),
                "{name}: H{dimension} pair {a:?} does not match GUDHI's {b:?}\n  betti: {ours:?}\n  gudhi: {theirs:?}"
            );
        }
    }
}

#[test]
fn square() {
    check("square");
    // The hand-checkable case also pins the exact values, not only agreement.
    let [h0, h1] = computed(&points("square"));
    assert_eq!(
        h0,
        vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, f64::INFINITY)]
    );
    assert_eq!(h1, vec![(1.0, 2f64.sqrt())]);
}

#[test]
fn noisy_circle() {
    check("noisy_circle");
}

#[test]
fn uniform_plane() {
    check("uniform_plane");
}

#[test]
fn uniform_cube() {
    check("uniform_cube");
}

#[test]
fn two_blobs() {
    check("two_blobs");
}
