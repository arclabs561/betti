//! `H0Decomposition::cluster_labels` against its specification.
//!
//! The fixed cases check geometric expectations that hold by construction:
//! blobs separated by more than the threshold, which is larger than either
//! blob's diameter. The random sweep compares every labelling with an
//! independent reference, the `ToMATo` union-find run online with the
//! threshold (Chazal, Guibas, Oudot and Skraba, 2013): edges in increasing
//! order, and a merge only when the younger root has lived less than the
//! threshold. Both must induce the same partition, with
//! `count_persistent(threshold)` parts.

mod common;

use std::collections::HashMap;

use betti::{H0Error, H0Filtration};
use common::{random_one_dimensional, rips, XorShift64};

/// Two square blobs of `per_blob` points with side `side`, the second shifted
/// by `gap` along x. Points `0..per_blob` are the first blob.
fn two_blobs(per_blob: usize, side: f64, gap: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = XorShift64::new(seed);
    let mut unit = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let mut points = Vec::new();
    for offset in [0.0, gap] {
        for _ in 0..per_blob {
            points.push(vec![offset + side * unit(), side * unit()]);
        }
    }
    points
}

/// Single-linkage filtration of a point cloud: vertices at 0, every edge at
/// its length.
fn complete_graph(points: &[Vec<f64>]) -> H0Filtration {
    let (births, edges) = rips(points, f64::INFINITY, false).one_skeleton();
    filtration(&births, &edges)
}

fn filtration(births: &[f64], edges: &[(usize, usize, f64)]) -> H0Filtration {
    let mut filtration = H0Filtration::new();
    for &birth in births {
        filtration.add_vertex(birth).unwrap();
    }
    for &(left, right, death) in edges {
        filtration.add_edge(left, right, death).unwrap();
    }
    filtration
}

fn distinct(labels: &[usize]) -> usize {
    let mut seen = labels.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

#[test]
fn two_separated_blobs_are_two_clusters() {
    // Blob diameter is at most 0.2 * sqrt(2) < 0.3; the blobs are at least
    // 4.8 apart. Any threshold in between separates exactly the blobs.
    let points = two_blobs(15, 0.2, 5.0, 3);
    let components = complete_graph(&points).decompose();

    for threshold in [0.5, 1.0, 4.0] {
        let labels = components.cluster_labels(threshold).unwrap();
        assert_eq!(distinct(&labels), 2, "threshold {threshold}");
        assert_eq!(components.count_persistent(threshold), 2);
        // Vertex 0 is in the first blob and wins the birth tie, so that blob
        // is label 0.
        assert!(labels[..15].iter().all(|&label| label == 0));
        assert!(labels[15..].iter().all(|&label| label == 1));
    }
}

#[test]
fn a_threshold_above_every_persistence_is_one_cluster() {
    let points = two_blobs(15, 0.2, 5.0, 3);
    let components = complete_graph(&points).decompose();
    let longest = components
        .pairs()
        .iter()
        .filter_map(|pair| pair.persistence())
        .fold(0.0, f64::max);

    for threshold in [longest + 1.0, f64::INFINITY] {
        assert_eq!(components.cluster_labels(threshold).unwrap(), vec![0; 30]);
        assert_eq!(components.count_persistent(threshold), 1);
    }
}

#[test]
fn a_nonpositive_threshold_keeps_every_vertex_apart() {
    // Every persistence is at least 0, so nothing merges.
    let points = two_blobs(5, 0.2, 5.0, 9);
    let components = complete_graph(&points).decompose();
    for threshold in [0.0, -1.0, f64::NEG_INFINITY] {
        let labels = components.cluster_labels(threshold).unwrap();
        assert_eq!(distinct(&labels), 10);
        assert_eq!(components.count_persistent(threshold), 10);
    }
}

#[test]
fn nan_matches_no_class_like_count_persistent() {
    let points = two_blobs(5, 0.2, 5.0, 9);
    let components = complete_graph(&points).decompose();
    assert_eq!(components.count_persistent(f64::NAN), 0);
    assert!(matches!(
        components.cluster_labels(f64::NAN),
        Err(H0Error::InvalidValue { value }) if value.is_nan()
    ));
}

#[test]
fn an_empty_filtration_has_no_labels() {
    let components = H0Filtration::new().decompose();
    assert_eq!(components.cluster_labels(1.0).unwrap(), Vec::<usize>::new());
}

#[test]
fn disconnected_components_never_merge() {
    // Two edges, no path between them: two essential classes at any threshold.
    let components = filtration(&[0.0; 4], &[(0, 1, 1.0), (2, 3, 1.0)]).decompose();
    assert_eq!(
        components.cluster_labels(f64::INFINITY).unwrap(),
        vec![0, 0, 1, 1]
    );
}

/// The `ToMATo` union-find with the threshold applied online, written
/// independently of the library: edges sorted by `(value, left, right)`, the
/// elder of two roots decided by `(birth, vertex id)`.
fn tomato_reference(births: &[f64], edges: &[(usize, usize, f64)], threshold: f64) -> Vec<usize> {
    fn find(parent: &mut [usize], mut vertex: usize) -> usize {
        while parent[vertex] != vertex {
            vertex = parent[vertex];
        }
        vertex
    }
    let older = |a: usize, b: usize| births[a].total_cmp(&births[b]).then(a.cmp(&b)).is_lt();

    let mut sorted = edges.to_vec();
    sorted.sort_by(|a, b| a.2.total_cmp(&b.2).then((a.0, a.1).cmp(&(b.0, b.1))));
    // Each root is the elder vertex of its component.
    let mut parent: Vec<usize> = (0..births.len()).collect();
    for (left, right, value) in sorted {
        let (a, b) = (find(&mut parent, left), find(&mut parent, right));
        if a == b {
            continue;
        }
        let (elder, younger) = if older(a, b) { (a, b) } else { (b, a) };
        if value - births[younger] < threshold {
            parent[younger] = elder;
        }
    }
    (0..births.len())
        .map(|vertex| find(&mut parent, vertex))
        .collect()
}

/// Whether two labellings induce the same partition of the vertices.
fn same_partition(left: &[usize], right: &[usize]) -> bool {
    let mut forward = HashMap::new();
    let mut backward = HashMap::new();
    left.iter()
        .zip(right)
        .all(|(&a, &b)| *forward.entry(a).or_insert(b) == b && *backward.entry(b).or_insert(a) == a)
}

#[test]
fn random_filtrations_match_the_online_tomato_rule() {
    let thresholds = [0.0, 0.1, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5, f64::INFINITY];
    for seed in 1..=300 {
        let mut rng = XorShift64::new(seed);
        let (births, edges) = random_one_dimensional(&mut rng).one_skeleton();
        let components = filtration(&births, &edges).decompose();
        for threshold in thresholds {
            let labels = components.cluster_labels(threshold).unwrap();
            let expected = tomato_reference(&births, &edges, threshold);
            assert_eq!(
                distinct(&labels),
                components.count_persistent(threshold),
                "seed {seed}, threshold {threshold}"
            );
            assert!(
                same_partition(&labels, &expected),
                "seed {seed}, threshold {threshold}: {labels:?} vs reference {expected:?}"
            );
        }
    }
}
