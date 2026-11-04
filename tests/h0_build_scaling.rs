//! Building a one-dimensional filtration must stay near-linear in the edge count.
//!
//! `add_edge` rejects duplicates, so each call has to find out whether the edge
//! is already present. A scan of the edge list makes building E edges cost
//! O(E^2); the bound below separates that from a hashed lookup by a wide margin.

use std::time::{Duration, Instant};

use betti::{H0Error, H0Filtration};

const EDGES: usize = 100_000;

#[test]
fn building_a_ring_is_not_quadratic() {
    let start = Instant::now();
    let mut filtration = H0Filtration::new();
    for i in 0..EDGES {
        filtration.add_vertex(i as f64).unwrap();
    }
    for i in 0..EDGES {
        let next = (i + 1) % EDGES;
        filtration
            .add_edge(i, next, (i.max(next) + 1) as f64)
            .unwrap();
    }
    let elapsed = start.elapsed();
    eprintln!("built {EDGES} edges in {elapsed:?}");

    assert_eq!(filtration.edge_count(), EDGES);
    // A ring has one essential component. Checking the result keeps the build
    // from being optimized into nothing and ties the timing to a real filtration.
    assert_eq!(filtration.decompose().essential().count(), 1);
    // A hashed lookup builds this in well under a second in a debug test
    // build; the quadratic scan needs more than 20 seconds.
    assert!(
        elapsed < Duration::from_secs(10),
        "building {EDGES} edges took {elapsed:?}"
    );
}

#[test]
fn a_duplicate_is_still_rejected_in_either_endpoint_order() {
    let mut filtration = H0Filtration::new();
    for _ in 0..3 {
        filtration.add_vertex(0.0).unwrap();
    }
    filtration.add_edge(0, 1, 1.0).unwrap();
    filtration.add_edge(1, 2, 1.0).unwrap();
    assert_eq!(
        filtration.add_edge(1, 0, 2.0),
        Err(H0Error::DuplicateEdge { edge: (0, 1) })
    );
    assert_eq!(filtration.edge_count(), 2);
}
