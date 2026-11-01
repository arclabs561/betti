//! Hand-checkable filtrations: point, edges, cycles, and a sphere.

use betti::{
    reduction, reduction_with, BoundaryMatrix, Decomposition, Filtration, FiltrationError, Pair,
    ReductionOptions, Simplex,
};

/// Build a boundary matrix from `(vertices, filtration value)` pairs.
fn matrix(simplices: &[(Vec<usize>, f64)]) -> BoundaryMatrix {
    let mut filtration = Filtration::new();
    for (vertices, value) in simplices {
        filtration
            .insert(vertices.clone(), *value)
            .expect("valid filtration");
    }
    BoundaryMatrix::new(&filtration).expect("valid boundary matrix")
}

/// A single vertex.
fn point() -> BoundaryMatrix {
    matrix(&[(vec![0], 0.0)])
}

/// Two disjoint edges: four vertices, two edges.
fn two_edges() -> BoundaryMatrix {
    matrix(&[
        (vec![0], 0.0),
        (vec![1], 0.0),
        (vec![2], 0.0),
        (vec![3], 0.0),
        (vec![0, 1], 1.0),
        (vec![2, 3], 1.0),
    ])
}

/// The triangle boundary: three vertices and three edges, no 2-simplex.
fn triangle_boundary() -> BoundaryMatrix {
    matrix(&[
        (vec![0], 0.0),
        (vec![1], 0.0),
        (vec![2], 0.0),
        (vec![0, 1], 1.0),
        (vec![0, 2], 1.0),
        (vec![1, 2], 1.0),
    ])
}

/// The filled triangle: the boundary plus its 2-simplex.
fn filled_triangle() -> BoundaryMatrix {
    matrix(&[
        (vec![0], 0.0),
        (vec![1], 0.0),
        (vec![2], 0.0),
        (vec![0, 1], 1.0),
        (vec![0, 2], 1.0),
        (vec![1, 2], 1.0),
        (vec![0, 1, 2], 2.0),
    ])
}

/// The boundary of a tetrahedron: four vertices, six edges, four triangles.
fn sphere() -> BoundaryMatrix {
    matrix(&[
        (vec![0], 0.0),
        (vec![1], 0.0),
        (vec![2], 0.0),
        (vec![3], 0.0),
        (vec![0, 1], 1.0),
        (vec![0, 2], 1.0),
        (vec![0, 3], 1.0),
        (vec![1, 2], 1.0),
        (vec![1, 3], 1.0),
        (vec![2, 3], 1.0),
        (vec![0, 1, 2], 2.0),
        (vec![0, 1, 3], 2.0),
        (vec![0, 2, 3], 2.0),
        (vec![1, 2, 3], 2.0),
    ])
}

/// Pair signatures that ignore discovery order but keep every value.
fn signature(decomposition: &Decomposition) -> Vec<Pair> {
    let mut pairs = decomposition.pairs();
    pairs.sort_by(|left, right| {
        left.0
            .simplex()
            .vertices()
            .cmp(right.0.simplex().vertices())
            .then_with(|| left.1.partial_cmp(&right.1).expect("no NaN"))
    });
    pairs
}

#[test]
fn single_vertex_has_one_component() {
    let decomposition = reduction(&point());
    assert_eq!(decomposition.betti_numbers(0), vec![1]);
    assert_eq!(decomposition.betti_numbers(2), vec![1, 0, 0]);
    assert!(decomposition.pairs()[0].1.is_none());
}

#[test]
fn two_disconnected_edges_have_two_components() {
    let decomposition = reduction(&two_edges());
    assert_eq!(decomposition.betti_numbers(1), vec![2, 0]);

    let pairs = decomposition.pairs();
    let finite = pairs.iter().filter(|(_, death)| death.is_some()).count();
    let essential = pairs.iter().filter(|(_, death)| death.is_none()).count();
    assert_eq!(finite, 2, "each edge merges one component");
    assert_eq!(essential, 2, "one surviving component per edge");
    assert!(pairs
        .iter()
        .all(|(birth, _)| birth.dim() == 0 && birth.value() == 0.0));
}

#[test]
fn triangle_boundary_has_one_loop() {
    let decomposition = reduction(&triangle_boundary());
    assert_eq!(decomposition.betti_numbers(2), vec![1, 1, 0]);

    let loop_class = decomposition
        .essential()
        .into_iter()
        .find(|birth| birth.dim() == 1)
        .expect("a 1-cycle survives");
    assert_eq!(loop_class.simplex().vertices(), &[1, 2]);
    assert_eq!(loop_class.value(), 1.0);
}

#[test]
fn filled_triangle_kills_the_loop() {
    let decomposition = reduction(&filled_triangle());
    assert_eq!(decomposition.betti_numbers(2), vec![1, 0, 0]);

    let death = decomposition
        .pairs()
        .into_iter()
        .find(|(birth, _)| birth.dim() == 1)
        .and_then(|(_, death)| death)
        .expect("the 1-simplex is paired");
    assert_eq!(death, 2.0);
}

#[test]
fn tetrahedron_boundary_is_a_sphere() {
    let decomposition = reduction(&sphere());
    assert_eq!(decomposition.betti_numbers(2), vec![1, 0, 1]);

    let essential: Vec<_> = decomposition
        .essential()
        .into_iter()
        .map(|birth| (birth.dim(), birth.simplex().vertices().to_vec()))
        .collect();
    assert_eq!(
        essential,
        vec![(0, vec![0]), (2, vec![1, 2, 3])],
        "vertex 0 and the last triangle are unpaired"
    );
}

#[test]
fn clearing_matches_plain_reduction_on_the_sphere() {
    let sphere = sphere();
    let plain = reduction(&sphere);
    let cleared = reduction_with(&sphere, ReductionOptions::clearing());

    assert_eq!(signature(&plain), signature(&cleared));
    assert_eq!(plain.betti_numbers(2), cleared.betti_numbers(2));
    assert_eq!(cleared.betti_numbers(2), vec![1, 0, 1]);
}

#[test]
fn face_valued_after_its_coface_is_a_typed_error() {
    let mut filtration = Filtration::new();
    for vertex in 0..3 {
        filtration.insert([vertex], 0.0).unwrap();
    }
    filtration.insert([0, 2], 0.0).unwrap();
    filtration.insert([1, 2], 0.0).unwrap();
    filtration.insert([0, 1, 2], 0.0).unwrap();

    // The missing edge arrives later, valued after the triangle that contains it.
    let error = filtration.insert([0, 1], 1.0).unwrap_err();
    match error {
        FiltrationError::FaceAfterCoface {
            face,
            face_value,
            coface,
            coface_value,
        } => {
            assert_eq!(face.vertices(), &[0, 1]);
            assert_eq!(face_value, 1.0);
            assert_eq!(coface.vertices(), &[0, 1, 2]);
            assert_eq!(coface_value, 0.0);
        }
        other => panic!("expected FaceAfterCoface, got {other}"),
    }
}

#[test]
fn boundary_matrix_rejects_a_simplex_without_its_faces() {
    let mut filtration = Filtration::new();
    filtration.insert([0, 1], 0.0).unwrap();

    let error = BoundaryMatrix::new(&filtration).unwrap_err();
    assert!(matches!(error, FiltrationError::MissingFace { .. }));
}

#[test]
fn equal_values_still_order_faces_first() {
    // Every simplex shares one value, so only dimension can break the tie.
    let mut filtration = Filtration::new();
    for vertices in [
        vec![0, 1, 2],
        vec![0, 1],
        vec![1, 2],
        vec![0, 2],
        vec![2],
        vec![0],
        vec![1],
    ] {
        filtration.insert(vertices, 0.0).unwrap();
    }

    let matrix = BoundaryMatrix::new(&filtration).unwrap();
    let dims: Vec<_> = matrix.simplices().iter().map(Simplex::dim).collect();
    assert_eq!(dims, vec![0, 0, 0, 1, 1, 1, 2]);
    assert_eq!(reduction(&matrix).betti_numbers(2), vec![1, 0, 0]);
}

#[test]
fn empty_filtration_has_no_classes() {
    let matrix = BoundaryMatrix::new(&Filtration::new()).unwrap();
    let decomposition = reduction(&matrix);
    assert!(decomposition.is_empty());
    assert_eq!(decomposition.betti_numbers(1), vec![0, 0]);
}
