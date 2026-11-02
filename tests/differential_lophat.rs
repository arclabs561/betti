//! Differential verification against `lophat`.
//!
//! Every random complex is reduced four ways -- `betti` with and without
//! clearing, and `lophat` serial plus its lock-free algorithm with clearing on
//! and off -- and the pairings must be the same multiset. `betti` feeds
//! `lophat` the columns in its own `(filtration value, dimension, vertices)`
//! order, so both sides see one filtration and one column ordering.

mod common;

use std::collections::BTreeMap;

use betti::{
    reduction, reduction_with, BoundaryMatrix, Decomposition, Filtration, ReductionOptions,
};
use common::{is_face, random_complex, Complex, XorShift64};
use lophat::{
    algorithms::{
        Decomposition as LophatDecomposition, DecompositionAlgo, LockFreeAlgorithm, SerialAlgorithm,
    },
    columns::VecColumn,
    options::LoPhatOptions,
    utils::{ExtendedUsize, PersistenceDiagram},
};

/// Seeds reduced by the random sweep: `1..=SEEDS`.
const SEEDS: u64 = 200;

/// Attempts the shrinker may spend on a minimal counterexample.
const SHRINK_BUDGET: usize = 300;

/// A pairing in a shape both crates can be normalised into.
///
/// The persistence pairing of a filtration with a fixed column order is
/// unique, and each simplex is the birth of at most one class, so a map from
/// birth simplex to death value plus the essential births describes it
/// completely.
#[derive(Debug, PartialEq)]
struct Normalized {
    finite: BTreeMap<Vec<usize>, f64>,
    essential: Vec<Vec<usize>>,
}

impl Normalized {
    fn from_betti(decomposition: &Decomposition) -> Self {
        let mut finite = BTreeMap::new();
        let mut essential = Vec::new();
        for (birth, death) in decomposition.pairs() {
            let vertices = birth.simplex().vertices().to_vec();
            match death {
                Some(value) => {
                    assert!(
                        finite.insert(vertices.clone(), value).is_none(),
                        "betti paired birth {vertices:?} twice"
                    );
                }
                None => essential.push(vertices),
            }
        }
        essential.sort();
        Self { finite, essential }
    }

    fn from_lophat(diagram: &PersistenceDiagram, matrix: &BoundaryMatrix) -> Self {
        let mut finite = BTreeMap::new();
        let mut essential = Vec::new();
        for (&birth, &death) in diagram.iter() {
            let vertices = matrix.simplex(birth).vertices().to_vec();
            match death {
                ExtendedUsize::Finite(column) => {
                    assert!(
                        finite
                            .insert(vertices.clone(), matrix.value(column))
                            .is_none(),
                        "lophat paired birth {vertices:?} twice"
                    );
                }
                ExtendedUsize::Infinity => essential.push(vertices),
            }
        }
        essential.sort();
        Self { finite, essential }
    }
}

/// The boundary columns of a matrix, in `betti`'s column order, for `lophat`.
fn columns(matrix: &BoundaryMatrix) -> Vec<VecColumn> {
    (0..matrix.len())
        .map(|column| VecColumn::from((matrix.dim(column), matrix.column(column).to_vec())))
        .collect()
}

fn serial_diagram(matrix: &BoundaryMatrix) -> PersistenceDiagram {
    SerialAlgorithm::<VecColumn>::init(None)
        .add_cols(columns(matrix).into_iter())
        .decompose()
        .diagram()
}

fn lock_free_diagram(matrix: &BoundaryMatrix, clearing: bool) -> PersistenceDiagram {
    let options = LoPhatOptions {
        clearing,
        ..Default::default()
    };
    LockFreeAlgorithm::init(Some(options))
        .add_cols(columns(matrix).into_iter())
        .decompose()
        .diagram()
}

/// Compare every reducer configuration on one complex.
fn compare(complex: &Complex) -> Result<(), String> {
    let filtration = complex.filtration();
    let matrix = BoundaryMatrix::new(&filtration)
        .map_err(|error| format!("betti rejected a generated filtration: {error}"))?;

    let plain = Normalized::from_betti(&reduction(&matrix));
    let cleared = Normalized::from_betti(&reduction_with(&matrix, ReductionOptions::clearing()));
    if plain != cleared {
        return Err(format!(
            "betti clearing disagrees with plain reduction\n  plain: {plain:?}\n  clearing: {cleared:?}"
        ));
    }

    let serial = Normalized::from_lophat(&serial_diagram(&matrix), &matrix);
    let without_clearing = Normalized::from_lophat(&lock_free_diagram(&matrix, false), &matrix);
    let with_clearing = Normalized::from_lophat(&lock_free_diagram(&matrix, true), &matrix);

    for (label, other) in [
        ("lophat serial", &serial),
        ("lophat lock-free without clearing", &without_clearing),
        ("lophat lock-free with clearing", &with_clearing),
    ] {
        if &plain != other {
            return Err(format!(
                "betti disagrees with {label}\n  betti:  {plain:?}\n  lophat: {other:?}"
            ));
        }
    }
    Ok(())
}

/// Greedily drop simplices, and every simplex above them, while the mismatch
/// survives. Only runs on failure, terminates because the empty complex
/// agrees, and stops after [`SHRINK_BUDGET`] attempts so a report stays fast.
fn shrink(complex: &Complex) -> Complex {
    let mut current = complex.clone();
    let mut budget = SHRINK_BUDGET;
    loop {
        let mut shrunk = false;
        for index in 0..current.simplices.len() {
            if budget == 0 {
                return current;
            }
            budget -= 1;
            let removed = current.simplices[index].0.clone();
            let candidate = Complex {
                simplices: current
                    .simplices
                    .iter()
                    .enumerate()
                    .filter(|&(other, (vertices, _))| {
                        other == index || !is_face(&removed, vertices)
                    })
                    .map(|(_, entry)| entry.clone())
                    .collect(),
            };
            if compare(&candidate).is_err() {
                current = candidate;
                shrunk = true;
                break;
            }
        }
        if !shrunk {
            return current;
        }
    }
}

#[test]
fn random_complexes_agree_with_lophat() {
    for seed in 1..=SEEDS {
        let mut rng = XorShift64::new(seed);
        let complex = random_complex(&mut rng);
        if let Err(discrepancy) = compare(&complex) {
            let minimal = shrink(&complex);
            panic!(
                "seed {seed}: {discrepancy}\nminimal counterexample ({} simplices):\n  {}",
                minimal.simplices.len(),
                minimal.describe()
            );
        }
    }
}

#[test]
fn empty_input_agrees() {
    let complex = Complex::default();
    compare(&complex).expect("the empty filtration has no pairs on either side");

    let matrix = BoundaryMatrix::new(&Filtration::new()).unwrap();
    assert!(reduction(&matrix).is_empty());
    assert!(serial_diagram(&matrix).is_empty());
}

#[test]
fn single_vertex_agrees() {
    let complex = Complex {
        simplices: vec![(vec![0], 0.5)],
    };
    compare(&complex).expect("one vertex is one essential class");

    let matrix = BoundaryMatrix::new(&complex.filtration()).unwrap();
    let decomposition = reduction(&matrix);
    assert_eq!(decomposition.essential().len(), 1);
    assert_eq!(decomposition.betti_numbers(0), vec![1]);

    let diagram = serial_diagram(&matrix);
    assert_eq!(
        diagram,
        [(0, ExtendedUsize::Infinity)].into_iter().collect()
    );
}

#[test]
fn duplicate_simplices_are_rejected_before_any_reduction() {
    // A duplicated simplex is not a filtration, so neither reducer ever sees a
    // repeated column; the matrix path refuses it as a typed error.
    let mut filtration = Filtration::new();
    filtration.insert([0, 1], 1.0).unwrap();
    let duplicate = filtration.insert([1, 0], 2.0).unwrap_err();
    assert!(matches!(
        duplicate,
        betti::FiltrationError::DuplicateSimplex { .. }
    ));
    assert_eq!(filtration.len(), 1);
}

#[test]
fn face_with_an_equal_filtration_value_agrees() {
    // The edge is valued exactly like its vertices, so only the dimension tie
    // break keeps the face first.
    let complex = Complex {
        simplices: vec![(vec![0], 0.0), (vec![1], 0.0), (vec![0, 1], 0.0)],
    };
    compare(&complex).expect("equal values still reduce face-first");

    let matrix = BoundaryMatrix::new(&complex.filtration()).unwrap();
    let pairs = reduction(&matrix).pairs();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0.simplex().vertices(), &[0]);
    assert_eq!(pairs[0].1, None);
    assert_eq!(pairs[1].0.simplex().vertices(), &[1]);
    assert_eq!(pairs[1].1, Some(0.0));
}

#[test]
fn complexes_with_only_essential_classes_agree() {
    let complex = Complex {
        simplices: (0..5)
            .map(|vertex| (vec![vertex], 0.25 * vertex as f64))
            .collect(),
    };
    compare(&complex).expect("five isolated vertices are five essential classes");

    let matrix = BoundaryMatrix::new(&complex.filtration()).unwrap();
    let decomposition = reduction(&matrix);
    assert_eq!(decomposition.essential().len(), 5);
    assert!(decomposition
        .pairs()
        .iter()
        .all(|(_, death)| death.is_none()));

    let mut normalized = Normalized::from_betti(&decomposition);
    normalized.essential.sort();
    assert_eq!(
        normalized.essential,
        vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
}

#[test]
fn hand_checkable_fixtures_agree() {
    // The tetrahedron boundary: one essential component and one essential loop.
    let complex = Complex {
        simplices: vec![
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
        ],
    };
    compare(&complex).expect("the sphere reduces identically on both sides");

    let matrix = BoundaryMatrix::new(&complex.filtration()).unwrap();
    assert_eq!(reduction(&matrix).betti_numbers(2), vec![1, 0, 1]);
}
