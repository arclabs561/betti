//! Persistent homology over GF(2).
//!
//! A filtration enters; boundary columns are reduced left to right; the result
//! is a list of persistence pairs and Betti numbers. This is the reducer, not a
//! topology suite: a sorted vertex set, a filtration, a boundary matrix, and
//! one reduction algorithm.
//!
//! ```
//! use betti::{reduction, BoundaryMatrix, Filtration};
//!
//! // The boundary of a triangle: a loop, no disk.
//! let mut filtration = Filtration::new();
//! for vertex in 0..3 {
//!     filtration.insert([vertex], 0.0)?;
//! }
//! for edge in [[0, 1], [0, 2], [1, 2]] {
//!     filtration.insert(edge, 1.0)?;
//! }
//!
//! let matrix = BoundaryMatrix::new(&filtration)?;
//! let decomposition = reduction(&matrix);
//!
//! assert_eq!(decomposition.betti_numbers(1), vec![1, 1]);
//! // Two vertex-edge pairs close the H0 classes that merge, and the surviving
//! // H0 class and the loop are essential: four pairs, two of them infinite.
//! assert_eq!(decomposition.pairs().len(), 4);
//! assert_eq!(decomposition.essential().len(), 2);
//! # Ok::<(), betti::FiltrationError>(())
//! ```
//!
//! # Guarantees and limits
//!
//! - Columns are ordered by `(filtration value, dimension, vertices)`, so each
//!   face precedes its coface, and reduction over GF(2) is exact integer
//!   arithmetic.
//! - A filtration that values a face after a coface, or omits a face of a
//!   simplex, is a typed [`FiltrationError`], never a panic.
//! - [`ReductionOptions::clear`] enables the clearing (twist) optimization. It
//!   skips work, and is required to return exactly the pairs and Betti numbers
//!   of the plain reduction.
//! - The cohomology variant and representative cycles are not implemented yet;
//!   only the homology reduction is.
//!
//! # Cluster counts
//!
//! When only connected components matter, [`H0Filtration`] returns the
//! dimension-zero pairs by union-find and skips the matrix entirely:
//!
//! ```
//! use betti::H0Filtration;
//!
//! let mut filtration = H0Filtration::new();
//! for _ in 0..3 {
//!     filtration.add_vertex(0.0)?;
//! }
//! filtration.add_edge(0, 1, 1.0)?;
//!
//! let components = filtration.decompose();
//! assert_eq!(components.count_components_above(2.0), 2);
//! assert_eq!(components.count_components_above(0.5), 3);
//! # Ok::<(), betti::H0Error>(())
//! ```

#![warn(missing_docs)]
#![warn(clippy::doc_markdown)]

mod boundary;
mod filtration;
mod h0;
mod reduction;
mod simplex;

pub use boundary::BoundaryMatrix;
pub use filtration::{Filtration, FiltrationError};
pub use h0::{H0Decomposition, H0Error, H0Filtration, H0Pair};
pub use reduction::{
    reduction, reduction_with, Birth, Death, Decomposition, Pair, ReductionOptions,
};
pub use simplex::{Simplex, SimplexError};
