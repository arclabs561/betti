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

#![warn(missing_docs)]
#![warn(clippy::doc_markdown)]

mod boundary;
mod filtration;
mod reduction;
mod simplex;

pub use boundary::BoundaryMatrix;
pub use filtration::{Filtration, FiltrationError};
pub use reduction::{
    reduction, reduction_with, Birth, Death, Decomposition, Pair, ReductionOptions,
};
pub use simplex::{Simplex, SimplexError};
