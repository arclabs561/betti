//! The GF(2) boundary matrix of a filtration.

use std::collections::HashMap;

use crate::filtration::{Filtration, FiltrationError};
use crate::simplex::Simplex;

/// The boundary matrix of a filtration, in column order.
///
/// Columns are ordered by `(filtration value, dimension, vertices)`, so every
/// face column precedes its coface columns. Entry `(row, column)` is set when
/// simplex `row` is a codimension-one face of simplex `column`; over GF(2) no
/// multiplicity is needed.
#[derive(Clone, Debug, Default)]
pub struct BoundaryMatrix {
    simplices: Vec<Simplex>,
    values: Vec<f64>,
    columns: Vec<Vec<usize>>,
}

impl BoundaryMatrix {
    /// Order the filtration and build its boundary columns.
    ///
    /// Returns [`FiltrationError::MissingFace`] when a simplex lacks a face and
    /// [`FiltrationError::FaceAfterCoface`] when a face is valued after its
    /// coface. Both are construction errors, never panics.
    pub fn new(filtration: &Filtration) -> Result<Self, FiltrationError> {
        let entries = filtration.entries();
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by(|&left, &right| {
            let (left_simplex, left_value) = &entries[left];
            let (right_simplex, right_value) = &entries[right];
            left_value
                .total_cmp(right_value)
                .then_with(|| left_simplex.dim().cmp(&right_simplex.dim()))
                .then_with(|| left_simplex.cmp(right_simplex))
        });

        let mut position: HashMap<&Simplex, usize> = HashMap::with_capacity(order.len());
        for (column, &entry) in order.iter().enumerate() {
            let (simplex, _) = &entries[entry];
            if position.insert(simplex, column).is_some() {
                return Err(FiltrationError::DuplicateSimplex {
                    simplex: simplex.clone(),
                });
            }
        }

        let simplices: Vec<Simplex> = order
            .iter()
            .map(|&entry| entries[entry].0.clone())
            .collect();
        let values: Vec<f64> = order.iter().map(|&entry| entries[entry].1).collect();

        let mut columns = Vec::with_capacity(simplices.len());
        for (column, simplex) in simplices.iter().enumerate() {
            let mut boundary = Vec::with_capacity(simplex.dim());
            for face in simplex.faces() {
                let Some(&row) = position.get(&face) else {
                    return Err(FiltrationError::MissingFace {
                        simplex: simplex.clone(),
                        face,
                    });
                };
                if values[row] > values[column] {
                    return Err(FiltrationError::FaceAfterCoface {
                        face,
                        face_value: values[row],
                        coface: simplex.clone(),
                        coface_value: values[column],
                    });
                }
                debug_assert!(row < column, "a face column must precede its coface");
                boundary.push(row);
            }
            boundary.sort_unstable();
            columns.push(boundary);
        }

        Ok(Self {
            simplices,
            values,
            columns,
        })
    }

    /// The number of columns, one per simplex.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Whether the matrix has no columns.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// The simplex owning column `column`, in filtration order.
    pub fn simplex(&self, column: usize) -> &Simplex {
        &self.simplices[column]
    }

    /// Every simplex, in column order.
    pub fn simplices(&self) -> &[Simplex] {
        &self.simplices
    }

    /// The filtration value of a column.
    pub fn value(&self, column: usize) -> f64 {
        self.values[column]
    }

    /// Every filtration value, in column order.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The dimension of a column's simplex.
    pub fn dim(&self, column: usize) -> usize {
        self.simplices[column].dim()
    }

    /// The row indices of nonzero entries in a column, sorted ascending.
    pub fn column(&self, column: usize) -> &[usize] {
        &self.columns[column]
    }

    /// Every boundary column, in column order.
    pub fn columns(&self) -> &[Vec<usize>] {
        &self.columns
    }

    /// The largest dimension present, if any.
    pub fn max_dim(&self) -> Option<usize> {
        self.simplices.iter().map(Simplex::dim).max()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_are_ordered_by_value_dimension_and_vertices() {
        let mut filtration = Filtration::new();
        filtration.insert([0], 0.5).unwrap();
        filtration.insert([1], 0.5).unwrap();
        filtration.insert([2], 0.5).unwrap();
        filtration.insert([3], 0.0).unwrap();
        filtration.insert([0, 2], 1.0).unwrap();
        filtration.insert([0, 1], 1.0).unwrap();

        let matrix = BoundaryMatrix::new(&filtration).unwrap();
        let order: Vec<_> = (0..matrix.len())
            .map(|column| {
                (
                    matrix.simplex(column).vertices().to_vec(),
                    matrix.value(column),
                )
            })
            .collect();

        assert_eq!(
            order,
            vec![
                (vec![3], 0.0),
                (vec![0], 0.5),
                (vec![1], 0.5),
                (vec![2], 0.5),
                (vec![0, 1], 1.0),
                (vec![0, 2], 1.0),
            ]
        );
        assert_eq!(matrix.column(4), &[1, 2]);
        assert_eq!(matrix.column(5), &[1, 3]);
        assert!(matrix.column(0).is_empty());
        assert_eq!(matrix.max_dim(), Some(1));
    }
}
