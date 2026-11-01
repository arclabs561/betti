//! Filtrations: simplices tagged with a filtration value.

use std::fmt;

use crate::simplex::{Simplex, SimplexError};

/// A simplexwise filtration, built by inserting simplices with values.
///
/// Insertion order does not matter. The boundary matrix sorts simplices by
/// `(filtration value, dimension, vertices)`, so every face precedes its
/// coface whenever the values respect the filtration condition.
#[derive(Clone, Debug, Default)]
pub struct Filtration {
    entries: Vec<(Simplex, f64)>,
}

/// Why a filtration cannot be built.
#[derive(Clone, Debug, PartialEq)]
pub enum FiltrationError {
    /// A face is valued strictly after one of its cofaces.
    FaceAfterCoface {
        /// The offending face.
        face: Simplex,
        /// The face's filtration value.
        face_value: f64,
        /// A coface that appears earlier.
        coface: Simplex,
        /// The coface's filtration value.
        coface_value: f64,
    },
    /// A simplex is present without one of its codimension-one faces.
    MissingFace {
        /// The simplex whose boundary is incomplete.
        simplex: Simplex,
        /// The absent face.
        face: Simplex,
    },
    /// The same simplex was inserted twice.
    DuplicateSimplex {
        /// The repeated simplex.
        simplex: Simplex,
    },
    /// A filtration value is `NaN`.
    InvalidValue {
        /// The rejected value.
        value: f64,
    },
    /// The vertex list does not describe a simplex.
    InvalidSimplex(SimplexError),
}

impl fmt::Display for FiltrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FaceAfterCoface {
                face,
                face_value,
                coface,
                coface_value,
            } => write!(
                f,
                "face {face} has filtration value {face_value} but coface {coface} has {coface_value}"
            ),
            Self::MissingFace { simplex, face } => {
                write!(f, "simplex {simplex} is missing its face {face}")
            }
            Self::DuplicateSimplex { simplex } => {
                write!(f, "simplex {simplex} was inserted more than once")
            }
            Self::InvalidValue { value } => {
                write!(f, "filtration value {value} is not a number")
            }
            Self::InvalidSimplex(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FiltrationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidSimplex(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SimplexError> for FiltrationError {
    fn from(error: SimplexError) -> Self {
        Self::InvalidSimplex(error)
    }
}

impl Filtration {
    /// An empty filtration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a simplex with its filtration value.
    ///
    /// Rejects `NaN` values, repeated simplices, and any pairing where a face
    /// is valued after a coface (in either insertion order). The check scans
    /// the simplices already present, so building a filtration of `n` simplices
    /// costs `O(n^2)` face comparisons; that is deliberate for a small reducer.
    pub fn add(&mut self, simplex: Simplex, value: f64) -> Result<(), FiltrationError> {
        if value.is_nan() {
            return Err(FiltrationError::InvalidValue { value });
        }
        if self
            .entries
            .iter()
            .any(|(existing, _)| existing == &simplex)
        {
            return Err(FiltrationError::DuplicateSimplex { simplex });
        }

        let faces: Vec<Simplex> = simplex.faces().collect();
        for face in faces {
            if let Some((_, face_value)) = self.entries.iter().find(|(other, _)| *other == face) {
                if *face_value > value {
                    return Err(FiltrationError::FaceAfterCoface {
                        face,
                        face_value: *face_value,
                        coface: simplex,
                        coface_value: value,
                    });
                }
            }
        }

        for (existing, existing_value) in &self.entries {
            if existing.dim() > simplex.dim()
                && simplex.is_face_of(existing)
                && *existing_value < value
            {
                return Err(FiltrationError::FaceAfterCoface {
                    face: simplex,
                    face_value: value,
                    coface: existing.clone(),
                    coface_value: *existing_value,
                });
            }
        }

        self.entries.push((simplex, value));
        Ok(())
    }

    /// Build a simplex from vertices and insert it.
    ///
    /// ```
    /// use betti::Filtration;
    ///
    /// let mut filtration = Filtration::new();
    /// filtration.insert([0, 1], 1.0).unwrap();
    /// assert_eq!(filtration.len(), 1);
    /// ```
    pub fn insert(
        &mut self,
        vertices: impl Into<Vec<usize>>,
        value: f64,
    ) -> Result<(), FiltrationError> {
        self.add(Simplex::new(vertices)?, value)
    }

    /// The number of simplices.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the filtration has no simplices.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The largest dimension present, if any.
    pub fn max_dim(&self) -> Option<usize> {
        self.entries.iter().map(|(simplex, _)| simplex.dim()).max()
    }

    /// The filtration value of `simplex`, if present.
    pub fn value_of(&self, simplex: &Simplex) -> Option<f64> {
        self.entries
            .iter()
            .find(|(other, _)| other == simplex)
            .map(|(_, value)| *value)
    }

    /// Iterate over inserted simplices and values, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&Simplex, f64)> {
        self.entries
            .iter()
            .map(|(simplex, value)| (simplex, *value))
    }

    pub(crate) fn entries(&self) -> &[(Simplex, f64)] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_membership_are_reported() {
        let mut filtration = Filtration::new();
        assert!(filtration.is_empty());
        filtration.insert([0], 0.0).unwrap();
        filtration.insert([1], 0.5).unwrap();

        assert_eq!(filtration.len(), 2);
        assert_eq!(filtration.max_dim(), Some(0));
        assert_eq!(filtration.value_of(&Simplex::new([1]).unwrap()), Some(0.5));
        assert_eq!(filtration.iter().count(), 2);
    }

    #[test]
    fn nan_is_rejected() {
        let mut filtration = Filtration::new();
        let error = filtration.insert([0], f64::NAN).unwrap_err();
        assert!(matches!(error, FiltrationError::InvalidValue { .. }));
    }

    #[test]
    fn duplicate_simplices_are_rejected() {
        let mut filtration = Filtration::new();
        filtration.insert([0, 1], 0.0).unwrap();
        let error = filtration.insert([1, 0], 1.0).unwrap_err();
        assert!(matches!(error, FiltrationError::DuplicateSimplex { .. }));
    }

    #[test]
    fn errors_display_and_carry_the_source() {
        let error = FiltrationError::from(SimplexError::Empty);
        assert!(error.to_string().contains("at least one vertex"));
        assert!(std::error::Error::source(&error).is_some());
    }
}
