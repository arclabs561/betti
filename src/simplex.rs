//! Simplices as sorted vertex sets.

use std::fmt;

/// A simplex: a non-empty, sorted set of vertex ids.
///
/// The dimension of a simplex is one less than its vertex count, so a vertex is
/// a 0-simplex, an edge a 1-simplex, and a triangle a 2-simplex.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Simplex {
    vertices: Vec<usize>,
}

/// Why a vertex list is not a simplex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimplexError {
    /// A simplex needs at least one vertex.
    Empty,
    /// A vertex id appears twice; a simplex is a set.
    DuplicateVertex {
        /// The repeated vertex id.
        vertex: usize,
    },
}

impl fmt::Display for SimplexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "a simplex needs at least one vertex"),
            Self::DuplicateVertex { vertex } => {
                write!(f, "simplex repeats vertex {vertex}")
            }
        }
    }
}

impl std::error::Error for SimplexError {}

impl Simplex {
    /// Build a simplex from any vertex sequence, sorting it into place.
    ///
    /// Returns [`SimplexError::Empty`] for no vertices and
    /// [`SimplexError::DuplicateVertex`] when an id repeats.
    ///
    /// ```
    /// use betti::Simplex;
    ///
    /// let edge = Simplex::new([2, 0]).unwrap();
    /// assert_eq!(edge.vertices(), &[0, 2]);
    /// assert_eq!(edge.dim(), 1);
    /// ```
    pub fn new(vertices: impl Into<Vec<usize>>) -> Result<Self, SimplexError> {
        let mut vertices = vertices.into();
        if vertices.is_empty() {
            return Err(SimplexError::Empty);
        }
        vertices.sort_unstable();
        for pair in vertices.windows(2) {
            if pair[0] == pair[1] {
                return Err(SimplexError::DuplicateVertex { vertex: pair[0] });
            }
        }
        Ok(Self { vertices })
    }

    /// The sorted vertex ids.
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Consume the simplex, returning its sorted vertex ids.
    pub fn into_vertices(self) -> Vec<usize> {
        self.vertices
    }

    /// The dimension, one less than the vertex count.
    pub fn dim(&self) -> usize {
        self.vertices.len() - 1
    }

    /// Whether `vertex` belongs to the simplex.
    pub fn contains(&self, vertex: usize) -> bool {
        self.vertices.binary_search(&vertex).is_ok()
    }

    /// Enumerate the codimension-one faces, each with one vertex removed.
    ///
    /// A vertex has no faces: its boundary is empty in the reduced chain
    /// complex used here, so the empty simplex is never produced.
    ///
    /// ```
    /// use betti::Simplex;
    ///
    /// let faces: Vec<_> = Simplex::new([0, 1, 2]).unwrap().faces().collect();
    /// assert_eq!(faces.len(), 3);
    /// assert_eq!(faces[0].vertices(), &[1, 2]);
    /// ```
    pub fn faces(&self) -> impl Iterator<Item = Simplex> + '_ {
        let vertices = self.vertices.as_slice();
        let len = vertices.len();
        (0..len).filter(move |_| len > 1).map(move |skip| {
            let mut face = Vec::with_capacity(len - 1);
            face.extend_from_slice(&vertices[..skip]);
            face.extend_from_slice(&vertices[skip + 1..]);
            Simplex { vertices: face }
        })
    }

    /// Whether every vertex of `self` also belongs to `coface`.
    ///
    /// A simplex is a face of itself under this test.
    pub fn is_face_of(&self, coface: &Simplex) -> bool {
        if self.vertices.len() > coface.vertices.len() {
            return false;
        }
        let mut available = coface.vertices.iter();
        self.vertices
            .iter()
            .all(|vertex| available.any(|other| other == vertex))
    }
}

impl fmt::Display for Simplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (index, vertex) in self.vertices.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{vertex}")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_sorts_vertices_and_computes_dimension() {
        let triangle = Simplex::new(vec![2, 0, 1]).unwrap();
        assert_eq!(triangle.vertices(), &[0, 1, 2]);
        assert_eq!(triangle.dim(), 2);
        assert!(triangle.contains(1));
        assert!(!triangle.contains(3));
    }

    #[test]
    fn empty_and_repeated_vertices_are_rejected() {
        assert_eq!(Simplex::new(Vec::<usize>::new()), Err(SimplexError::Empty));
        assert_eq!(
            Simplex::new([0, 1, 1]),
            Err(SimplexError::DuplicateVertex { vertex: 1 })
        );
    }

    #[test]
    fn faces_are_codimension_one_and_a_vertex_has_none() {
        let faces: Vec<_> = Simplex::new([0, 1, 2]).unwrap().faces().collect();
        assert_eq!(
            faces.iter().map(Simplex::vertices).collect::<Vec<_>>(),
            vec![&[1, 2][..], &[0, 2][..], &[0, 1][..]]
        );
        assert_eq!(Simplex::new([7]).unwrap().faces().count(), 0);
    }

    #[test]
    fn is_face_of_requires_a_subset() {
        let triangle = Simplex::new([0, 1, 2]).unwrap();
        assert!(Simplex::new([0, 2]).unwrap().is_face_of(&triangle));
        assert!(!Simplex::new([0, 3]).unwrap().is_face_of(&triangle));
        assert!(triangle.is_face_of(&triangle));
    }
}
