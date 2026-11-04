//! Dimension-zero persistence by filtration-aware union-find.
//!
//! A filtration that stops at dimension one -- vertices with birth times and
//! edges with death times -- needs no boundary matrix to describe its
//! components. Each edge that joins two components kills the younger of them,
//! and the elder rule decides which class survives, so a union-find over the
//! edges in death order returns exactly the pairs that
//! [`reduction`](crate::reduction) produces on the same filtration, in
//! `O(n log n)` time instead of matrix reduction.
//!
//! This is the path for a cluster count, where only connected components
//! matter and a matrix reducer is wasted work. Two counts are available:
//! [`H0Decomposition::count_components_above`] gives the components alive at
//! one filtration value (a level set of the merge tree), and
//! [`H0Decomposition::count_persistent`] gives the classes that persist at
//! least a threshold, which is the count `ToMATo` uses for a prominence
//! threshold.
//!
//! ```
//! use betti::H0Filtration;
//!
//! let mut filtration = H0Filtration::new();
//! for _ in 0..4 {
//!     filtration.add_vertex(0.0).unwrap();
//! }
//! filtration.add_edge(0, 1, 1.0).unwrap();
//! filtration.add_edge(2, 3, 2.0).unwrap();
//!
//! let components = filtration.decompose();
//! // A class is alive from its birth up to, but not including, its death.
//! assert_eq!(components.count_components_above(0.5), 4);
//! assert_eq!(components.count_components_above(1.5), 3);
//! assert_eq!(components.count_components_above(2.0), 2);
//! ```

use std::collections::HashSet;
use std::fmt;

/// One dimension-zero class: a component born at a vertex and, when an older
/// component absorbs it, killed by an edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct H0Pair {
    birth_vertex: usize,
    birth: f64,
    death: Option<f64>,
    death_edge: Option<(usize, usize)>,
}

impl H0Pair {
    /// The vertex that created the class.
    pub fn birth_vertex(&self) -> usize {
        self.birth_vertex
    }

    /// The filtration value at which the class appeared.
    pub fn birth(&self) -> f64 {
        self.birth
    }

    /// The filtration value at which the class merged into an older one, or
    /// [`None`] for a class that survives the end of the filtration.
    pub fn death(&self) -> Option<f64> {
        self.death
    }

    /// The edge that ended the class, as a sorted vertex pair.
    pub fn death_edge(&self) -> Option<(usize, usize)> {
        self.death_edge
    }

    /// The persistence of the class, or [`None`] for an essential class.
    ///
    /// This is the significance a merge-tree based cluster count compares
    /// against its threshold.
    pub fn persistence(&self) -> Option<f64> {
        self.death.map(|death| death - self.birth)
    }
}

/// A one-dimensional filtration: vertices with birth times and edges with
/// death times.
///
/// Vertex ids are the insertion order of [`Self::add_vertex`], so they are
/// dense. An edge dies no earlier than either of its vertices is born, which
/// is the filtration condition for dimension one.
#[derive(Clone, Debug, Default)]
pub struct H0Filtration {
    vertices: Vec<f64>,
    edges: Vec<(usize, usize, f64)>,
    // Sorted endpoint pairs already in `edges`, so a duplicate is found in
    // constant time instead of by scanning every earlier edge.
    seen: HashSet<(usize, usize)>,
}

/// Why a one-dimensional filtration cannot be built.
#[derive(Clone, Debug, PartialEq)]
pub enum H0Error {
    /// A filtration value is `NaN`, so it has no position in the order.
    InvalidValue {
        /// The rejected value.
        value: f64,
    },
    /// An edge names a vertex that was never added.
    UnknownVertex {
        /// The offending vertex id.
        vertex: usize,
        /// The edge that named it.
        edge: (usize, usize),
    },
    /// An edge joins a vertex to itself, which is not a simplex boundary.
    SelfLoop {
        /// The repeated vertex id.
        vertex: usize,
    },
    /// The same undirected edge was added twice.
    DuplicateEdge {
        /// The repeated edge, as a sorted vertex pair.
        edge: (usize, usize),
    },
    /// An edge dies strictly before one of its vertices is born.
    BirthAfterDeath {
        /// The vertex born after the edge dies.
        vertex: usize,
        /// The vertex's birth value.
        birth: f64,
        /// The edge's death value.
        death: f64,
    },
}

impl fmt::Display for H0Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue { value } => {
                write!(f, "filtration value {value} is not a number")
            }
            Self::UnknownVertex { vertex, edge } => write!(
                f,
                "edge [{}, {}] names vertex {vertex}, which was never added",
                edge.0, edge.1
            ),
            Self::SelfLoop { vertex } => {
                write!(f, "edge joins vertex {vertex} to itself")
            }
            Self::DuplicateEdge { edge } => {
                write!(f, "edge [{}, {}] was added more than once", edge.0, edge.1)
            }
            Self::BirthAfterDeath {
                vertex,
                birth,
                death,
            } => write!(
                f,
                "vertex {vertex} is born at {birth} but its edge dies at {death}"
            ),
        }
    }
}

impl std::error::Error for H0Error {}

impl H0Filtration {
    /// An empty filtration: no vertices, no edges.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a vertex born at `birth` and return its id.
    pub fn add_vertex(&mut self, birth: f64) -> Result<usize, H0Error> {
        if birth.is_nan() {
            return Err(H0Error::InvalidValue { value: birth });
        }
        let id = self.vertices.len();
        self.vertices.push(birth);
        Ok(id)
    }

    /// Add an edge between two existing vertices that dies at `death`.
    ///
    /// The endpoints are stored in sorted order, so `add_edge(1, 0, _)` and
    /// `add_edge(0, 1, _)` are the same edge.
    pub fn add_edge(&mut self, left: usize, right: usize, death: f64) -> Result<(), H0Error> {
        if death.is_nan() {
            return Err(H0Error::InvalidValue { value: death });
        }
        let edge = (left.min(right), left.max(right));
        for vertex in [left, right] {
            if vertex >= self.vertices.len() {
                return Err(H0Error::UnknownVertex { vertex, edge });
            }
        }
        if left == right {
            return Err(H0Error::SelfLoop { vertex: left });
        }
        if self.seen.contains(&edge) {
            return Err(H0Error::DuplicateEdge { edge });
        }
        for vertex in [left, right] {
            let birth = self.vertices[vertex];
            if death < birth {
                return Err(H0Error::BirthAfterDeath {
                    vertex,
                    birth,
                    death,
                });
            }
        }
        self.edges.push((edge.0, edge.1, death));
        self.seen.insert(edge);
        Ok(())
    }

    /// The number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// The number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Whether the filtration has no vertices.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Reduce the filtration to its dimension-zero persistence pairs.
    ///
    /// Edges are processed in increasing death order; an edge whose endpoints
    /// already lie in one component is a 1-cycle and is ignored, and any other
    /// edge kills the younger of the two components it joins. The elder rule
    /// breaks equal birth values by vertex id, matching the
    /// `(filtration value, dimension, vertices)` column order of the matrix
    /// reducer, so both paths return the same multiset of pairs.
    pub fn decompose(&self) -> H0Decomposition {
        let count = self.vertices.len();

        // The matrix reducer breaks ties between equal values by vertices, so
        // "older" means earlier in this order.
        let mut birth_rank = vec![0; count];
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&left, &right| {
            self.vertices[left]
                .total_cmp(&self.vertices[right])
                .then_with(|| left.cmp(&right))
        });
        for (rank, &vertex) in order.iter().enumerate() {
            birth_rank[vertex] = rank;
        }

        let mut edges: Vec<(f64, usize, usize)> = self
            .edges
            .iter()
            .map(|&(left, right, death)| (death, left, right))
            .collect();
        edges.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| (left.1, left.2).cmp(&(right.1, right.2)))
        });

        let mut forest = Forest::new(count);
        let mut pairs = Vec::with_capacity(count);
        for (death, left, right) in edges {
            let left_root = forest.find(left);
            let right_root = forest.find(right);
            if left_root == right_root {
                continue;
            }
            let left_birth = forest.elder(left_root);
            let right_birth = forest.elder(right_root);
            let (survivor, absorbed) = if birth_rank[left_birth] <= birth_rank[right_birth] {
                (left_root, right_root)
            } else {
                (right_root, left_root)
            };
            let absorbed_vertex = forest.elder(absorbed);
            pairs.push(H0Pair {
                birth_vertex: absorbed_vertex,
                birth: self.vertices[absorbed_vertex],
                death: Some(death),
                death_edge: Some((left, right)),
            });
            forest.merge(survivor, absorbed);
        }

        for vertex in 0..count {
            if forest.find(vertex) == vertex {
                let elder = forest.elder(vertex);
                pairs.push(H0Pair {
                    birth_vertex: elder,
                    birth: self.vertices[elder],
                    death: None,
                    death_edge: None,
                });
            }
        }

        pairs.sort_by_key(|pair| birth_rank[pair.birth_vertex]);
        H0Decomposition { pairs }
    }
}

/// The dimension-zero pairs of a filtration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct H0Decomposition {
    pairs: Vec<H0Pair>,
}

impl H0Decomposition {
    /// Every class, in birth order.
    ///
    /// There is one pair per vertex: a class either dies at an edge or
    /// survives as an essential class.
    pub fn pairs(&self) -> &[H0Pair] {
        &self.pairs
    }

    /// The classes that survive the end of the filtration.
    pub fn essential(&self) -> impl Iterator<Item = &H0Pair> {
        self.pairs.iter().filter(|pair| pair.death.is_none())
    }

    /// The number of components alive at `threshold`.
    ///
    /// A class counts when its birth is at most `threshold` and its death is
    /// strictly greater, the same half-open convention the matrix reducer
    /// uses: a component that appears and merges at one value is already gone
    /// at that value.
    pub fn count_components_above(&self, threshold: f64) -> usize {
        self.pairs
            .iter()
            .filter(|pair| {
                pair.birth <= threshold && pair.death.is_none_or(|death| death > threshold)
            })
            .count()
    }

    /// The number of classes whose persistence is at least `min_persistence`,
    /// counting essential classes as infinitely persistent.
    ///
    /// This is the cluster count `ToMATo` reads off for a prominence threshold:
    /// a mode counts when it survives at least `min_persistence` before
    /// merging into an older one. It differs from
    /// [`Self::count_components_above`], which counts the components alive at
    /// one filtration value regardless of how long they live. For a superlevel
    /// density filtration, negate the densities so births precede deaths. A
    /// `NaN` threshold matches nothing and returns 0.
    ///
    /// ```
    /// use betti::H0Filtration;
    ///
    /// let mut filtration = H0Filtration::new();
    /// let old = filtration.add_vertex(0.0).unwrap();
    /// let young = filtration.add_vertex(5.0).unwrap();
    /// filtration.add_edge(old, young, 5.5).unwrap();
    /// let components = filtration.decompose();
    /// // Both components are alive at 5.2, but the young one lives only 0.5.
    /// assert_eq!(components.count_components_above(5.2), 2);
    /// assert_eq!(components.count_persistent(1.0), 1);
    /// ```
    pub fn count_persistent(&self, min_persistence: f64) -> usize {
        if min_persistence.is_nan() {
            return 0;
        }
        self.pairs
            .iter()
            .filter(|pair| {
                pair.persistence()
                    .is_none_or(|persistence| persistence >= min_persistence)
            })
            .count()
    }

    /// A cluster label for every vertex at a persistence threshold, the
    /// `ToMATo` merge rule applied to the merge tree.
    ///
    /// A class whose persistence is below `min_persistence` merges, at its
    /// death edge, into the component it met there; a class that persists at
    /// least `min_persistence`, or never dies, stays a cluster of its own. The
    /// labels are the connected components of the death edges of the merged
    /// classes, so there are exactly [`Self::count_persistent`] distinct labels.
    /// They are dense, `0..count_persistent(min_persistence)`, and numbered in
    /// the birth order of each cluster's surviving class, so label 0 is the
    /// oldest class.
    ///
    /// The result is indexed by vertex id. For a density, build the
    /// filtration from negated densities, as for [`Self::count_persistent`]:
    /// a vertex is born at `-density` and an edge enters at the larger of its
    /// endpoints' negated densities, so each cluster is a density mode and the
    /// threshold is a prominence. There is no separate superlevel input.
    ///
    /// A `NaN` threshold matches no class, which [`Self::count_persistent`]
    /// reports as zero clusters; since no labelling has zero clusters unless
    /// there are no vertices, it is a [`H0Error::InvalidValue`] here.
    ///
    /// ```
    /// use betti::H0Filtration;
    ///
    /// // Densities along a path with peaks at vertices 1 and 3 and a valley
    /// // at vertex 2.
    /// let density = [0.5, 1.0, 0.25, 0.75, 0.25];
    /// let mut filtration = H0Filtration::new();
    /// for value in density {
    ///     filtration.add_vertex(-value)?;
    /// }
    /// for v in 0..4 {
    ///     filtration.add_edge(v, v + 1, -density[v].min(density[v + 1]))?;
    /// }
    /// let components = filtration.decompose();
    ///
    /// // The smaller peak rises 0.5 above the valley.
    /// assert_eq!(components.cluster_labels(0.4)?, vec![0, 0, 0, 1, 1]);
    /// assert_eq!(components.cluster_labels(0.6)?, vec![0; 5]);
    /// # Ok::<(), betti::H0Error>(())
    /// ```
    pub fn cluster_labels(&self, min_persistence: f64) -> Result<Vec<usize>, H0Error> {
        if min_persistence.is_nan() {
            return Err(H0Error::InvalidValue {
                value: min_persistence,
            });
        }
        let persists = |pair: &H0Pair| {
            pair.persistence()
                .is_none_or(|persistence| persistence >= min_persistence)
        };

        // One pair per vertex, so vertex ids are 0..len.
        let mut forest = Forest::new(self.pairs.len());
        for pair in &self.pairs {
            if persists(pair) {
                continue;
            }
            if let Some((left, right)) = pair.death_edge {
                let left = forest.find(left);
                let right = forest.find(right);
                forest.merge(left, right);
            }
        }

        // A class outlives every younger class it absorbed, and was born no
        // later, so it persists at least as long as each of them: a merged
        // class absorbed only merged classes, and each component holds
        // exactly one surviving class.
        let mut root_label = vec![None; self.pairs.len()];
        for (label, pair) in self.pairs.iter().filter(|pair| persists(pair)).enumerate() {
            let root = forest.find(pair.birth_vertex);
            debug_assert!(root_label[root].is_none(), "two clusters share a root");
            root_label[root] = Some(label);
        }
        Ok((0..self.pairs.len())
            .map(|vertex| {
                root_label[forest.find(vertex)].expect("every component holds a surviving class")
            })
            .collect())
    }

    /// The number of classes, finite and essential.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether no class was recorded.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Union-find with path halving, union by size, and one elder pointer per
/// component.
#[derive(Clone, Debug)]
struct Forest {
    parent: Vec<usize>,
    size: Vec<usize>,
    elder: Vec<usize>,
}

impl Forest {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
            elder: (0..count).collect(),
        }
    }

    fn find(&mut self, mut index: usize) -> usize {
        while self.parent[index] != index {
            let grandparent = self.parent[self.parent[index]];
            self.parent[index] = grandparent;
            index = grandparent;
        }
        index
    }

    fn elder(&self, root: usize) -> usize {
        self.elder[root]
    }

    /// Attach the smaller tree to the larger, keeping `survivor`'s elder.
    fn merge(&mut self, survivor: usize, absorbed: usize) {
        let elder = self.elder[survivor];
        let (small, large) = if self.size[survivor] < self.size[absorbed] {
            (survivor, absorbed)
        } else {
            (absorbed, survivor)
        };
        self.parent[small] = large;
        self.size[large] += self.size[small];
        self.elder[large] = elder;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filtration(vertices: &[f64], edges: &[(usize, usize, f64)]) -> H0Filtration {
        let mut filtration = H0Filtration::new();
        for &birth in vertices {
            filtration.add_vertex(birth).unwrap();
        }
        for &(left, right, death) in edges {
            filtration.add_edge(left, right, death).unwrap();
        }
        filtration
    }

    #[test]
    fn hand_computed_merge_tree() {
        // Two pairs merge first, then the surviving pair absorbs the rest.
        let filtration = filtration(&[0.0; 4], &[(0, 1, 1.0), (2, 3, 2.0), (1, 2, 3.0)]);
        let decomposition = filtration.decompose();

        let expected: Vec<(usize, Option<f64>)> = decomposition
            .pairs()
            .iter()
            .map(|pair| (pair.birth_vertex(), pair.death()))
            .collect();
        assert_eq!(
            expected,
            vec![(0, None), (1, Some(1.0)), (2, Some(3.0)), (3, Some(2.0))]
        );

        assert_eq!(decomposition.pairs()[1].persistence(), Some(1.0));
        assert_eq!(decomposition.pairs()[0].persistence(), None);
        assert_eq!(decomposition.pairs()[1].death_edge(), Some((0, 1)));
        assert_eq!(decomposition.essential().count(), 1);
        assert_eq!(decomposition.len(), 4);
        assert!(!decomposition.is_empty());
    }

    #[test]
    fn counts_follow_the_half_open_convention() {
        let filtration = filtration(&[0.0; 4], &[(0, 1, 1.0), (2, 3, 2.0)]);
        let decomposition = filtration.decompose();

        assert_eq!(decomposition.count_components_above(-1.0), 0);
        assert_eq!(decomposition.count_components_above(0.5), 4);
        assert_eq!(decomposition.count_components_above(1.0), 3);
        assert_eq!(decomposition.count_components_above(1.5), 3);
        assert_eq!(decomposition.count_components_above(2.0), 2);
        assert_eq!(decomposition.count_components_above(100.0), 2);
    }

    #[test]
    fn equal_births_break_ties_by_vertex_id() {
        let filtration = filtration(&[0.0; 3], &[(0, 1, 1.0), (1, 2, 1.0)]);
        let decomposition = filtration.decompose();

        let pairs: Vec<(usize, Option<f64>)> = decomposition
            .pairs()
            .iter()
            .map(|pair| (pair.birth_vertex(), pair.death()))
            .collect();
        assert_eq!(
            pairs,
            vec![(0, None), (1, Some(1.0)), (2, Some(1.0))],
            "the lowest vertex id survives each merge"
        );
    }

    #[test]
    fn later_birth_survives_an_earlier_merge_target() {
        // Vertex 0 is born late, so the class born at 0.0 in vertex 1 survives.
        let filtration = filtration(&[1.0, 0.0, 0.5], &[(0, 1, 2.0), (1, 2, 3.0)]);
        let decomposition = filtration.decompose();

        let pairs: Vec<(usize, Option<f64>)> = decomposition
            .pairs()
            .iter()
            .map(|pair| (pair.birth_vertex(), pair.death()))
            .collect();
        assert_eq!(
            pairs,
            vec![(1, None), (2, Some(3.0)), (0, Some(2.0))],
            "the elder class is the earliest birth, not the lowest id"
        );
    }

    #[test]
    fn zero_length_interval_is_already_dead() {
        let filtration = filtration(&[0.0, 1.0], &[(0, 1, 1.0)]);
        let decomposition = filtration.decompose();

        assert_eq!(decomposition.pairs()[1].birth(), 1.0);
        assert_eq!(decomposition.pairs()[1].death(), Some(1.0));
        assert_eq!(decomposition.pairs()[1].persistence(), Some(0.0));
        assert_eq!(decomposition.count_components_above(1.0), 1);
    }

    #[test]
    fn isolated_vertices_are_all_essential() {
        let filtration = filtration(&[0.0, 0.5, 2.0], &[]);
        let decomposition = filtration.decompose();

        assert_eq!(decomposition.essential().count(), 3);
        assert!(decomposition
            .pairs()
            .iter()
            .all(|pair| pair.death().is_none()));
        assert_eq!(decomposition.count_components_above(0.25), 1);
        assert_eq!(decomposition.count_components_above(2.0), 3);
    }

    #[test]
    fn empty_filtration_has_no_classes() {
        let decomposition = H0Filtration::new().decompose();
        assert!(decomposition.is_empty());
        assert_eq!(decomposition.count_components_above(0.0), 0);
    }

    #[test]
    fn malformed_input_is_a_typed_error() {
        let mut filtration = H0Filtration::new();
        assert!(matches!(
            filtration.add_vertex(f64::NAN),
            Err(H0Error::InvalidValue { .. })
        ));
        filtration.add_vertex(0.0).unwrap();
        filtration.add_vertex(1.0).unwrap();

        assert!(matches!(
            filtration.add_edge(0, 1, f64::NAN),
            Err(H0Error::InvalidValue { .. })
        ));
        assert_eq!(
            filtration.add_edge(0, 5, 1.0),
            Err(H0Error::UnknownVertex {
                vertex: 5,
                edge: (0, 5)
            })
        );
        assert_eq!(
            filtration.add_edge(1, 1, 1.0),
            Err(H0Error::SelfLoop { vertex: 1 })
        );
        assert_eq!(
            filtration.add_edge(1, 0, 0.5),
            Err(H0Error::BirthAfterDeath {
                vertex: 1,
                birth: 1.0,
                death: 0.5
            })
        );
        filtration.add_edge(0, 1, 1.0).unwrap();
        assert_eq!(
            filtration.add_edge(1, 0, 2.0),
            Err(H0Error::DuplicateEdge { edge: (0, 1) })
        );
        assert_eq!(filtration.edge_count(), 1);

        assert!(H0Error::SelfLoop { vertex: 3 }
            .to_string()
            .contains("itself"));
        assert!(H0Error::InvalidValue { value: 0.0 }
            .to_string()
            .contains("not a number"));
        assert!(std::error::Error::source(&H0Error::SelfLoop { vertex: 0 }).is_none());
    }

    #[test]
    fn duplicate_edges_are_rejected_not_counted() {
        let mut filtration = H0Filtration::new();
        filtration.add_vertex(0.0).unwrap();
        filtration.add_vertex(0.0).unwrap();
        filtration.add_edge(0, 1, 1.0).unwrap();
        assert!(filtration.add_edge(0, 1, 1.0).is_err());
        // One edge merges the two components once, as in a simplicial complex.
        assert_eq!(filtration.decompose().essential().count(), 1);
    }

    #[test]
    fn persistent_count_uses_lifetimes_not_a_level() {
        // Four vertices born at 0; edges kill one class at 1 and one at 2.
        let mut filtration = H0Filtration::new();
        for _ in 0..4 {
            filtration.add_vertex(0.0).unwrap();
        }
        filtration.add_edge(0, 1, 1.0).unwrap();
        filtration.add_edge(2, 3, 2.0).unwrap();
        let decomposition = filtration.decompose();
        // Persistences are 1 and 2, plus two essential classes.
        assert_eq!(decomposition.count_persistent(0.0), 4);
        assert_eq!(decomposition.count_persistent(1.0), 4);
        assert_eq!(decomposition.count_persistent(1.5), 3);
        assert_eq!(decomposition.count_persistent(2.0), 3);
        assert_eq!(decomposition.count_persistent(2.5), 2);
        assert_eq!(decomposition.count_persistent(f64::INFINITY), 2);
        assert_eq!(decomposition.count_persistent(f64::NAN), 0);
    }

    #[test]
    fn persistent_and_level_counts_differ_for_late_births() {
        let mut filtration = H0Filtration::new();
        filtration.add_vertex(0.0).unwrap();
        filtration.add_vertex(5.0).unwrap();
        filtration.add_edge(0, 1, 5.5).unwrap();
        let decomposition = filtration.decompose();
        assert_eq!(decomposition.count_components_above(5.2), 2);
        assert_eq!(decomposition.count_persistent(1.0), 1);
        assert_eq!(decomposition.count_persistent(0.5), 2);
    }
}
