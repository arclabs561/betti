//! Boundary matrix reduction and the resulting persistence pairs.

use crate::boundary::BoundaryMatrix;
use crate::simplex::Simplex;

/// A class birth: the simplex that created it and when it appeared.
#[derive(Clone, Debug, PartialEq)]
pub struct Birth {
    simplex: Simplex,
    value: f64,
}

impl Birth {
    /// The simplex that created the class.
    pub fn simplex(&self) -> &Simplex {
        &self.simplex
    }

    /// The filtration value at which the class appeared.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// The dimension of the class.
    pub fn dim(&self) -> usize {
        self.simplex.dim()
    }
}

/// The filtration value at which a class dies.
///
/// `None` marks an essential class: it is still alive when the filtration
/// ends, and it contributes to a Betti number.
pub type Death = Option<f64>;

/// One persistence pair: a birth and, for a finite class, its death.
pub type Pair = (Birth, Death);

/// How much of the reduction to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReductionOptions {
    /// Use the clearing (twist) optimization.
    ///
    /// When a column `j` pairs with a lower-index column `i`, the pair is
    /// final; clearing skips the reduction of column `i` entirely and processes
    /// dimensions from high to low. It changes the work done, never the
    /// resulting pairs or Betti numbers.
    pub clear: bool,
}

impl ReductionOptions {
    /// Options with clearing enabled.
    pub fn clearing() -> Self {
        Self { clear: true }
    }
}

/// The reduced matrix, expressed as persistence pairs.
#[derive(Clone, Debug, Default)]
pub struct Decomposition {
    simplices: Vec<Simplex>,
    values: Vec<f64>,
    pairs: Vec<(usize, Option<usize>)>,
}

impl Decomposition {
    /// Every pair, in birth order.
    ///
    /// A finite pair reports the death value; an essential class reports
    /// [`None`] and is counted by [`Self::betti_numbers`].
    pub fn pairs(&self) -> Vec<Pair> {
        self.pairs
            .iter()
            .map(|&(birth, death)| {
                (
                    Birth {
                        simplex: self.simplices[birth].clone(),
                        value: self.values[birth],
                    },
                    death.map(|column| self.values[column]),
                )
            })
            .collect()
    }

    /// The essential classes, in birth order.
    pub fn essential(&self) -> Vec<Birth> {
        self.pairs
            .iter()
            .filter(|(_, death)| death.is_none())
            .map(|&(birth, _)| Birth {
                simplex: self.simplices[birth].clone(),
                value: self.values[birth],
            })
            .collect()
    }

    /// Betti numbers for dimensions `0..=max_dim`.
    ///
    /// A Betti number counts the essential classes in its dimension, that is,
    /// the unpaired columns of the reduced matrix.
    ///
    /// ```
    /// use betti::{reduction, BoundaryMatrix, Filtration};
    ///
    /// let mut filtration = Filtration::new();
    /// filtration.insert([0], 0.0).unwrap();
    /// filtration.insert([1], 0.0).unwrap();
    /// let matrix = BoundaryMatrix::new(&filtration).unwrap();
    ///
    /// // Two isolated vertices: two components.
    /// assert_eq!(reduction(&matrix).betti_numbers(0), vec![2]);
    /// ```
    pub fn betti_numbers(&self, max_dim: usize) -> Vec<usize> {
        let mut betti = vec![0; max_dim.saturating_add(1)];
        for &(birth, death) in &self.pairs {
            if death.is_none() {
                let dimension = self.simplices[birth].dim();
                if let Some(slot) = betti.get_mut(dimension) {
                    *slot += 1;
                }
            }
        }
        betti
    }

    /// The number of persistence pairs, finite and essential.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether no class was recorded.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Reduce the boundary matrix over GF(2), left to right.
///
/// Columns are added while their lowest set row, the pivot, collides with an
/// earlier column. A column that reduces to zero is a birth; a column whose
/// pivot removes an earlier birth records a death.
pub fn reduction(matrix: &BoundaryMatrix) -> Decomposition {
    reduction_with(matrix, ReductionOptions::default())
}

/// Reduce with explicit options; see [`ReductionOptions::clear`].
pub fn reduction_with(matrix: &BoundaryMatrix, options: ReductionOptions) -> Decomposition {
    let n = matrix.len();
    let dims: Vec<usize> = matrix.simplices().iter().map(Simplex::dim).collect();
    let mut columns: Vec<BitSet> = matrix
        .columns()
        .iter()
        .map(|column| BitSet::from_indices(column, n))
        .collect();

    // owner[low] is the column whose reduced pivot is `low`.
    let mut owner: Vec<Option<usize>> = vec![None; n];
    let mut paired = vec![false; n];
    let mut cleared = vec![false; n];
    let mut pairs: Vec<(usize, Option<usize>)> = Vec::new();

    let mut dimensions: Vec<usize> = (0..=matrix.max_dim().unwrap_or(0)).collect();
    if options.clear {
        dimensions.reverse();
    }

    for dimension in dimensions {
        for (column, &dim) in dims.iter().enumerate() {
            if dim != dimension || cleared[column] {
                continue;
            }
            while let Some(low) = columns[column].highest() {
                match owner[low] {
                    Some(partner) => add_into(&mut columns, column, partner),
                    None => {
                        owner[low] = Some(column);
                        paired[low] = true;
                        pairs.push((low, Some(column)));
                        if options.clear {
                            cleared[low] = true;
                        }
                        break;
                    }
                }
            }
        }
    }

    for (column, bits) in columns.iter().enumerate() {
        if bits.is_empty() && !paired[column] {
            pairs.push((column, None));
        }
    }
    pairs.sort_by_key(|&(birth, _)| birth);

    Decomposition {
        simplices: matrix.simplices().to_vec(),
        values: matrix.values().to_vec(),
        pairs,
    }
}

/// XOR column `source` into column `target` without cloning either.
fn add_into(columns: &mut [BitSet], target: usize, source: usize) {
    debug_assert_ne!(target, source, "a column cannot be added to itself");
    if target > source {
        let (head, tail) = columns.split_at_mut(target);
        tail[0].xor_assign(&head[source]);
    } else {
        let (head, tail) = columns.split_at_mut(source);
        head[target].xor_assign(&tail[0]);
    }
}

/// A fixed-width set of row indices, one bit per column of the matrix.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    fn new(bits: usize) -> Self {
        Self {
            words: vec![0; bits.div_ceil(64)],
        }
    }

    fn from_indices(indices: &[usize], bits: usize) -> Self {
        let mut set = Self::new(bits);
        for &index in indices {
            set.insert(index);
        }
        set
    }

    fn insert(&mut self, index: usize) {
        self.words[index / 64] |= 1 << (index % 64);
    }

    fn xor_assign(&mut self, other: &Self) {
        for (word, mask) in self.words.iter_mut().zip(&other.words) {
            *word ^= mask;
        }
    }

    fn highest(&self) -> Option<usize> {
        for (index, word) in self.words.iter().enumerate().rev() {
            if *word != 0 {
                return Some(index * 64 + (63 - word.leading_zeros() as usize));
            }
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.words.iter().all(|word| *word == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_pivots_and_xor() {
        let mut left = BitSet::from_indices(&[0, 3], 4);
        let right = BitSet::from_indices(&[3], 4);
        assert_eq!(left.highest(), Some(3));
        assert!(!left.is_empty());
        left.xor_assign(&right);
        assert_eq!(left.highest(), Some(0));
        left.xor_assign(&BitSet::from_indices(&[0], 4));
        assert!(left.is_empty());
        assert_eq!(left.highest(), None);
    }

    #[test]
    fn bitset_spans_several_words() {
        let mut set = BitSet::from_indices(&[0, 64, 130], 131);
        assert_eq!(set.highest(), Some(130));
        set.xor_assign(&BitSet::from_indices(&[130], 131));
        assert_eq!(set.highest(), Some(64));
    }

    #[test]
    fn empty_matrix_has_no_classes() {
        let matrix = BoundaryMatrix::default();
        let decomposition = reduction(&matrix);
        assert!(decomposition.is_empty());
        assert_eq!(decomposition.betti_numbers(3), vec![0, 0, 0, 0]);
    }
}
