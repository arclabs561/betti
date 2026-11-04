//! Shared test support: a deterministic PRNG, random filtrations, and a
//! Vietoris-Rips builder.
//!
//! Each integration test binary loads this module on its own and uses a
//! different subset, so unused helpers are expected.
#![allow(dead_code)]

use betti::Filtration;

/// A 64-bit xorshift PRNG.
///
/// The generator is small, dependency-free, and bit-identical on every
/// platform: a seed reproduces a failure exactly.
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Seed the generator; zero is remapped because it is a fixed point.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    /// The next output of the shift register.
    pub fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    /// A value in `0..bound`; `bound` must be nonzero.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// True with probability `numerator / denominator`.
    pub fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.below(denominator) < numerator
    }
}

/// A face-closed simplicial complex with filtration values, as plain data.
///
/// Values live on a grid of quarters, so ties are common, every value is
/// exactly representable, and no comparison needs an epsilon.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Complex {
    /// `(sorted vertices, filtration value)`, in no particular order.
    pub simplices: Vec<(Vec<usize>, f64)>,
}

impl Complex {
    /// Build a [`Filtration`], panicking if the data violates the filtration
    /// condition. The generators here keep it valid by construction.
    pub fn filtration(&self) -> Filtration {
        let mut filtration = Filtration::new();
        for (vertices, value) in &self.simplices {
            filtration
                .insert(vertices.clone(), *value)
                .expect("the generator keeps a valid filtration");
        }
        filtration
    }

    /// The one-skeleton as `(vertex births, edges)`, in vertex-id order.
    ///
    /// Vertex ids are assumed to be exactly `0..count`.
    pub fn one_skeleton(&self) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
        let count = self
            .simplices
            .iter()
            .filter(|(vertices, _)| vertices.len() == 1)
            .count();
        let mut births = vec![f64::NAN; count];
        let mut edges = Vec::new();
        for (vertices, value) in &self.simplices {
            match vertices.len() {
                1 => births[vertices[0]] = *value,
                2 => edges.push((vertices[0], vertices[1], *value)),
                _ => {}
            }
        }
        assert!(
            births.iter().all(|birth| !birth.is_nan()),
            "every vertex in 0..count must be present"
        );
        (births, edges)
    }

    /// A compact, reproducible description for a failure report.
    pub fn describe(&self) -> String {
        let mut lines: Vec<String> = self
            .simplices
            .iter()
            .map(|(vertices, value)| format!("{vertices:?} at {value}"))
            .collect();
        lines.sort();
        lines.join("\n  ")
    }
}

/// Whether every vertex of `face` also belongs to `coface`.
pub fn is_face(face: &[usize], coface: &[usize]) -> bool {
    face.len() <= coface.len() && face.iter().all(|vertex| coface.contains(vertex))
}

/// The Euclidean distance between two points of equal dimension.
pub fn distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// The Vietoris-Rips complex of a point cloud, truncated at `radius`.
///
/// Vertices enter at 0, an edge at its length when that is at most `radius`,
/// and, when `triangles` is set, each triangle whose three edges are present
/// at its longest edge. This is the convention GUDHI's `RipsComplex` and
/// Ripser use, so their diagrams are directly comparable.
pub fn rips(points: &[Vec<f64>], radius: f64, triangles: bool) -> Complex {
    let count = points.len();
    let mut length = vec![vec![None; count]; count];
    let mut simplices: Vec<(Vec<usize>, f64)> = (0..count).map(|v| (vec![v], 0.0)).collect();
    let mut edges = Vec::new();
    for i in 0..count {
        for j in (i + 1)..count {
            let d = distance(&points[i], &points[j]);
            if d <= radius {
                length[i][j] = Some(d);
                length[j][i] = Some(d);
                edges.push((i, j, d));
                simplices.push((vec![i, j], d));
            }
        }
    }
    if triangles {
        for &(i, j, ij) in &edges {
            let pairs = length[i].iter().zip(&length[j]).enumerate();
            for (k, (ik, jk)) in pairs.skip(j + 1) {
                if let (Some(ik), Some(jk)) = (ik, jk) {
                    simplices.push((vec![i, j, k], ij.max(*ik).max(*jk)));
                }
            }
        }
    }
    Complex { simplices }
}

/// A random one-dimensional filtration: `3..=10` vertices and a random edge
/// subset, with every value on the quarter grid.
///
/// Vertex births come from `{0, 0.25, 0.5, 0.75, 1}`, so equal values are
/// frequent and the column order is exercised; an edge dies no earlier than
/// either endpoint is born.
pub fn random_one_dimensional(rng: &mut XorShift64) -> Complex {
    let vertex_count = 3 + rng.below(8) as usize;
    let mut simplices = Vec::new();
    let mut births = Vec::with_capacity(vertex_count);
    for vertex in 0..vertex_count {
        let birth = 0.25 * rng.below(5) as f64;
        births.push(birth);
        simplices.push((vec![vertex], birth));
    }

    let edge_tenths = 2 + rng.below(7);
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for left in 0..vertex_count {
        for right in (left + 1)..vertex_count {
            if !rng.chance(edge_tenths, 10) {
                continue;
            }
            let base = births[left].max(births[right]);
            let death = base + 0.25 * rng.below(4) as f64;
            edges.push((left, right, death));
            simplices.push((vec![left, right], death));
        }
    }

    Complex { simplices }
}

/// A random complex: the one-skeleton plus a random subset of the triangles
/// whose three edges are present.
pub fn random_complex(rng: &mut XorShift64) -> Complex {
    let mut complex = random_one_dimensional(rng);
    let (_, edges) = complex.one_skeleton();
    let value_of = |left: usize, right: usize| {
        let edge = (left.min(right), left.max(right));
        edges
            .iter()
            .find(|&&(a, b, _)| (a, b) == edge)
            .map(|&(_, _, value)| value)
    };

    let count = complex
        .simplices
        .iter()
        .filter(|(vertices, _)| vertices.len() == 1)
        .count();
    for left in 0..count {
        for middle in (left + 1)..count {
            for right in (middle + 1)..count {
                let (Some(first), Some(second), Some(third)) = (
                    value_of(left, middle),
                    value_of(left, right),
                    value_of(middle, right),
                ) else {
                    continue;
                };
                if !rng.chance(3, 10) {
                    continue;
                }
                let death = first.max(second).max(third) + 0.25 * rng.below(3) as f64;
                complex.simplices.push((vec![left, middle, right], death));
            }
        }
    }
    complex
}
