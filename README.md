# betti

Persistent homology over GF(2) with a boundary matrix reducer, the
twist/clearing optimization, and Betti numbers.

## Install

```toml
betti = "0.1.1"
```

```rust
use betti::{reduction, BoundaryMatrix, Filtration};

// The boundary of a triangle: three vertices and three edges, no 2-simplex.
let mut filtration = Filtration::new();
for vertex in 0..3 {
    filtration.insert([vertex], 0.0).unwrap();
}
for edge in [[0, 1], [0, 2], [1, 2]] {
    filtration.insert(edge, 1.0).unwrap();
}

let matrix = BoundaryMatrix::new(&filtration).unwrap();
let decomposition = reduction(&matrix);

// One component and one loop.
assert_eq!(decomposition.betti_numbers(1), vec![1, 1]);
```

## What it is

- `Simplex`: a sorted `Vec<usize>` of vertex ids with `dim()` and codimension-one
  face enumeration.
- `Filtration`: simplices tagged with a filtration value. It rejects a face
  valued after its coface, a missing face, a duplicate simplex and a `NaN`
  value as a typed `FiltrationError`, never a panic.
- `BoundaryMatrix`: the GF(2) boundary columns, ordered by
  `(filtration value, dimension, vertices)` so every face precedes its coface.
- `reduction`: standard left-to-right column addition with `low` as the highest
  set row. It returns a `Decomposition` of `(Birth, Death)` pairs, where an
  essential class has `Death = None`, plus `betti_numbers(max_dim)`.
- `ReductionOptions::clear`: the clearing (twist) optimization. It skips
  columns whose classes are already paired and must return exactly the same
  pairs and Betti numbers as the plain reduction.
- `H0Filtration`: a one-dimensional filtration (vertices with birth times,
  edges with death times) that returns its dimension-zero pairs by
  filtration-aware union-find, without a boundary matrix.
  It agrees pair for pair with the matrix reducer on the same filtration.
  `H0Decomposition::count_components_above(level)` counts the components alive
  at one filtration value; `count_persistent(min_persistence)` counts the
  classes that persist at least that long (essential classes included), which
  is the cluster count `ToMATo` takes for a prominence threshold.

## Boundary

This is a reducer, not a TDA suite. Version 0.1.0 implements homology reduction
only; the cohomology variant and representative cycle extraction named in the
purpose above are not implemented yet. There is no multiparameter persistence,
no Mapper, no GPU backend, and no simplex generation from point clouds: build
the filtration yourself, or feed one in from a tool such as `skel`.

Filtration values are `f64`; `NaN` is rejected because it has no order and
would make the column order ambiguous.

## Testing

`just check` runs formatting, Clippy with `-D warnings`, the test suite and
docs. The hand-checkable cases are a single vertex, two disconnected edges, a
triangle boundary, a filled triangle, and the tetrahedron boundary as a sphere,
plus checks that clearing returns the same pairs as the plain reduction and
that a face valued after its coface yields `FiltrationError::FaceAfterCoface`.

Two differential suites back the claims:

- `tests/differential_lophat.rs` reduces 200 random complexes (3-10 vertices,
  random edges and triangles, quarter-grid values so ties are common) with
  `betti` and with `lophat`, under both clearing settings, and compares the
  pairings as a multiset. Seeds are a small built-in xorshift; a failure prints
  the seed and a greedily shrunk counterexample.
- `tests/h0_matrix_agreement.rs` compares the union-find `H0` pairs and
  component counts against the matrix reducer on 300 random one-dimensional
  filtrations.

## License

MIT OR Apache-2.0.
