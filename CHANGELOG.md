# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Criterion benchmark (`benches/reduction.rs`) of the reduction and the `H0`
  union-find against `lophat` on the same boundary matrix.
- External-oracle tests: Vietoris-Rips H0 and H1 diagrams of five fixed
  point clouds compared with GUDHI 3.13.0 output, with the generator script
  in `scripts/oracle_fixtures.py`.
- `H0Decomposition::cluster_labels(min_persistence)`: `ToMATo`-style cluster
  labels per vertex. Classes with persistence below the threshold merge into
  the component they met at death; the label count equals
  `count_persistent(min_persistence)`. A `NaN` threshold is
  `H0Error::InvalidValue`.

### Changed

- The reducer stores each column as a sorted list of row indices instead of
  a dense bitset over every column, so a column addition costs the two
  columns' lengths rather than `n / 64` words and memory is linear in the
  entries rather than `n^2 / 8` bytes. Pairs are unchanged. On Rips
  2-skeletons with clearing it is 3.8x faster at 15k columns and 9-11x at
  36k-45k.

## [0.1.1] - 2026-11-04

### Added

- `H0Filtration` and `H0Decomposition`: dimension-zero persistence by
  filtration-aware union-find over vertices with birth times and edges with
  death times. `count_components_above(level)` counts the components alive at
  one filtration value; `count_persistent(min_persistence)` counts the classes
  that persist at least that long, the cluster count `ToMATo` takes for a
  prominence threshold. The pairs agree with the matrix reducer on the same
  filtration.
- Differential test suite against `lophat` 0.14 (dev-dependency only): 200
  random complexes, both clearing settings, compared as a multiset of
  (birth, death) pairs with essential classes compared separately.

## 0.1.0

Initial release.

### Added

- `Simplex`: a sorted vertex set with `dim()`, codimension-one `faces()` and
  subset testing.
- `Filtration`: simplexwise filtration that rejects `FaceAfterCoface`,
  `MissingFace`, `DuplicateSimplex` and `InvalidValue` as typed errors.
- `BoundaryMatrix`: GF(2) boundary columns in
  `(filtration value, dimension, vertices)` order.
- `reduction` and `ReductionOptions`: standard left-to-right column reduction,
  with the optional clearing (twist) optimization that returns identical pairs.
- `Decomposition::pairs`, `essential` and `betti_numbers` for essential classes.
