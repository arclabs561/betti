# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Needs a 0.1.1 release: `0.1.0` is published and does not contain the `H0`
path below.

### Added

- `H0Filtration` and `H0Decomposition`: dimension-zero persistence by
  filtration-aware union-find over vertices with birth times and edges with
  death times, with `count_components_above(threshold)` for a merge-tree or
  `ToMATo` component count. The pairs agree with the matrix reducer on the same
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
