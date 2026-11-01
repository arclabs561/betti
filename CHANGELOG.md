# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
