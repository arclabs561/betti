# betti

Local draft spot. Not a crate yet.

Purpose: persistent homology and Betti-number computation: boundary matrix
reduction, twist/clearing optimization, cohomology variant, and representative
cycles.

Crates.io exact-name check with `cargo info betti`: no crate found on
2026-06-26. PyPI and npm also returned 404 for `betti` on the same date.

First proof: consume a small boundary matrix emitted by `skel`, reduce it, and
print intervals.

Current verdict: do not promote yet. `skel` already emits filtration boundary
matrices, `skel/examples/persistence_reduction.rs` reduces one boundary matrix
to intervals, and `skel` documents `lophat` / `phlite` as downstream persistent
homology consumers. A standalone `betti` crate only makes sense if an in-tree
reducer is needed repeatedly and cannot live as a small `skel` module.

Extraction rule: start as a tiny reducer around one artifact. Do not build a
full TDA suite, multiparameter persistence, GPU backend, or Mapper here.
