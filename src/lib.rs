//! Canonical graph labelling.
//!
//! Leverages [nauty and Traces](http://pallini.di.uniroma1.it/) to
//! find [canonical
//! labellings](https://en.wikipedia.org/wiki/Graph_canonization) for
//! [petgraph](https://github.com/petgraph/petgraph) graphs.
//!
//! Version 2.6 or 2.7 of the nauty and Traces library has to be
//! installed separately before installing this crate.
//!
//! # Example
//!
//! ```rust
//! use petgraph::graph::UnGraph;
//! use nauty_pet::prelude::*;
//!
//! // Two different vertex labellings for the tree graph with two edges
//! let g1 = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)]);
//! let g2 = UnGraph::<(), ()>::from_edges([(0, 1), (0, 2)]);
//!
//! // There are two equivalent labellings
//! let automorphism_info = g1.clone().try_into_autom().unwrap();
//! assert_eq!(automorphism_info.grpsize(), 2.);
//!
//! // The canonical forms are identical
//! let c1 = g1.clone().into_canon();
//! let c2 = g2.clone().into_canon();
//! assert!(c1.is_identical(&c2));
//!
//! // Alternatively, we can use a dedicated `struct` for canonically
//! // labelled graphs
//! let c1 = CanonGraph::from(g1);
//! let c2 = CanonGraph::from(g2);
//! assert_eq!(c1, c2);
//! ```
//!
//! # Features
//!
//! * `libc`: Allows using sparse graphs for canonisation. Before
//!           enabling this feature, ensure that the nauty and Traces
//!           library is linked to the same C library as this crate.
//!
//! * `serde-1`: Enables serialisation of
//!              [CanonGraph](graph::CanonGraph) objects using
//!              [serde](https://crates.io/crates/serde).
//!
//! * `stable`: Ensures deterministic behaviour when node or edge
//!             weights are distinguishable, but compare equal.
//!
//! To enable features `feature1`, `feature2` add the following to
//! your Cargo.toml:
//! ```toml
//! [dependencies]
//! nauty-pet = { version = "0.7", features = ["feature1", "feature2"] }
//! ```
mod autom;
pub mod canon;
mod cmp;
pub mod error;
pub mod graph;
mod nauty_graph;
pub mod prelude;

pub use canon::IntoCanon;
#[cfg(feature = "libc")]
pub use canon::{IntoCanonNautySparse, TryIntoCanonTraces};
pub use cmp::IsIdentical;

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use petgraph::{
        graph::{Graph, IndexType, UnGraph},
        prelude::EdgeIndex,
        EdgeType,
    };

    fn add_edge<N, E, Ty: EdgeType, Ix: IndexType>(
        g: &mut Graph<N, E, Ty, Ix>,
        v1: usize,
        v2: usize,
        wt: E,
    ) -> EdgeIndex<Ix> {
        use petgraph::visit::NodeIndexable;
        g.add_edge(g.from_index(v1), g.from_index(v2), wt)
    }

    #[test]
    fn nautyex8() {
        let n_range = (2..20).step_by(2);

        for n in n_range {
            let mut g1 = UnGraph::<(), ()>::with_capacity(n, 3 * n);
            for _ in 0..n {
                g1.add_node(());
            }

            // Spokes
            for i in (0..n).step_by(2) {
                add_edge(&mut g1, i, i + 1, ());
            }
            // Cycle
            for i in 0..n - 2 {
                add_edge(&mut g1, i, i + 2, ());
            }
            add_edge(&mut g1, 1, n - 2, ());
            add_edge(&mut g1, 0, n - 1, ());

            let mut g2 = UnGraph::<(), ()>::with_capacity(n, 3 * n);
            for _ in 0..n {
                g2.add_node(());
            }

            for i in 0..n {
                add_edge(&mut g2, i, (i + 1) % n, ()); /* Rim */
            }
            for i in 0..(n / 2) {
                add_edge(&mut g2, i, i + n / 2, ()); /* Diagonals */
            }

            /* Create canonical graphs */

            let cg1 = g1.into_canon();
            let cg2 = g2.into_canon();

            assert!(cg1.is_identical(&cg2));
        }
    }
}
