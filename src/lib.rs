//! Canonical graph labelling.
//!
//! Leverages [nauty and Traces](http://pallini.di.uniroma1.it/) to
//! find [canonical
//! labellings](https://en.wikipedia.org/wiki/Graph_canonization) for
//! [petgraph](https://github.com/petgraph/petgraph) graphs.
//!
//! # Usage
//!
//! Version 2.6 or 2.7 of the nauty and Traces library has to be
//! installed first and _linked to the same C library as this crate_.
//!
//! Afterwards, add this to your Cargo.toml:
//! ```toml
//! [dependencies]
//! nauty-pet = "0.1"
//! ```
//!
//! # Example
//!
//! ```rust
//! use petgraph::graph::UnGraph;
//! use nauty_pet::ToCanon;
//!
//! // Two different vertex labellings for the tree graph with two edges
//! let g1 = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)]);
//! let g2 = UnGraph::<(), ()>::from_edges([(0, 1), (0, 2)]);
//!
//! let c1 = g1.to_canon();
//! let c2 = g2.to_canon();
//! // c1 and c2 now have the same edges, up to permutation
//! #
//! # let mut edges = [Vec::new(), Vec::new()];
//! # for (e, c) in edges.iter_mut().zip([c1, c2].into_iter()) {
//! #    let (_, ee) = c.into_nodes_edges();
//! #    *e = Vec::from_iter(
//! #       ee.into_iter().map(|e| (e.source(), e.target()))
//! #    );
//! #    e.sort_unstable();
//! # }
//! # assert_eq!(edges[0], edges[1]);
//! ```
//!
//! # Caveats
//!
//! - Edge weights are ignored and discarded.
//! - Only undirected graphs without self-loops have been tested so far.
//!
mod canon;
mod sparse_graph;

pub use canon::{
    ToCanon,
    ToCanonNautySparse,
    ToCanonTraces,
    TracesError,
};

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::{
        graph::{Graph, IndexType, UnGraph},
        EdgeType,
        prelude::EdgeIndex
    };

    fn add_edge<N, E, Ty: EdgeType, Ix: IndexType>(
        g: &mut Graph<N, E, Ty, Ix>,
        v1: usize,
        v2: usize,
        wt: E
    ) -> EdgeIndex<Ix> {
        use petgraph::visit::NodeIndexable;
        g.add_edge(g.from_index(v1), g.from_index(v2), wt)
    }

    fn minmax<T: Ord>(t1: T, t2: T) -> (T, T) {
        if t1 <= t2 {
            (t1, t2)
        } else {
            (t2, t1)
        }
    }

    #[test]
    fn nautyex8() {
        let n_range = (2..20).step_by(2);

        for n in n_range {
            let mut g1 = UnGraph::<(), ()>::with_capacity(n, 3*n);
            for _ in 0..n {
                g1.add_node(());
            }

            // Spokes
            for i in (0..n).step_by(2) {
                add_edge(&mut g1, i, i + 1, ());
            }
            // Cycle
            for i in 0..n-2 {
                add_edge(&mut g1, i, i + 2, ());
            }
            add_edge(&mut g1, 1, n - 2, ());
            add_edge(&mut g1, 0, n - 1, ());

            let mut g2 = UnGraph::<(), ()>::with_capacity(n, 3*n);
            for _ in 0..n {
                g2.add_node(());
            }

            for i in 0..n {
                add_edge(&mut g2, i, (i + 1) % n, ());     /* Rim */
            }
            for i in 0..(n / 2) {
                add_edge(&mut g2, i, i + n / 2, ()); /* Diagonals */
            }

            /* Create canonical graphs */

            let cg1 = g1.to_canon();
            let cg2 = g2.to_canon();

            let (_, e1) = cg1.into_nodes_edges();
            let mut e1 = Vec::from_iter(
                e1.into_iter().map(|e| minmax(e.source(), e.target()))
            );
            e1.sort_unstable();
            let (_, e2) = cg2.into_nodes_edges();
            let mut e2 = Vec::from_iter(
                e2.into_iter().map(|e| minmax(e.source(), e.target()))
            );
            e2.sort_unstable();
            assert_eq!(e1, e2);
        }
    }
}
