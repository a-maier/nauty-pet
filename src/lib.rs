mod canon;
mod sparse_graph;

pub use canon::{
    ToCanon, ToCanonNautyDense, ToCanonNautySparse, ToCanonTraces, TracesError,
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

            println!("{cg1:#?}");
            println!("{cg2:#?}");
        }
    }
}
