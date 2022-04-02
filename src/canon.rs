use crate::sparse_graph::SparseGraph;

use std::cmp::Ord;

use nauty_Traces_sys::{
    optionblk, sparsegraph, sparsenauty, statsblk, Traces, TracesOptions,
    TracesStats, FALSE, SG_FREE, TRUE,
};
use petgraph::{
    graph::{DiGraph, Graph, IndexType, UnGraph},
    visit::EdgeRef,
    EdgeType,
};
use thiserror::Error;

/// Find the canonical labelling for a graph
///
/// Internally, this uses Traces for undirected graphs without
/// self-loops and sparse nauty otherwise.
pub trait IntoCanon {
    fn into_canon(self) -> Self;
}

/// Use sparse nauty to find the canonical labelling
pub trait IntoCanonNautySparse {
    fn into_canon_nauty_sparse(self) -> Self;
}

/// Use dense nauty to find the canonical labelling
pub trait IntoCanonNautyDense {
    fn into_canon_nauty_dense(self) -> Self;
}

/// Use Traces to find the canonical labelling
pub trait TryIntoCanonTraces {
    type Error;

    fn try_into_canon_traces(self) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

impl<N, E, Ix: IndexType> IntoCanon for UnGraph<N, E, Ix>
where
    N: Ord,
    E: Ord,
{
    fn into_canon(self) -> Self {
        match self.try_into_canon_traces() {
            Ok(c) => c,
            Err(TracesError::SelfLoop(g)) => g.into_canon_nauty_sparse(),
        }
    }
}

impl<N, E, Ix: IndexType> IntoCanon for DiGraph<N, E, Ix>
where
    N: Ord,
    E: Ord,
{
    fn into_canon(self) -> Self {
        self.into_canon_nauty_sparse()
    }
}

impl<N, E, Ty, Ix: IndexType> IntoCanonNautySparse for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Ord,
    Ty: EdgeType,
{
    fn into_canon_nauty_sparse(self) -> Self {
        let mut options = optionblk::default_sparse();
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        options.digraph = TRUE;
        let mut stats = statsblk::default();
        let mut orbits = vec![0; self.node_count()];
        let mut sg = SparseGraph::from(self);
        let mut cg = sparsegraph::default();
        unsafe {
            sparsenauty(
                &mut (&mut sg.g).into(),
                sg.node_weights.lab.as_mut_ptr(),
                sg.node_weights.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            SG_FREE(&mut cg);
        }
        sg.into()
    }
}

impl<N, E, Ix: IndexType> TryIntoCanonTraces for UnGraph<N, E, Ix>
where
    N: Ord,
    E: Ord,
{
    type Error = TracesError<N, E, Ix>;

    fn try_into_canon_traces(self) -> Result<Self, Self::Error> {
        if has_self_loop(&self) {
            return Err(TracesError::SelfLoop(self));
        }
        let mut options = TracesOptions {
            getcanon: TRUE,
            defaultptn: FALSE,
            digraph: TRUE,
            ..Default::default()
        };
        let mut stats = TracesStats::default();
        let mut orbits = vec![0; self.node_count()];
        let mut sg = SparseGraph::from(self);
        let mut cg = sparsegraph::default();
        unsafe {
            Traces(
                &mut (&mut sg.g).into(),
                sg.node_weights.lab.as_mut_ptr(),
                sg.node_weights.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            SG_FREE(&mut cg);
        }
        Ok(sg.into())
    }
}

#[derive(Error, Debug)]
pub enum TracesError<N, E, Ix: IndexType> {
    #[error("Graph has at least one self-loop")]
    SelfLoop(UnGraph<N, E, Ix>),
}

fn has_self_loop<N, E, Ty: EdgeType, Ix: IndexType>(
    g: &Graph<N, E, Ty, Ix>,
) -> bool {
    g.edge_references().any(|e| e.source() == e.target())
}

#[cfg(test)]
mod tests {
    use super::super::cmp::IsIdentical;
    use super::*;
    use itertools::izip;
    use petgraph::{
        algo::isomorphism::is_isomorphic,
        graph::{Graph, IndexType},
        Directed, EdgeType, Undirected,
    };
    use rand::{distributions::Uniform, prelude::*};
    use testing::GraphIter;

    use rand_xoshiro::Xoshiro256Plus;

    use log::debug;

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn randomize_labels<N, E, Ty, Ix>(
        g: Graph<N, E, Ty, Ix>,
        rng: &mut impl Rng,
    ) -> Graph<N, E, Ty, Ix>
    where
        Ty: EdgeType,
        Ix: IndexType,
    {
        use petgraph::visit::NodeIndexable;
        let mut res = Graph::with_capacity(g.node_count(), g.edge_count());
        let edges = Vec::from_iter(g.edge_references().map(|e| {
            let source = g.to_index(e.source());
            let target = g.to_index(e.target());
            (source, target)
        }));
        let (nodes, edge_wts) = g.into_nodes_edges();
        let mut nodes =
            Vec::from_iter(nodes.into_iter().map(|n| n.weight).enumerate());
        nodes.shuffle(rng);
        let mut relabel = vec![0; nodes.len()];
        for (new, (old, w)) in nodes.into_iter().enumerate() {
            res.add_node(w);
            relabel[old] = new;
        }
        let edges = izip!(edges, edge_wts).map(|((source, target), w)| {
            (relabel[source], relabel[target], w.weight)
        });
        for (source, target, w) in edges {
            res.add_edge(res.from_index(source), res.from_index(target), w);
        }
        res
    }

    #[test]
    fn random_canon_undirected() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let mut graphs = GraphIter::<Undirected>::default();
        graphs.edge_wt_distr = Uniform::from(0..=0);

        for g in graphs.take(1000) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            let g = g.into_canon();
            debug!("Canonical graph (from initial): {g:#?}");
            assert!(is_isomorphic(&g, &gg));
            let gg = gg.into_canon();
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            assert!(g.is_identical(&gg));
        }
    }

    #[test]
    fn random_canon_directed() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let mut graphs = GraphIter::<Directed>::default();
        graphs.edge_wt_distr = Uniform::from(0..=0);

        for g in graphs.take(700) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            let g = g.into_canon();
            debug!("Canonical graph (from initial): {g:#?}");
            assert!(is_isomorphic(&g, &gg));
            let gg = gg.into_canon();
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            assert!(g.is_identical(&gg));
        }
    }
}
