use crate::error::NautyError;
use crate::nauty_graph::DenseGraph;
use crate::nauty_graph::RawGraphData;
use crate::nauty_graph::SparseGraph;
use crate::nauty_graph::inv_perm;

use crate::autom::{
    AUTOM_GENERATORS, AutomGenerators, AutomStats, store_generator,
    store_generator_traces, undo_orbit_relabelling, undo_vertex_relabelling,
};
use std::os::raw::c_int;

use std::cmp::Ord;
use std::convert::Infallible;
use std::fmt::Debug;
use std::hash::Hash;

use nauty_Traces_sys::{
    FALSE, MTOOBIG, NTOOBIG, TRUE, densenauty, empty_graph, optionblk, statsblk,
};
use nauty_Traces_sys::{
    SG_FREE, Traces, TracesOptions, TracesStats, sparsegraph, sparsenauty,
};
use petgraph::graph::UnGraph;
use petgraph::{
    EdgeType,
    graph::{Graph, IndexType},
};

/// Find the canonical labelling for a graph
pub trait IntoCanon {
    fn into_canon(self) -> Self;
}

/// Try to find the canonical labelling for a graph
pub trait TryIntoCanon {
    type Error;

    fn try_into_canon(self) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// Use sparse nauty to find the canonical labelling
pub trait IntoCanonNautySparse {
    fn into_canon_nauty_sparse(self) -> Self;
}

/// Use sparse nauty to find the canonical labelling
pub trait TryIntoCanonNautySparse {
    type Error;

    fn try_into_canon_nauty_sparse(self) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// Use dense nauty to find the canonical labelling
pub trait IntoCanonNautyDense {
    fn into_canon_nauty_dense(self) -> Self;
}

/// Use dense nauty to find the canonical labelling
pub trait TryIntoCanonNautyDense {
    type Error;

    fn try_into_canon_nauty_dense(self) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// Use Traces to find the canonical labelling
pub trait IntoCanonTraces {
    fn into_canon_traces(self) -> Self;
}

/// Use Traces to find the canonical labelling
pub trait TryIntoCanonTraces {
    type Error;

    fn try_into_canon_traces(self) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

impl<N, E, Ty: EdgeType, Ix: IndexType> IntoCanon for Graph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: TryIntoCanon,
    <Graph<N, E, Ty, Ix> as TryIntoCanon>::Error: Debug,
{
    fn into_canon(self) -> Self {
        self.try_into_canon().unwrap()
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> TryIntoCanon for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
{
    type Error = NautyError;

    fn try_into_canon(self) -> Result<Self, Self::Error> {
        self.try_into_canon_nauty_dense()
    }
}

impl<N, E, Ty, Ix: IndexType> TryIntoCanonNautySparse for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
{
    type Error = Infallible;

    fn try_into_canon_nauty_sparse(self) -> Result<Self, Self::Error> {
        if self.node_count() == 0 {
            return Ok(self);
        }
        let mut options = if self.is_directed() {
            optionblk::default_sparse_digraph()
        } else {
            optionblk::default_sparse()
        };
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() { TRUE } else { FALSE };
        let mut stats = statsblk::default();
        let mut sg = SparseGraph::from(self);
        let mut orbits = vec![0; sg.g.v.len()];
        let mut cg = sparsegraph::default();
        unsafe {
            sparsenauty(
                &mut (&mut sg.g).into(),
                sg.nodes.lab.as_mut_ptr(),
                sg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            SG_FREE(&mut cg);
        }
        debug_assert_eq!(stats.errstatus, 0);
        Ok(sg.into())
    }
}

impl<N, E, Ty, Ix> IntoCanonNautySparse for Graph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: TryIntoCanonNautySparse,
    <Graph<N, E, Ty, Ix> as TryIntoCanonNautySparse>::Error: Debug,
{
    fn into_canon_nauty_sparse(self) -> Self {
        self.try_into_canon_nauty_sparse().unwrap()
    }
}

impl<N, E, Ty, Ix: IndexType> TryIntoCanonNautyDense for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
{
    type Error = NautyError;

    fn try_into_canon_nauty_dense(self) -> Result<Self, Self::Error> {
        use ::std::os::raw::c_int;
        use NautyError::*;

        if self.node_count() == 0 {
            return Ok(self);
        }
        let mut options = if self.is_directed() {
            optionblk::default_digraph()
        } else {
            optionblk::default()
        };
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() { TRUE } else { FALSE };
        let mut stats = statsblk::default();
        let mut dg = DenseGraph::from(self);
        let mut orbits = vec![0; dg.n];
        let mut cg = empty_graph(dg.m, dg.n);
        unsafe {
            densenauty(
                dg.g.as_mut_ptr(),
                dg.nodes.lab.as_mut_ptr(),
                dg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                dg.m as c_int,
                dg.n as c_int,
                cg.as_mut_ptr(),
            );
        }
        match stats.errstatus {
            0 => Ok(dg.into()),
            MTOOBIG => Err(MTooBig),
            NTOOBIG => Err(NTooBig),
            _ => unreachable!(),
        }
    }
}

impl<N, E, Ty, Ix> IntoCanonNautyDense for Graph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: TryIntoCanonNautyDense,
    <Graph<N, E, Ty, Ix> as TryIntoCanonNautyDense>::Error: Debug,
{
    fn into_canon_nauty_dense(self) -> Self {
        self.try_into_canon_nauty_dense().unwrap()
    }
}

impl<N, E, Ix: IndexType> TryIntoCanonTraces for UnGraph<N, E, Ix>
where
    N: Ord,
    E: Hash + Ord,
{
    type Error = Infallible;

    fn try_into_canon_traces(self) -> Result<Self, Self::Error> {
        if self.node_count() == 0 {
            return Ok(self);
        }
        let mut options = TracesOptions {
            getcanon: TRUE,
            defaultptn: FALSE,
            digraph: FALSE,
            ..Default::default()
        };
        let mut stats = TracesStats::default();
        let mut sg = SparseGraph::from(self);
        let mut orbits = vec![0; sg.g.v.len()];
        let mut cg = sparsegraph::default();
        unsafe {
            Traces(
                &mut (&mut sg.g).into(),
                sg.nodes.lab.as_mut_ptr(),
                sg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            SG_FREE(&mut cg);
        }
        debug_assert_eq!(stats.errstatus, 0);
        Ok(sg.into())
    }
}

impl<N, E, Ix> IntoCanonTraces for UnGraph<N, E, Ix>
where
    UnGraph<N, E, Ix>: TryIntoCanonTraces,
    <UnGraph<N, E, Ix> as TryIntoCanonTraces>::Error: Debug,
{
    fn into_canon_traces(self) -> Self {
        self.try_into_canon_traces().unwrap()
    }
}

/// A graph's canonical labelling, with the automorphisms found on the same run
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct CanonLabelling {
    /// Each vertex's canonical position, a permutation of `0..n`
    pub labelling: Vec<usize>,
    /// The generators, orbits, and group size found on the same run
    pub automorphisms: AutomGenerators,
}

// Rank the real vertices by canonical position, dropping the auxiliary
// vertices (which have no petgraph counterpart) so the result is a clean
// permutation of `0..n` in the petgraph's vertex labels
fn relabel_to_canonical(lab: &[c_int], relabel: &[usize]) -> Vec<usize> {
    let canon_pos = inv_perm(lab);
    let num_real = relabel.len();
    let mut order = Vec::from_iter(0..num_real);
    order.sort_by_key(|&v| canon_pos[relabel[v]]);
    let mut labelling = vec![0; num_real];
    for (rank, &v) in order.iter().enumerate() {
        labelling[v] = rank;
    }
    labelling
}

// Assemble the result of a getcanon run: take the generators the callback
// stored, and map the labelling and orbits back to the petgraph's vertex
// labels
fn assemble(
    lab: &[c_int],
    orbits: &[c_int],
    relabel: &[usize],
    stats: AutomStats,
) -> CanonLabelling {
    let generators =
        undo_vertex_relabelling(AUTOM_GENERATORS.with(|g| g.take()), relabel);
    CanonLabelling {
        labelling: relabel_to_canonical(lab, relabel),
        automorphisms: AutomGenerators {
            generators,
            orbits: undo_orbit_relabelling(orbits, relabel),
            stats,
        },
    }
}

/// Find the canonical labelling and automorphisms for a graph
pub trait IntoCanonLabelling {
    fn into_canon_labelling(self) -> CanonLabelling;
}

/// Try to find the canonical labelling and automorphisms for a graph
pub trait TryIntoCanonLabelling {
    type Error;

    fn try_into_canon_labelling(self) -> Result<CanonLabelling, Self::Error>;
}

/// Use sparse nauty to find the canonical labelling and automorphisms
pub trait IntoCanonLabellingNautySparse {
    fn into_canon_labelling_nauty_sparse(self) -> CanonLabelling;
}

/// Use sparse nauty to find the canonical labelling and automorphisms
pub trait TryIntoCanonLabellingNautySparse {
    type Error;

    fn try_into_canon_labelling_nauty_sparse(
        self,
    ) -> Result<CanonLabelling, Self::Error>;
}

/// Use dense nauty to find the canonical labelling and automorphisms
pub trait IntoCanonLabellingNautyDense {
    fn into_canon_labelling_nauty_dense(self) -> CanonLabelling;
}

/// Use dense nauty to find the canonical labelling and automorphisms
pub trait TryIntoCanonLabellingNautyDense {
    type Error;

    fn try_into_canon_labelling_nauty_dense(
        self,
    ) -> Result<CanonLabelling, Self::Error>;
}

/// Use Traces to find the canonical labelling and automorphisms
pub trait IntoCanonLabellingTraces {
    fn into_canon_labelling_traces(self) -> CanonLabelling;
}

/// Use Traces to find the canonical labelling and automorphisms
pub trait TryIntoCanonLabellingTraces {
    type Error;

    fn try_into_canon_labelling_traces(
        self,
    ) -> Result<CanonLabelling, Self::Error>;
}

impl<N, E, Ty: EdgeType, Ix: IndexType> IntoCanonLabelling
    for Graph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: TryIntoCanonLabelling,
    <Graph<N, E, Ty, Ix> as TryIntoCanonLabelling>::Error: Debug,
{
    fn into_canon_labelling(self) -> CanonLabelling {
        self.try_into_canon_labelling().unwrap()
    }
}

impl<N, E, Ty: EdgeType, Ix: IndexType> TryIntoCanonLabelling
    for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
{
    type Error = NautyError;

    fn try_into_canon_labelling(self) -> Result<CanonLabelling, Self::Error> {
        self.try_into_canon_labelling_nauty_dense()
    }
}

impl<N, E, Ty, Ix> IntoCanonLabellingNautySparse for Graph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: TryIntoCanonLabellingNautySparse,
    <Graph<N, E, Ty, Ix> as TryIntoCanonLabellingNautySparse>::Error: Debug,
{
    fn into_canon_labelling_nauty_sparse(self) -> CanonLabelling {
        self.try_into_canon_labelling_nauty_sparse().unwrap()
    }
}

impl<N, E, Ty, Ix: IndexType> TryIntoCanonLabellingNautySparse
    for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
{
    type Error = Infallible;

    fn try_into_canon_labelling_nauty_sparse(
        self,
    ) -> Result<CanonLabelling, Self::Error> {
        if self.node_count() == 0 {
            return Ok(CanonLabelling::default());
        }
        let mut options = if self.is_directed() {
            optionblk::default_sparse_digraph()
        } else {
            optionblk::default_sparse()
        };
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() { TRUE } else { FALSE };
        options.userautomproc = Some(store_generator);
        let mut stats = statsblk::default();
        let mut g = RawGraphData::from(self);
        let relabel = std::mem::take(&mut g.relabel);
        let mut sg = SparseGraph::from(g);
        let mut orbits = vec![0; sg.g.v.len()];
        let mut cg = sparsegraph::default();
        AUTOM_GENERATORS.with(|g| g.borrow_mut().clear());
        unsafe {
            sparsenauty(
                &mut (&mut sg.g).into(),
                sg.nodes.lab.as_mut_ptr(),
                sg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            SG_FREE(&mut cg);
        }
        debug_assert_eq!(stats.errstatus, 0);
        Ok(assemble(&sg.nodes.lab, &orbits, &relabel, stats.into()))
    }
}

impl<N, E, Ty, Ix> IntoCanonLabellingNautyDense for Graph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: TryIntoCanonLabellingNautyDense,
    <Graph<N, E, Ty, Ix> as TryIntoCanonLabellingNautyDense>::Error: Debug,
{
    fn into_canon_labelling_nauty_dense(self) -> CanonLabelling {
        self.try_into_canon_labelling_nauty_dense().unwrap()
    }
}

impl<N, E, Ty, Ix: IndexType> TryIntoCanonLabellingNautyDense
    for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
{
    type Error = NautyError;

    fn try_into_canon_labelling_nauty_dense(
        self,
    ) -> Result<CanonLabelling, Self::Error> {
        use NautyError::*;

        if self.node_count() == 0 {
            return Ok(CanonLabelling::default());
        }
        let mut options = if self.is_directed() {
            optionblk::default_digraph()
        } else {
            optionblk::default()
        };
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() { TRUE } else { FALSE };
        options.userautomproc = Some(store_generator);
        let mut stats = statsblk::default();
        let mut g = RawGraphData::from(self);
        let relabel = std::mem::take(&mut g.relabel);
        let mut dg = DenseGraph::from(g);
        let mut orbits = vec![0; dg.n];
        let mut cg = empty_graph(dg.m, dg.n);
        AUTOM_GENERATORS.with(|g| g.borrow_mut().clear());
        unsafe {
            densenauty(
                dg.g.as_mut_ptr(),
                dg.nodes.lab.as_mut_ptr(),
                dg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                dg.m as c_int,
                dg.n as c_int,
                cg.as_mut_ptr(),
            );
        }
        match stats.errstatus {
            0 => Ok(assemble(&dg.nodes.lab, &orbits, &relabel, stats.into())),
            MTOOBIG => Err(MTooBig),
            NTOOBIG => Err(NTooBig),
            _ => unreachable!(),
        }
    }
}

impl<N, E, Ix> IntoCanonLabellingTraces for UnGraph<N, E, Ix>
where
    UnGraph<N, E, Ix>: TryIntoCanonLabellingTraces,
    <UnGraph<N, E, Ix> as TryIntoCanonLabellingTraces>::Error: Debug,
{
    fn into_canon_labelling_traces(self) -> CanonLabelling {
        self.try_into_canon_labelling_traces().unwrap()
    }
}

impl<N, E, Ix: IndexType> TryIntoCanonLabellingTraces for UnGraph<N, E, Ix>
where
    N: Ord,
    E: Hash + Ord,
{
    type Error = Infallible;

    fn try_into_canon_labelling_traces(
        self,
    ) -> Result<CanonLabelling, Self::Error> {
        if self.node_count() == 0 {
            return Ok(CanonLabelling::default());
        }
        let mut options = TracesOptions {
            getcanon: TRUE,
            defaultptn: FALSE,
            digraph: FALSE,
            userautomproc: Some(store_generator_traces),
            ..Default::default()
        };
        let mut stats = TracesStats::default();
        let mut g = RawGraphData::from(self);
        let relabel = std::mem::take(&mut g.relabel);
        let mut sg = SparseGraph::from(g);
        let mut orbits = vec![0; sg.g.v.len()];
        let mut cg = sparsegraph::default();
        AUTOM_GENERATORS.with(|g| g.borrow_mut().clear());
        unsafe {
            Traces(
                &mut (&mut sg.g).into(),
                sg.nodes.lab.as_mut_ptr(),
                sg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            SG_FREE(&mut cg);
        }
        debug_assert_eq!(stats.errstatus, 0);
        Ok(assemble(&sg.nodes.lab, &orbits, &relabel, stats.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::cmp::IsIdentical;
    use super::*;
    use crate::autom::TryIntoAutomGroup;
    use crate::nauty_graph;
    use petgraph::visit::EdgeRef;
    use petgraph::{
        Directed, Undirected,
        algo::isomorphism::is_isomorphic,
        graph::{Graph, UnGraph},
    };
    use rand::prelude::*;
    use std::collections::BTreeSet;
    use testing::{GraphIter, randomize_labels};

    use rand_xoshiro::Xoshiro256Plus;

    use log::debug;

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn triangle() {
        log_init();

        use petgraph::visit::NodeIndexable;
        let mut g1 = UnGraph::<u8, ()>::from_edges([
            (0, 0),
            (1, 1),
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 2),
        ]);
        *g1.node_weight_mut(g1.from_index(0)).unwrap() = 2;
        *g1.node_weight_mut(g1.from_index(1)).unwrap() = 2;
        let g1 = g1.into_canon();

        let mut g2 = UnGraph::<u8, ()>::from_edges([
            (0, 0),
            (1, 1),
            (0, 1),
            (0, 2),
            (0, 2),
            (1, 2),
        ]);
        *g2.node_weight_mut(g2.from_index(0)).unwrap() = 2;
        *g2.node_weight_mut(g2.from_index(1)).unwrap() = 2;
        let g2 = g2.into_canon();

        assert!(g1.is_identical(&g2));
    }

    #[test]
    fn random_canon_nauty_sparse_undirected() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let graphs = GraphIter::<Undirected>::default();

        for g in graphs.take(1000) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            let g = g.into_canon_nauty_sparse();
            debug!("Canonical graph (from initial): {g:#?}");
            assert!(is_isomorphic(&g, &gg));
            let gg = gg.into_canon_nauty_sparse();
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            assert!(g.is_identical(&gg));
        }
    }

    #[test]
    fn random_canon_nauty_sparse_directed() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let graphs = GraphIter::<Directed>::default();

        for g in graphs.take(700) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            let g = g.into_canon_nauty_sparse();
            debug!("Canonical graph (from initial): {g:#?}");
            assert!(is_isomorphic(&g, &gg));
            let gg = gg.into_canon_nauty_sparse();
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            assert!(g.is_identical(&gg));
        }
    }

    #[test]
    fn random_canon_nauty_dense_undirected() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let graphs = GraphIter::<Undirected>::default();

        for g in graphs.take(1000) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            let g = g.into_canon_nauty_dense();
            debug!("Canonical graph (from initial): {g:#?}");
            assert!(is_isomorphic(&g, &gg));
            let gg = gg.into_canon_nauty_dense();
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            assert!(g.is_identical(&gg));
        }
    }

    #[test]
    fn random_canon_nauty_dense_directed() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let graphs = GraphIter::<Directed>::default();

        for g in graphs.take(700) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            let g = g.into_canon_nauty_dense();
            debug!("Canonical graph (from initial): {g:#?}");
            assert!(is_isomorphic(&g, &gg));
            let gg = gg.into_canon_nauty_dense();
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            assert!(g.is_identical(&gg));
        }
    }

    #[test]
    fn random_canon_traces_undirected() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let graphs = GraphIter::<Undirected>::default();

        for g in graphs.take(1000) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            let g = g.into_canon_traces();
            debug!("Canonical graph (from initial): {g:#?}");
            assert!(is_isomorphic(&g, &gg));
            let gg = gg.into_canon_traces();
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert!(is_isomorphic(&g, &gg));
            assert!(g.is_identical(&gg));
        }
    }

    #[test]
    fn asym() {
        log_init();

        let g = UnGraph::<(), ()>::from_edges([(0, 1), (1, 0)]);
        assert!(is_isomorphic(&g, &g.clone().into_canon()));
    }

    #[test]
    fn empty() {
        log_init();

        let g = Graph::<(), (), _>::new_undirected();
        assert!(g.is_identical(&g.clone().into_canon()));
    }

    fn generated_group(gens: &[Vec<usize>], n: usize) -> BTreeSet<Vec<usize>> {
        let id = Vec::from_iter(0..n);
        let mut seen = BTreeSet::from([id.clone()]);
        let mut todo = vec![id];
        while let Some(p) = todo.pop() {
            for g in gens {
                let q = Vec::from_iter(p.iter().map(|&i| g[i]));
                if seen.insert(q.clone()) {
                    todo.push(q);
                }
            }
        }
        seen
    }

    fn apply_perm<N, E, Ty: EdgeType, Ix: IndexType>(
        g: Graph<N, E, Ty, Ix>,
        perm: Vec<usize>,
    ) -> Graph<N, E, Ty, Ix> {
        use petgraph::visit::NodeIndexable;

        let mut res = Graph::with_capacity(g.node_count(), g.edge_count());
        let edges = Vec::from_iter(g.edge_references().map(|e| {
            let source = perm[g.to_index(e.source())];
            let target = perm[g.to_index(e.target())];
            (source, target)
        }));
        let (nodes, edge_wts) = g.into_nodes_edges();
        let mut nodes = Vec::from_iter(nodes.into_iter().map(|n| n.weight));
        nauty_graph::apply_perm(&mut nodes, perm);
        for node in nodes {
            res.add_node(node);
        }
        let edges = edges.into_iter().zip(edge_wts);
        for ((source, target), w) in edges {
            res.add_edge(
                res.from_index(source),
                res.from_index(target),
                w.weight,
            );
        }
        res
    }

    // A graph reduced to its node weights (in vertex order) and its
    // orientation-normalised, sorted edge set. Two graphs with the same
    // labelling compare equal here regardless of edge insertion order,
    // which `is_identical` is sensitive to but a labelling does not fix.
    fn canon_key<N, E, Ty, Ix>(
        g: &Graph<N, E, Ty, Ix>,
    ) -> (Vec<N>, Vec<(usize, usize, E)>)
    where
        N: Clone + Ord,
        E: Clone + Ord,
        Ty: EdgeType,
        Ix: IndexType,
    {
        use petgraph::visit::NodeIndexable;
        let directed = g.is_directed();
        let nodes = Vec::from_iter(g.node_weights().cloned());
        let mut edges = Vec::from_iter(g.edge_references().map(|e| {
            let s = g.to_index(e.source());
            let t = g.to_index(e.target());
            let (s, t) = if !directed && s > t { (t, s) } else { (s, t) };
            (s, t, e.weight().clone())
        }));
        edges.sort();
        (nodes, edges)
    }

    #[test]
    fn random_canon_labelling_nauty_sparse_undirected() {
        log_init();

        for g in GraphIter::<Undirected>::default().take(1000) {
            let cl = g.clone().try_into_canon_labelling_nauty_sparse().unwrap();
            let canon = g.clone().into_canon_nauty_sparse();
            assert_eq!(
                canon_key(&apply_perm(g, cl.labelling)),
                canon_key(&canon)
            );
        }
    }

    #[test]
    fn random_canon_labelling_nauty_sparse_directed() {
        log_init();

        for g in GraphIter::<Directed>::default().take(700) {
            let cl = g.clone().try_into_canon_labelling_nauty_sparse().unwrap();
            let canon = g.clone().into_canon_nauty_sparse();
            assert_eq!(
                canon_key(&apply_perm(g, cl.labelling)),
                canon_key(&canon)
            );
        }
    }

    #[test]
    fn random_canon_labelling_nauty_dense_undirected() {
        log_init();

        for g in GraphIter::<Undirected>::default().take(1000) {
            let cl = g.clone().try_into_canon_labelling_nauty_dense().unwrap();
            let canon = g.clone().into_canon_nauty_dense();
            assert_eq!(
                canon_key(&apply_perm(g, cl.labelling)),
                canon_key(&canon)
            );
        }
    }

    #[test]
    fn random_canon_labelling_nauty_dense_directed() {
        log_init();

        for g in GraphIter::<Directed>::default().take(700) {
            let cl = g.clone().try_into_canon_labelling_nauty_dense().unwrap();
            let canon = g.clone().into_canon_nauty_dense();
            assert_eq!(
                canon_key(&apply_perm(g, cl.labelling)),
                canon_key(&canon)
            );
        }
    }

    #[test]
    fn random_canon_labelling_traces_undirected() {
        log_init();

        // Traces exercises `store_generator_traces`; also check the collected
        // generators generate the whole automorphism group
        for g in GraphIter::<Undirected>::default().take(1000) {
            let n = g.node_count();
            let cl = g.clone().try_into_canon_labelling_traces().unwrap();
            let group = g.clone().try_into_autom_group().unwrap();
            assert_eq!(
                generated_group(&cl.automorphisms.generators, n),
                BTreeSet::from_iter(group.0)
            );
            let canon = g.clone().into_canon_traces();
            assert_eq!(
                canon_key(&apply_perm(g, cl.labelling)),
                canon_key(&canon)
            );
        }
    }

    #[test]
    fn empty_canon_labelling() {
        log_init();

        let g = Graph::<(), (), _>::new_undirected();
        let cl = g.try_into_canon_labelling().unwrap();
        assert!(cl.labelling.is_empty());
        assert!(cl.automorphisms.generators.is_empty());
        assert!(cl.automorphisms.orbits.is_empty());
    }
}
