use crate::error::NautyError;
use crate::graph::DenseGraph;
#[cfg(feature = "libc")]
use crate::graph::SparseGraph;

use std::cmp::Ord;
#[cfg(feature = "libc")]
use std::convert::Infallible;
use std::hash::Hash;
use std::fmt::Debug;

use nauty_Traces_sys::{
    densenauty, empty_graph, optionblk, statsblk, MTOOBIG, NTOOBIG,
    FALSE, TRUE
};
#[cfg(feature = "libc")]
use nauty_Traces_sys::{
    SG_FREE, sparsegraph, sparsenauty, Traces, TracesOptions,
    TracesStats,
};
use petgraph::{
    graph::{IndexType, Graph},
    EdgeType,
};
#[cfg(feature = "libc")]
use petgraph::graph::UnGraph;

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

#[cfg(feature = "libc")]
/// Use sparse nauty to find the canonical labelling
pub trait IntoCanonNautySparse {
    fn into_canon_nauty_sparse(self) -> Self;
}

#[cfg(feature = "libc")]
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

pub trait TryIntoCanonNautyDense {
    type Error;

    fn try_into_canon_nauty_dense(self) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// Use Traces to find the canonical labelling
#[cfg(feature = "libc")]
pub trait IntoCanonTraces {
    fn into_canon_traces(self) -> Self;
}

/// Use Traces to find the canonical labelling
#[cfg(feature = "libc")]
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

#[cfg(feature = "libc")]
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
        let mut options = optionblk::default_sparse();
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() {
            TRUE
        } else {
            FALSE
        };
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

#[cfg(feature = "libc")]
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
        use NautyError::*;
        use ::std::os::raw::c_int;

        if self.node_count() == 0 {
            return Ok(self);
        }
        let mut options = optionblk::default();
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() {
            TRUE
        } else {
            FALSE
        };
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
            _ => unreachable!()
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

#[cfg(feature = "libc")]
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

#[cfg(feature = "libc")]
impl<N, E, Ix> IntoCanonTraces for UnGraph<N, E, Ix>
where
    UnGraph<N, E, Ix>: TryIntoCanonTraces,
    <UnGraph<N, E, Ix> as TryIntoCanonTraces>::Error: Debug,
{
    fn into_canon_traces(self) -> Self {
        self.try_into_canon_traces().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::super::cmp::IsIdentical;
    use super::*;
    use petgraph::{
        algo::isomorphism::is_isomorphic,
        graph::{Graph, UnGraph},
        Directed, Undirected,
    };
    use rand::prelude::*;
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

    #[cfg(feature = "libc")]
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

    #[cfg(feature = "libc")]
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

    #[cfg(feature = "libc")]
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
    fn empty() {
        log_init();

        let g = Graph::<(), (), _>::new_undirected();
        assert!(g.is_identical(&g.clone().into_canon()));
    }
}
