use std::cmp::Ord;
use std::convert::From;
use std::convert::Infallible;
use std::hash::Hash;

use crate::error::NautyError;
use crate::nauty_graph::DenseGraph;
use crate::nauty_graph::SparseGraph;

use nauty_Traces_sys::{
    densenauty, optionblk, statsblk, FALSE, MTOOBIG, NTOOBIG, TRUE,
};
use nauty_Traces_sys::{sparsenauty, Traces, TracesOptions, TracesStats};
use petgraph::{
    graph::{Graph, IndexType},
    EdgeType,
};

#[deprecated(note = "use `TryIntoAutomStats` instead")]
pub trait TryIntoAutom {
    type Error;

    fn try_into_autom(self) -> Result<AutomStats, Self::Error>;
}

#[allow(deprecated)]
impl<T: TryIntoAutomStats> TryIntoAutom for T {
    type Error = <T as TryIntoAutomStats>::Error;

    fn try_into_autom(self) -> Result<AutomStats, Self::Error> {
        self.try_into_autom_stats()
    }
}

/// Information on automorphism group of a graph
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub struct AutomStats {
    /// The size of the automorphism group is approximately `grpsize_base` * 10.pow(`grpsize_exp`)
    pub grpsize_base: f64,
    /// The size of the automorphism group is approximately `grpsize_base` * 10.pow(`grpsize_exp`)
    pub grpsize_exp: u32,
    /// Number of orbits of the automorphism group
    pub num_orbits: u32,
    /// Number of generators
    pub num_generators: u32,
}

impl AutomStats {
    /// The size of the automorphism group
    pub fn grpsize(&self) -> f64 {
        self.grpsize_base * 10f64.powi(self.grpsize_exp as i32)
    }
}

impl From<TracesStats> for AutomStats {
    fn from(o: TracesStats) -> Self {
        Self {
            grpsize_base: o.grpsize1,
            grpsize_exp: o.grpsize2 as u32,
            num_orbits: o.numorbits as u32,
            num_generators: o.numgenerators as u32,
        }
    }
}

impl From<statsblk> for AutomStats {
    fn from(o: statsblk) -> Self {
        Self {
            grpsize_base: o.grpsize1,
            grpsize_exp: o.grpsize2 as u32,
            num_orbits: o.numorbits as u32,
            num_generators: o.numgenerators as u32,
        }
    }
}

/// Statistics for a graph's automorphism group
pub trait TryIntoAutomStats {
    type Error;

    fn try_into_autom_stats(self) -> Result<AutomStats, Self::Error>;
}

/// Statistics for a graph's automorphism group using sparse nauty
pub trait TryIntoAutomStatsNautySparse {
    type Error;

    fn try_into_autom_stats_nauty_sparse(self) -> Result<AutomStats, Self::Error>;
}

/// Statistics for a graph's automorphism group using dense nauty
pub trait TryIntoAutomStatsNautyDense {
    type Error;

    fn try_into_autom_stats_nauty_dense(self) -> Result<AutomStats, Self::Error>;
}

/// Statistics for a graph's automorphism group using Traces
pub trait TryIntoAutomStatsTraces {
    type Error;

    fn try_into_autom_stats_traces(self) -> Result<AutomStats, Self::Error>;
}

impl<N, E, Ty, Ix> TryIntoAutomStats for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = NautyError;

    fn try_into_autom_stats(self) -> Result<AutomStats, Self::Error> {
        self.try_into_autom_stats_nauty_dense()
    }
}

impl<N, E, Ty, Ix> TryIntoAutomStatsNautySparse for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = Infallible;

    fn try_into_autom_stats_nauty_sparse(self) -> Result<AutomStats, Self::Error> {
        let mut options = optionblk::default_sparse();
        options.getcanon = FALSE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() { TRUE } else { FALSE };
        let mut stats = statsblk::default();
        let mut sg = SparseGraph::from(self);
        let mut orbits = vec![0; sg.g.v.len()];
        unsafe {
            sparsenauty(
                &mut (&mut sg.g).into(),
                sg.nodes.lab.as_mut_ptr(),
                sg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                std::ptr::null_mut(),
            );
        }
        debug_assert_eq!(stats.errstatus, 0);
        Ok(stats.into())
    }
}

impl<N, E, Ty, Ix> TryIntoAutomStatsNautyDense for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = NautyError;

    fn try_into_autom_stats_nauty_dense(self) -> Result<AutomStats, Self::Error> {
        use ::std::os::raw::c_int;
        use NautyError::*;

        let mut options = optionblk {
            getcanon: FALSE,
            defaultptn: FALSE,
            digraph: if self.is_directed() { TRUE } else { FALSE },
            ..Default::default()
        };
        let mut stats = statsblk::default();
        let mut dg = DenseGraph::from(self);
        let mut orbits = vec![0; dg.n];
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
                std::ptr::null_mut(),
            );
        }
        match stats.errstatus {
            0 => Ok(stats.into()),
            MTOOBIG => Err(MTooBig),
            NTOOBIG => Err(NTooBig),
            _ => unreachable!(),
        }
    }
}

impl<N, E, Ty, Ix> TryIntoAutomStatsTraces for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = Infallible;

    fn try_into_autom_stats_traces(self) -> Result<AutomStats, Self::Error> {
        let mut options = TracesOptions {
            getcanon: FALSE,
            defaultptn: FALSE,
            digraph: TRUE,
            ..Default::default()
        };
        let mut stats = TracesStats::default();
        let mut sg = SparseGraph::from(self);
        let mut orbits = vec![0; sg.g.v.len()];
        unsafe {
            Traces(
                &mut (&mut sg.g).into(),
                sg.nodes.lab.as_mut_ptr(),
                sg.nodes.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                std::ptr::null_mut(),
            );
        }
        debug_assert_eq!(stats.errstatus, 0);
        Ok(stats.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::{graph::DiGraph, Undirected};

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn simple() {
        log_init();

        use petgraph::visit::NodeIndexable;
        let g = DiGraph::<u8, ()>::from_edges([(0, 1)]);
        let autom = g.clone().try_into_autom_stats().unwrap();
        assert_eq!(autom.grpsize_base, 1.);
        assert_eq!(autom.grpsize_exp, 0);
        let g = g.into_edge_type::<Undirected>();
        let autom = g.clone().try_into_autom_stats().unwrap();
        assert_eq!(autom.grpsize_base, 2.);
        assert_eq!(autom.grpsize_exp, 0);
        let mut g = g;
        *g.node_weight_mut(g.from_index(0)).unwrap() = 2;
        let autom = g.clone().try_into_autom_stats().unwrap();
        assert_eq!(autom.grpsize_base, 1.);
        assert_eq!(autom.grpsize_exp, 0);
    }

    #[test]
    fn triangle() {
        log_init();

        use petgraph::visit::EdgeIndexable;
        let g = DiGraph::<(), u8>::from_edges([(0, 1), (1, 2), (2, 0)]);
        let autom = g.clone().try_into_autom_stats().unwrap();
        assert_eq!(autom.grpsize_base, 3.);
        assert_eq!(autom.grpsize_exp, 0);
        let g = g.into_edge_type::<Undirected>();
        let autom = g.clone().try_into_autom_stats().unwrap();
        assert_eq!(autom.grpsize_base, 6.);
        assert_eq!(autom.grpsize_exp, 0);
        let mut g = g;
        *g.edge_weight_mut(g.from_index(0)).unwrap() = 2;
        let autom = g.clone().try_into_autom_stats().unwrap();
        assert_eq!(autom.grpsize_base, 2.);
        assert_eq!(autom.grpsize_exp, 0);
    }
}
