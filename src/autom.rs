use std::cell::RefCell;
use std::cmp::Ord;
use std::convert::From;
use std::convert::Infallible;
use std::hash::Hash;
use std::ops::Deref;
use std::ops::DerefMut;
use std::os::raw::c_int;
use std::slice;

use crate::error::NautyError;
use crate::nauty_graph::DenseGraph;
use crate::nauty_graph::RawGraphData;
use crate::nauty_graph::SparseGraph;
use crate::nauty_graph::inv_perm;

use nauty_Traces_sys::allgroup;
use nauty_Traces_sys::groupautomproc;
use nauty_Traces_sys::grouplevelproc;
use nauty_Traces_sys::groupptr;
use nauty_Traces_sys::makecosetreps;
use nauty_Traces_sys::{
    FALSE, MTOOBIG, NTOOBIG, TRUE, densenauty, optionblk, statsblk,
};
use nauty_Traces_sys::{Traces, TracesOptions, TracesStats, sparsenauty};
use petgraph::{
    EdgeType,
    graph::{Graph, IndexType},
};

/// A graph's complete automorphism group
///
/// Each element`perm` of the contained vector corresponds to an
/// automorphism, i.e. a permutation of vertices. Applying the
/// permutation corresponds to replacing each node index `n` in a
/// graph `g` by `g.from_index(perm[g.to_index(n)])`
#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AutomGroup(pub Vec<Vec<usize>>);

impl Deref for AutomGroup {
    type Target = Vec<Vec<usize>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for AutomGroup {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Determine all elements of a graph's automorphism group
pub trait TryIntoAutomGroup {
    type Error;

    fn try_into_autom_group(self) -> Result<AutomGroup, Self::Error>;
}

/// Determine all elements of a graph's automorphism group using sparse nauty
pub trait TryIntoAutomGroupNautySparse {
    type Error;

    fn try_into_autom_group_nauty_sparse(
        self,
    ) -> Result<AutomGroup, Self::Error>;
}

/// Determine all elements of a graph's automorphism group using dense nauty
pub trait TryIntoAutomGroupNautyDense {
    type Error;

    fn try_into_autom_group_nauty_dense(
        self,
    ) -> Result<AutomGroup, Self::Error>;
}

impl<N, E, Ty, Ix> TryIntoAutomGroup for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = NautyError;

    fn try_into_autom_group(self) -> Result<AutomGroup, Self::Error> {
        self.try_into_autom_group_nauty_dense()
    }
}

impl<N, E, Ty, Ix> TryIntoAutomGroupNautySparse for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = Infallible;

    fn try_into_autom_group_nauty_sparse(
        self,
    ) -> Result<AutomGroup, Self::Error> {
        if self.node_count() == 0 {
            return Ok(AutomGroup(vec![vec![]]))
        }
        let mut options = optionblk::default_sparse();
        options.getcanon = FALSE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() { TRUE } else { FALSE };
        options.userautomproc = Some(groupautomproc);
        options.userlevelproc = Some(grouplevelproc);
        let mut stats = statsblk::default();
        let mut g = RawGraphData::from(self);
        // remember how the vertex labels were changed so we can undo it
        // when converting back to a petgraph
        let relabel = std::mem::take(&mut g.relabel);
        let mut sg = SparseGraph::from(g);
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
            let group = groupptr(FALSE);
            makecosetreps(group);
            allgroup(group, Some(store_perm));
        }
        debug_assert_eq!(stats.errstatus, 0);
        let res = AUTOM_GROUP.with(|g| g.take());
        let res = undo_vertex_relabelling(res, &relabel);
        Ok(AutomGroup(res))
    }
}

impl<N, E, Ty, Ix> TryIntoAutomGroupNautyDense for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = NautyError;

    fn try_into_autom_group_nauty_dense(
        self,
    ) -> Result<AutomGroup, Self::Error> {
        use NautyError::*;

        if self.node_count() == 0 {
            return Ok(AutomGroup(vec![vec![]]))
        }
        let mut options = optionblk {
            getcanon: FALSE,
            defaultptn: FALSE,
            digraph: if self.is_directed() { TRUE } else { FALSE },
            userautomproc: Some(groupautomproc),
            userlevelproc: Some(grouplevelproc),
            ..Default::default()
        };
        let mut stats = statsblk::default();
        let mut g = RawGraphData::from(self);
        // remember how the vertex labels were changed so we can undo it
        // when converting back to a petgraph
        let relabel = std::mem::take(&mut g.relabel);
        let mut dg = DenseGraph::from(g);
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
            let group = groupptr(FALSE);
            makecosetreps(group);
            allgroup(group, Some(store_perm));
        }
        match stats.errstatus {
            0 => {
                let res = AUTOM_GROUP.with(|g| g.take());
                let res = undo_vertex_relabelling(res, &relabel);
                Ok(AutomGroup(res))
            }
            MTOOBIG => Err(MTooBig),
            NTOOBIG => Err(NTooBig),
            _ => unreachable!(),
        }
    }
}

pub(crate) fn undo_vertex_relabelling(
    autom: Vec<Vec<c_int>>,
    relabel: &[usize],
) -> Vec<Vec<usize>> {
    let inv_relabel = inv_perm(relabel);
    let mut res = Vec::with_capacity(autom.len());
    for perm in autom {
        let mut res_perm = vec![0; inv_relabel.len()];
        for (from, to) in perm.into_iter().take(inv_relabel.len()).enumerate() {
            res_perm[inv_relabel[from]] = inv_relabel[to as usize];
        }
        res.push(res_perm);
    }
    res.sort_unstable();
    res.dedup();
    res
}

// Ideally, the signature would include a data pointer, which would
// allow us to store the output locally. Since nauty doesn't include
// any data pointer, we have to resort to a global variable.
extern "C" fn store_perm(p: *mut c_int, n: i32) {
    let perm = unsafe { slice::from_raw_parts(p, n as usize) };
    AUTOM_GROUP.with(|g| g.borrow_mut().push(perm.to_vec()));
}

thread_local! {
    static AUTOM_GROUP: RefCell<Vec<Vec<c_int>>> = const { RefCell::new(Vec::new()) };
}

// nauty's `userautomproc`, called once for each generator found in the search
pub(crate) extern "C" fn store_generator(
    _count: c_int,
    perm: *mut c_int,
    _orbits: *mut c_int,
    _numorbits: c_int,
    _stabvertex: c_int,
    n: c_int,
) {
    let perm = unsafe { slice::from_raw_parts(perm, n as usize) };
    AUTOM_GENERATORS.with(|g| g.borrow_mut().push(perm.to_vec()));
}

// Traces' `userautomproc` has a different, three-argument signature
pub(crate) extern "C" fn store_generator_traces(
    _count: c_int,
    perm: *mut c_int,
    n: c_int,
) {
    let perm = unsafe { slice::from_raw_parts(perm, n as usize) };
    AUTOM_GENERATORS.with(|g| g.borrow_mut().push(perm.to_vec()));
}

thread_local! {
    pub(crate) static AUTOM_GENERATORS: RefCell<Vec<Vec<c_int>>> =
        const { RefCell::new(Vec::new()) };
}

// nauty's orbit array (nauty numbering) mapped to one representative per
// petgraph vertex, so two vertices share a value iff they share an orbit
pub(crate) fn undo_orbit_relabelling(
    orbits: &[c_int],
    relabel: &[usize],
) -> Vec<usize> {
    let inv_relabel = inv_perm(relabel);
    relabel
        .iter()
        .map(|r| inv_relabel[orbits[*r] as usize])
        .collect()
}

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

    fn try_into_autom_stats_nauty_sparse(
        self,
    ) -> Result<AutomStats, Self::Error>;
}

/// Statistics for a graph's automorphism group using dense nauty
pub trait TryIntoAutomStatsNautyDense {
    type Error;

    fn try_into_autom_stats_nauty_dense(
        self,
    ) -> Result<AutomStats, Self::Error>;
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

    fn try_into_autom_stats_nauty_sparse(
        self,
    ) -> Result<AutomStats, Self::Error> {
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

    fn try_into_autom_stats_nauty_dense(
        self,
    ) -> Result<AutomStats, Self::Error> {
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

/// A generating set of a graph's automorphism group, with its orbits and size
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct AutomGenerators {
    /// A generating set; each element a permutation of vertices
    pub generators: Vec<Vec<usize>>,
    /// Orbit representative per vertex; two share a value iff in one orbit
    pub orbits: Vec<usize>,
    /// The group size and the orbit and generator counts
    pub stats: AutomStats,
}

impl AutomGenerators {
    // Assemble a run's generators and orbits, mapping them back to the
    // petgraph's vertex labels
    pub(crate) fn from_run(
        generators: Vec<Vec<c_int>>,
        orbits: &[c_int],
        relabel: &[usize],
        stats: AutomStats,
    ) -> Self {
        Self {
            generators: undo_vertex_relabelling(generators, relabel),
            orbits: undo_orbit_relabelling(orbits, relabel),
            stats,
        }
    }
}

/// Determine a generating set of a graph's automorphism group
pub trait TryIntoAutomGenerators {
    type Error;

    fn try_into_autom_generators(self) -> Result<AutomGenerators, Self::Error>;
}

/// Determine a generating set of a graph's automorphism group using sparse nauty
pub trait TryIntoAutomGeneratorsNautySparse {
    type Error;

    fn try_into_autom_generators_nauty_sparse(
        self,
    ) -> Result<AutomGenerators, Self::Error>;
}

/// Determine a generating set of a graph's automorphism group using dense nauty
pub trait TryIntoAutomGeneratorsNautyDense {
    type Error;

    fn try_into_autom_generators_nauty_dense(
        self,
    ) -> Result<AutomGenerators, Self::Error>;
}

impl<N, E, Ty, Ix> TryIntoAutomGenerators for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = NautyError;

    fn try_into_autom_generators(self) -> Result<AutomGenerators, Self::Error> {
        self.try_into_autom_generators_nauty_dense()
    }
}

impl<N, E, Ty, Ix> TryIntoAutomGeneratorsNautySparse for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = Infallible;

    fn try_into_autom_generators_nauty_sparse(
        self,
    ) -> Result<AutomGenerators, Self::Error> {
        let mut options = optionblk::default_sparse();
        options.getcanon = FALSE;
        options.defaultptn = FALSE;
        options.digraph = if self.is_directed() { TRUE } else { FALSE };
        options.userautomproc = Some(store_generator);
        let mut stats = statsblk::default();
        let mut g = RawGraphData::from(self);
        // remember how the vertex labels were changed so the generators and
        // orbits can be converted back to the petgraph
        let relabel = std::mem::take(&mut g.relabel);
        let mut sg = SparseGraph::from(g);
        let mut orbits = vec![0; sg.g.v.len()];
        // filled by the run's callback and taken out below; must be empty here
        debug_assert!(AUTOM_GENERATORS.with(|g| g.borrow().is_empty()));
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
        let generators = AUTOM_GENERATORS.with(|g| g.take());
        Ok(AutomGenerators::from_run(
            generators,
            &orbits,
            &relabel,
            stats.into(),
        ))
    }
}

impl<N, E, Ty, Ix> TryIntoAutomGeneratorsNautyDense for Graph<N, E, Ty, Ix>
where
    N: Ord,
    E: Hash + Ord,
    Ty: EdgeType,
    Ix: IndexType,
{
    type Error = NautyError;

    fn try_into_autom_generators_nauty_dense(
        self,
    ) -> Result<AutomGenerators, Self::Error> {
        use NautyError::*;

        let mut options = optionblk {
            getcanon: FALSE,
            defaultptn: FALSE,
            digraph: if self.is_directed() { TRUE } else { FALSE },
            userautomproc: Some(store_generator),
            ..Default::default()
        };
        let mut stats = statsblk::default();
        let mut g = RawGraphData::from(self);
        // remember how the vertex labels were changed so the generators and
        // orbits can be converted back to the petgraph
        let relabel = std::mem::take(&mut g.relabel);
        let mut dg = DenseGraph::from(g);
        let mut orbits = vec![0; dg.n];
        // filled by the run's callback and taken out below; must be empty here
        debug_assert!(AUTOM_GENERATORS.with(|g| g.borrow().is_empty()));
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
        let generators = AUTOM_GENERATORS.with(|g| g.take());
        match stats.errstatus {
            0 => Ok(AutomGenerators::from_run(
                generators,
                &orbits,
                &relabel,
                stats.into(),
            )),
            MTOOBIG => Err(MTooBig),
            NTOOBIG => Err(NTooBig),
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::CanonGraph;
    use crate::test_util::{apply_perm, generated_group};

    use super::*;
    use log::debug;
    use petgraph::{Directed, Undirected, graph::DiGraph};
    use std::collections::{BTreeMap, BTreeSet};
    use testing::GraphIter;

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn simple_stats() {
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
    fn empty_stats() {
        log_init();

        let g = Graph::<(), ()>::new();
        let autom = g.clone().try_into_autom_stats().unwrap();
        assert_eq!(autom.grpsize_base, 1.);
        assert_eq!(autom.grpsize_exp, 0);
        let autom = g.clone().try_into_autom_stats_nauty_dense().unwrap();
        assert_eq!(autom.grpsize_base, 1.);
        assert_eq!(autom.grpsize_exp, 0);
        let autom = g.try_into_autom_stats_nauty_sparse().unwrap();
        assert_eq!(autom.grpsize_base, 1.);
        assert_eq!(autom.grpsize_exp, 0);
    }

    #[test]
    fn triangle_stats() {
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

    #[test]
    fn simple_group() {
        log_init();

        use petgraph::visit::NodeIndexable;
        let g = DiGraph::<u8, ()>::from_edges([(0, 1)]);
        let autom = g.clone().try_into_autom_group().unwrap();
        assert_eq!(autom.0, [[0, 1]]);
        let g = g.into_edge_type::<Undirected>();
        let mut autom = g.clone().try_into_autom_group().unwrap();
        autom.sort();
        assert_eq!(autom.0, [[0, 1], [1, 0]]);
        let mut g = g;
        *g.node_weight_mut(g.from_index(0)).unwrap() = 2;
        let autom = g.try_into_autom_group().unwrap();
        assert_eq!(autom.0, [[0, 1]]);
    }

    #[test]
    fn empty_group() {
        log_init();

        let g = Graph::<(), ()>::new();
        let autom = g.clone().try_into_autom_group().unwrap();
        assert_eq!(autom.0, vec![vec![]]);
        let autom = g.clone().try_into_autom_group_nauty_dense().unwrap();
        assert_eq!(autom.0, vec![vec![]]);
        let autom = g.try_into_autom_group_nauty_sparse().unwrap();
        assert_eq!(autom.0, vec![vec![]]);
    }

    #[test]
    fn triangle_group() {
        log_init();

        let g = DiGraph::<u8, u8>::from_edges([(0, 1), (1, 2), (2, 0)]);
        let mut autom = g.clone().try_into_autom_group().unwrap();
        autom.sort();
        assert_eq!(autom.0, [[0, 1, 2], [1, 2, 0], [2, 0, 1]]);
        let g = g.into_edge_type::<Undirected>();
        let mut autom = g.clone().try_into_autom_group().unwrap();
        autom.sort();
        assert_eq!(
            autom.0,
            [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ]
        );
        let mut g = g;
        {
            use petgraph::visit::EdgeIndexable;
            *g.edge_weight_mut(g.from_index(0)).unwrap() = 2;
        }
        debug!("{g:#?}");
        let mut autom = g.clone().try_into_autom_group().unwrap();
        autom.sort();
        assert_eq!(autom.0, [[0, 1, 2], [1, 0, 2],]);
        {
            use petgraph::visit::EdgeIndexable;
            *g.edge_weight_mut(g.from_index(0)).unwrap() = 0;
        }
        {
            use petgraph::visit::NodeIndexable;
            *g.node_weight_mut(g.from_index(0)).unwrap() = 2;
        }
        debug!("{g:#?}");
        let mut autom = g.clone().try_into_autom_group().unwrap();
        autom.sort();
        assert_eq!(autom.0, [[0, 1, 2], [0, 2, 1],]);
    }

    #[test]
    fn random_autom_nauty_sparse_undirected() {
        log_init();

        let graphs = GraphIter::<Undirected>::default();

        for g in graphs.take(1000) {
            let g_canon = CanonGraph::from(g.clone());
            debug!("Initial graph: {g:#?}");
            let autom = g.clone().try_into_autom_group_nauty_sparse().unwrap();
            debug!("Automorphisms: {autom:#?}");
            for perm in autom.0 {
                let g_perm = CanonGraph::from(apply_perm(g.clone(), perm));
                assert_eq!(g_canon, g_perm);
            }
        }
    }

    #[test]
    fn random_autom_nauty_sparse_directed() {
        log_init();

        let graphs = GraphIter::<Directed>::default();

        for g in graphs.take(700) {
            let g_canon = CanonGraph::from(g.clone());
            debug!("Initial graph: {g:#?}");
            let autom = g.clone().try_into_autom_group_nauty_sparse().unwrap();
            debug!("Automorphisms: {autom:#?}");
            for perm in autom.0 {
                let g_perm = CanonGraph::from(apply_perm(g.clone(), perm));
                assert_eq!(g_canon, g_perm);
            }
        }
    }

    #[test]
    fn random_autom_nauty_dense_undirected() {
        log_init();

        let graphs = GraphIter::<Undirected>::default();

        for g in graphs.take(1000) {
            let g_canon = CanonGraph::from(g.clone());
            debug!("Initial graph: {g:#?}");
            let autom = g.clone().try_into_autom_group_nauty_dense().unwrap();
            debug!("Automorphisms: {autom:#?}");
            for perm in autom.0 {
                let g_perm = CanonGraph::from(apply_perm(g.clone(), perm));
                assert_eq!(g_canon, g_perm);
            }
        }
    }

    #[test]
    fn random_autom_nauty_dense_directed() {
        log_init();

        let graphs = GraphIter::<Directed>::default();

        for g in graphs.take(700) {
            let g_canon = CanonGraph::from(g.clone());
            debug!("Initial graph: {g:#?}");
            let autom = g.clone().try_into_autom_group_nauty_dense().unwrap();
            debug!("Automorphisms: {autom:#?}");
            for perm in autom.0 {
                let g_perm = CanonGraph::from(apply_perm(g.clone(), perm));
                assert_eq!(g_canon, g_perm);
            }
        }
    }

    fn orbit_partition(reps: &[usize]) -> BTreeSet<BTreeSet<usize>> {
        let mut classes: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for (vertex, &rep) in reps.iter().enumerate() {
            classes.entry(rep).or_default().insert(vertex);
        }
        classes.into_values().collect()
    }

    fn group_orbits(
        group: &[Vec<usize>],
        n: usize,
    ) -> BTreeSet<BTreeSet<usize>> {
        (0..n)
            .map(|vertex| group.iter().map(|perm| perm[vertex]).collect())
            .collect()
    }

    #[test]
    fn triangle_generators() {
        log_init();

        // C_3 for the directed triangle, S_3 for the undirected one; in both
        // the returned generators must generate exactly that group
        let g = DiGraph::<u8, u8>::from_edges([(0, 1), (1, 2), (2, 0)]);
        let a = g.clone().try_into_autom_generators().unwrap();
        assert_eq!(
            generated_group(&a.generators, 3),
            BTreeSet::from_iter([vec![0, 1, 2], vec![1, 2, 0], vec![2, 0, 1]])
        );

        let g = g.into_edge_type::<Undirected>();
        let a = g.try_into_autom_generators().unwrap();
        assert_eq!(
            generated_group(&a.generators, 3),
            BTreeSet::from_iter([
                vec![0, 1, 2],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![1, 2, 0],
                vec![2, 0, 1],
                vec![2, 1, 0],
            ])
        );
    }

    #[test]
    fn empty_generators() {
        log_init();

        let g = Graph::<(), ()>::new();
        let a = g.clone().try_into_autom_generators().unwrap();
        assert!(a.generators.is_empty());
        assert!(a.orbits.is_empty());
        assert_eq!(a.stats.grpsize_base, 1.);
        assert_eq!(a.stats.grpsize_exp, 0);
        let a = g.clone().try_into_autom_generators_nauty_dense().unwrap();
        assert!(a.generators.is_empty());
        assert!(a.orbits.is_empty());
        let a = g.try_into_autom_generators_nauty_sparse().unwrap();
        assert!(a.generators.is_empty());
        assert!(a.orbits.is_empty());
    }

    #[test]
    fn random_autom_generators_nauty_sparse_undirected() {
        log_init();

        let graphs = GraphIter::<Undirected>::default();
        for g in graphs.take(1000) {
            let n = g.node_count();
            let a = g.clone().try_into_autom_generators_nauty_sparse().unwrap();
            let group = g.clone().try_into_autom_group().unwrap();
            assert_eq!(
                generated_group(&a.generators, n),
                BTreeSet::from_iter(group.0.clone())
            );
            assert_eq!(orbit_partition(&a.orbits), group_orbits(&group.0, n));
        }
    }

    #[test]
    fn random_autom_generators_nauty_sparse_directed() {
        log_init();

        let graphs = GraphIter::<Directed>::default();
        for g in graphs.take(700) {
            let n = g.node_count();
            let a = g.clone().try_into_autom_generators_nauty_sparse().unwrap();
            let group = g.clone().try_into_autom_group().unwrap();
            assert_eq!(
                generated_group(&a.generators, n),
                BTreeSet::from_iter(group.0.clone())
            );
            assert_eq!(orbit_partition(&a.orbits), group_orbits(&group.0, n));
        }
    }

    #[test]
    fn random_autom_generators_nauty_dense_undirected() {
        log_init();

        let graphs = GraphIter::<Undirected>::default();
        for g in graphs.take(1000) {
            let n = g.node_count();
            let a = g.clone().try_into_autom_generators_nauty_dense().unwrap();
            let group = g.clone().try_into_autom_group().unwrap();
            assert_eq!(
                generated_group(&a.generators, n),
                BTreeSet::from_iter(group.0.clone())
            );
            assert_eq!(orbit_partition(&a.orbits), group_orbits(&group.0, n));
        }
    }

    #[test]
    fn random_autom_generators_nauty_dense_directed() {
        log_init();

        let graphs = GraphIter::<Directed>::default();
        for g in graphs.take(700) {
            let n = g.node_count();
            let a = g.clone().try_into_autom_generators_nauty_dense().unwrap();
            let group = g.clone().try_into_autom_group().unwrap();
            assert_eq!(
                generated_group(&a.generators, n),
                BTreeSet::from_iter(group.0.clone())
            );
            assert_eq!(orbit_partition(&a.orbits), group_orbits(&group.0, n));
        }
    }
}
