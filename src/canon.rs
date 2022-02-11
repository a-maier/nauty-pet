use crate::sparse_graph::{SparseDiGraph, SparseUnGraph};

use std::hash::Hash;

use nauty_Traces_sys::{
    FALSE, optionblk, sparsegraph, sparsenauty, statsblk, SparseGraph, Traces,
    TracesOptions, TracesStats, SG_FREE, TRUE,
};
use petgraph::{
    graph::{DefaultIx, DiGraph, Graph, IndexType, UnGraph},
    visit::EdgeRef,
    EdgeType,
};
use thiserror::Error;

/// Find the canonical labelling for a graph
///
/// Internally, this uses Traces for undirected graphs without
/// self-loops and sparse nauty otherwise.
pub trait ToCanon {
    type Output;

    fn to_canon(self) -> Self::Output;
}

/// Use sparse nauty to find the canonical labelling
pub trait ToCanonNautySparse {
    type Output;

    fn to_canon_nauty_sparse(self) -> Self::Output;
}

/// Use dense nauty to find the canonical labelling
pub trait ToCanonNautyDense {
    type Output;

    fn to_canon_nauty_dense(self) -> Self::Output;
}

/// Use Traces to find the canonical labelling
pub trait ToCanonTraces {
    type Output;

    fn to_canon_traces(self) -> Self::Output;
}

impl<N, Ix: IndexType> ToCanon for &UnGraph<N, (), Ix>
where
    N: Clone + Default + Hash + Eq,
{
    type Output = UnGraph<N, (), DefaultIx>;

    fn to_canon(self) -> Self::Output {
        match self.to_canon_traces() {
            Ok(c) => c,
            Err(TracesError::SelfLoop) => self.to_canon_nauty_sparse(),
        }
    }
}

impl<N, Ix: IndexType> ToCanon for &DiGraph<N, (), Ix>
where
    N: Clone + Default + Hash + Eq,
{
    type Output = DiGraph<N, (), DefaultIx>;

    fn to_canon(self) -> Self::Output {
        self.to_canon_nauty_sparse()
    }
}

impl<N, Ix: IndexType> ToCanonNautySparse for &UnGraph<N, (), Ix>
where
    N: Clone + Default + Hash + Eq,
{
    type Output = UnGraph<N, (), DefaultIx>;

    fn to_canon_nauty_sparse(self) -> Self::Output {
        let mut options = optionblk::default_sparse();
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        let mut stats = statsblk::default();
        let mut orbits = vec![0; self.node_count()];
        let mut sg: SparseUnGraph<_> = self.into();
        let mut cg = sparsegraph::default();
        unsafe {
            sparsenauty(
                &mut (&mut sg.g).into(),
                sg.weights.lab.as_mut_ptr(),
                sg.weights.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            copy_sg(&cg, &mut sg.g);
            SG_FREE(&mut cg);
        }
        (&sg).into()
    }
}

// TODO: code duplication
impl<N, Ix: IndexType> ToCanonNautySparse for &DiGraph<N, (), Ix>
where
    N: Clone + Default + Hash + Eq,
{
    type Output = DiGraph<N, (), DefaultIx>;

    fn to_canon_nauty_sparse(self) -> Self::Output {
        let mut options = optionblk::default_sparse();
        options.getcanon = TRUE;
        options.defaultptn = FALSE;
        let mut stats = statsblk::default();
        let mut orbits = vec![0; self.node_count()];
        let mut sg: SparseDiGraph<_> = self.into();
        let mut cg = sparsegraph::default();
        unsafe {
            sparsenauty(
                &mut (&mut sg.g).into(),
                sg.weights.lab.as_mut_ptr(),
                sg.weights.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            copy_sg(&cg, &mut sg.g);
            SG_FREE(&mut cg);
        }
        (&sg).into()
    }
}

impl<N, Ix: IndexType> ToCanonTraces for &UnGraph<N, (), Ix>
where
    N: Clone + Default + Hash + Eq,
{
    type Output = Result<UnGraph<N, (), DefaultIx>, TracesError>;

    fn to_canon_traces(self) -> Self::Output {
        if has_self_loop(self) {
            return Err(TracesError::SelfLoop);
        }
        let mut options = TracesOptions {
            getcanon: TRUE,
            defaultptn: FALSE,
            ..Default::default()
        };
        let mut stats = TracesStats::default();
        let mut orbits = vec![0; self.node_count()];
        let mut sg: SparseUnGraph<_> = self.into();
        let mut cg = sparsegraph::default();
        unsafe {
            Traces(
                &mut (&mut sg.g).into(),
                sg.weights.lab.as_mut_ptr(),
                sg.weights.ptn.as_mut_ptr(),
                orbits.as_mut_ptr(),
                &mut options,
                &mut stats,
                &mut cg,
            );
            copy_sg(&cg, &mut sg.g);
            SG_FREE(&mut cg);
        }
        Ok((&sg).into())
    }
}

#[derive(Error, Debug)]
pub enum TracesError {
    #[error("Graph has at least one self-loop")]
    SelfLoop,
}

unsafe fn copy_sg(from: &sparsegraph, to: &mut SparseGraph) {
    use std::slice::from_raw_parts;
    debug_assert_eq!(from.nv as usize, to.v.len());
    debug_assert_eq!(from.vlen as usize, to.v.len());
    debug_assert_eq!(from.dlen as usize, to.d.len());
    debug_assert_eq!(from.nde as usize, to.e.len());
    debug_assert_eq!(from.elen as usize, to.e.len());
    let v = from_raw_parts(from.v, from.vlen as usize);
    let d = from_raw_parts(from.d, from.dlen as usize);
    let e = from_raw_parts(from.e, from.elen as usize);
    to.v.copy_from_slice(v);
    to.d.copy_from_slice(d);
    to.e.copy_from_slice(e);
}

fn has_self_loop<N, E, Ty: EdgeType, Ix: IndexType>(
    g: &Graph<N, E, Ty, Ix>,
) -> bool {
    g.edge_references().any(|e| e.source() == e.target())
}
