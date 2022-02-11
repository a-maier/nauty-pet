use std::collections::HashMap;
use std::convert::From;
use std::hash::Hash;
use std::os::raw::c_int;

use nauty_Traces_sys::{size_t, SparseGraph};
use petgraph::{
    graph::{DefaultIx, DiGraph, Graph, IndexType, UnGraph},
    visit::EdgeRef,
    EdgeType,
};

pub(crate) struct SparseDiGraph<N> {
    pub(crate) g: SparseGraph,
    pub(crate) weights: Weights<N>,
}

pub(crate) struct SparseUnGraph<N> {
    pub(crate) g: SparseGraph,
    pub(crate) weights: Weights<N>,
}

pub(crate) struct Weights<N> {
    pub(crate) lab: Vec<c_int>,
    pub(crate) ptn: Vec<c_int>,
    pub(crate) weights: Vec<N>,
}

impl<N, E, Ix> From<&DiGraph<N, E, Ix>> for SparseDiGraph<N>
where
    Ix: IndexType,
    N: Hash + Clone + Eq,
{
    fn from(g: &DiGraph<N, E, Ix>) -> Self {
        let Sg(sg) = g.into();
        let weights = g.get_weights();
        SparseDiGraph { g: sg, weights }
    }
}

impl<N> From<&SparseDiGraph<N>> for DiGraph<N, (), DefaultIx>
where
    N: Clone + Default,
{
    fn from(sg: &SparseDiGraph<N>) -> Self {
        let mut edges = Vec::new();
        for (v1, (d, begin_e)) in sg.g.d.iter().zip(sg.g.v.iter()).enumerate() {
            let begin_e = *begin_e as usize;
            let end_e = begin_e + *d as usize;
            let endpoints = &sg.g.e[begin_e..end_e];
            for v2 in endpoints {
                edges.push((v1 as u32, *v2 as u32, ()))
            }
        }
        let mut g = Graph::<N, ()>::from_edges(edges);
        clone_weights(&sg.weights, &mut g);
        g
    }
}

impl<N> From<&SparseUnGraph<N>> for UnGraph<N, (), DefaultIx>
where
    N: Clone + Default,
{
    fn from(sg: &SparseUnGraph<N>) -> Self {
        let mut edges = Vec::new();
        for (v1, (d, begin_e)) in sg.g.d.iter().zip(sg.g.v.iter()).enumerate() {
            let begin_e = *begin_e as usize;
            let end_e = begin_e + *d as usize;
            let endpoints = sg.g.e[begin_e..end_e].iter();
            for v2 in endpoints.filter(|&&v| v as usize >= v1) {
                edges.push((v1 as u32, *v2 as u32, ()))
            }
        }
        let mut g = UnGraph::<N, ()>::from_edges(edges);
        clone_weights(&sg.weights, &mut g);
        g
    }
}

impl<N, E, Ix> From<&UnGraph<N, E, Ix>> for SparseUnGraph<N>
where
    Ix: IndexType,
    N: Hash + Clone + Eq,
{
    fn from(g: &UnGraph<N, E, Ix>) -> Self {
        let Sg(sg) = g.into();
        let weights = g.get_weights();
        SparseUnGraph { g: sg, weights }
    }
}

// newtype wrapper so we can implement From
struct Sg(SparseGraph);

impl<N, E, Ty, Ix> From<&Graph<N, E, Ty, Ix>> for Sg
where
    Ix: IndexType,
    Ty: EdgeType,
    Graph<N, E, Ty, Ix>: CopyDeg + CopyEdges,
{
    fn from(g: &Graph<N, E, Ty, Ix>) -> Self {
        let mut sg = SparseGraph::new(g.node_count(), g.edge_count());
        g.copy_degrees(&mut sg);
        update_edges_addr(&mut sg);
        g.copy_edges(&mut sg);
        Sg(sg)
    }
}

trait CopyDeg {
    fn copy_degrees(&self, sg: &mut SparseGraph);
}

impl<N, E, Ix: IndexType> CopyDeg for DiGraph<N, E, Ix> {
    fn copy_degrees(&self, sg: &mut SparseGraph) {
        use petgraph::visit::NodeIndexable;
        for v in self.node_indices() {
            let n = self.to_index(v);
            sg.d[n] = self.edges(v).count() as c_int;
        }
    }
}

impl<N, E, Ix: IndexType> CopyDeg for UnGraph<N, E, Ix> {
    fn copy_degrees(&self, sg: &mut SparseGraph) {
        use petgraph::visit::NodeIndexable;
        for v in self.node_indices() {
            let n = self.to_index(v);
            sg.d[n] = 2 * self.edges(v).count() as c_int;
        }
    }
}

fn update_edges_addr(sg: &mut SparseGraph) {
    let mut v_idx = 0;
    for (n, d) in sg.d.iter().enumerate() {
        sg.v[n] = v_idx;
        v_idx += *d as size_t;
    }
}

trait CopyEdges {
    fn copy_edges(&self, sg: &mut SparseGraph);
}

impl<N, E, Ix: IndexType> CopyEdges for DiGraph<N, E, Ix> {
    fn copy_edges(&self, sg: &mut SparseGraph) {
        use petgraph::visit::NodeIndexable;
        for v in self.node_indices() {
            let e = &mut sg.e[self.to_index(v)..];
            for (target, edge) in e.iter_mut().zip(self.edges(v)) {
                debug_assert_eq!(edge.source(), v);
                *target = self.to_index(edge.target()) as c_int;
            }
        }
    }
}

impl<N, E, Ix: IndexType> CopyEdges for UnGraph<N, E, Ix> {
    fn copy_edges(&self, sg: &mut SparseGraph) {
        use petgraph::visit::NodeIndexable;
        for v in self.node_indices() {
            let e = &mut sg.e[self.to_index(v)..];
            for (target, edge) in e.iter_mut().zip(self.edges(v)) {
                let v2 = if v == edge.source() {
                    edge.target()
                } else {
                    debug_assert_eq!(edge.target(), v);
                    edge.source()
                };
                *target = self.to_index(v2) as c_int;
            }
        }
    }
}

trait GetWeights<N> {
    fn get_weights(&self) -> Weights<N>;
}

impl<N, E, Ty: EdgeType, Ix: IndexType> GetWeights<N> for Graph<N, E, Ty, Ix>
where
    N: Hash + Clone + Eq,
{
    fn get_weights(&self) -> Weights<N> {
        use petgraph::visit::NodeIndexable;
        let mut weight_map: HashMap<_, Vec<_>> = HashMap::new();
        for v in self.node_indices() {
            let wt = self.node_weight(v).unwrap();
            let n = self.to_index(v);
            weight_map.entry(wt).or_default().push(n as c_int);
        }
        let mut lab = Vec::with_capacity(self.node_count());
        let mut ptn = vec![1; self.node_count()];
        let mut weights = Vec::with_capacity(weight_map.len());
        for (weight, mut nodes) in weight_map {
            weights.push(weight.clone());
            lab.append(&mut nodes);
            ptn[lab.len() - 1] = 0;
        }
        Weights { lab, ptn, weights }
    }
}

fn clone_weights<N, E, Ty, Ix>(wt: &Weights<N>, g: &mut Graph<N, E, Ty, Ix>)
where
    N: Clone,
    Ix: IndexType,
    Ty: EdgeType,
{
    let mut wt_idx = 0;
    for (&v, &end_mark) in wt.lab.iter().zip(wt.ptn.iter()) {
        use petgraph::visit::NodeIndexable;
        let v_idx = g.from_index(v as usize);
        *g.node_weight_mut(v_idx).unwrap() = wt.weights[wt_idx].clone();
        if end_mark == 0 {
            wt_idx += 1
        }
    }
}
