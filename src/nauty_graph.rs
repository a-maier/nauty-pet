use std::cmp::{Ord, Ordering};
use std::convert::From;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::os::raw::c_int;

use ahash::RandomState;
use itertools::izip;
use nauty_Traces_sys::SparseGraph as NautySparse;
use nauty_Traces_sys::{empty_graph, graph, ADDONEARC, SETWORDSNEEDED};

use petgraph::{
    graph::{Graph, IndexType},
    visit::EdgeRef,
    EdgeType,
};

#[cfg(feature = "stable")]
type HashMap<K, V> = indexmap::IndexMap<K, V, RandomState>;
#[cfg(feature = "stable")]
fn sort<T: Ord>(slice: &mut [T]) {
    slice.sort()
}
#[cfg(feature = "stable")]
fn sort_by<T: Ord, F>(slice: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    slice.sort_by(cmp)
}
#[cfg(feature = "stable")]
fn sort_by_key<T, K, F>(slice: &mut [T], f: F)
where
    F: FnMut(&T) -> K,
    K: Ord,
{
    slice.sort_by_key(f)
}

#[cfg(not(feature = "stable"))]
type HashMap<K, V> = ahash::AHashMap<K, V, RandomState>;
#[cfg(not(feature = "stable"))]
fn sort<T: Ord>(slice: &mut [T]) {
    slice.sort_unstable()
}
#[cfg(not(feature = "stable"))]
fn sort_by<T: Ord, F>(slice: &mut [T], cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    slice.sort_unstable_by(cmp)
}
#[cfg(not(feature = "stable"))]
fn sort_by_key<T, K, F>(slice: &mut [T], f: F)
where
    F: FnMut(&T) -> K,
    K: Ord,
{
    slice.sort_unstable_by_key(f)
}

#[derive(Debug, Default, Clone)]
pub(crate) struct SparseGraph<N, E, D> {
    pub(crate) g: NautySparse,
    pub(crate) nodes: Nodes<N>,
    edges: HashMap<(usize, usize), Vec<E>>,
    dir: PhantomData<D>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct DenseGraph<N, E, D> {
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) g: Vec<graph>,
    pub(crate) nodes: Nodes<N>,
    edges: HashMap<(usize, usize), Vec<E>>,
    dir: PhantomData<D>,
}

#[derive(Debug, Default, Clone, Hash)]
pub(crate) struct Nodes<N> {
    pub(crate) lab: Vec<c_int>,
    pub(crate) ptn: Vec<c_int>,
    pub(crate) weights: Vec<N>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct RawGraphData<N, E, D> {
    adj: Vec<Vec<c_int>>,
    nodes: Nodes<N>,
    edges: HashMap<(usize, usize), Vec<E>>,
    num_nauty_edges: usize,
    dir: PhantomData<D>,
    pub(crate) relabel: Vec<usize>,
}

fn relabel_to_contiguous_node_weights<N: Ord>(nodes: &mut [N]) -> Vec<usize> {
    let mut new_ord = Vec::from_iter(0..nodes.len());
    sort_by(&mut new_ord, |&i, &j| nodes[i].cmp(&nodes[j]));
    let mut renumber = vec![0; new_ord.len()];
    for (new_idx, old_idx) in new_ord.iter().enumerate() {
        renumber[*old_idx] = new_idx;
    }
    apply_perm(nodes, renumber.clone());
    for n in nodes.windows(2) {
        debug_assert!(n[0] <= n[1]);
    }
    renumber
}

pub(crate) fn apply_perm<T>(slice: &mut [T], mut new_pos: Vec<usize>) {
    const CORRECT_POS: usize = usize::MAX;
    for idx in 0..slice.len() {
        let mut next_idx = new_pos[idx];
        if next_idx == CORRECT_POS {
            continue;
        }
        while next_idx != idx {
            slice.swap(idx, next_idx);
            next_idx = std::mem::replace(&mut new_pos[next_idx], CORRECT_POS);
        }
    }
}

impl<N, E, Ty, Ix> From<Graph<N, E, Ty, Ix>>
    for RawGraphData<(N, Vec<E>), E, Ty>
where
    Ty: EdgeType,
    Ix: IndexType,
    N: Ord,
    E: Hash + Ord,
{
    fn from(g: Graph<N, E, Ty, Ix>) -> Self {
        use petgraph::visit::NodeIndexable;
        let is_directed = g.is_directed();
        let edges = Vec::from_iter(
            g.edge_references()
                .map(|e| (g.to_index(e.source()), g.to_index(e.target()))),
        );
        let (nodes, e) = g.into_nodes_edges();
        let mut node_weights =
            Vec::from_iter(nodes.into_iter().map(|n| (n.weight, Vec::new())));

        // edge weights
        // we combine multiple edges into a single one with an
        // effective weight given by the sorted vector of the
        // individual weights
        // self-loops are removed and their weights instead appended
        // to the corresponding node weight
        let mut edge_weights: HashMap<_, Vec<E>> = HashMap::default();
        for (mut edge, wt) in izip!(edges, e.into_iter().map(|e| e.weight)) {
            if edge.0 == edge.1 {
                node_weights[edge.0].1.push(wt);
            } else {
                if !is_directed && edge.0 > edge.1 {
                    std::mem::swap(&mut edge.0, &mut edge.1)
                }
                edge_weights.entry(edge).or_default().push(wt);
            }
        }
        for v in &mut node_weights {
            sort(&mut v.1);
        }
        for v in edge_weights.values_mut() {
            sort(v);
        }
        let relabel = relabel_to_contiguous_node_weights(&mut node_weights);
        let edge_weights: HashMap<_, _> = edge_weights
            .into_iter()
            .map(|(e, wt)| {
                let mut e = (relabel[e.0], relabel[e.1]);
                if !is_directed && e.0 > e.1 {
                    std::mem::swap(&mut e.0, &mut e.1)
                }
                (e, wt)
            })
            .collect();

        // the edge weight that appears most often is taken to be the default
        // for all other edge weights we introduce auxiliary vertices
        // each non-default edge weight has its own vertex type (colour)
        let mut edge_weight_counts: HashMap<&[E], u32> = HashMap::default();
        for v in edge_weights.values() {
            *edge_weight_counts.entry(v).or_default() += 1;
        }
        let mut edge_weight_counts = Vec::from_iter(edge_weight_counts);
        sort(&mut edge_weight_counts);
        let max_pos = edge_weight_counts
            .iter()
            .enumerate()
            .max_by_key(|(_n, (_k, v))| v)
            .map(|(n, _)| n);
        if let Some(max_pos) = max_pos {
            edge_weight_counts.remove(max_pos);
        }

        let aux_vertex_type: HashMap<_, _> = edge_weight_counts
            .into_iter()
            .enumerate()
            .map(|(n, (k, _))| (k, n))
            .collect();

        // We introduce one auxiliar vertex for each edge with
        // non-default weight.
        let mut num_aux = vec![0; aux_vertex_type.len()];
        for ((source, target), w) in &edge_weights {
            debug_assert_ne!(source, target);
            if let Some(&aux_type) = aux_vertex_type.get(w.as_slice()) {
                num_aux[aux_type] += 1;
            }
        }

        let total_num_aux: usize = num_aux.iter().sum();
        let num_nauty_vertices = node_weights.len() + total_num_aux;

        let mut aux_vx_idx = Vec::from_iter(num_aux.iter().scan(
            node_weights.len(),
            |partial_sum, n| {
                let idx = *partial_sum;
                *partial_sum += n;
                Some(idx)
            },
        ));
        let mut adj = vec![vec![]; num_nauty_vertices];
        fn add_edge(
            adj: &mut [Vec<c_int>],
            source: usize,
            target: usize,
            is_directed: bool,
        ) {
            adj[source].push(target as c_int);
            if !is_directed {
                adj[target].push(source as c_int);
            }
        }

        for (edge, wt) in &edge_weights {
            let (source, target) = *edge;
            debug_assert_ne!(source, target);
            if let Some(&aux_vx_type) = aux_vertex_type.get(wt.as_slice()) {
                // non-default weight, introduce an extra vertex
                add_edge(
                    &mut adj,
                    source,
                    aux_vx_idx[aux_vx_type],
                    is_directed,
                );
                add_edge(
                    &mut adj,
                    aux_vx_idx[aux_vx_type],
                    target,
                    is_directed,
                );
                aux_vx_idx[aux_vx_type] += 1;
            } else {
                add_edge(&mut adj, source, target, is_directed);
            }
        }
        let mut ptn = vec![1; num_nauty_vertices];
        for (ptn, wts) in izip!(&mut ptn, node_weights.windows(2)) {
            if wts[1] > wts[0] {
                *ptn = 0;
            }
        }
        if !ptn.is_empty() {
            ptn[node_weights.len() - 1] = 0;
            for i in aux_vx_idx {
                ptn[i - 1] = 0;
            }
        }
        let lab = (0..num_nauty_vertices as i32).collect();
        let nodes = Nodes {
            weights: node_weights,
            lab,
            ptn,
        };
        let num_nauty_edges = {
            let num_nauty_edges = edge_weights.len() + total_num_aux;
            if is_directed {
                num_nauty_edges
            } else {
                2 * num_nauty_edges
            }
        };
        Self {
            adj,
            edges: edge_weights,
            num_nauty_edges,
            nodes,
            relabel,
            dir: PhantomData,
        }
    }
}

impl<N, E, Ty> From<RawGraphData<N, E, Ty>> for SparseGraph<N, E, Ty>
where
    Ty: EdgeType,
    N: Ord,
    E: Hash + Ord,
{
    fn from(g: RawGraphData<N, E, Ty>) -> Self {
        let adj = g.adj;
        let mut sg = NautySparse::new(adj.len(), g.num_nauty_edges);
        let mut vpos = 0;
        for (adj, d, v) in izip!(adj, &mut sg.d, &mut sg.v) {
            *d = adj.len() as c_int;
            *v = vpos;
            let start = vpos;
            let end = start + *d as usize;
            sg.e[start..end].copy_from_slice(&adj);
            vpos += *d as usize
        }
        debug_assert!(sg.v.len() >= g.nodes.weights.len());

        Self {
            g: sg,
            edges: g.edges,
            nodes: g.nodes,
            dir: PhantomData,
        }
    }
}

impl<N, E, Ty, Ix> From<Graph<N, E, Ty, Ix>> for SparseGraph<(N, Vec<E>), E, Ty>
where
    Ty: EdgeType,
    Ix: IndexType,
    N: Ord,
    E: Hash + Ord,
{
    fn from(g: Graph<N, E, Ty, Ix>) -> Self {
        let g: RawGraphData<_, _, _> = g.into();
        g.into()
    }
}

impl<N, E, Ty> From<RawGraphData<N, E, Ty>> for DenseGraph<N, E, Ty>
where
    Ty: EdgeType,
    N: Ord,
    E: Hash + Ord,
{
    fn from(g: RawGraphData<N, E, Ty>) -> Self {
        let adj = g.adj;
        let n = adj.len();
        let m = SETWORDSNEEDED(n);
        let mut dg = empty_graph(m, n);
        for (source, adj) in adj.into_iter().enumerate() {
            for target in adj {
                ADDONEARC(&mut dg, source, target as usize, m);
            }
        }

        Self {
            n,
            m,
            g: dg,
            edges: g.edges,
            nodes: g.nodes,
            dir: PhantomData,
        }
    }
}

impl<N, E, Ty, Ix> From<Graph<N, E, Ty, Ix>> for DenseGraph<(N, Vec<E>), E, Ty>
where
    Ty: EdgeType,
    Ix: IndexType,
    N: Ord,
    E: Hash + Ord,
{
    fn from(g: Graph<N, E, Ty, Ix>) -> Self {
        let g: RawGraphData<_, _, _> = g.into();
        g.into()
    }
}

pub(crate) fn inv_perm<I>(perm: &[I]) -> Vec<usize>
where
    I: Copy + TryInto<usize>,
    <I as TryInto<usize>>::Error: Debug,
{
    let mut relabel = vec![0; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        let old = old.try_into().unwrap();
        relabel[old] = new;
    }
    relabel
}

fn into_graph<N, E, Ty, Ix>(
    node_weights: Vec<(N, Vec<E>)>,
    edge_weights: HashMap<(usize, usize), Vec<E>>,
    lab: &[c_int],
) -> Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    E: Ord,
{
    // TODO: check if precalculating the number of edges helps
    let mut res = Graph::with_capacity(node_weights.len(), 0);
    let relabel = inv_perm(lab);
    let mut edges = Vec::new();
    let is_directed = Ty::is_directed();

    // nodes + self-loops
    let mut node_weights =
        Vec::from_iter(izip!(relabel.iter().copied(), node_weights));
    sort_by_key(&mut node_weights, |e| e.0);
    for (n, i) in node_weights.iter().map(|(i, _w)| i).enumerate() {
        debug_assert_eq!(n, *i)
    }
    for (_, (w, loops)) in node_weights {
        for w in loops {
            edges.push((res.node_count(), res.node_count(), w))
        }
        res.add_node(w);
    }

    // edges
    for ((source, target), weights) in edge_weights {
        let mut source = relabel[source];
        let mut target = relabel[target];
        if !is_directed && source > target {
            std::mem::swap(&mut source, &mut target);
        }
        for w in weights {
            edges.push((source, target, w));
        }
    }
    sort(&mut edges);
    for (source, target, weight) in edges {
        use petgraph::visit::NodeIndexable;
        let source = res.from_index(source);
        let target = res.from_index(target);
        res.add_edge(source, target, weight);
    }

    res
}

impl<N, E, Ty, Ix> From<DenseGraph<(N, Vec<E>), E, Ty>> for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    E: Ord,
{
    fn from(g: DenseGraph<(N, Vec<E>), E, Ty>) -> Self {
        into_graph(g.nodes.weights, g.edges, &g.nodes.lab)
    }
}

impl<N, E, Ty, Ix> From<SparseGraph<(N, Vec<E>), E, Ty>> for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    E: Ord,
{
    fn from(g: SparseGraph<(N, Vec<E>), E, Ty>) -> Self {
        into_graph(g.nodes.weights, g.edges, &g.nodes.lab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::Debug;

    use log::debug;
    use petgraph::{
        algo::isomorphism::is_isomorphic,
        graph::{DiGraph, Graph, IndexType, UnGraph},
        Directed, EdgeType, Undirected,
    };
    use testing::GraphIter;

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn tst_conv_sparse<N, E, Ty, Ix>(g: Graph<N, E, Ty, Ix>)
    where
        N: Clone + Debug + Ord,
        E: Clone + Debug + Ord + Hash,
        Ty: Debug + EdgeType,
        Ix: IndexType,
    {
        debug!("Initial graph: {g:#?}");
        let s = SparseGraph::from(g.clone());
        debug!("Nauty graph: {s:#?}");
        let gg: Graph<N, E, Ty, Ix> = s.into();
        debug!("Final graph: {gg:#?}");
        assert!(is_isomorphic(&g, &gg));
    }

    fn tst_conv_dense<N, E, Ty, Ix>(g: Graph<N, E, Ty, Ix>)
    where
        N: Clone + Debug + Ord,
        E: Clone + Debug + Ord + Hash,
        Ty: Debug + EdgeType,
        Ix: IndexType,
    {
        debug!("Initial graph: {g:#?}");
        let s = DenseGraph::from(g.clone());
        debug!("Nauty graph: {s:#?}");
        let gg: Graph<N, E, Ty, Ix> = s.into();
        debug!("Final graph: {gg:#?}");
        assert!(is_isomorphic(&g, &gg));
    }

    #[test]
    fn simple_conversion_sparse() {
        log_init();

        tst_conv_sparse(Graph::<(), (), _>::new_undirected());
        tst_conv_sparse(UnGraph::<(), ()>::from_edges([(0, 1), (2, 0)]));
        tst_conv_sparse(UnGraph::<(), i32>::from_edges([
            (0, 1, -1),
            (2, 0, 1),
        ]));
        tst_conv_sparse(DiGraph::<(), ()>::from_edges([
            (0, 1),
            (1, 1),
            (0, 2),
            (2, 0),
        ]));
        tst_conv_sparse(DiGraph::<(), u32>::from_edges([
            (0, 1, 0),
            (1, 1, 0),
            (0, 2, 0),
            (2, 0, 1),
        ]));
    }

    #[test]
    fn simple_conversion_dense() {
        log_init();

        tst_conv_dense(Graph::<(), (), _>::new_undirected());
        tst_conv_dense(UnGraph::<(), ()>::from_edges([(0, 1), (2, 0)]));
        tst_conv_dense(UnGraph::<(), i32>::from_edges([(0, 1, -1), (2, 0, 1)]));
        tst_conv_dense(DiGraph::<(), ()>::from_edges([
            (0, 1),
            (1, 1),
            (0, 2),
            (2, 0),
        ]));
        tst_conv_dense(DiGraph::<(), u32>::from_edges([
            (0, 1, 0),
            (1, 1, 0),
            (0, 2, 0),
            (2, 0, 1),
        ]));
    }

    #[test]
    fn random_conversion_sparse_undirected() {
        log_init();
        for g in GraphIter::<Undirected>::default().take(1000) {
            tst_conv_sparse(g);
        }
    }

    #[test]
    fn random_conversion_sparse_directed() {
        log_init();
        for g in GraphIter::<Directed>::default().take(700) {
            tst_conv_sparse(g);
        }
    }

    #[test]
    fn random_conversion_dense_undirected() {
        log_init();
        for g in GraphIter::<Undirected>::default().take(1000) {
            tst_conv_dense(g);
        }
    }

    #[test]
    fn random_conversion_dense_directed() {
        log_init();
        for g in GraphIter::<Directed>::default().take(700) {
            tst_conv_dense(g);
        }
    }

    #[test]
    fn asym_conversion() {
        log_init();

        let g = UnGraph::<(), ()>::from_edges([(0, 1), (1, 0)]);
        tst_conv_sparse(g.clone());
        tst_conv_dense(g);
    }
}
