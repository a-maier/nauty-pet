use std::cmp::Ord;
use std::convert::From;
use std::hash::Hash;
use std::marker::PhantomData;
use std::os::raw::c_int;

use ahash::AHashMap;
use itertools::izip;
use nauty_Traces_sys::{size_t, SparseGraph as NautySparse};
use petgraph::{
    graph::{Graph, IndexType, Node},
    visit::EdgeRef,
    EdgeType,
};

#[derive(Debug, Default, Clone)]
pub(crate) struct SparseGraph<N, E, D> {
    pub(crate) g: NautySparse,
    pub(crate) nodes: Nodes<N>,
    edges: AHashMap<(usize, usize), Vec<E>>,
    dir: PhantomData<D>,
}

#[derive(Debug, Default, Clone, Hash)]
pub(crate) struct Nodes<N> {
    pub(crate) lab: Vec<c_int>,
    pub(crate) ptn: Vec<c_int>,
    pub(crate) weights: Vec<N>,
}

fn relabel_to_contiguous_node_weights<N: Ord, Ix: IndexType>(
    node_weights: impl IntoIterator<Item = Node<N, Ix>>,
    edges: &mut [(usize, usize)],
    is_directed: bool,
) -> Vec<N> {
    let mut node_weights = Vec::from_iter(
        node_weights
            .into_iter()
            .enumerate()
            .map(|(idx, n)| (n.weight, idx)),
    );
    // TODO: use sort_by_key, but that has lifetime problems?
    node_weights.sort_unstable_by(|i, j| i.0.cmp(&j.0));
    let mut renumber = vec![0; node_weights.len()];
    for (new_idx, (_, old_idx)) in node_weights.iter().enumerate() {
        renumber[*old_idx] = new_idx;
    }
    for edge in edges {
        edge.0 = renumber[edge.0];
        edge.1 = renumber[edge.1];
        if !is_directed && edge.0 > edge.1 {
            std::mem::swap(&mut edge.0, &mut edge.1);
        }
    }
    node_weights.into_iter().map(|(wt, _old_idx)| wt).collect()
}

impl<N, E, Ty, Ix> From<Graph<N, E, Ty, Ix>> for SparseGraph<N, E, Ty>
where
    Ty: EdgeType,
    Ix: IndexType,
    N: Ord,
    E: Hash + Ord,
{
    fn from(g: Graph<N, E, Ty, Ix>) -> Self {
        use petgraph::visit::NodeIndexable;
        let is_directed = g.is_directed();
        let mut edges = Vec::from_iter(
            g.edge_references()
                .map(|e| (g.to_index(e.source()), g.to_index(e.target()))),
        );
        let (nodes, e) = g.into_nodes_edges();
        let e_weights = Vec::from_iter(e.into_iter().map(|e| e.weight));

        let node_weights =
            relabel_to_contiguous_node_weights(nodes, &mut edges, is_directed);

        // edge weights
        // we combine multiple edges into a single one with an
        // effective weight given by the sorted vector of the
        // individual weights
        let mut edge_weights: AHashMap<_, Vec<E>> = AHashMap::new();
        for (edge, wt) in izip!(edges, e_weights) {
            edge_weights.entry(edge).or_default().push(wt)
        }
        for v in edge_weights.values_mut() {
            v.sort_unstable();
        }

        // the edge weight that appears most often is taken to be the default
        // for all other edge weights we introduce auxiliary vertices
        // each non-default edge weight has its own vertex type (colour)
        let mut edge_weight_counts: AHashMap<&[E], u32> = AHashMap::new();
        for v in edge_weights.values() {
            *edge_weight_counts.entry(v).or_default() += 1;
        }
        let mut edge_weight_counts = Vec::from_iter(edge_weight_counts);
        edge_weight_counts.sort_unstable();
        let max_pos = edge_weight_counts
            .iter()
            .enumerate()
            .max_by_key(|(_n, (_k, v))| v)
            .map(|(n, _)| n);
        if let Some(max_pos) = max_pos {
            edge_weight_counts.remove(max_pos);
        }

        let aux_vertex_type: AHashMap<_, _> = edge_weight_counts
            .into_iter()
            .enumerate()
            .map(|(n, (k, _))| (k, n))
            .collect();

        // We introduce one auxiliar vertex for each edge with
        // non-default weight. To eliminate self-loops, we instead
        // introduce _two_ additional vertices, even if the original
        // edge had the default weight
        let mut num_aux = vec![0; aux_vertex_type.len() + 1];
        for ((source, target), w) in &edge_weights {
            if let Some(&aux_type) = aux_vertex_type.get(w.as_slice()) {
                if source == target {
                    num_aux[aux_type] += 2;
                } else {
                    num_aux[aux_type] += 1;
                }
            } else if source == target {
                *num_aux.last_mut().unwrap() += 2;
            }
        }

        let total_num_aux: usize = num_aux.iter().sum();
        let num_nauty_vertices = node_weights.len() + total_num_aux;
        let mut num_nauty_edges = edge_weights.len() + total_num_aux;
        if !is_directed {
            num_nauty_edges *= 2
        }

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
            if let Some(&aux_vx_type) = aux_vertex_type.get(wt.as_slice()) {
                // non-default weight, introduce an extra vertex
                add_edge(
                    &mut adj,
                    source,
                    aux_vx_idx[aux_vx_type],
                    is_directed,
                );
                if source == target {
                    // self-loop, introduce one more vertex
                    let start = aux_vx_idx[aux_vx_type];
                    add_edge(&mut adj, start, start + 1, is_directed);
                    aux_vx_idx[aux_vx_type] += 1;
                }
                add_edge(
                    &mut adj,
                    aux_vx_idx[aux_vx_type],
                    target,
                    is_directed,
                );
                aux_vx_idx[aux_vx_type] += 1;
            } else if source == target {
                // self-loop with default weight, add two auxiliary vertices
                let vx_idx = aux_vx_idx.last_mut().unwrap();
                add_edge(&mut adj, source, *vx_idx, is_directed);
                add_edge(&mut adj, *vx_idx, *vx_idx + 1, is_directed);
                add_edge(&mut adj, *vx_idx + 1, target, is_directed);
                *vx_idx += 2;
            } else {
                add_edge(&mut adj, source, target, is_directed);
            }
        }

        let mut g = NautySparse::new(num_nauty_vertices, num_nauty_edges);
        let mut vpos = 0;
        for (adj, d, v) in izip!(adj, &mut g.d, &mut g.v) {
            *d = adj.len() as c_int;
            *v = vpos;
            let start = vpos as usize;
            let end = start + *d as usize;
            g.e[start..end].copy_from_slice(&adj);
            vpos += *d as size_t
        }
        debug_assert!(g.v.len() >= node_weights.len());

        let mut ptn = vec![1; g.v.len()];
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
        let lab = (0..g.v.len() as i32).collect();
        let nodes = Nodes {
            weights: node_weights,
            lab,
            ptn,
        };
        Self {
            g,
            edges: edge_weights,
            nodes,
            dir: PhantomData,
        }
    }
}

impl<N, E, Ty, Ix> From<SparseGraph<N, E, Ty>> for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    E: Ord,
{
    fn from(mut g: SparseGraph<N, E, Ty>) -> Self {
        debug_assert_eq!(g.g.v.len(), g.g.d.len());
        debug_assert_eq!(g.g.v.len(), g.nodes.ptn.len());
        debug_assert_eq!(g.g.v.len(), g.nodes.lab.len());
        debug_assert!(g.g.v.len() >= g.nodes.weights.len());
        let is_directed = Ty::is_directed();
        let nedges = if is_directed {
            g.g.e.len()
        } else {
            debug_assert_eq!(g.g.e.len() % 2, 0);
            g.g.e.len() / 2
        };
        let mut res = Graph::with_capacity(g.nodes.weights.len(), nedges);

        // find relabelling from `lab`
        let mut relabel = vec![0; g.nodes.lab.len()];
        for (new, old) in g.nodes.lab.into_iter().enumerate() {
            relabel[old as usize] = new;
        }

        // nodes
        let mut node_weights =
            Vec::from_iter(izip!(relabel.iter().copied(), g.nodes.weights));
        node_weights.sort_unstable_by_key(|e| e.0);
        for (n, i) in node_weights.iter().map(|(i, _w)| i).enumerate() {
            debug_assert_eq!(n, *i as usize)
        }
        for (_, w) in node_weights {
            res.add_node(w);
        }

        // edges
        let mut edges = Vec::new();
        let node_info = g.g.v.iter().zip(g.g.d.iter()).take(res.node_count());
        for (source, (&adj_pos, &degree)) in node_info.enumerate() {
            let source = source as c_int;
            let start = adj_pos as usize;
            let end = start + degree as usize;
            for mut target in g.g.e[start..end].iter().copied() {
                // by construction there can't be any self-loops
                debug_assert_ne!(source, target);
                if !is_directed && source > target {
                    continue;
                }
                // remove auxiliary nodes
                let mut previous = source;
                while target as usize >= res.node_count() {
                    let d = g.g.d[target as usize];
                    debug_assert_eq!(d, if is_directed { 1 } else { 2 });
                    let start = g.g.v[target as usize] as usize;
                    let end = start + d as usize;
                    let next = g.g.e[start..end]
                        .iter()
                        .copied()
                        .find(|&t| t != previous)
                        .unwrap();
                    previous = target;
                    target = next;
                }
                if !is_directed && source > target {
                    continue;
                }
                let mut new_source = relabel[source as usize];
                let mut new_target = relabel[target as usize];
                if !is_directed && new_source > new_target {
                    std::mem::swap(&mut new_source, &mut new_target);
                }

                let weights =
                    g.edges.remove(&(source as usize, target as usize));
                if let Some(weights) = weights {
                    for w in weights {
                        edges.push((new_source, new_target, w));
                    }
                } else {
                    // the same edge can only be hit twice in self-loops
                    debug_assert_eq!(source, target);
                }
            }
        }
        edges.sort_unstable();
        for (source, target, weight) in edges {
            use petgraph::visit::NodeIndexable;
            let source = res.from_index(source as usize);
            let target = res.from_index(target as usize);
            res.add_edge(source, target, weight);
        }
        res
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

    fn tst_conv<N, E, Ty, Ix>(g: Graph<N, E, Ty, Ix>)
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

    #[test]
    fn simple_conversion() {
        log_init();

        tst_conv(Graph::<(), (), _>::new_undirected());
        tst_conv(UnGraph::<(), ()>::from_edges([(0, 1), (2, 0)]));
        tst_conv(UnGraph::<(), i32>::from_edges([(0, 1, -1), (2, 0, 1)]));
        tst_conv(DiGraph::<(), ()>::from_edges([
            (0, 1),
            (1, 1),
            (0, 2),
            (2, 0),
        ]));
        tst_conv(DiGraph::<(), u32>::from_edges([
            (0, 1, 0),
            (1, 1, 0),
            (0, 2, 0),
            (2, 0, 1),
        ]));
    }

    #[test]
    fn random_conversion_undirected() {
        log_init();
        for g in GraphIter::<Undirected>::default().take(1000) {
            tst_conv(g);
        }
    }

    #[test]
    fn random_conversion_directed() {
        log_init();
        for g in GraphIter::<Directed>::default().take(700) {
            tst_conv(g);
        }
    }
}
