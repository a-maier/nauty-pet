use std::cmp::Ord;
use std::convert::From;
use std::hash::Hash;
use std::marker::PhantomData;
use std::os::raw::c_int;

use ahash::AHashMap;
use itertools::izip;
use nauty_Traces_sys::{size_t, SparseGraph as NautySparse};
use petgraph::{
    graph::{Graph, IndexType},
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

fn relabel_to_contiguous_node_weights<N: Ord>(
    nodes: &mut[N],
) -> Vec<usize> {
    let mut new_ord = Vec::from_iter(0..nodes.len());
    new_ord.sort_unstable_by(|&i, &j| nodes[i].cmp(&nodes[j]));
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

fn apply_perm<T>(slice: &mut [T], mut new_pos: Vec<usize>) {
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

impl<N, E, Ty, Ix> From<Graph<N, E, Ty, Ix>> for SparseGraph<(N, Vec<E>), E, Ty>
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
        let mut node_weights = Vec::from_iter(
            nodes.into_iter().map(
                |n| (n.weight, Vec::new())
            )
        );

        // edge weights
        // we combine multiple edges into a single one with an
        // effective weight given by the sorted vector of the
        // individual weights
        // self-loops are removed and their weights instead appended
        // to the corresponding node weight
        let mut edge_weights: AHashMap<_, Vec<E>> = AHashMap::new();
        for (edge, wt) in izip!(edges, e.into_iter().map(|e| e.weight)) {
            if edge.0 == edge.1 {
                node_weights[edge.0].1.push(wt);
            } else {
                edge_weights.entry(edge).or_default().push(wt);
            }
        }
        for v in edge_weights.values_mut() {
            v.sort_unstable();
        }
        let relabel = relabel_to_contiguous_node_weights(&mut node_weights);
        let edge_weights: AHashMap<_, _> = edge_weights.into_iter().map(
            |(e, wt)| {
                let mut e = (relabel[e.0], relabel[e.1]);
                if !is_directed && e.0 > e.1 {
                    std::mem::swap(&mut e.0, &mut e.1)
                }
                (e, wt)
            }
        ).collect();

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

impl<N, E, Ty, Ix> From<SparseGraph<(N, Vec<E>), E, Ty>> for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    E: Ord,
{
    fn from(mut g: SparseGraph<(N, Vec<E>), E, Ty>) -> Self {
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
        } + g.nodes.weights.iter().map(|(_, loops)| loops.len()).sum::<usize>();
        let mut res = Graph::with_capacity(g.nodes.weights.len(), nedges);

        // find relabelling from `lab`
        let mut relabel = vec![0; g.nodes.lab.len()];
        for (new, old) in g.nodes.lab.into_iter().enumerate() {
            relabel[old as usize] = new;
        }

        let mut edges = Vec::new();

        // nodes + self-loops
        let mut node_weights =
            Vec::from_iter(izip!(relabel.iter().copied(), g.nodes.weights));
        node_weights.sort_unstable_by_key(|e| e.0);
        for (n, i) in node_weights.iter().map(|(i, _w)| i).enumerate() {
            debug_assert_eq!(n, *i as usize)
        }
        for (_, (w, loops)) in node_weights {
            for w in loops {
                edges.push((res.node_count(), res.node_count(), w))
            }
            res.add_node(w);
        }

        // edges
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
                if target as usize >= res.node_count() {
                    let d = g.g.d[target as usize];
                    debug_assert_eq!(d, if is_directed { 1 } else { 2 });
                    let start = g.g.v[target as usize] as usize;
                    let end = start + d as usize;
                    target = g.g.e[start..end]
                        .iter()
                        .copied()
                        .find(|&t| t != source)
                        .unwrap();
                }
                if !is_directed && source > target {
                    continue;
                }
                let mut new_source = relabel[source as usize];
                let mut new_target = relabel[target as usize];
                if !is_directed && new_source > new_target {
                    std::mem::swap(&mut new_source, &mut new_target);
                }

                let weights = g.edges
                    .remove(&(source as usize, target as usize))
                    .unwrap();
                for w in weights {
                    edges.push((new_source, new_target, w));
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
