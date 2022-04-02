use std::cmp::{Ord, Ordering};
use std::convert::From;
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
    pub(crate) node_weights: Weights<N>,
    edge_weights: AHashMap<(c_int, c_int), Vec<E>>,
    dir: PhantomData<D>,
}

#[derive(Debug, Default, Clone, Hash)]
pub(crate) struct Weights<N> {
    pub(crate) lab: Vec<c_int>,
    pub(crate) ptn: Vec<c_int>,
    pub(crate) weights: Vec<N>,
}

impl<N, E, Ty, Ix> From<Graph<N, E, Ty, Ix>> for SparseGraph<N, E, Ty>
where
    Ty: EdgeType,
    Ix: IndexType,
    N: Ord,
{
    fn from(g: Graph<N, E, Ty, Ix>) -> Self {
        use petgraph::visit::NodeIndexable;
        let is_directed = g.is_directed();
        let mut edges = Vec::from_iter(g.edge_references().map(|e| {
            (
                g.to_index(e.source()) as c_int,
                g.to_index(e.target()) as c_int,
            )
        }));
        let (nodes, e) = g.into_nodes_edges();
        let e_weights = Vec::from_iter(e.into_iter().map(|e| e.weight));
        let mut node_weights = Vec::from_iter(
            nodes
                .into_iter()
                .enumerate()
                .map(|(idx, n)| (n.weight, idx)),
        );
        // TODO: use sort_by_key, but that has lifetime problems?
        node_weights.sort_unstable_by(|i, j| i.0.cmp(&j.0));
        let mut renumber = vec![0; node_weights.len()];
        for (new_idx, (_, old_idx)) in node_weights.iter().enumerate() {
            renumber[*old_idx] = new_idx as c_int;
        }
        let node_weights =
            Vec::from_iter(node_weights.into_iter().map(|(wt, _old_idx)| wt));
        for edge in &mut edges {
            edge.0 = renumber[edge.0 as usize];
            edge.1 = renumber[edge.1 as usize];
            if !is_directed && edge.0 > edge.1 {
                std::mem::swap(&mut edge.0, &mut edge.1);
            }
        }
        let g = sg_from(node_weights.len(), &edges, is_directed);
        let mut edge_weights: AHashMap<_, Vec<E>> = AHashMap::new();
        for (edge, wt) in izip!(edges, e_weights) {
            edge_weights.entry(edge).or_default().push(wt)
        }
        let mut ptn = vec![1; node_weights.len()];
        for (ptn, wts) in izip!(&mut ptn, node_weights.windows(2)) {
            if wts[1] > wts[0] {
                *ptn = 0;
            }
        }
        if let Some(last) = ptn.last_mut() {
            *last = 0;
        }
        let lab = (0..node_weights.len() as i32).collect();
        let node_weights = Weights {
            weights: node_weights,
            lab,
            ptn,
        };
        Self {
            g,
            edge_weights,
            node_weights,
            dir: PhantomData,
        }
    }
}

fn sg_from(
    num_nodes: usize,
    edges: &[(c_int, c_int)],
    is_directed: bool,
) -> NautySparse {
    let mut sg = NautySparse::new(
        num_nodes,
        if is_directed {
            edges.len()
        } else {
            2 * edges.len()
        },
    );
    let mut adj = vec![vec![]; num_nodes];
    for (source, target) in edges {
        adj[*source as usize].push(*target);
        if !is_directed {
            adj[*target as usize].push(*source);
        }
    }
    let mut vpos = 0;
    for (adj, d, v) in izip!(adj, &mut sg.d, &mut sg.v) {
        *d = adj.len() as c_int;
        *v = vpos;
        let start = vpos as usize;
        let end = start + *d as usize;
        sg.e[start..end].copy_from_slice(&adj);
        vpos += *d as size_t
    }
    sg
}

impl<N, E, Ty, Ix> From<SparseGraph<N, E, Ty>> for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    E: Ord,
{
    fn from(mut g: SparseGraph<N, E, Ty>) -> Self {
        debug_assert_eq!(g.g.v.len(), g.g.d.len());
        debug_assert_eq!(g.g.v.len(), g.node_weights.lab.len());
        debug_assert_eq!(g.g.v.len(), g.node_weights.ptn.len());
        debug_assert_eq!(g.g.v.len(), g.node_weights.weights.len());
        let is_directed = Ty::is_directed();
        let nedges = if is_directed {
            g.g.e.len()
        } else {
            debug_assert_eq!(g.g.e.len() % 2, 0);
            g.g.e.len() / 2
        };
        let mut res = Graph::with_capacity(g.g.v.len(), nedges);

        // find relabelling from `lab`
        let mut relabel = vec![0; g.node_weights.lab.len()];
        for (new, old) in g.node_weights.lab.into_iter().enumerate() {
            relabel[old as usize] = new;
        }

        // nodes
        let mut node_weights = Vec::from_iter(izip!(
            relabel.iter().copied(),
            g.node_weights.weights
        ));
        node_weights.sort_unstable_by_key(|e| e.0);
        for (n, i) in node_weights.iter().map(|(i, _w)| i).enumerate() {
            debug_assert_eq!(n, *i as usize)
        }
        for (_, w) in node_weights {
            res.add_node(w);
        }

        // edges
        let mut edges = Vec::new();
        for (source, (adj_pos, degree)) in izip!(g.g.v, g.g.d).enumerate() {
            let source = source as c_int;
            let start = adj_pos as usize;
            let end = start + degree as usize;
            let mut odd_self_loop = false;
            for target in g.g.e[start..end].iter().copied() {
                if !is_directed {
                    match source.cmp(&target) {
                        Ordering::Greater => continue,
                        Ordering::Less => {}
                        Ordering::Equal => {
                            // ignore exactly half of the self-loops
                            odd_self_loop = !odd_self_loop;
                            if odd_self_loop {
                                continue;
                            };
                        }
                    };
                }
                let w = g
                    .edge_weights
                    .get_mut(&(source, target))
                    .unwrap()
                    .pop()
                    .unwrap();

                // relabel
                let mut source = relabel[source as usize];
                let mut target = relabel[target as usize];
                if !is_directed && source > target {
                    std::mem::swap(&mut source, &mut target);
                }
                edges.push((source, target, w));
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
        E: Clone + Debug + Ord,
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
