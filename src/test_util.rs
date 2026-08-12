use std::collections::BTreeSet;

use petgraph::{
    EdgeType,
    graph::{Graph, IndexType},
    visit::EdgeRef,
};

use crate::nauty_graph;

pub(crate) fn generated_group(
    gens: &[Vec<usize>],
    n: usize,
) -> BTreeSet<Vec<usize>> {
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

pub(crate) fn apply_perm<N, E, Ty: EdgeType, Ix: IndexType>(
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
        res.add_edge(res.from_index(source), res.from_index(target), w.weight);
    }
    res
}
