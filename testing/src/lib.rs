use std::marker::PhantomData;

use petgraph::{
    algo::connected_components,
    graph::{IndexType, Graph},
    EdgeType,
    visit::EdgeRef,
};
use rand::{
    distributions::Uniform,
    prelude::*
};
use rand_distr::Normal;
use rand_xoshiro::Xoshiro256Plus;

pub struct GraphIter<Ty: EdgeType> {
    rng: Xoshiro256Plus,
    node_distr: Uniform<usize>,
    node_wt_distr: Uniform<u8>,
    edge_wt_distr: Uniform<u8>,
    edge_distr: Normal<f64>,
    edge_type: PhantomData<Ty>,
}

impl<Ty: EdgeType> Default for GraphIter<Ty> {
    fn default() -> Self {
        Self {
            rng: Xoshiro256Plus::seed_from_u64(0),
            node_distr: Uniform::from(1..10),
            node_wt_distr: Uniform::from(0..3),
            edge_wt_distr: Uniform::from(0..3),
            edge_distr: Normal::new(0., 1.0).unwrap(),
            edge_type: PhantomData
        }
    }
}

impl<Ty: EdgeType> Iterator for GraphIter<Ty> {
    type Item = Graph<u8, u8, Ty>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut rng = &mut self.rng;
        let mut g = Graph::default();
        let nnodes = self.node_distr.sample(&mut rng);
        for _ in 0..nnodes {
            g.add_node(self.node_wt_distr.sample(&mut rng));
        }
        for i in 0..nnodes {
            let start = if Ty::is_directed() { 0 } else { i };
            for j in start..nnodes {
                let nedges = self.edge_distr.sample(&mut rng)
                    .clamp(0.0, 1.0)
                    .round() as u64;
                for _ in 0..nedges {
                    use petgraph::visit::NodeIndexable;
                    let source = g.from_index(i);
                    let target = g.from_index(j);
                    g.add_edge(source, target, self.edge_wt_distr.sample(&mut rng));
                }
            }
        }
        if connected_components(&g) > 1 {
            return self.next()
        }
        Some(g)
    }
}

pub fn randomize_labels<N, E, Ty, Ix>(
    g: Graph<N, E, Ty, Ix>,
    rng: &mut impl Rng,
) -> Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    use petgraph::visit::NodeIndexable;
    let mut res = Graph::with_capacity(g.node_count(), g.edge_count());
    let edges = Vec::from_iter(g.edge_references().map(|e| {
        let source = g.to_index(e.source());
        let target = g.to_index(e.target());
        (source, target)
    }));
    let (nodes, edge_wts) = g.into_nodes_edges();
    let mut nodes =
        Vec::from_iter(nodes.into_iter().map(|n| n.weight).enumerate());
    nodes.shuffle(rng);
    let mut relabel = vec![0; nodes.len()];
    for (new, (old, w)) in nodes.into_iter().enumerate() {
        res.add_node(w);
        relabel[old] = new;
    }
    let edges = edges.into_iter().zip(edge_wts.into_iter()).map(|((source, target), w)| {
        (relabel[source], relabel[target], w.weight)
    });
    for (source, target, w) in edges {
        res.add_edge(res.from_index(source), res.from_index(target), w);
    }
    res
}
