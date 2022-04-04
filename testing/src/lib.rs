use std::marker::PhantomData;

use petgraph::{
    algo::connected_components,
    graph::Graph,
    EdgeType,
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
    pub edge_wt_distr: Uniform<u8>,
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
            edge_distr: Normal::new(0.5, 1.0).unwrap(),
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
