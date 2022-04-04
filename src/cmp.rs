use petgraph::{graph::IndexType, visit::EdgeRef, EdgeType, Graph};

/// Check if two objects are identical
pub trait IsIdentical {
    fn is_identical(&self, other: &Self) -> bool;
}

impl<N: Eq, E: Eq, Ty: EdgeType, Ix: IndexType> IsIdentical
    for Graph<N, E, Ty, Ix>
{
    fn is_identical(&self, other: &Self) -> bool {
        self.node_weights().eq(other.node_weights())
            && self.edge_references().zip(other.edge_references()).all(
                |(e1, e2)| {
                    e1.source() == e2.source()
                        && e1.target() == e2.target()
                        && e1.weight() == e2.weight()
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn ident() {
        let g1 = UnGraph::<(), ()>::from_edges([(0, 1), (0, 2)]);
        let g2 = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)]);
        assert!(g1.is_identical(&g1));
        assert!(g2.is_identical(&g2));
        assert!(!g1.is_identical(&g2));
    }

    #[test]
    fn ident_node_wt() {
        use petgraph::visit::NodeIndexable;

        let g1 = UnGraph::<u8, ()>::from_edges([(0, 1), (0, 1)]);
        let mut g2 = UnGraph::<u8, ()>::from_edges([(0, 1), (0, 1)]);
        assert!(g1.is_identical(&g2));
        *g2.node_weight_mut(g2.from_index(0)).unwrap() = 1;
        assert!(!g1.is_identical(&g2));
    }

    #[test]
    fn ident_edge_wt() {
        let g1 = UnGraph::<(), u8>::from_edges([(0, 1, 0), (0, 1, 0)]);
        let g2 = UnGraph::<(), u8>::from_edges([(0, 1, 0), (0, 1, 1)]);
        assert!(!g1.is_identical(&g2));
    }
}
