use petgraph::{graph::IndexType, EdgeType, Graph};

/// Check if two objects are identical
pub trait IsIdentical {
    fn is_identical(&self, other: &Self) -> bool;
}

impl<N: Eq, E: Eq, Ty: EdgeType, Ix: IndexType> IsIdentical
    for Graph<N, E, Ty, Ix>
{
    fn is_identical(&self, other: &Self) -> bool {
        self.node_weights().eq(other.node_weights())
            && self.edge_references().eq(other.edge_references())
    }
}
