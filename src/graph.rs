use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::convert::From;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use crate::{IntoCanon, IsIdentical};

use petgraph::{
    graph::{DefaultIx, Graph, IndexType, NodeIndex},
    stable_graph::StableGraph,
    visit::EdgeRef,
    Directed, EdgeType, IntoWeightedEdge, Undirected,
};

pub type CanonDiGraph<N, E, Ix> = CanonGraph<N, E, Directed, Ix>;
pub type CanonUnGraph<N, E, Ix> = CanonGraph<N, E, Undirected, Ix>;

/// Canonically labelled graph
///
/// The interface closely mimics
/// [petgraph::Graph](https://docs.rs/petgraph/latest/petgraph/graph/struct.Graph.html).
/// The exception are mutating methods, which could potentially
/// be misused to destroy the canonical labelling.
///
/// # Example
///
/// ```rust
/// use std::collections::HashSet;
/// use petgraph::graph::UnGraph;
/// use nauty_pet::prelude::*;
///
/// let g = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)]);
///
/// // canonical labelling
/// let g = CanonGraph::from(g);
///
/// // we can now compare `g` to other canonically labelled graphs and
/// // use it in hash sets and tables.
/// assert_eq!(g, g);
/// let mut graphs = HashSet::new();
/// graphs.insert(g);
/// ```
///
#[cfg_attr(feature = "serde-1", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default)]
pub struct CanonGraph<N, E, Ty: EdgeType = Directed, Ix: IndexType = DefaultIx>(
    Graph<N, E, Ty, Ix>,
);

impl<N, E, Ty: EdgeType, Ix: IndexType> Deref for CanonGraph<N, E, Ty, Ix> {
    type Target = Graph<N, E, Ty, Ix>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<N, E, Ty: EdgeType, Ix: IndexType> AsRef<Graph<N, E, Ty, Ix>> for CanonGraph<N, E, Ty, Ix> {
    fn as_ref(&self) -> &Graph<N, E, Ty, Ix> {
        &self.0
    }
}

impl<N, E, Ty, Ix> CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    Graph<N, E, Ty, Ix>: IntoCanon,
{
    pub fn from_edges<I>(iterable: I) -> Self
    where
        I: IntoIterator,
        I::Item: IntoWeightedEdge<E>,
        <I::Item as IntoWeightedEdge<E>>::NodeId: Into<NodeIndex<Ix>>,
        N: Default,
    {
        Self(Graph::from_edges(iterable).into_canon())
    }
}

impl<N, E, Ty, Ix> From<Graph<N, E, Ty, Ix>> for CanonGraph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: IntoCanon,
    Ty: EdgeType,
    Ix: IndexType,
{
    fn from(g: Graph<N, E, Ty, Ix>) -> Self {
        Self(g.into_canon())
    }
}

impl<N, E, Ty, Ix> From<CanonGraph<N, E, Ty, Ix>> for Graph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn from(g: CanonGraph<N, E, Ty, Ix>) -> Self {
        g.0
    }
}

impl<N, E, Ty, Ix> From<CanonGraph<N, E, Ty, Ix>> for StableGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn from(g: CanonGraph<N, E, Ty, Ix>) -> Self {
        g.0.into()
    }
}

impl<N, E, Ty, Ix> From<StableGraph<N, E, Ty, Ix>> for CanonGraph<N, E, Ty, Ix>
where
    Graph<N, E, Ty, Ix>: IntoCanon,
    Ty: EdgeType,
    Ix: IndexType,
{
    fn from(g: StableGraph<N, E, Ty, Ix>) -> Self {
        let g: Graph<_, _, _, _> = g.into();
        Self(g.into_canon())
    }
}

impl<N, E, Ty, Ix> PartialEq for CanonGraph<N, E, Ty, Ix>
where
    N: PartialEq,
    E: PartialEq,
    Ty: EdgeType,
    Ix: IndexType,
{
    fn eq(&self, other: &Self) -> bool {
        self.is_identical(&other)
    }
}
impl<N: Eq, E: Eq, Ty: EdgeType, Ix: IndexType> Eq
    for CanonGraph<N, E, Ty, Ix>
{
}

impl<N: Hash, E: Hash, Ty: EdgeType, Ix: IndexType> Hash
    for CanonGraph<N, E, Ty, Ix>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for w in self.node_weights() {
            w.hash(state)
        }
        for e in self.edge_references() {
            e.source().hash(state);
            e.target().hash(state);
            e.weight().hash(state);
        }
    }
}

// Doesn't have to make much sense, just give a reproducible ordering
impl<N: Ord, E: Ord, Ty: EdgeType, Ix: IndexType> Ord
    for CanonGraph<N, E, Ty, Ix>
{
    fn cmp(&self, other: &Self) -> Ordering {
        let cmp = self.node_weights().cmp(other.node_weights());
        if cmp == Ordering::Equal {
            let my_edges = self
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight()));
            let other_edges = other
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight()));
            my_edges.cmp(other_edges)
        } else {
            cmp
        }
    }
}

impl<N, E, Ty, Ix> PartialOrd for CanonGraph<N, E, Ty, Ix>
where
    N: PartialOrd,
    E: PartialOrd,
    Ty: EdgeType,
    Ix: IndexType,
{
    // TODO: code duplication
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let cmp = self.node_weights().partial_cmp(other.node_weights());
        if cmp == Some(Ordering::Equal) {
            let my_edges = self
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight()));
            let other_edges = other
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight()));
            my_edges.partial_cmp(other_edges)
        } else {
            cmp
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use log::debug;
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256Plus;
    use testing::{randomize_labels, GraphIter};
    use petgraph::graph::{UnGraph, DiGraph};

    fn log_init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn random_canon_graph() {
        log_init();

        let mut rng = Xoshiro256Plus::seed_from_u64(0);
        let graphs = GraphIter::<Undirected>::default();

        for g in graphs.take(1000) {
            debug!("Initial graph: {g:#?}");
            let gg = randomize_labels(g.clone(), &mut rng);
            debug!("Randomised graph: {gg:#?}");
            let g = CanonGraph::from(g);
            debug!("Canonical graph (from initial): {g:#?}");
            let gg = CanonGraph::from(gg);
            debug!("Canonical graph (from randomised): {gg:#?}");
            assert_eq!(g, gg);
        }
    }

    #[test]
    fn test_eq_ord() {
        assert_eq!(CanonGraph::from(UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)])), CanonGraph::from(UnGraph::<(), ()>::from_edges([(0, 1), (0, 2)])));
        assert_ne!(CanonGraph::from(DiGraph::<(), ()>::from_edges([(0, 1), (1, 2)])), CanonGraph::from(DiGraph::<(), ()>::from_edges([(0, 1), (0, 2)])));

        assert_eq!(CanonGraph::from(UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)])).cmp(&CanonGraph::from(UnGraph::<(), ()>::from_edges([(0, 1), (0, 2)]))), Ordering::Equal);
        assert_ne!(CanonGraph::from(DiGraph::<(), ()>::from_edges([(0, 1), (1, 2)])).cmp(&CanonGraph::from(DiGraph::<(), ()>::from_edges([(0, 1), (0, 2)]))), Ordering::Equal);
    }
}
