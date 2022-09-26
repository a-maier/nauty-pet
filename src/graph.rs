use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::convert::From;
use std::hash::{Hash, Hasher};
use std::ops::Index;

use crate::{IntoCanon, IsIdentical};

use petgraph::{
    data::DataMap,
    graph::{
        DefaultIx, Edge, EdgeIndex, EdgeIndices, EdgeReference, EdgeReferences,
        Edges, EdgesConnecting, Externals, Graph, IndexType, Neighbors, Node,
        NodeIndex, NodeIndices, NodeReferences,
    },
    stable_graph::StableGraph,
    visit::{
        Data, EdgeCount, EdgeIndexable, EdgeRef, GetAdjacencyMatrix, GraphBase,
        GraphProp, IntoEdgeReferences, IntoEdges, IntoEdgesDirected,
        IntoNeighbors, IntoNeighborsDirected, IntoNodeIdentifiers,
        IntoNodeReferences, NodeCount, NodeIndexable,
    },
    Directed, Direction, EdgeType, IntoWeightedEdge, Undirected,
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

impl<N, E, Ty, Ix> CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    /// Gets a reference to the underlying `petgraph::Graph`
    pub fn get(&self) -> &Graph<N, E, Ty, Ix> {
        &self.0
    }

    pub fn node_count(&self) -> usize {
        self.0.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    pub fn is_directed(&self) -> bool {
        self.0.is_directed()
    }

    pub fn node_weight(&self, a: NodeIndex<Ix>) -> Option<&N> {
        self.0.node_weight(a)
    }

    pub fn edge_weight(&self, e: EdgeIndex<Ix>) -> Option<&E> {
        self.0.edge_weight(e)
    }

    pub fn edge_endpoints(&self, e: EdgeIndex<Ix>) -> Option<&E> {
        self.0.edge_weight(e)
    }

    pub fn neighbors(&self, a: NodeIndex<Ix>) -> Neighbors<'_, E, Ix> {
        self.0.neighbors(a)
    }

    pub fn neighbors_directed(
        &self,
        a: NodeIndex<Ix>,
        dir: Direction,
    ) -> Neighbors<'_, E, Ix> {
        self.0.neighbors_directed(a, dir)
    }

    pub fn neighbors_undirected(
        &self,
        a: NodeIndex<Ix>,
    ) -> Neighbors<'_, E, Ix> {
        self.0.neighbors_undirected(a)
    }

    pub fn edges(&self, a: NodeIndex<Ix>) -> Edges<'_, E, Ty, Ix> {
        self.0.edges(a)
    }

    pub fn edges_directed(
        &self,
        a: NodeIndex<Ix>,
        dir: Direction,
    ) -> Edges<'_, E, Ty, Ix> {
        self.0.edges_directed(a, dir)
    }

    pub fn edges_connecting(
        &self,
        a: NodeIndex<Ix>,
        b: NodeIndex<Ix>,
    ) -> EdgesConnecting<'_, E, Ty, Ix> {
        self.0.edges_connecting(a, b)
    }

    pub fn contains_edge(&self, a: NodeIndex<Ix>, b: NodeIndex<Ix>) -> bool {
        self.0.contains_edge(a, b)
    }

    pub fn find_edge_undirected(
        &self,
        a: NodeIndex<Ix>,
        b: NodeIndex<Ix>,
    ) -> Option<(EdgeIndex<Ix>, Direction)> {
        self.0.find_edge_undirected(a, b)
    }

    pub fn externals(&self, dir: Direction) -> Externals<'_, N, Ty, Ix> {
        self.0.externals(dir)
    }

    pub fn node_indices(&self) -> NodeIndices<Ix> {
        self.0.node_indices()
    }

    // TODO: return `NodeWeights<'_, N, Ix>`, but it's private in petgraph 0.6.0
    pub fn node_weights(&self) -> impl Iterator<Item = &N> {
        self.0.node_weights()
    }

    pub fn edge_indices(&self) -> EdgeIndices<Ix> {
        self.0.edge_indices()
    }

    pub fn edge_references(&self) -> EdgeReferences<'_, E, Ix> {
        self.0.edge_references()
    }

    // TODO: return `EdgeWeights<'_, E, Ix>`, but it's private in petgraph 0.6.0
    pub fn edge_weights(&self) -> impl Iterator<Item = &E> {
        self.0.edge_weights()
    }

    pub fn raw_nodes(&self) -> &[Node<N, Ix>] {
        self.0.raw_nodes()
    }

    pub fn raw_edges(&self) -> &[Edge<E, Ix>] {
        self.0.raw_edges()
    }

    pub fn into_nodes_edges(self) -> (Vec<Node<N, Ix>>, Vec<Edge<E, Ix>>) {
        self.0.into_nodes_edges()
    }

    pub fn next_edge(
        &self,
        e: EdgeIndex<Ix>,
        dir: Direction,
    ) -> Option<EdgeIndex<Ix>> {
        self.0.next_edge(e, dir)
    }

    pub fn capacity(&self) -> (usize, usize) {
        self.0.capacity()
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

impl<N, E, Ty, Ix> AsRef<Graph<N, E, Ty, Ix>> for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn as_ref(&self) -> &Graph<N, E, Ty, Ix> {
        &self.0
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

impl<N, E, Ty, Ix> Data for CanonGraph<N, E, Ty, Ix>
where
    CanonGraph<N, E, Ty, Ix>: GraphBase,
    Graph<N, E, Ty, Ix>: Data,
    Ty: EdgeType,
    Ix: IndexType,
{
    type NodeWeight = N;
    type EdgeWeight = E;
}

impl<N, E, Ty, Ix> DataMap for CanonGraph<N, E, Ty, Ix>
where
    CanonGraph<N, E, Ty, Ix>:
        GraphBase<NodeId = NodeIndex<Ix>, EdgeId = EdgeIndex<Ix>>,
    CanonGraph<N, E, Ty, Ix>: Data<NodeWeight = N, EdgeWeight = E>,
    Ty: EdgeType,
    Ix: IndexType,
{
    fn node_weight(&self, id: Self::NodeId) -> Option<&Self::NodeWeight> {
        Self::node_weight(self, id)
    }
    fn edge_weight(&self, id: Self::EdgeId) -> Option<&Self::EdgeWeight> {
        Self::edge_weight(self, id)
    }
}

impl<N, E, Ty, Ix> EdgeCount for CanonGraph<N, E, Ty, Ix>
where
    CanonGraph<N, E, Ty, Ix>:
        GraphBase<NodeId = NodeIndex<Ix>, EdgeId = EdgeIndex<Ix>>,
    Ty: EdgeType,
    Ix: IndexType,
{
    fn edge_count(&self) -> usize {
        self.0.edge_count()
    }
}

impl<N, E, Ty, Ix> EdgeIndexable for CanonGraph<N, E, Ty, Ix>
where
    CanonGraph<N, E, Ty, Ix>:
        GraphBase<NodeId = NodeIndex<Ix>, EdgeId = EdgeIndex<Ix>>,
    Ty: EdgeType,
    Ix: IndexType,
{
    fn edge_bound(&self) -> usize {
        self.0.edge_bound()
    }

    fn to_index(&self, ix: EdgeIndex<Ix>) -> usize {
        EdgeIndexable::to_index(&self.0, ix)
    }

    fn from_index(&self, ix: usize) -> Self::EdgeId {
        EdgeIndexable::from_index(&self.0, ix)
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

impl<N, E, Ty, Ix> GetAdjacencyMatrix for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
    CanonGraph<N, E, Ty, Ix>:
        GraphBase<NodeId = NodeIndex<Ix>, EdgeId = EdgeIndex<Ix>>,
    Graph<N, E, Ty, Ix>:
        GraphBase<NodeId = NodeIndex<Ix>, EdgeId = EdgeIndex<Ix>>,
    Graph<N, E, Ty, Ix>: GetAdjacencyMatrix,
{
    type AdjMatrix = <Graph<N, E, Ty, Ix> as GetAdjacencyMatrix>::AdjMatrix;

    fn adjacency_matrix(&self) -> Self::AdjMatrix {
        self.0.adjacency_matrix()
    }

    fn is_adjacent(
        &self,
        matrix: &Self::AdjMatrix,
        a: NodeIndex<Ix>,
        b: NodeIndex<Ix>,
    ) -> bool {
        self.0.is_adjacent(matrix, a, b)
    }
}

impl<N, E, Ty, Ix> GraphBase for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type NodeId = NodeIndex<Ix>;

    type EdgeId = EdgeIndex<Ix>;
}

impl<N, E, Ty, Ix> GraphProp for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type EdgeType = Ty;

    fn is_directed(&self) -> bool {
        self.0.is_directed()
    }
}

impl<N, E, Ty, Ix> Index<EdgeIndex<Ix>> for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type Output = E;

    fn index(&self, index: EdgeIndex<Ix>) -> &E {
        self.0.index(index)
    }
}

impl<N, E, Ty, Ix> Index<NodeIndex<Ix>> for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type Output = N;

    fn index(&self, index: NodeIndex<Ix>) -> &N {
        self.0.index(index)
    }
}

impl<'a, N: 'a, E: 'a, Ty, Ix> IntoEdgeReferences
    for &'a CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type EdgeRef = EdgeReference<'a, E, Ix>;
    type EdgeReferences = EdgeReferences<'a, E, Ix>;

    fn edge_references(self) -> Self::EdgeReferences {
        self.0.edge_references()
    }
}

impl<'a, N, E, Ty, Ix> IntoEdges for &'a CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type Edges = Edges<'a, E, Ty, Ix>;

    fn edges(self, a: Self::NodeId) -> Self::Edges {
        self.0.edges(a)
    }
}

impl<'a, N, E, Ty, Ix> IntoEdgesDirected for &'a CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type EdgesDirected = Edges<'a, E, Ty, Ix>;

    fn edges_directed(
        self,
        a: Self::NodeId,
        dir: Direction,
    ) -> Self::EdgesDirected {
        self.0.edges_directed(a, dir)
    }
}

impl<'a, N, E: 'a, Ty, Ix> IntoNeighbors for &'a CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type Neighbors = Neighbors<'a, E, Ix>;

    fn neighbors(self, n: NodeIndex<Ix>) -> Neighbors<'a, E, Ix> {
        self.neighbors(n)
    }
}

impl<'a, N, E: 'a, Ty, Ix> IntoNeighborsDirected
    for &'a CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type NeighborsDirected = Neighbors<'a, E, Ix>;

    fn neighbors_directed(
        self,
        n: NodeIndex<Ix>,
        d: Direction,
    ) -> Neighbors<'a, E, Ix> {
        self.neighbors_directed(n, d)
    }
}

impl<'a, N, E: 'a, Ty, Ix> IntoNodeIdentifiers for &'a CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type NodeIdentifiers = NodeIndices<Ix>;

    fn node_identifiers(self) -> NodeIndices<Ix> {
        self.0.node_identifiers()
    }
}

impl<'a, N, E, Ty, Ix> IntoNodeReferences for &'a CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    type NodeRef = (NodeIndex<Ix>, &'a N);
    type NodeReferences = NodeReferences<'a, N, Ix>;

    fn node_references(self) -> Self::NodeReferences {
        self.0.node_references()
    }
}

impl<N, E, Ty, Ix> NodeCount for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn node_count(&self) -> usize {
        self.0.node_count()
    }
}

impl<N, E, Ty, Ix> NodeIndexable for CanonGraph<N, E, Ty, Ix>
where
    Ty: EdgeType,
    Ix: IndexType,
{
    fn node_bound(&self) -> usize {
        self.0.node_bound()
    }

    fn to_index(&self, ix: NodeIndex<Ix>) -> usize {
        NodeIndexable::to_index(&self.0, ix)
    }

    fn from_index(&self, ix: usize) -> Self::NodeId {
        NodeIndexable::from_index(&self.0, ix)
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
        self.0.is_identical(&other.0)
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
        for w in self.0.node_weights() {
            w.hash(state)
        }
        for e in self.0.edge_references() {
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
        let cmp = self.0.node_weights().cmp(other.0.node_weights());
        if cmp == Ordering::Equal {
            let my_edges = self
                .0
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight()));
            let other_edges = self
                .0
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
        let cmp = self.0.node_weights().partial_cmp(other.0.node_weights());
        if cmp == Some(Ordering::Equal) {
            let my_edges = self
                .0
                .edge_references()
                .map(|e| (e.source(), e.target(), e.weight()));
            let other_edges = self
                .0
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
}
