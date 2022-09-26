# nauty-pet

Canonical graph labelling.

Leverages [nauty and Traces](http://pallini.di.uniroma1.it/) to
find [canonical
labellings](https://en.wikipedia.org/wiki/Graph_canonization) for
[petgraph](https://github.com/petgraph/petgraph) graphs.

## Example

```rust
use petgraph::graph::UnGraph;
use nauty_pet::prelude::*;

// Two different vertex labellings for the tree graph with two edges
let g1 = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)]);
let g2 = UnGraph::<(), ()>::from_edges([(0, 1), (0, 2)]);

// There are two equivalent labellings
let automorphism_info = g1.clone().try_into_autom().unwrap();
assert_eq!(automorphism_info.grpsize(), 2.);

// The canonical forms are identical
let c1 = g1.clone().into_canon();
let c2 = g2.clone().into_canon();
assert!(c1.is_identical(&c2));

// Alternatively, we can use a dedicated `struct` for canonically
// labelled graphs
let c1 = CanonGraph::from(g1);
let c2 = CanonGraph::from(g2);
assert_eq!(c1, c2);
```

## Features

* `serde-1`: Enables serialisation of
             [CanonGraph](graph::CanonGraph) objects using
             [serde](https://crates.io/crates/serde).

* `stable`: Ensures deterministic behaviour when node or edge
            weights are distinguishable, but compare equal.

To enable features `feature1`, `feature2` add the following to
your Cargo.toml:
```toml
[dependencies]
nauty-pet = { version = "0.8", features = ["feature1", "feature2"] }
```

License: Apache-2.0
