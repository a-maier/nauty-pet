# nauty-pet

Canonical graph labelling.

Leverages [nauty and Traces](http://pallini.di.uniroma1.it/) to
find [canonical
labellings](https://en.wikipedia.org/wiki/Graph_canonization) for
[petgraph](https://github.com/petgraph/petgraph) graphs.

## Usage

Version 2.6 or 2.7 of the nauty and Traces library has to be
installed first and _linked to the same C library as this crate_.

Afterwards, add this to your Cargo.toml:
```toml
[dependencies]
nauty-pet = "0.1"
```

## Example

```rust
use petgraph::graph::UnGraph;
use nauty_pet::ToCanon;

// Two different vertex labellings for the tree graph with two edges
let g1 = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2)]);
let g2 = UnGraph::<(), ()>::from_edges([(0, 1), (0, 2)]);

let c1 = g1.to_canon();
let c2 = g2.to_canon();
// c1 and c2 now have the same edges, up to permutation
#
```

## Caveats

- Edge weights are ignored and discarded.
- Only undirected graphs without self-loops have been tested so far.


License: Apache-2.0
