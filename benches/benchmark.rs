use nauty_pet::prelude::*;
use nauty_pet::canon::{IntoCanonNautySparse, IntoCanonTraces};
use testing::{GraphIter, randomize_labels};

use criterion::{BatchSize, black_box, criterion_group, criterion_main, Criterion};
use petgraph::{
    EdgeType,
    graph::{Graph, IndexType, UnGraph},
    Directed, Undirected,
};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256Plus;

fn iso_nauty<Ty: EdgeType, Ix: IndexType>(
    graphs: impl IntoIterator<Item = (Graph<u8, u8, Ty, Ix>, Graph<u8, u8, Ty, Ix>)>
) -> bool {
    graphs.into_iter().all(|(g, h)| {
        g.into_canon_nauty_sparse().is_identical(
            &h.into_canon_nauty_sparse()
        )
    })
}

fn iso_traces<Ix: IndexType>(
    graphs: impl IntoIterator<Item = (UnGraph<u8, u8, Ix>, UnGraph<u8, u8, Ix>)>
) -> bool {
    graphs.into_iter().all(|(g, h)| {
        g.into_canon_traces().is_identical(
            &h.into_canon_traces()
        )
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(0);

    let graphs = Vec::from_iter(
        GraphIter::<Undirected>::default()
            .take(1000)
            .map(|g| (g.clone(), randomize_labels(g, &mut rng))),
    );
    let g = graphs.clone();
    c.bench_function("undirected nauty", move |b| {
        b.iter_batched(|| g.clone(), |g| iso_nauty(black_box(g)), BatchSize::SmallInput)
    });
    c.bench_function("undirected traces", move |b| {
        b.iter_batched(|| graphs.clone(), |g| iso_traces(black_box(g)), BatchSize::SmallInput)
    });

    let graphs = Vec::from_iter(
        GraphIter::<Directed>::default()
            .take(1000)
            .map(|g| (g.clone(), randomize_labels(g, &mut rng))),
    );
    c.bench_function("directed nauty", move |b| {
        b.iter_batched(|| graphs.clone(), |g| iso_nauty(black_box(g)), BatchSize::SmallInput)
    });

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
