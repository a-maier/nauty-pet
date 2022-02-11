mod canon;
mod sparse_graph;

pub use canon::{
    ToCanon, ToCanonNautyDense, ToCanonNautySparse, ToCanonTraces, TracesError,
};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
