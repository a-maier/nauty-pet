use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NautyError {
     #[error("Too much memory needed")]
    MTooBig,
     #[error("Too many nodes")]
    NTooBig,
    //  #[error("Aborted by user code")]
    // Aborted,
}
