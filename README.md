# rust-ocl-histogram
An image luminance (histogram) generator written in Rust. Uses both CPU multithreading and GPU acceleration via OpenCL in order to maximize performance while maintaining thread and memory safety. 

For the moment, histogram is displayed in the terminal itself using the tui library.

To run:
cargo run PATH_TO_IMAGE
