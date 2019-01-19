extern crate image;
extern crate num_cpus;
extern crate ocl;
mod gpu;
mod threads;
mod ui;

use image::GenericImageView;
use ocl::core::Uchar3;
use std::env::args;
use std::iter::FromIterator;
use std::sync::mpsc;
use std::thread;

fn main() {
    let ar: Vec<String> = args().collect();
    let img = image::open(&ar[1]).unwrap();
    let (width, height) = img.dimensions();
    let pix: Vec<(u32, u32, image::Rgba<u8>)> = Vec::from_iter(img.pixels());
    let num_threads = num_cpus::get();
    let mut threads: Vec<thread::JoinHandle<()>> = Vec::with_capacity(num_threads - 1);
    let (tx, rx) = mpsc::channel();
    for i in 0..num_threads - 1 {
        let chunk: Vec<(u32, u32, image::Rgba<u8>)> =
            pix[i * pix.len() / (num_threads - 1)..(i + 1) * pix.len() / (num_threads - 1)].to_vec(); // split input equally among available threads
        let tmp = tx.clone();
        threads.push(thread::spawn(move || {
            for p in chunk {
                tmp.send(Uchar3::new(p.2[0], p.2[1], p.2[2])).unwrap(); // create 3-d vector with rgb values
            }
        }));
    }
    let gpu_thread = thread::spawn(move || {
        return gpu::luminance(rx, width as usize, height as usize);
    });
    for thread in threads {
        thread.join().unwrap();
    }
    let (lum, sums) = gpu_thread.join().unwrap().unwrap();
    println!("{:?}", lum.len());
    println!("{:?}", sums.len());
}
