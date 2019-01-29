extern crate image;
extern crate num_cpus;
extern crate ocl;
mod gpu;
mod ui;

use image::GenericImageView;
use ocl::core::Uchar3;
use std::env::args;
use std::iter::FromIterator;
use std::sync::mpsc;
use std::thread;

fn main() {
    let ar: Vec<String> = args().collect();
    println!("Opening image {:?}", &ar[1]);
    let img = image::open(&ar[1]).unwrap(); // the image library's open function seems to be the largest performance bottleneck with large files
    println!("Opened image");
    let (width, height) = img.dimensions();
    let pix: Vec<(u32, u32, image::Rgba<u8>)> = Vec::from_iter(img.pixels());
    let num_threads = num_cpus::get();
    let mut threads: Vec<thread::JoinHandle<()>> = Vec::with_capacity(num_threads);
    let (tx, rx) = mpsc::channel();
    for i in 0..num_threads - 1 {
        println!("Spawned thread {:?}", i);
        let chunk: Vec<(u32, u32, image::Rgba<u8>)> = pix
            [i * pix.len() / (num_threads - 1)..(i + 1) * pix.len() / (num_threads - 1)]
            .to_vec();
        let tx = tx.clone();
        threads.push(thread::spawn(move || {
            println!("Spawned thread {:?}", i);
            for p in chunk {
                tx.send(Uchar3::new(p.2[0], p.2[1], p.2[2])).unwrap();
            }
            println!("Thread {:?} finished task", i);
        }));
    }
    let gpu_thread = thread::spawn(move || {
        println!("GPU thread spawned");
        return gpu::luminance(rx, width as usize, height as usize);
    });
    for thread in threads {
        thread.join().unwrap();
    }
    let (_lum, sums) = gpu_thread.join().unwrap().unwrap();
    println!("Preparing ui");
    ui::tuihisto(sums).unwrap();
}
