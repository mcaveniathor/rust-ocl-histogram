extern crate image;
extern crate ocl;

use image::GenericImageView;
use ocl::core::Uchar3;
use std::env;

fn luminance(pixels: Vec<Uchar3>, width: usize, height: usize) -> ocl::core::Result<()> {
    let wh: usize = width * height;
    use ocl::builders::ContextProperties;
    use ocl::core::Int2;
    use ocl::enums::ArgVal;
    use ocl::{core, flags};
    use std::ffi::CString;

    let src = r#"

        __kernel void luminance(__global uchar3 *bufin, int2 dims, __global uchar *bufout, __volatile __global int *sums) {
            __volatile __local int tmpSums[256];
            int lidx = get_local_id(0);
            int lidy = get_local_id(1);
            int gidx = get_global_id(0);
            int gidy = get_global_id(1);
            int gid_1d = gidx + dims[1] * gidy;
            if(lidx == 0 && lidy == 0) {
                for(int i=0; i < 256; i++) {
                    tmpSums[i] = 0;
                }
            }
            uchar3 pixel = bufin[gid_1d];
            /* luminance per ITU BT.709-6 */
            uchar lum = pixel.x * .2126 + pixel.y * .0722 + pixel.z * .7152;
            bufout[gid_1d] = lum;
            barrier(CLK_LOCAL_MEM_FENCE);
            atomic_inc(&tmpSums[lum]);
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lidx == get_local_size(0) -1 && lidy == get_local_size(1) -1) {
                for(int i=0; i<256; i++) {
                    atomic_add(&sums[i], tmpSums[i]);
                }
            }
        }

    "#;
    // opencl boilerplate
    let platform = core::default_platform()?;
    let devices = core::get_device_ids(&platform, None, None)?;
    let device = devices[0];
    let context_properties = ContextProperties::new().platform(platform);
    let context = core::create_context(Some(&context_properties), &[device], None, None)?;
    let src_c = CString::new(src)?;
    let program = core::create_program_with_source(&context, &[src_c])?;
    core::build_program(&program, Some(&[device]), &CString::new("")?, None, None)?;
    let queue = core::create_command_queue(&context, &device, None)?;
    // dimensions
    let work_dim = 2;
    let global_work_size: [usize; 3] = [width, height, 0];
    let local_work_size: [usize; 3] = [10, 10, 0];
    let dims: Int2 = Int2::new(width as i32, height as i32);
    let mut vec_lum_out = vec![0u8; wh];
    let mut vec_sums_out = vec![0i32; 256];
    let buf_in = unsafe {
        core::create_buffer(
            &context,
            flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR,
            wh,
            Some(&pixels),
        )?
    };

    let buf_lum_out = unsafe {
        core::create_buffer(
            &context,
            flags::MEM_WRITE_ONLY | flags::MEM_COPY_HOST_PTR,
            wh,
            Some(&vec_lum_out),
        )?
    };

    let buf_sums_out = unsafe {
        core::create_buffer(
            &context,
            flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR,
            256,
            Some(&vec_sums_out),
        )?
    };

    let kernel = core::create_kernel(&program, "luminance")?;
    core::set_kernel_arg(&kernel, 0, ArgVal::mem(&buf_in))?;
    core::set_kernel_arg(&kernel, 1, ArgVal::scalar(&dims))?;
    core::set_kernel_arg(&kernel, 2, ArgVal::mem(&buf_lum_out))?;
    core::set_kernel_arg(&kernel, 3, ArgVal::mem(&buf_sums_out))?;

    //run kernel
    unsafe {
        core::enqueue_kernel(
            &queue,
            &kernel,
            work_dim,
            None,
            &global_work_size,
            Some(local_work_size),
            None::<core::Event>,
            None::<&mut core::Event>,
        )?;
    }

    // read from output buffers
    unsafe {
        core::enqueue_read_buffer(
            &queue,
            &buf_lum_out,
            true,
            0,
            &mut vec_lum_out,
            None::<core::Event>,
            None::<&mut core::Event>,
        )?;
    }

    unsafe {
        core::enqueue_read_buffer(
            &queue,
            &buf_sums_out,
            true,
            0,
            &mut vec_sums_out,
            None::<core::Event>,
            None::<&mut core::Event>,
        )?;
    }
    println!("{:?}", vec_lum_out);
    println!("{:?}", vec_sums_out);
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let img = image::open(&args[1]).unwrap();
    let (width, height) = img.dimensions();
    println!("width: {:?}", width);
    println!("height: {:?}", height);
    let mut pix = img.pixels();
    let mut pixels: Vec<Uchar3> = Vec::new(); // vector of 3-dimensional 8bit opencl vectors
    for _ in 0..width * height {
        let px = pix.next().unwrap();
        let rgbvec = Uchar3::new(px.2[0], px.2[1], px.2[2]);
        pixels.push(rgbvec);
    }
    luminance(pixels, width as usize, height as usize).unwrap();
}
