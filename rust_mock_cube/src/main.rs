mod util;

use ndarray as nd;
use image as img;

use rand::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rayon::prelude::*;
use rustronomy_watershed::prelude::*;

const DATA_ENV: &str = "DATA_PATH";

//We make a 2250x1000x2000Pc cube with 0.5Pc sized pixels
const CUBE_SIZE: (usize, usize, usize) = (2250, 1000, 2100);//(4500, 2000, 4200);
const CUBE_SCALE: (f64, f64, f64) = (
  CUBE_SIZE.0 as f64 / 2250.0,
  CUBE_SIZE.1 as f64 / 1000.0,
  CUBE_SIZE.2 as f64 / 2100.0
);
const STDEV: f64 = 250.0;

fn main() {
  //Yes get the root
  let root_path = util::get_root_path();
  println!("Using TMP dir {root_path:?}");

  //Get the appropriate data (hmhm)
  let holes = util::load_csv(&root_path.join("full_distr.csv"));
  let holes = nd::Array2::from_shape_vec((holes.len() / 2, 2), holes).unwrap();

  //Get the slice positions :c
  let slices = util::load_csv(&root_path.join("PERSEUS_DDX.csv"));
  let slices = nd::Array2::from_shape_vec((slices.len() / 2, 2), slices).unwrap();
  let slices: Vec<(f64, f64)> = slices
    .axis_iter(nd::Axis(0))
    //Convert from kPc to Pc starting at slice0 = distance 0
    .map(|x| ((x[0] - slices[(0,0)]) * 1000.0, x[1]))
    .collect();

  /*
    GAUßIAN DISTRIBUTION
  */
  println!("Generating noisy cluster cube of size {CUBE_SIZE:?}...");
  let dis_x = rand_distr::Normal::new((CUBE_SIZE.0 as f64) / 2.0, STDEV).unwrap();
  let dis_y = rand_distr::Normal::new((CUBE_SIZE.1 as f64) / 2.0, STDEV).unwrap();
  let dis_z = rand_distr::Normal::new((CUBE_SIZE.2 as f64) / 2.0, STDEV).unwrap();
  let mut rng = thread_rng();
  let cube = make_cube(holes.view(), || (
    dis_x.sample(&mut rng) as isize,
    dis_y.sample(&mut rng) as isize,
    dis_z.sample(&mut rng) as isize
  ));

  //I want to see if it worked :3 Let's plot every ehhh 10th slice along z
  println!("Plotting clustered cube...");
  let cmap = rustronomy_watershed::plotting::viridis;
  let plot = rustronomy_watershed::plotting::plot_slice;
  cube
    .axis_iter(nd::Axis(2))
    .step_by(10)
    .enumerate()
    .par_bridge()
    .for_each(|(i, slice)|
      plot(slice, &root_path.join(&format!("out/kaas_pics/CLUSTER_slice_{i}.png")), cmap).unwrap()
    );

  //Next let's make 2250x1000 images out of the cube
  let imgs = make_imgs(cube, &slices, &root_path);

  //Now do watershed
  let watershed = TransformBuilder::new_merging().build().unwrap();
  let output_dir = &root_path.join("out/CLUSTER_KAAS/");

  imgs
    .into_par_iter()
    .enumerate()
    .for_each(|(idx, img)| {
      println!("Starting watershed on slice #{idx}...");
      let mins = &watershed.find_local_minima(img.view());
      let lakes = watershed.transform_to_list(img.view(), mins);
      drop(img); //would be nice to dealloc

      //Save data to disk
      println!("Saved output of slice #{idx}!");
      let out_path = output_dir.join(&format!("depth_{idx}.csv"));
      let mut writer = csv::WriterBuilder::new().from_path(out_path).unwrap();
      lakes.into_iter().for_each(|(_water_level, lake_sizes)| {
        writer.write_record(lake_sizes.iter().map(|&x| format!("{x}"))).unwrap();
      });
      writer.flush().unwrap();
    });

  /*
    UNIFORM DISTRIBUTION
  */
  println!("Generating noisy cube of size {CUBE_SIZE:?}...");
  let mut rng = thread_rng();
  let cube = make_cube(holes.view(), || ((
    rng.gen_range(0..CUBE_SIZE.0) as isize,
    rng.gen_range(0..CUBE_SIZE.1) as isize,
    rng.gen_range(0..CUBE_SIZE.2) as isize
  )));

  //I want to see if it worked :3 Let's plot every ehhh 10th slice along z
  println!("Plotting cube...");
  let cmap = rustronomy_watershed::plotting::viridis;
  let plot = rustronomy_watershed::plotting::plot_slice;
  cube
    .axis_iter(nd::Axis(2))
    .step_by(10)
    .enumerate()
    .par_bridge()
    .for_each(|(i, slice)|
      plot(slice, &root_path.join(&format!("out/kaas_pics/slice_{i}.png")), cmap).unwrap()
    );

  //Next let's make 2250x1000 images out of the cube
  let imgs = make_imgs(cube, &slices, &root_path);

  //Now do watershed
  let watershed = TransformBuilder::new_merging().build().unwrap();
  let output_dir = &root_path.join("out/KAAS/");

  imgs
    .into_par_iter()
    .enumerate()
    .for_each(|(idx, img)| {
      println!("Starting watershed on slice #{idx}...");
      let mins = &watershed.find_local_minima(img.view());
      let lakes = watershed.transform_to_list(img.view(), mins);
      drop(img); //would be nice to dealloc

      //Save data to disk
      println!("Saved output of slice #{idx}!");
      let out_path = output_dir.join(&format!("depth_{idx}.csv"));
      let mut writer = csv::WriterBuilder::new().from_path(out_path).unwrap();
      lakes.into_iter().for_each(|(_water_level, lake_sizes)| {
        writer.write_record(lake_sizes.iter().map(|&x| format!("{x}"))).unwrap();
      });
      writer.flush().unwrap();
    });
}

fn make_cube<F>(holes: nd::ArrayView2<f64>, mut rand: F) -> nd::Array3<u8>
where
  F: FnMut() -> (isize, isize, isize)
{
  let mut cube = nd::Array3::<u8>::random(CUBE_SIZE, Uniform::new(0, 254));

  //Now make holes!
  let mut rng = thread_rng();
  holes
    .axis_iter(nd::Axis(0))
    .map(|x| {
      //Multiply num density of holes with volume to get number
      const VOL: f64 = (CUBE_SIZE.0 * CUBE_SIZE.1 * CUBE_SIZE.2) as f64 / 1e9;
      (x[0], (VOL * x[1]).round() as usize)
    })
    .filter(|(_size, num)| *num != 0)
    .for_each(|(size, num)| {
      println!("Seeding {num} spheres of radius {size}");
      let mut count = 0;
      while count < num {
        //Generate a random position
        let rand_position = rand();

        //Check all indices in a 2R cube around the random position. set them to
        //zero if they are within the circle
        for x in -size.round() as isize..size.round() as isize {
          for y in -size.round() as isize..size.round() as isize {
            for z in -size.round() as isize..size.round() as isize {
              if x*x + y*y + z*z <= (size * size).round() as isize {
                  if let Some(val) = cube.get_mut((
                    (x + rand_position.0) as usize,
                    (y + rand_position.1) as usize,
                    (z + rand_position.2) as usize
                  )) {
                    *val /= 10;
                  };
                }
              }
            }
          }
        
        //Increment counter
        count += 1;
        }
      }
    );

    cube
}

fn make_imgs(
  cube: nd::Array3<u8>,
  slices: &[(f64, f64)],
  root_path: &std::path::Path
) -> Vec<nd::Array2<u8>> {
  let cmap = rustronomy_watershed::plotting::viridis;
  let plot = rustronomy_watershed::plotting::plot_slice;
  slices
    .into_par_iter()
    .map(|(dist, px_size)| {
      //Calculate size of a 2250x1000 image in Pc at this distance given the 
      //size of each pixel, then convert that into a slice of the same PHYSICAL
      //size but a different PIXEL size. Then scale-up that slice to cover the
      //new 2250x1000 image.
      let z = (dist / CUBE_SCALE.2).round() as usize;
      let x_max = ((2250.0 * px_size) / CUBE_SCALE.0).round() as usize;
      let y_max = ((1000.0 * px_size) / CUBE_SCALE.1).round() as usize;
      println!("Rescaling Slice #{z} ({x_max}x{y_max}) to (2250x1000)...");

      let source_slice = cube.slice(nd::s![..x_max, ..y_max, z]);
      plot(source_slice.clone(), &root_path.join(&format!("out/kaas_pics/unedited_slice_{z}.png")), cmap).unwrap();

      let output_slice = img::imageops::resize(
        &img::ImageBuffer::<img::Luma<u8>, _>::from_raw(
          y_max as u32,
          x_max as u32,
          source_slice.iter().cloned().collect::<Vec<_>>()
        ).unwrap(),
        1000,
        2250,
        img::imageops::Gaussian
      );
      println!("Smoothing Slice #{z}...");
      let output_slice = img::imageops::blur(&output_slice, 4.0).to_vec();
      let array = nd::Array2::from_shape_vec((2250, 1000), output_slice).unwrap();

      //Plot edited image
      plot(array.view(), &root_path.join(&format!("out/kaas_pics/edited_slice_{z}.png")), cmap).unwrap();

      //Yield image
      array
    })
    .collect()
}