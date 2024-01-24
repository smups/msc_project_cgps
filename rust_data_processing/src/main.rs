use ndarray as nd;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rayon::prelude::*;
use rustronomy_watershed::prelude::*;

mod util;

pub const DATA_ENV: &str = "DATA_PATH";

fn main() {
  //get the root path from the env. vars and make an output directory
  let root_path = util::get_root_path();
  println!("Using TMP dir {root_path:?}");
  let output_dir_1 = &root_path.join("out/CUBE1/"); 
  let output_dir_2 = &root_path.join("out/CUBE2/"); 
  println!("Using OUT dir {output_dir_1:?} nad {output_dir_2:?}");

  //load slices
  let norma = util::load_csv(&root_path.join("SLICE_NORMA.csv"));
  let cygnus = util::load_csv(&root_path.join("SLICE_CYGNUS.csv"));
  let perseus = util::load_csv(&root_path.join("SLICE_PERSEUS.csv"));
  let local = util::load_csv(&root_path.join("SLICE_LOCAL.csv"));
  let outer = util::load_csv(&root_path.join("SLICE_OUTER.csv"));

  let slices_1 = concat_slices(vec![norma, cygnus, perseus]);
  let slices_2 = concat_slices(vec![local, outer]);
  println!("Will be performing watershed for images {slices_1:?} in cube 1");
  println!("Will be performing watershed for images {slices_2:?} in cube 2");

  //Watershed ALL the slices in parallel
  let cube_1 = util::open_image(&root_path.join("full_cube_smoothed.fits"));
  cube_shed(cube_1.view(), output_dir_1, &slices_1);
  drop(cube_1);

  //Watershed ALL the slices in parallel (again)
  let cube_2 = util::open_image(&root_path.join("full_cube_smoothed-farfield.fits"));
  cube_shed(cube_2.view(), output_dir_2, &slices_2);
  drop(cube_2);

  //Do da computation
  stat_script(&root_path);
}

fn concat_slices(slices: Vec<Vec<f64>>) -> Vec<usize> {
  slices
    .into_iter()
    .map(|vec| {
      vec
        .into_iter()
        .enumerate()
        .filter_map(|(idx, truth_val)| if truth_val as usize == 1 { Some(idx) } else { None })
        .collect::<Vec<usize>>()
    })
    .flatten()
    .collect()
}

fn cube_shed(cube: nd::ArrayView3<f64>, output_dir: &std::path::Path, slice: &[usize]) {
  //Check if the output dir exists lol
  assert!(output_dir.exists());

  slice.into_par_iter().for_each(|idx| {
    //Get image and run pre-processor
    let img = cube.slice(nd::s![*idx, .., ..]);
    let watershed = TransformBuilder::new_merging().build().unwrap();
    let img = watershed.pre_processor(img.view());

    //Collect and save watershed data
    println!("Starting watershed on slice #{idx}");
    let mins = watershed.find_local_minima(img.view());
    let data = watershed.transform_to_list(img.view(), &mins);

    //Save data to disk
    println!("Saved output of slice #{idx}");
    let out_path = output_dir.join(&format!("depth_{idx}.csv"));
    let mut writer = csv::WriterBuilder::new().from_path(out_path).unwrap();
    data.into_iter().for_each(|(_water_level, lake_sizes)| {
      writer.write_record(lake_sizes.iter().map(|&x| format!("{x}"))).unwrap();
    });
    writer.flush().unwrap();
  });
}

fn stat_script(root: &std::path::Path) {
  //We prepare three different watershed images:
  // 1. A real image
  // 2. A gaußian random field with power = 3
  // 3. A uniform random field (gaussian field with power = 0)

  //First we do the Gaußian
  let gauß = {
    let hdu = rustronomy_fits::Fits::open(&root.join("GAUß.fits")).unwrap().remove_hdu(0).unwrap();
    println!("{hdu}");
    match hdu.to_parts().1.unwrap() {
      rustronomy_fits::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
      _ => panic!("shitties"),
    }
  }
  .into_dimensionality::<nd::Ix2>()
  .unwrap();

  //Configure watershed transfo
  let watershed = TransformBuilder::new_merging().build().unwrap();

  let gauß = watershed.pre_processor(gauß.view());
  let gauß_shape = (gauß.shape()[0], gauß.shape()[1]);
  let gauß_mins = watershed.find_local_minima(gauß.view());

  //Next we do the uniform field
  let unif = nd::Array2::<u8>::random(gauß_shape, Uniform::new(0, 254));
  let unif_mins = watershed.find_local_minima(unif.view());

  //Now we do the actual water-shed, in parallel ofc.
  let package = vec![(gauß, gauß_mins), (unif, unif_mins)];

  let results = package
    .into_par_iter()
    .map(|(field, mins)| watershed.transform_to_list(field.view(), &mins))
    .collect::<Vec<Vec<(u8, Vec<usize>)>>>();

  results.into_iter().zip(["gauß", "unif"]).for_each(|(data, name)| {
    let mut writer =
      csv::WriterBuilder::new().from_path(&root.join(format!("{name}_ws.csv"))).unwrap();
    data.into_iter().for_each(|(_water_level, lake_sizes)| {
      writer.write_record(lake_sizes.iter().map(|&x| format!("{x}"))).unwrap();
    });
    writer.flush().unwrap();
  });
}
