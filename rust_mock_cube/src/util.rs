use std::path::Path;

use ndarray as nd;
use rustronomy_fits as rsf;

pub fn get_root_path() -> std::path::PathBuf {
  const DATA_ENV: &str = crate::DATA_ENV;
  let root_path =
    std::env::var(DATA_ENV).expect(&format!("enviroment variable ${DATA_ENV} not set"));
  std::path::Path::new(&root_path)
    .canonicalize()
    .expect(&format!("could not canonicalize path found in ${DATA_ENV} env. variable"))
}

pub fn open_image(path: &Path) -> nd::Array<f64, nd::Ix3> {
  let mut fits_file = rsf::Fits::open(std::path::Path::new(path)).unwrap();

  let (header, data) = fits_file.remove_hdu(0).unwrap().to_parts();
  print!("{header}");

  let array = match data.unwrap() {
    rsf::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
    _ => panic!(),
  };

  //Datacube is 3D: we hebben 2D image in 272 verschillende channels.
  array.into_dimensionality().unwrap()
}

pub fn load_csv(path: &Path) -> Vec<f64> {
  let mut reader = csv::Reader::from_path(path).unwrap();
  reader
    .records()
    .into_iter()
    .map(|entry| {
      let str_entry = entry.unwrap();
      let size: f64 = str::parse(str_entry.get(0).unwrap()).unwrap();
      let num: f64 = str::parse(str_entry.get(1).unwrap()).unwrap();
      [size, num]
    })
    .flatten()
    .collect()
}
