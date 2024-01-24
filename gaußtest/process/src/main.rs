use std::path::Path;

use ndarray as nd;
use rustronomy_fits as rsf;
use rayon::prelude::*;
use rustronomy_watershed::prelude::*;

pub fn load_csv(path: &Path) -> Vec<f64> {
  let mut reader = csv::Reader::from_path(path).unwrap();
  reader
    .records()
    .into_iter()
    .map(|entry| {
      let str_entry = entry.unwrap();
      str_entry.as_slice().parse::<f64>().unwrap()
    })
    .collect()
}

fn main() {
  println!("Hello, world!");
  std::fs::read_dir("/home/raulwolters/Documents/school/MRP1-HI_Cavity_Distribution/gaußtest/").unwrap()
  .into_iter()
  .filter_map(|file| if let Ok(entry) = file {
    if entry.file_name().to_str().unwrap().ends_with(".fits") {
      Some(entry.path())
    } else { None }
  } else { None })
  .map(|path| (
    {
      let filename = path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .split('.')
        .next()
        .unwrap();
      path.parent().unwrap().join(&format!("{filename}.csv"))
    },{
      let hdu = rsf::Fits::open(&path).unwrap().remove_hdu(0).unwrap();
      println!("{hdu}");
      match hdu.to_parts().1.unwrap() {
        rustronomy_fits::Extension::Image(img) => img.as_owned_f64_array().unwrap(),
        _ => panic!("shitties"),
      }.into_dimensionality::<nd::Ix2>().unwrap()
    }
  ))
  .par_bridge()
  .for_each(|(ref path, gauß)| {
    //Configure watershed transform
    println!("Doing ws on {path:?}");
    let watershed = TransformBuilder::new_merging().build().unwrap();
    let gauß = watershed.pre_processor(gauß.view());
    let gauß_mins = watershed.find_local_minima(gauß.view());
    let gauß_out = watershed.transform(gauß.view(), &gauß_mins);
    
    //Write results to disk
    println!("{:?}", path.join(".csv"));
    let mut writer = csv::WriterBuilder::new().from_path(path).unwrap();
    gauß_out.into_iter().for_each(|(_water_level, lake_sizes)| {
      writer.write_record(lake_sizes.iter().map(|&x| format!("{x}"))).unwrap();
    });
    writer.flush().unwrap();
  });
}
