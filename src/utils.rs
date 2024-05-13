use std::collections::HashMap;

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use tch::{Device, Tensor};

pub fn unique_index(tensor: &Tensor) -> (Tensor, Tensor) {
    let mut index_map: HashMap<u8, i64> = HashMap::new();
    let mut count = 0;
    let (unique, indices, _) = tensor.unique_dim(0, false, true, false);

    for (i, v) in Vec::try_from(indices).unwrap().into_iter().enumerate() {
        if index_map.contains_key(&v) {
            continue;
        }

        index_map.insert(v, i as i64);
        count += 1;

        if count >= unique.size()[0] {
            break;
        }
    }

    let index = Tensor::from_slice(
        &index_map
            .iter()
            .sorted_by_key(|(&k, _)| k)
            .map(|(_, &v)| v)
            .collect::<Vec<_>>(),
    )
    .to(Device::cuda_if_available());

    (unique, index)
}

pub fn progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] - {pos:>6}/{len} - {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    pb
}
