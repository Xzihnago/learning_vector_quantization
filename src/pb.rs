use indicatif::{ProgressBar, ProgressStyle};

pub fn create(total_size: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] - {pos:>6}/{len} - {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    pb
}
