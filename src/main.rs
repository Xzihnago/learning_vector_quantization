use anyhow::Result;
use pb::create;
use std::{fs::File, io::Write};

mod digits;
mod lvq;
mod pb;
mod utils;

fn main() -> Result<()> {
    let train_data = digits::Digits::train_data()?;
    let mut lvq = lvq::LVQ::new(train_data.data, train_data.target, 0.1, 10);

    lvq.fit();

    let mut file = File::create("output.csv")?;
    file.write_all(b"ImageId,Label\n")?;
    let test_data = digits::Digits::test_data()?;

    let total_size = test_data.size()[0];
    let pb = create(total_size as u64);
    println!("Predict...");
    for index in 0..total_size {
        let x = test_data.get(index);
        let predict = lvq.predict(&x);
        file.write_all(format!("{},{predict}\n", index + 1).as_bytes())?;
        pb.set_position(index as u64);
    }
    pb.finish();

    Ok(())
}
