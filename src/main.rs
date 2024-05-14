use std::{fs::File, io::Write};

use anyhow::Result;
use lvq::{Dataset, LVQ};

fn main() -> Result<()> {
    let train_data = Dataset::train_data()?;
    let test_data = Dataset::test_data()?;

    let mut file = File::create("output.csv")?;
    file.write_all(b"ImageId,Label\n")?;

    for (i, tensor) in LVQ::new(0.1, 10)
        .fit(&train_data.data, &train_data.target)?
        .predict(&test_data)?
        .iter()
        .enumerate()
    {
        let res: i64 = tensor.try_into()?;
        file.write_all(format!("{},{}\n", i + 1, res).as_bytes())?;
    }

    Ok(())
}
