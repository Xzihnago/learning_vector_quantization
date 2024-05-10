use anyhow::Result;
use tch::{Device, Tensor};

pub struct Digits {
    pub target: Tensor,
    pub data: Tensor,
}

impl Digits {
    pub fn train_data() -> Result<Self> {
        let mut rdr = csv::Reader::from_path("data/train.csv")?;

        let mut targets: Vec<i64> = Vec::new();
        let mut datas: Vec<Vec<i64>> = Vec::new();

        for record in rdr.records().filter_map(|r| r.ok()) {
            let v = record
                .iter()
                .map(|s| s.parse::<i64>().unwrap())
                .collect::<Vec<_>>();

            if let Some((target, data)) = v.split_first() {
                datas.push(data.to_vec());
                targets.push(*target);
            }
        }

        Ok(Digits {
            target: Tensor::from_slice(&targets).to(Device::cuda_if_available()),
            data: Tensor::from_slice2(&datas).to(Device::cuda_if_available()),
        })
    }

    pub fn test_data() -> Result<Tensor> {
        let mut rdr = csv::Reader::from_path("data/test.csv")?;

        let mut datas: Vec<Vec<i64>> = Vec::new();

        for record in rdr.records().filter_map(|r| r.ok()) {
            let v = record
                .iter()
                .map(|s| s.parse::<i64>().unwrap())
                .collect::<Vec<_>>();

            datas.push(v);
        }

        Ok(Tensor::from_slice2(&datas).to(Device::cuda_if_available()))
    }
}
