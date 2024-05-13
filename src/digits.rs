use anyhow::Result;
use tch::{Device, Tensor};

pub struct Dataset {
    pub target: Tensor,
    pub data: Tensor,
}

impl Dataset {
    pub fn train_data() -> Result<Self> {
        let mut targets = Vec::new();
        let mut datas = Vec::new();

        csv::Reader::from_path("data/train.csv")?
            .records()
            .filter_map(Result::ok)
            .for_each(|record| {
                if let Some((target, data)) = record
                    .iter()
                    .map(|s| s.parse().unwrap())
                    .collect::<Vec<i64>>()
                    .split_first()
                {
                    targets.push(*target);
                    datas.push(data.to_vec())
                }
            });

        Ok(Self {
            target: Tensor::from_slice(&targets).to(Device::cuda_if_available()),
            data: Tensor::from_slice2(&datas).to(Device::cuda_if_available()),
        })
    }

    pub fn test_data() -> Result<Tensor> {
        let datas = csv::Reader::from_path("data/test.csv")?
            .records()
            .filter_map(Result::ok)
            .map(|record| {
                record
                    .iter()
                    .map(|s| s.parse().unwrap())
                    .collect::<Vec<i64>>()
            })
            .collect::<Vec<_>>();

        Ok(Tensor::from_slice2(&datas).to(Device::cuda_if_available()))
    }
}
