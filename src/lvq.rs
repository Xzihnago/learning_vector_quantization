use anyhow::{anyhow, Result};
use tch::{Kind, Tensor};

use crate::utils;

#[derive(Debug)]
pub struct LVQ {
    pub epochs: u8,
    pub learning_rate: f64,
    pub w: Option<Tensor>,
    pub labels: Option<Tensor>,
    pub losses: Vec<f64>,
}

impl LVQ {
    pub fn new(learning_rate: f64, epochs: u8) -> Self {
        Self {
            epochs,
            learning_rate,
            w: None,
            labels: None,
            losses: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &Tensor, y: &Tensor) -> Result<&mut Self> {
        let size = x.size()[0];
        if size != y.size()[0] {
            return Err(anyhow!("The length of x and y must be the same"));
        }

        let (labels, index) = utils::unique_index(y);
        let w = x.index_select(0, &index).totype(Kind::BFloat16);
        self.w = Some(w);
        self.labels = Some(labels);

        let calc_avg_loss = |loss: i64, size: i64| loss as f64 / size as f64;
        for epoch in 1..=self.epochs {
            println!("Epoch {}/{}:", epoch, self.epochs);
            let pb = utils::progress_bar(size as u64);

            let mut loss_sum = 0;
            for i in 0..size {
                loss_sum += self.update_weight(&x.get(i), &y.get(i))?;
                pb.set_position(i as u64);
                pb.set_message(format!("loss: {:.4}", calc_avg_loss(loss_sum, i + 1)))
            }

            let loss = calc_avg_loss(loss_sum, size);
            pb.finish_with_message(format!("loss: {:.4}", loss));

            self.losses.push(loss);
            self.learning_rate /= epoch as f64;
        }

        Ok(self)
    }

    pub fn predict(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut res = Vec::new();

        println!("Predicting...");
        let size = x.size()[0];
        let pb = utils::progress_bar(size as u64);
        for i in 0..size {
            pb.set_position(i as u64);
            res.push(self.predict_single(&x.get(i))?);
        }
        pb.finish();

        Ok(res)
    }

    fn predict_single(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(tensor) = &self.labels {
            Ok(tensor.get(self.similar_w_index(x)?))
        } else {
            Err(anyhow!("no labels"))
        }
    }

    fn update_weight(&self, x: &Tensor, y: &Tensor) -> Result<i64> {
        let w_index = self.similar_w_index(x)?;
        let mut w = self.w.as_ref().unwrap().get(w_index);

        let learning_rate = match &self.labels.as_ref().unwrap().get(w_index) == y {
            true => self.learning_rate,
            false => -self.learning_rate,
        };
        let _ = w.g_add_(&((x - &w) * learning_rate));

        let predict: i64 = self.predict_single(x)?.try_into()?;
        let result: i64 = y.try_into()?;
        match predict == result {
            true => Ok(0),
            false => Ok(1),
        }
    }

    fn similar_w_index(&self, x: &Tensor) -> Result<i64> {
        if let Some(w) = &self.w {
            Ok((w - x)
                .pow_tensor_scalar(2)
                .sum_dim_intlist(1, true, Kind::BFloat16)
                .squeeze()
                .argmin(0, false)
                .try_into()?)
        } else {
            Err(anyhow!("no weight"))
        }
    }
}
