use tch::{Device, Kind, Tensor};

use crate::{pb::create, utils};

#[derive(Debug)]
pub struct LVQ {
    pub x: Tensor,
    pub y: Tensor,
    pub w: Tensor,
    pub labels: Tensor,
    pub epochs: u8,
    pub learning_rate: f64,
    pub loss: Vec<f64>,
}

impl LVQ {
    pub fn new(x: Tensor, y: Tensor, learning_rate: f64, epochs: u8) -> LVQ {
        let (labels, index) = utils::unique_index(&y);
        let w = x
            .index_select(0, &index.to(Device::cuda_if_available()))
            .totype(Kind::BFloat16);

        LVQ {
            x,
            y,
            w,
            labels,
            epochs,
            learning_rate,
            loss: Vec::new(),
        }
    }

    fn similar_w_index(&self, x: &Tensor) -> i64 {
        (&self.w - x)
            .pow_tensor_scalar(2)
            .sum_dim_intlist(1, true, Kind::BFloat16)
            .squeeze()
            .argmin(0, false)
            .try_into()
            .unwrap()
    }

    fn update_weight(&mut self, j: i64) -> i64 {
        let x = self.x.get(j);
        let w_index = self.similar_w_index(&x);
        let mut w = self.w.get(w_index);

        let learning_rate = match self.labels.get(w_index) == self.y.get(j) {
            true => self.learning_rate,
            false => -self.learning_rate,
        };
        let _ = w.g_add_(&((&x - &w) * learning_rate));

        self.calculate_loss(&x, j)
    }

    pub fn fit(&mut self) {
        let cacl_loss = |loss: i64, amount: i64| loss as f64 / amount as f64 * 100.0;

        for epoch in 1..=self.epochs {
            println!("  Epoch {epoch}/{}:", self.epochs);

            let total_size = self.x.size()[0];
            let pb = create(total_size as u64);
            let mut loss = 0;
            for j in 0..total_size {
                loss += self.update_weight(j);
                pb.set_position(j as u64);
                pb.set_message(format!("loss: {:.4}", cacl_loss(loss, j + 1)))
            }

            pb.finish_with_message(format!("loss: {:.4}", cacl_loss(loss, total_size)));
            self.loss.push(cacl_loss(loss, total_size));
            println!("\n");

            self.learning_rate /= epoch as f64;
        }
    }

    fn calculate_loss(&self, x: &Tensor, j: i64) -> i64 {
        let predict = self.predict(x);
        let ans: i64 = self.y.get(j).try_into().unwrap();

        match predict == ans {
            true => 0,
            false => 1,
        }
    }

    pub fn predict(&self, x: &Tensor) -> i64 {
        self.labels.get(self.similar_w_index(x)).try_into().unwrap()
    }
}
