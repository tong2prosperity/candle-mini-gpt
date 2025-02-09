use candle_core::{IndexOp, Result, Tensor};
use rand::rngs::ThreadRng;
use rand::Rng;

pub struct Dataset {
    pub training_data: Tensor,
    pub training_size: usize,
    pub validation_data: Tensor,
    pub validation_size: usize,
    rng: ThreadRng,
}
impl Dataset {
    pub fn new(data: Tensor, training_ratio: f64) -> Self {
        let data_size = *data.shape().dims().first().unwrap();
        let training_size = (data_size as f64 * training_ratio) as usize;
        let training_data = data.i(0..training_size).unwrap();

        let validation_size = data_size - training_size;
        let validation_data = data.i(0..validation_size).unwrap();
        let rng: ThreadRng = rand::thread_rng();

        Self {
            training_data,
            training_size,
            validation_data,
            validation_size,
            rng,
        }
    }

    pub fn random_training_batch(
        &mut self,
        block_size: usize,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let max_block_indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.training_size - block_size))
            .collect();

        let context_rows = max_block_indices.iter().map(|&max_index| {
            self.training_data
                .i(max_index..max_index + block_size)
                .unwrap()
        });
        let stacked_contexts = Tensor::stack(&context_rows.collect::<Vec<_>>(), 0)?;

        let target_rows = max_block_indices.iter().map(|&max_index| {
            self.training_data
                .i(max_index + 1..max_index + block_size + 1)
                .unwrap()
        });
        let stacked_targets = Tensor::stack(&target_rows.collect::<Vec<_>>(), 0)?;

        Ok((stacked_contexts, stacked_targets))
    }

    // 计算总共可能的训练窗口数量
    pub fn get_total_training_windows(&self, context_size: usize) -> Result<usize> {
        let total_tokens = self.training_data.shape().dims1()?;
        Ok(total_tokens - context_size + 1)
    }

    // 获取连续的训练batch
    pub fn get_sequential_training_batch(
        &self,
        start_idx: usize,
        batch_size: usize,
        context_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut input_indices: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut target_indices: Vec<Tensor> = Vec::with_capacity(batch_size);
        // let mut input_indices_tensor = Tensor::zeros((batch_size, context_size), DType::I32)?;
        // let mut target_indices_tensor = Tensor::zeros((batch_size, context_size), DType::I32)?;

        for i in 0..batch_size {
            let window_start = start_idx + i;
            let window_end = window_start + context_size;

            // 确保不会超出训练数据范围
            if window_end > self.training_data.shape().dims1()? {
                break;
            }

            input_indices.push(self.training_data.i(window_start..window_end)?);
            target_indices.push(self.training_data.i((window_start + 1)..(window_end + 1))?);
        }

        let inputs = Tensor::stack(&input_indices, 0)?;
        let targets = Tensor::stack(&target_indices, 0)?;

        Ok((inputs, targets))
    }
}
