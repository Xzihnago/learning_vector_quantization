use std::collections::HashMap;
use tch::Tensor;

pub fn unique_index(v: &Tensor) -> (Tensor, Tensor) {
    let mut index_map: HashMap<u8, i64> = HashMap::new();
    let mut count = 0;
    let (unique, indices, _) = v.unique_dim(0, false, true, false);

    for (idx, x) in Vec::try_from(indices).unwrap().into_iter().enumerate() {
        if index_map.contains_key(&x) {
            continue;
        }

        index_map.insert(x, idx as i64);
        count += 1;

        if count >= unique.size()[0] {
            break;
        }
    }

    let mut index: Vec<_> = index_map.iter().collect();
    index.sort_by_key(|&(&k, _)| k);
    let index = index.iter().map(|(_, &v)| v).collect::<Vec<_>>();
    let index = Tensor::from_slice(&index);

    (unique, index)
}
