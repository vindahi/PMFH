# PFMH

**Learning Together Securely: Prototype-Based Federated Multi-Modal Hashing for Safe and Efficient Multi-Modal Retrieval**

**Authors:** Ruifan Zuo, Chaoqun Zheng, Lei Zhu, Wenpeng Lu, Yuanyuan Xiang, Zhao Li, Xiaofeng Qu  
**Conference:** Proceedings of the AAAI Conference on Artificial Intelligence

## Abstract

With the proliferation of multi-modal data, safe and efficient multi-modal hashing retrieval has become a pressing research challenge, particularly due to concerns over data privacy during centralized processing. To address this, we propose Prototype-based Federated Multi-modal Hashing (PFMH), an innovative framework that seamlessly integrates federated learning with multi-modal hashing techniques. PFMH achieves fine-grained fusion of heterogeneous multi-modal data, enhancing retrieval accuracy while ensuring data privacy through prototype-based communication, thereby reducing communication costs and mitigating risks of data leakage. Furthermore, using a prototype completion strategy, PFMH tackles class imbalance and statistical heterogeneity in multi-modal data, improving model generalization and performance across diverse data distributions. Extensive experiments demonstrate the efficiency and effectiveness of PFMH within the federated learning framework, enabling distributed training for secure and precise multi-modal retrieval in real-world scenarios.

## Datasets

| Dataset    | Categories | Training Samples | Retrieval Samples | Query Samples |
|------------|------------|------------------|-------------------|---------------|
| MIRFlickr  | 24         | 5,000            | 17,772            | 2,243         |
| MS COCO    | 80         | 18,000           | 82,783            | 5,981         |
| NUS-WIDE   | 21         | 21,000           | 193,749           | 2,085         |

**Datasets Download:**  
[Download Link](https://pan.baidu.com/s/1-_XwzUb8w-UMupa_U6aWnw)  
**Code:** u7gu

## Experimental Environment

- **Python Version:** 3.8.18
- **Torch Version:** 1.10.1
- **Hardware:** 2 x RTX 3090 GPUs

## Modal

For more information, please refer to the [MODAL.pdf](MODAL.pdf).

## Usage

Execute the following command to train the model:

```bash
bash flickr.sh
```

## E-mail

Ruifan Zuo zrfan9928@gmail.com


## Citation
If you find this work useful, please consider citing it:
```bash
@inproceedings{zuo2025learning,
  title={Learning Together Securely: Prototype-Based Federated Multi-Modal Hashing for Safe and Efficient Multi-Modal Retrieval},
  author={Zuo, Ruifan and Zheng, Chaoqun and Zhu, Lei and Lu, Wenpeng and Xiang, Yuanyuan and Li, Zhao and Qu, Xiaofeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={21},
  pages={23108--23116},
  year={2025}
}
```

