# CUDA Optimized Simple Recurrent Unit (SRU)

- [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)

#### Todo

- [x] CUDA optimized forward computation
- [x] CUDA optimized backward computation
- [x] Testing
- [x] Training language models
- [x] Benchmark

## Requirements

- Chainer 2+
- CuPy
- Python 2 or 3

## Language Modeling

### Penn Treebank

| Model | #layers | d   | Perplexity |
|-------|---------|-----|------------|
| LSTM  | 2       | 320 | 93         |
| SRU   | 2       | 320 | 95         |
| LSTM  | 2       | 128 | 117        |
| SRU   | 2       | 128 | 118        |

## Benchmark

![result](https://github.com/musyoku/images/blob/master/sru/result.png?raw=true)