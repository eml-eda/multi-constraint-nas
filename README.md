Copyright (C) 2021 Politecnico di Torino, Italy. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Matteo Risso, Alessio Burrello, Luca Benini, Enrico Macii, Massimo Poncino, Daniele Jahier Pagliari

# multi-objective-nas

## Reference
If you use our code in your experiments, please make sure to cite our paper:
```
@inproceedings{10.1145/3531437.3539720,
author = {Risso, Matteo and Burrello, Alessio and Benini, Luca and Macii, Enrico and Poncino, Massimo and Jahier Pagliari, Daniele},
title = {Multi-Complexity-Loss DNAS for Energy-Efficient and Memory-Constrained Deep Neural Networks},
year = {2022},
isbn = {9781450393546},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3531437.3539720},
doi = {10.1145/3531437.3539720},
booktitle = {Proceedings of the ACM/IEEE International Symposium on Low Power Electronics and Design},
articleno = {28},
numpages = {6},
keywords = {Deep Learning, NAS, TinyML, Energy-efficiency},
location = {Boston, MA, USA},
series = {ISLPED '22}
}
```
## Datasets
The current version support the following datasets and tasks taken from the benchmark suite MLPerf Tiny:
- CIFAR10 - Image Classification.
- MSCOCO - Visual Wake Words.
- Google Speech Commands v2 - Keyword Spotting.

## How to run
1. Run the provided `Makefile` to download the desired dataset: e.g., `make cifar10-init`.
2. Visit the desired folder: e.g., `cd cifar10`.
3. Run the provided shell script `run.sh`: 
```
source run.sh <size_target> <ops_regularization_strenght> search ft
```

## License
This code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.