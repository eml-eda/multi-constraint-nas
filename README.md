Copyright (C) 2021 Politecnico di Torino, Italy. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Alessio Burrello, Matteo Risso, Beatrice Alessandra Motetti, Luca Benini, Enrico Macii, Massimo Poncino, Daniele Jahier Pagliari

# multi-objective-nas

## Reference
If you use our code in your experiments, please make sure to cite our papers:
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

```
@ARTICLE{10278089,
  author={Burrello, Alessio and Risso, Matteo and Motetti, Beatrice Alessandra and Macii, Enrico and Benini, Luca and Pagliari, Daniele Jahier},
  journal={IEEE Transactions on Emerging Topics in Computing}, 
  title={Enhancing Neural Architecture Search with Multiple Hardware Constraints for Deep Learning Model Deployment on Tiny IoT Devices}, 
  year={2023},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TETC.2023.3322033}}
```

## Datasets
The current version support the following datasets and tasks taken from the benchmark suite MLPerf Tiny:
- CIFAR10 - Image Classification.
- MSCOCO - Visual Wake Words.
- Google Speech Commands v2 - Keyword Spotting.
- TinyImagenet

## How to install
```
git clone https://github.com/eml-eda/multi-constraint-nas.git
cd multi-constraint-nas
git submodule init
git submodule update
cd plinio
git checkout 1251ee62fbac3fb0070087167f1627a12aff0d81
python3 -m pip install -e .
```

## How to run
1. Visit the source scripts folder: e.g., `cd source_scripts`.
2. Run the provided shell script `run_server_simple.sh`: 
```
source run_server_simple.sh 
```
3. Change the script to define the desired input parameters.

## License
This code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.