# MPPMCTS--Multi-Player Parallel Monte Carlo Tree

**Multi-Player Parallel Monte Carlo Tree search combined with COSMO-SAC model to design Ionic Liquids**

# install and run

## install

```shell
# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# u can assigh specific installation prefix 
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
# create conda enviroment and install cCOSMO
conda create -n mppmcts rdkit pytorch[cpu] numpy matplotlib scipy pandas cmake pybind11
conda activate mppmcts
wget https://download.fastgit.org/usnistgov/COSMOSAC/releases/download/v1.0.1/COSMOSAC_v1.0.1.zip
mkdir tmp
mv COSMOSAC_v1.0.1.zip tmp
cd tmp
unzip COSMOSAC_v1.0.1.zip
python setup.py install
rm -rf tmp
# install MPPMCTS
pip install MPPMCTS
# do a test
mppmcts --turn 5 1.tree 1.out
```

## run with scripts

```shell
# shell
# please make sure that g09 is in your system enviroment
mppmcts --help # see more information about flags meaning
mppmcts test.tree test.out # run. different jobs can use same .tree and .out file.
						   # need to test later
```

```python
# python
from MPPMCTS import MCTS_all
MCTS_all.P_MCTS().run_n(
        n=20,
        tree="test.tree",
        results_file="test.out",
        ty="h2",
        sac=False
)
```

## run on HPC

create a job file and use shell command or run  another python script

## Dependence

* [COSMO-SAC](https://github.com/usnistgov/COSMOSAC)
* [RDKit](http://rdkit.org/docs/api-docs.html)
* [Gaussian09](http://gaussian.com/)
* [molecularsets](https://github.com/molecularsets/moses/blob/master/data/dataset_v1.csv)
* [pytorch](https://pytorch.org/)
* [zss](https://pypi.org/project/zss/)

## Structure of program

```shell
.
├── ASDI
│   ├── asdi.py
│   ├── calculate_gamma.py
│   ├── __init__.py
│   ├── SigmaDB.zip
│   └── to_sigma3.py
├── __init__.py
├── MCTS_all.py
├── _MCTS_carbon_test.py
├── run.py
└── utils
    ├── carbon_smiles.py
    ├── fpscores.pkl.gz
    ├── __init__.py
    ├── meltingpoint.py
    ├── rnn_model_parameters
    ├── sascorer.py
    ├── smiles_dict
    ├── smiles.py
    └── train_model.py
```

## Simulation process

1. Prepare RNN(GRU or LSTM) model or ZSS tree editing distance model to genrate SMILES
2. Run P-MCTS (all elements cases) or MP-P-MCTS (alkyl cases).
3. In the simulation step, ASDI and SAC scores are calculated and are combined to update MCT.
4. The tree and results are store in results folder

## Hyperparameters

### RNN (GRU or LSTM)

The input sequence can be encode by embedding layer or just one-hot layer.

(After testing, there exist no big difference)

1. sequence length: 45
2. one-hot length: 23
3. embedding layer shape: (23, 256)  ! if using (23, 23),  the same as one-hot 
4. hidden layer size of RNN: 512
5. RNN layer numbers: 2
6. dropout during training: 0.5
7. batch size: 6000
8. learning rate: 3e-3

### MCTS

#### UCB

$$
\begin{align*}
\text{UCB}&=\text{exploitation}+\text{exploration}\\&=\frac{s_i}{v_i}+C\sqrt{\frac{\ln v_p}{v_i}}
\end{align*}
$$

* $s=\sum r$ and $v$ are the cumulative reward value and total visits of node i.
* $v_p$ is the cumulative visits of node i's father node.

> In ordinary MCTS, the results are neither 0 or 1, and C=2 re the best choice.
>
> However, In this simulation, the simulation results are not discrete, so we need to find some way to normalize reward value between 0 and 1. 

#### Reward

$$
r_1=\text{tanh}\,(A_1\times\frac{1}{ASDI})\\
r_2=\text{tanh}\,(A_2\times\frac{1}{ASDI\times SAC})\\
$$

### 

