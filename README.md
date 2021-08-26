# Multi-Player Parallel Monte Carlo Tree search combined with COSMO-SAC model to design Ionic Liquids

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
├── all                      $ all elements cases
│   ├── asdi.py                # calculate ASDI
│   ├── calculate_gamma.py     # calculate infinite gamma in liquids mixture
│   ├── cCOSMO.cpython-37m-x86_64-linux-gnu.so # COSMOSAC lib (can be installed customly)
│   ├── mcts.lsf          
│   ├── MCTS.py                # Multi-Player MCTS 
│   ├── sascorer.py            # SAC score
│   ├── smiles.py              # handle SMILES
│   ├── to_sigma3.py           # convert cosmo file to sigma file
│   └── train_model.py         # include RNN model class
├── c                        $ alkyl cases
│   ├── asdi.py
│   ├── calculate_gamma.py
│   ├── carbon_smiles.py       # handle alkyl SMILES
│   ├── cCOSMO.cpython-37m-x86_64-linux-gnu.so
│   ├── mcts.lsf
│   ├── MCTS.py                
│   ├── meltingpoint.py        # calculate melting point of ILs(NTF2 based)
│   ├── sascorer.py           
│   └── to_sigma3.py
├── model
│   ├── fpscores.pkl.gz        # database that SAC calculating needed
│   ├── rnn_model_parameters   # RNN model parameters
│   └── smiles_dict            # store alkyl SMILES
├── profiles_rnn.zip           # sigma profiles of All elements SMILES
├── profiles.zip               # sigma profiles of carbon SMILES
├── README.md
├── results                    # MCTS trees and results
└── rnn                        # prepare database and train RNN model
    ├── database
    │   ├── dataset1.txt
    │   ├── dataset2.txt
    │   ├── dataset3           # torch saved binary database (410K molecules)
    │   ├── dataset3label.txt
    │   ├── dataset3.txt
    │   ├── dataset.py
    │   ├── dataset_v1.csv     # primary databases from
    │   └── symbols.txt
    ├── dataset3
    ├── rnn_model_parameters   # The trained RNN model parameters
    ├── train_model_old.py
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

## How to run on HPC

1. Copy `c` pr `all` forder and change the target of your task in `MCTS.py`
2. `bsub` or `qsub` your jobs in the specific folder. 

### choices in `MCTS.py`

```python
def run_n(self,n,tree,results_file,ty,sac):
    '''
    n: determine valid molecules in this run
    tree: the location of MCT
    result_file: the location of simulation results
    ty: ["h2"|"h2s"|"n2"]
    sac: [True|False] determine whether to use SAC score or not
    '''
    pass

```



