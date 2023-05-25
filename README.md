# Temporal-network-Generative-Dynamic-Activity-Model
This repository contains implementations of the generative dynamic activity model, in which the bursty behavior of nodes and links and the structural scale-freeness of the static counterparts are governed at the same time. 

![image](https://github.com/Guyu98/Temporal-network-Generative-Dynamic-Activity-Model/blob/main/pic/aggregated%20static%20network.png)

Users can start with loading the package
```python
from model import GDAM
```
The proposed temporal network model -- GDAM -- requires five configuration parameters that are briefly described below. In this repository, the initial network structures are set to be complete graph with `n` nodes, while it is easy to modify the source code in the `__init__()` to achieve any desired structure. Parameter `gamma` reflects the busrty activity of nodes (and links) to engage in communications, which can be obtained by fitting the real power-law interaction sequence using the algorithm proposed in [Power-Law Distributions in Empirical Data](https://doi.org/10.1137/070710111).
```python
class GDAM:
    def __init__(self, n, m, gamma, rho, gam):
    # n     : [int] initial size of the network
    # m     : [int] num of links that created by a newly coming node, also the minimum degree of the aggregated network
    # gamma : [float] the exponent of the power-law distribution of bursty node and link activities
    # rho   : [float] the parameter that regulates the probability of an active node creating a new link according to the size of the network
    # gam   : [float] the parameter that regulates the probability of an active node creating a new link according to its own degree
```
