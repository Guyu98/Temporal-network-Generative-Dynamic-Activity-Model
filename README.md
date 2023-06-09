# Temporal-network-Generative-Dynamic-Activity-Model
This repository contains implementations of the generative dynamic activity model, in which the bursty behavior of nodes and links and the structural scale-freeness of the static counterparts are governed at the same time. 

![image](https://github.com/Guyu98/Temporal-network-Generative-Dynamic-Activity-Model/blob/main/pic/aggregated%20static%20network.png)

Users can start with loading the package:
```python
from model import GDAM
```
  
  
  
## Settings and evolution
The proposed temporal network model -- GDAM -- requires five configuration parameters that are briefly described below. In this repository, the initial network structures are set to be complete graph with `n` nodes, while it is easy to modify the source code in the `__init__()` to achieve any desired structure. Parameter `gamma` reflects the busrty activity of nodes (and links) to engage in communications, which can be obtained by fitting the real power-law interaction sequence using the algorithm proposed in [Power-Law Distributions in Empirical Data](https://doi.org/10.1137/070710111) by Clauset *et al.*. The parameters `rho` and `gam` regulate the probability of an active node to explore a new neighbor, which can be obtained by general linear regression from empirical data. Other details can be found in the paper.

```python
class GDAM:
    def __init__(self, n, m, gamma, rho, gam):
    # n     : [int] initial size of the network
    # m     : [int] num of links that created by a newly coming node, also the minimum degree of the aggregated network
    # gamma : [float] the exponent of the power-law distribution of bursty node and link activities
    # rho   : [float] the parameter that regulates the probability of an active node to create a new link according to the size of the network
    # gam   : [float] the parameter that regulates the probability of an active node to create a new link according to its own degree
```

After setting up a GDAM instance, there are two implementations for the network to evovle. When a fixed-size temporal network is desired, set parameter `finite_grow` in the `evo` method to be the wanted size.
if `finite_grow` is 0, the network will keep growing ending up with the network size = `n + N`.
The `evo` method will return a neighbor list of the corresponding aggregated network.
```python
net = GDAM(5, 2, 2.2, 1., 0.7)
nl = net.evo(N, finite_grow)
# N           : [int] total step for the network to evovle  
# finite_grow : [int] if `finite_grow = 0`, then the network will embrace a newly coming node every step ending up with the network size = `n + N`, e.g. net.evo(1000, 0).
#                      if `finite_grow > n`, then the network will not take in any newly coming node after the network size = `finite_grow`, e.g. net.evo(1000, 20).
# Note that `N >= finite_grow`. 
```

## Snapshots
After setting up the temporal network, one can query any snapshot through the method `snapshot`.The `snapshot` method takes in the order of the queried snapshot ranging from 0 to `N` and returns the adjacent matrix of the snapshot. The source code saves snapshots for all time steps by default, which can make the program run slower and take up more memory when the total evo-time step `N` is longer. It is recommended to choose the appropriate snapshot storage according to specific needs. The `visualize_graph` method help to visualize any network given either its adjacent matrix or neighbor list.
```python
adj = net.snapshot(num)
# num     : [int] the order of the required snapshot ranging from 0 to N, e.g. net.snapshot(65).
net.visualize_graph(adj)
# adj     : [nd.array] adjacent matrix or neighbor list.
```
![image](https://github.com/Guyu98/Temporal-network-Generative-Dynamic-Activity-Model/blob/main/pic/snapshot.png)

## Get node and link activities
To see the activity of nodes and link, one can query the active time list of any node or link by the method `get_activenode_t` or `get_edge_t` respectively. The inputs are the order of the queried node and the orders of the ending nodes of the queried link. To see the statistical features, use `plot_CCDF` to generate the complementary cumulative distributions of the inter-event time.

```python
N_t = net.get_activenode_t(2)
net.plot_CCDF(np.diff(N_t))

E_t = net.get_edge_t(0,3)
net.plot_CCDF(np.diff(E_t))
```
![image](https://github.com/Guyu98/Temporal-network-Generative-Dynamic-Activity-Model/blob/main/pic/activity.png)

## Hypothesis testing 
To test the power-law hypothesis and goodness of fit, we follow the work below:


A. Clauset, C.R. Shalizi, and M.E.J. Newman, "[Power-Law Distributions in Empirical Data](https://doi.org/10.1137/070710111)" SIAM Review 51(4), 661-703 (2009).


Y. Virkar and A. Clauset, "[Power-law distributions in binned empirical data](https://arxiv.org/abs/1208.3524)". Annals of Applied Statistics 8(1), 89 - 119 (2014).





