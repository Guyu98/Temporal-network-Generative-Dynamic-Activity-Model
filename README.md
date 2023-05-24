# Temporal-network-Generative-Dynamic-Activity-Model
implementation of the generative dynamic activity model, in which the bursty behavior of nodes and links and the structural scale-freeness of the static counterparts are governed at the same time.

![image](https://github.com/Guyu98/Temporal-network-Generative-Dynamic-Activity-Model/blob/main/pic/aggregated%20static%20network.png)

```python
class GDAM:
    def __init__(self, n, m, gamma, rho, gam):
    # n     : [int] initial size of the network
    # m     : [int] num of links that created by a newly coming node, also the minimum degree of the aggregated network
    # gamma : [float] the exponent of the power-law distribution of bursty node and link activities
    # rho   : [float] the parameter that regulates the probability of an active node creating a new link according to the size of the network
    # gam   : [float] the parameter that regulates the probability of an active node creating a new link according to its own degree
```
