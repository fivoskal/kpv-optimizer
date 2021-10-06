# kpv-optimizer

A simple pytorch optim class optimizer that is based on Gradient Descent and the use of [Washout-Filters](https://ieeexplore.ieee.org/abstract/document/1383925).

Figuratively, the (possibly unstable) process of running gradient descent on an objective function is treated as a "plant" and a feedback loop is added to it that uses a washout filter.
