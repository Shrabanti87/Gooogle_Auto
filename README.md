# Gooogle_Auto (Group Regularization for Zero-inflated Regression Models with Application to Insurance Rate-making)
# Introduction
In many risk analysis problems, covariates are naturally grouped, where variables in the same group
are either mechanistically related or statistically correlated. Under such settings, variable selection
must be conducted at both group and individual variable level. Motivated by the widespread
availability of zero-inflted count outcomes and grouped covariates in many practical applications, we
consider group regularization for zero-inflated Poisson (ZIP) regression models. Using a least squares
approximation of the mixture likelihood and a group-wise L1 penalty on the coefficients, we propose
a unified algorithm (Gooogle: Group Regularization for Zero Inflated Count Regression Models)
to efficiently compute the entire regularization path of the estimator. We also derive theoretical
properties of the proposed group variable selection procedure under certain regularity conditions,
which further provides deeper insight into the asymptotic behaviour of the method. On simulated
datasets, we show that Gooogle outperforms published methods in estimation and prediction across
a wide range of scenarios. Finally, we apply Gooogle to an auto insurance claim dataset from the
SAS Enterprise Miner database for illustrative purposes.

# Simulation Example 1
We construct a simulation example similar to Huang et al. (2009) [ https://doi.org/10.1093/biomet/asp020] which can be used to simulate a dataset according to the zero-inflated Poisson model. We have 40 covariates in 5 groups with 8 covariates in each group. For this example, we assume the covariates in the count part (X) and in the zero inflation part (Z) to be the same (i.e set Z=X). Below we explain each function that we have used to generate the data and calculate the predictive as well as group selection measures that we have presented in our simulation study in the manuscript.

## Data generating function
datagen.sim1.func<-function(n,p,ngrp,beta,gamma,rho,family): This is the data generating function which takes the following arguments:
n: sample size in each of training and testing group
p: Number of covariates
ngrp: Number of covariates in each group (constant for this example)
beta: True regression coeffficients in the count model
gamma: True regression coeffficients in the zero model
rho: Correlation parameter of AR(1) of the covariance matrix for multivariate normal distribution
family: It specifies the distribution of the count model which is "Poisson" in this case. 

## 
