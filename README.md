# Gooogle_Auto (Group Regularization for Zero-inflated Regression Models with Application to Insurance Rate-making)
# Introduction
In many risk analysis problems, covariates are naturally grouped, where variables in the same group
are either mechanistically related or statistically correlated. Under such settings, variable selection
must be conducted at both group and individual variable level. Motivated by the widespread
availability of zero-inflted count outcomes and grouped covariates in many practical applications, we
consider group regularization for zero-inflated Poisson (ZIP) regression models. Using a least squares
approximation of the mixture likelihood and a group-wise L1 penalty on the coefficients, we propose
a unied algorithm (Gooogle: Group Regularization for Zero Inflated Count Regression Models)
to eciently compute the entire regularization path of the estimator. We also derive theoretical
properties of the proposed group variable selection procedure under certain regularity conditions,
which further provides deeper insight into the asymptotic behaviour of the method. On simulated
datasets, we show that Gooogle outperforms published methods in estimation and prediction across
a wide range of scenarios. Finally, we apply Gooogle to an auto insurance claim dataset from the
SAS Enterprise Miner database for illustrative purposes.
