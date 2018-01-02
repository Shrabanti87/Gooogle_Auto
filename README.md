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

# Simulation 
We conduct two extensive simulation studies to investigate the statistical properties of our proposed group regularization approach across a wide range of scenarios. To effectively evaluate the group selection performance of various methods, we prefix the number of groups to a moderately large number, and vary the group sizes from equal to unequal, and within each group. We partition each generated synthetic data into a training set and a test set; models are fitted on the training set and MASE's are calculated on the test set. We also note the percentage of times correct groups are selected by each method. We consider the sample size in training and test group as {200,200}, {500,500} and {1000,1000}, the zero inflation parameter \phi=0.3,0.4,0.5, the correlation parameter of the covariance matrix of the covariates \rho=0,0.4,0.8. Below we explain each function that we have used to generate the data and calculate the predictive as well as group selection measures that are presented in our simulation study in the manuscript.

## List of Functions

### Data generating function: `datagen_sim.R`
This R file contains three functions described below:

1. datagen.sim1.func & datage.sim2.func: These are the data generating functions for simulation 1 and 2 respectively which generate one dataset according to the specified parameters. 

n: sample size in each of training and testing group
p: Number of covariates
ngrp: Number of covariates in each group (constant for this example)
beta: True regression coeffficients in the count model
gamma: True regression coeffficients in the zero model
rho: Correlation parameter of AR(1) of the covariance matrix for multivariate normal distribution
sim: Takes "1" for simulation 1, else takes "2"
family: It specifies the distribution of the count model which is "Poisson" in this case. 

These functions generate the  dataset, zero-inflated outcome variable, covariates for the count model (X) which is assumed to be equal to that of the zero model (Z), and the proportion of zero inflation.

2. datagen.sim.all<-function(n,p,ngrp,beta,gamma,rho,family,ITER): This function calls the function datagen.sim1.func "ITER" times to generate the simulated data set "ITER" times for a given set of parameters. It outputs all the datasets in the list form `data.list`.

### Function to fit the gooogle method on the training dataset: `fit_method`
This R file contains the function `fit.method` which takes the the following as arguments:

dataset: generated dataset using datagen.sim1.func
yvar,xvars,zvars: corresponding variables
method: `EM_LASSO` or `gbridge` 
group: grouping of the covariates
dist : "Poisson"

Depending on the method specified in the argument, this function calls the gooogle function from the Gooogle package to fit the zero inflated dataset with grouped covariates or calls the function `func_EMLasso` (also present in this R file) which fits the data with `Lasso` penalty using the zipath function. The functon `fit.method` outputs the fitted coefficients for count model and zero model along with the AIC, BIC and loglikelihood of the zero-inflated model.

### Calculate the predictive measure `MASE`: `predict_measures.R`
This R file contains two functions described below:

1. `measures.func`: This function takes the following arguments:

train: Training dataset obtained from the original dataset 
test: Complement of the training dataset used for prediction
fit: The output of the function fit.method
yvar,xvars,zvars: Corresponding variables

For a given training and test dataset and the fitted coefficients this function calculates and returns the predictive measure MASE by using the function accuracy from the forecast package.

2. `measures.summary`: This function takes the following arguments:

n:  Sample size in the training dataset
data.list: output of datagen.sim1.all
method: `EM_LASSO` or `gbridge` 
ITER: number of simulations
group: grouping of the covariates
family: "negbin"

This function fits the gooogle method to the training part of a dataset in data.list and calculate the predicted value using the fitted coefficients from the test data. It calls the function `measures.func` to calculate MASE. This is repeated for "ITER" number of datasets and outputs the median `MASE`.

### Calculate the group selecton measure: `grpresult.R`

This R file contains the function `grpresult.func` which takes the following argument:

betahat: Fitted value of the regression coefficients
betatrue: True value of the regression coefficients
sim: Takes "1" for simulation 1, else takes "2"

This function returns the group selection measure `pgrp.correct` which is the proportion of groups correctly selected. 

## Example 1
In this example we generate 40 continuous predictors which are grouped into 5 groups consisting of 8 predictors each. The covariates are generated from multivariate normal distribution with `rho` as the corrrelation parameter of the AR(1) covariance matrix. The following function master.func generates 10 datasets for a given n, rho and phi from ZIP model and fits the google function and calculates median MASE as well percentage of correct group selection for both the count and zero model. The values of the regression coefficients for the count model (beta) and those for the zero model (gamma) are given inside the function. 

```
## Load the packages

require(MASS)
require(forecast)
require(Gooogle)
require(mpath)
require(dummies)
```

```
 ## Specify the true parameter values
  
 count model: beta<-c(5,-1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, rep(0.2,8), rep(0,24))    
    
 zero model: gamma<-c(-1,-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, rep(0.2,8), rep(0,24))
 For phi=0.3, gamma_0 (intercept) of the zero model is -1
 
 group: (8,8,8,8,8)
   ```
    
  ```
 ## Generate the list of datasets
  
  data.list<-datagen.sim.all(n=200,beta=beta,gamma=gamma,rho=0.4,phi=0.3,family="negbin",sim=1,ITER=10)
  ```
  
  ```
 ## Calculate MASE
  
  measures<-measures.summary(n=200,data.list=data.list,method="gBridge",ITER=10,group=group,family="negbin")
  ```
  
  ```
  ## Calculate the Percentage of correct group selection for both the count and zero model
  
  betahat<-measures$betahat
  gammahat<-measures$gammahat
  
  grp.count<- grpresult.func(betahat=betahat,betatrue=beta,sim=1)
  
  grp.zero<- grpresult.func(betahat=gammahat,betatrue=gamma,sim=1)
  
  pgrp.corr.count<- grp.count$pgrp.correct
  pgrp.corr.zero<- grp.zero$pgrp.correct
  
  result<-(data.frame(measures$output,pgrp.corr.count,pgrp.corr.zero))
  names(result)<- c("MASE","corr_group_count","corr_group_zero")
  return(result)
  ```

 ```
 ## Output 
      MASE corr_group_count corr_group_zero
1 0.99965                1             0.8
```
