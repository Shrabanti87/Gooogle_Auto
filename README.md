# Simulation Codes for Gooogle_Auto (Group Regularization for Zero-inflated Regression Models with Application to Insurance Rate-making)


### Example 1
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
    
 zero model: 
 For phi=0.3,gamma<-c(-1,-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, rep(0.2,8), rep(0,24))
 For phi=0.4,gamma<-c(-0.5,-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, rep(0.2,8), rep(0,24))
 For phi=0.5,gamma<-c(0,-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, rep(0.2,8), rep(0,24))
  ```
    
  ```
 ## The main function for running the simulation and calculating the predictive measures
 
 methods<-c("EMLasso","grLasso", "grMCP", "grSCAD", "gBridge")

 master.func<-function(n,rho,phi,family,method,sim,ITER=10)
{
  if(sim==1)
  {
    # size for each group
    size=rep(8,5)
    p=sum(size)
    # number of groups
    ngrp<-length(size) 
  }else{
    size=c(1,1,3,1,1,3,4,4,4,4,4)
    p=sum(size)
    ngrp<-length(size)
  }
  group<-NULL
  for (k in 1:ngrp)
  {
    group<-c(group,rep(k,size[k]))
  }
  ### Generate the data ###
  data.list<-datagen.sim.all(n=n,beta=beta,gamma=gamma,rho=rho,phi=phi,family=family,sim=sim,ITER=10)
  
  ### Calculate the predictive measures ###
  measures<-measures.summary(n=n,data.list=data.list,method=method,ITER=ITER,group=group,family=family)
  
  return(measures)
}

master.func(n=200,beta=beta,gamma=gamma,rho=0,phi=0.3,family="negbin",method="grLasso",ITER=10)
  ```

 ```
 ## Output for phi=0.3
           MAE      MASE 
       144.87630  0.71405 

 ## Output for phi=0.4
           MAE     MASE 
       103.6743   0.6996 
 
 ## Output for phi=0.5
           MAE    MASE 
        85.2746  0.6618 
```
## Real Data Example

We illustrate our proposal by re-analyzing the auto insurance claim dataset from SAS Enterprise Miner database. The response variable of interest (y) is the aggregate claim loss of an auto insurance policy. Considering only policy records corresponding to the new customers the reduced dataset has 2,812 observations with 56 predictors being grouped into 21 groups of different group sizes. For the comparison of Gooogle methods with EMLasso we employ a repeated 5-fold cross validation (CV) procedure in which the dataset is randomly partitioned into 5 equal folds, iteratively taking each fold as the test set and the remaining set as the trainng set. We fit the models on the training sets and predictions are based on the test sets. We calculate average and median of the Mean Absolute Scaled Error (MASE) as the metric of evaluation, calculated over 100 iterations. 

```
Here is the code for calculating the MAE and MASE using 5-fold CV for the Gooogle method "gBridge" and for "EMLasso" for 5 iterations:

result.gooogle<-realdata.func(data=data,yvar=yvar,xvars=xvars,zvars=zvars,cv.iter=5,k.fold=5,seedval=123,method="Gooogle")

result.em<-realdata.func(data=data,yvar=yvar,xvars=xvars,zvars=zvars,cv.iter=5,k.fold=5,seedval=123,method="EMLasso")
```
```
Here is the output of "gBridge" :

     iter    MAE     MASE       
[1,]    1 2.767285 0.8916563         
[2,]    2 2.984444 0.9616278        
[3,]    3 2.986073 0.9621526        
[4,]    4 2.924540 0.9423260        
[5,]    5 3.142137 1.0124386

Here is the output of "EMLasso" :

     iter    MAE     MASE
[1,]    1 3.088494 0.995154
[2,]    2 3.122427 1.006088
[3,]    3 3.362784 1.083534
[4,]    4 3.227218 1.039853
[5,]    5 3.187387 1.027019

```
