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
  ```

 ```
 ## phi=0.3

 master.func(n=200,beta=beta,gamma=gamma,rho=0,phi=0.3,family="negbin",method="grLasso",sim=1,ITER=10)
 
 |Method | MAE     | MASE  |
 |-------|:-------:|------:|
 |grLasso|144.8763 |0.7141 | 
 |EMLasso|251.2550 |1.2517 |
          
        
## phi=0.4

 master.func(n=200,beta=beta,gamma=gamma,rho=0,phi=0.4,family="negbin",method="grLasso",sim=1,ITER=10)
 
 Output:
           MAE     MASE 
       103.6743   0.6996
       
 EM Output:
           MAE     MASE      
        181.7793   1.0558       
 
## phi=0.5

 master.func(n=200,beta=beta,gamma=gamma,rho=0,phi=0.5,family="negbin",method="grLasso",sim=1,ITER=10)
 
 Output:
          MAE    MASE 
        85.2746  0.6618  
        
 EM Output:
          MAE    MASE        
        136.2116   1.1475 
```

### Example 2

In example 2 we generate 6 continuous covariates each of which forms a singleton group. Four more variables (two each from X3 and X6) are further polynomially constructed, giving rise to a total of 10 continuous predictors. For constructing the categorical variables, we generate 5 continuous variables from a multivariate normal distribution and quantile-discretize each of them into 5 new variables based on their quintiles. This leads to a combination of 20 categorical predictors with 5 non-overlapping groups of equal size.

```
## Specify the true parameter values

For count model: 
betag1<-c(0)
betag2<-c(0)
betag3<-c(-0.1,0.2,0.1)
betag4<-c(0)
betag5<-c(0)
betag6<-c(2/3,-1,1/3)
betag7<-c(-2,-1,1,2)
betag8<-c(0,0,0,0)
betag9<-c(0,0,0,0)
betag10<-rep(0,4)
betag11<-c(0,0,0,0)

beta<-c(5,betag1,betag2,betag3,betag4,betag5,betag6,betag7,betag8,betag9,betag10,betag11)

For zero model:
phi=0.3: gamma<-c(-1.4,beta[-1])
phi=0.4: gamma<-c(-.7,beta[-1])
phi=0.5: gamma<-c(-.15,beta[-1])
```

```
We run the master.func to calculate MAE and MASE. Below we give the code and the output:

 ## phi=0.3

 master.func(n=200,beta=beta,gamma=gamma,rho=0,phi=0.3,family="negbin",method="grLasso",sim=2,ITER=10)
 
 Output:
           MAE      MASE 
       144.87630  0.71405 
       
 EM Output:      
      MAE      MASE 
256.19715   1.03615 


 ## phi=0.4

 master.func(n=200,beta=beta,gamma=gamma,rho=0,phi=0.4,family="negbin",method="grLasso",sim=2,ITER=10)
 
 Output:
            MAE     MASE 
        125.7181   0.8215 
 Output EM:
           MAE     MASE 
       189.3503   1.1017 

 ## phi=0.5

 master.func(n=200,beta=beta,gamma=gamma,rho=0,phi=0.5,family="negbin",method="grLasso",sim=2,ITER=10)
 
 Output:                                      
          MAE    MASE 
        82.1168  0.9727 
        
        output EM
     MAE     MASE 
142.1029   1.1952 
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
