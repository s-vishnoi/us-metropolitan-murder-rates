import pystan
import pickle
# import json
import numpy as np
import pandas as pd
import arviz as az
import os
# os.environ["CC"] = "clang"
# os.environ["CXX"] = "clang"

NITER     = 7000
THIN      = 1
NCHAINS   = 2  
NCORES    = 2 
WARMUP    = 6000

def compute_mean(data_info, exog_var):
    results=[]

    hdi_interval=az.hdi(data_info['a'].flatten(),hdi_prob=0.95)
    results.append(['Intercept',data_info['a'].mean(),data_info['a'].std(),hdi_interval[0],hdi_interval[1]])
    
    shape=len(data_info['b'].shape)
    if shape==2:
        for i in range(0,len(exog_var)):
            hdi_interval=az.hdi(data_info['b'][:,i].flatten(),hdi_prob=0.95)
            results.append([exog_var[i],data_info['b'][:,i].mean(),data_info['b'][:,i].std(),hdi_interval[0],hdi_interval[1]])        
    elif shape==3:
        for i in range(0,len(exog_var)):
            hdi_interval=az.hdi(data_info['b'][:,:,i].flatten(),hdi_prob=0.95)
            results.append([exog_var[i],data_info['b'][:,:,i].mean(),data_info['b'][:,:,i].std(),hdi_interval[0],hdi_interval[1]])        
    elif shape==4:
        for i in range(0,len(exog_var)):
            hdi_interval=az.hdi(data_info['b'][:,:,:,i].flatten(),hdi_prob=0.95)
            results.append([exog_var[i],data_info['b'][:,:,:,i].mean(),data_info['b'][:,:,:,i].std(),hdi_interval[0],hdi_interval[1]])

    df=pd.DataFrame(results,columns=['Variable','Mean','Std','2.5%','97.5%'])
    return df

def read_data(path):
    data=pd.read_csv(path,index_col=0)
    return data

def center_data(data,exog_var):
    data_c=data.copy()
    for column in exog_var:
        data_c[column]=data_c[column]-data_c[column].mean()
    return data_c

def generate_entity_time_ids(data,entity='Regions'):
    df=data.copy()

    # Create entity and time effects labels:
    # df['Year']=df['Year']-np.min(df['Year'])
    # time_ids=df['Year'].values

    years=np.unique(df['Year'])           
    years_dict = {years[i]:i for i in range(len(years))} 
    df['Year']=df['Year'].map(years_dict)
    time_ids= df['Year'].values   

    entities=np.unique(df[entity])           
    entity_dict = {entities[i]:i for i in range(len(entities))} 
    df[entity]=df[entity].map(entity_dict)
    entity_ids= df[entity].values
    return entity_ids,time_ids

def compile_stanmodel(model):
    # Compile model
    smodel = pystan.StanModel(model_code=model)
    return smodel

def MCMCinference(smodel,data):
    # Inference
    fit = smodel.sampling(data    = data, 
                          iter    = NITER, 
                          chains  = NCHAINS,
                          thin    = THIN,
                          n_jobs  = NCORES,
                          warmup  = WARMUP,
                          control = {"adapt_delta": 0.9, "max_treedepth":13}
                           )         
    return fit

def bayes_r2(y_true,y_pred):
    
    """ Compute the Bayesian R-square according to the formula proposed by 
    Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019). 
    R-squared for Bayesian regression models. American Statistician, 1-6. 
    https://doi.org/10.1080/00031305.2018.1549100"""
    var_y_est = np.var( y_pred.mean(0))
    var_e = np.var((y_true - y_pred),axis=0)
    r_squared = var_y_est / (var_y_est + var_e)
    return r_squared

def compute_waic(fit):
    log_lik  = fit['log_lik']
    scale_value=1
    shape         = log_lik.shape
    n_samples     = shape[0]
    n_data_points = shape[1]
    vars_lpd      = np.var(log_lik,axis=0)
    lppd_i        = np.log(np.mean(np.exp(log_lik),axis=0))


    waic_i        = scale_value * (lppd_i - vars_lpd)
    waic_se       = (n_data_points * np.var(waic_i)) ** 0.5
    waic_sum      = np.sum(waic_i)
    p_waic        = np.sum(vars_lpd)
    return [waic_sum,waic_se,p_waic]

def save_results(fit,data,save_path):  
    # Save fitted model!
    summary_data=fit.summary()
    df_summary=pd.DataFrame(summary_data['summary'],
                            columns=summary_data['summary_colnames'],
                            index=summary_data['summary_rownames'])

    #Compute R-squared
    y_pred=fit.extract()['yhat']
    y_true=data['y']
    r2=bayes_r2(y_true,y_pred)

    # data_info
    data_samples=fit.extract()
    # data_samples=remove_first_half_of_chains(data_samples)

    # df=fit.to_dataframe()
    data={ 'summary':df_summary,
           'waic':compute_waic(fit),
           'data':data_samples,
           'R-squared':r2
        }


    path=save_path+'.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return data
    
def load_results(load_path):
    # Read fitted model!
    path=load_path+'.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################

#### Pooled model ####

# Model 1
def pooled(df,dependent_var,exog_var):
    """
    y_ijt = a + b x_ijt + N(0,sigma)
    
    """
    
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors
        real y[N];
        real x[N,K];

    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a;
        real b[K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a;
           for (k in 1:K){
               mu[n] = mu[n] + b[k] * x[n,k];
           }
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }

        // prior
        a ~ normal(mu_a, sigma_a);
        for (k in 1:K){
            b[k] ~ normal(mu_b[k], sigma_b[k]);
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma); 
        }      
        
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma);
            yhat[n] = normal_rng( mu[n], sigma);
        }
    }
    '''

    return model,data

#### Time-varying models ####

# Model 2
def time_varying_intercept(df,dependent_var,exog_var):
    """
    y_ijt = a_t + b x_ijt + N(0,sigma)
    
    """
    
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'year'  : time_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[T];
        real b[K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[year[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[k] * x[n,k];
           }
        }
    }
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }

        // prior
        for (t in 1:T){
            a[t] ~ normal(mu_a, sigma_a);
        }
        for (k in 1:K){
            b[k] ~ normal(mu_b[k], sigma_b[k]);
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma);     
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma); 
            yhat[n] = normal_rng( mu[n], sigma); 
        }
    }
    '''

    return model,data

# Model 3
def time_varying_slope(df,dependent_var,exog_var):
    """
    y_ijt = a + b_t x_ijt + N(0,sigma)
    
    """
    
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'year'  : time_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a;
        real b[T,K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a;
           for (k in 1:K){
               mu[n] = mu[n] + b[year[n],k] * x[n,k];
           }
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25); 
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }

        // prior
        a ~ normal(mu_a, sigma_a);
        for (t in 1:T){
            for (k in 1:K){
                b[t,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma); 
            yhat[n] = normal_rng( mu[n], sigma); 
        }
    }
    '''

    return model,data

# Model 4
def time_varying_intercept_slope(df,dependent_var,exog_var):
    """
    y_ijt = a_t + b_t x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'year'  : time_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[T];
        real b[T,K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[year[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[year[n],k] * x[n,k] ;
           }
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }

        // prior
        for (t in 1:T){
            a[t] ~ normal(mu_a, sigma_a);
            for (k in 1:K){
                b[t,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma);
            
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma);
            yhat[n] = normal_rng( mu[n], sigma);
        }
    }
    '''
    return model, data

#### Entity-varying models ####

# Model 5
def entity_varying_intercept(df,dependent_var,exog_var):
    """
    y_ijt = a_j + b x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'J'     : len(n_entities),
            'entity': entity_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of entity groups
        int entity[N];    // lookup table with entity ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[J];
        real b[K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[entity[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[k] * x[n,k];
           }
        }
    }

    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }

        // prior
        for (j in 1:J){
            a[j] ~ normal(mu_a, sigma_a);
        }
        for (k in 1:K){
            b[k] ~ normal(mu_b[k], sigma_b[k]);
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma); 
            yhat[n] = normal_rng( mu[n], sigma); 
        }
    }
    '''
    return model, data

# Model 6
def entity_varying_slope(df,dependent_var,exog_var):
    """
    y_ijt = a + b_j x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'J'     : len(n_entities),
            'entity': entity_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of entity groups
        int entity[N];    // lookup table with entity ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a;
        real b[J,K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a;
           for (k in 1:K){
               mu[n] = mu[n]+b[entity[n],k] * x[n,k] ;
           }
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }

        // prior
        a ~ normal(mu_a, sigma_a);
        for (j in 1:J){
            for (k in 1:K){
                b[j,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma); 
            yhat[n] = normal_rng( mu[n], sigma); 
        }
    }
    '''
    return model, data

# Model 7
def entity_varying_intercept_slope(df,dependent_var,exog_var):
    """
    y_ijt = a_j + b_j x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'J'     : len(n_entities),
            'entity': entity_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of entity groups
        int entity[N];    // lookup table with entity ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma;
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[J];
        real b[J,K];
    }
    
    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[entity[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[entity[n],k] * x[n,k] ;
           }
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }

        // prior
        for (j in 1:J){
            a[j] ~ normal(mu_a, sigma_a);
            for (k in 1:K){
                b[j,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] |  mu[n], sigma); 
            yhat[n] = normal_rng( mu[n], sigma);
        }
    }
    '''
    return model, data

#### Entity- and time-varying models ####

# Model 8
def time_entity_varying_intercept(df,dependent_var,exog_var):
    """
    y_ijt = a_j + g_t + b x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
      real<lower=0> sigma;
      real<lower=0> sigma_a;
      real<lower=0> sigma_b[K];
      real a[J,T];
      real b[K];
      real mu_a;
      real mu_b[K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[entity[n],year[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[k] * x[n,k] ;
           }
        }
    }
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }
            
        // prior
        for (j in 1:J){
            for (t in 1:T){
                a[j,t] ~ normal(mu_a, sigma_a);   
            }
        }
        for (k in 1:K){
            b[k] ~ normal(mu_b[k] , sigma_b[k]);
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
             log_lik[n] = normal_lpdf(y[n] |  mu[n], sigma);
            yhat[n] = normal_rng( mu[n], sigma);
        }
    }

    '''

    return model,data

# Model 9
def time_entity_varying_intercept_time_varying_slope(df,dependent_var,exog_var):
    """
    y_ijt = a_jt + b_t x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
      real<lower=0> sigma;
      real<lower=0> sigma_a;
      real<lower=0> sigma_b[K];
      real a[J,T];
      real b[T,K];
      real mu_a;
      real mu_b[K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[entity[n],year[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[year[n],k] * x[n,k];
           }
        }
    }
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }
            
        // prior
        for (j in 1:J){
            for (t in 1:T){
                a[j,t] ~ normal(mu_a, sigma_a);
            }
        }        

        for (t in 1:T){
            for (k in 1:K){
                b[t,k] ~ normal(mu_b[k] , sigma_b[k]);
            }
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
             log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
            yhat[n] = normal_rng( mu[n], sigma);
        }
    } 

    '''
    return model,data 

# Model 10
def time_entity_varying_intercept_slope(df,dependent_var,exog_var):
    """
    y_ijt = a_jt + b_jt x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
      real<lower=0> sigma;
      real<lower=0> sigma_a;
      real<lower=0> sigma_b[K];
      real a[J,T];
      real b[J,T,K];
      real mu_a;
      real mu_b[K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[entity[n],year[n]] ;
           for (k in 1:K){
               mu[n] = mu[n] +  b[entity[n],year[n],k] * x[n,k];
           }
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
        }
            
        // prior
        for (j in 1:J){
            for (t in 1:T){
                a[j,t] ~ normal(mu_a, sigma_a);
                for (k in 1:K){
                    b[j,t,k] ~ normal(mu_b[k], sigma_b[k]);
                }
            }
        }
        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
            yhat[n] = normal_rng(mu[n], sigma);
        }
             
    }

    '''

    return model,data

# Model 11
def random_walk_model(df,dependent_var,exog_var):
    """
    y_ijt = a_jt + b_jt x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
      real<lower=0> sigma;
      real<lower=0> sigma_a;
      real<lower=0> sigma_b[K];
      real a[J,T];
      real b[J,T,K];
      real mu_a[J];
      real mu_b[J,K];
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[entity[n],year[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[entity[n],year[n],k] * x[n,k] ;
           }
        }
    }
    
    model{
        // hyperprior
        sigma_a ~ gamma(0.25, 0.25);
        for (j in 1:J){
            mu_a[j] ~ normal(0, 10);
        }

        for (k in 1:K){
            sigma_b[k] ~ gamma(0.25, 0.25);
            for (j in 1:J){
                mu_b[j,k] ~ normal(0, 10);
            }
        }
            
        // prior
        for (j in 1:J){
            a[j,1] ~ normal(mu_a[j], sigma_a);
            for (t in 2:T){
                a[j,t] ~ normal(a[j,t-1], sigma_a);
            }
        }

        for (j in 1:J){
            for (k in 1:K){
                b[j,1,k] ~ normal(mu_b[j,k], sigma_b[k]);
                for (t in 2:T){
                    b[j,t,k] ~ normal(b[j,t-1,k], sigma_b[k]);
                }
            }
        }

        // model error
        sigma ~ gamma(0.25, 0.25);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
            yhat[n] = normal_rng(mu[n], sigma);
        }
             
    }

    '''

    return model,data

# Model 12
def lasso_model(df,dependent_var,exog_var):
    """
    y_ijt = a_jt + b_jt x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
      real<lower=0> sigma;
      real<lower=0> sigma_a;
      real<lower=0> sigma_b[K];
      real a[J,T];
      real b[J,T,K];
      real mu_a;
    }

    transformed parameters {
        vector[N] mu;
        for (n in 1:N){
           mu[n] = a[entity[n],year[n]];
           for (k in 1:K){
               mu[n] = mu[n] + b[entity[n],year[n],k] * x[n,k];
           }
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 1);
        sigma_a ~ cauchy(0, 1);
        for (k in 1:K){
            sigma_b[k] ~ cauchy(0, 1);
        }
            
        // prior
        for (j in 1:J){
            for (t in 1:T){
                a[j,t] ~ normal(mu_a, sigma_a);
                for (k in 1:K){
                    b[j,t,k] ~ double_exponential(0, sigma_b[k]);
                }
            }
        }
        // model error
        sigma ~ cauchy(0, 1);
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
             log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
            yhat[n] = normal_rng(mu[n], sigma);
        }
    }

    '''

    return model,data

#########################################################################################################################################
############################################################ Variance models ############################################################
#########################################################################################################################################

#### Pooled model ####

# Model 13
def pooled_var(df,dependent_var,exog_var):
    """
    y_ijt = a + b x_ijt + N(0,sigma)
    
    """
    
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors
        real y[N];
        real x[N,K];

    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a;
        real b[K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a;
            sigma[n]  = g;
            for (k in 1:K){
                mu[n] = mu[n] + b[k] * x[n,k];
                sigma[n] = sigma[n] + delta[k] * x[n,k];
            }
            sigma[n] = exp(sigma[n]);
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        g ~ normal(0,10);
        sigma_a ~ gamma(0.25, 0.25);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }

        // prior
        a ~ normal(mu_a, sigma_a);
        for (k in 1:K){
            b[k] ~ normal(mu_b[k], sigma_b[k]);
        }

        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma[n]); 
        }      
        
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma[n]);
            yhat[n] = normal_rng( mu[n], sigma[n]);
        }
    }
    '''

    return model,data

#### Time-varying models ####

# Model 14
def time_varying_intercept_var(df,dependent_var,exog_var):
    """
    y_ijt = a_t + b x_ijt + N(0,sigma)
    
    """
    
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'year'  : time_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[T];
        real b[K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[year[n]];
            sigma[n]  = g;
            for (k in 1:K){
                mu[n] = mu[n] + b[k] * x[n,k];
                sigma[n] = sigma[n] + delta[k] * x[n,k];
            }
            sigma[n] = exp(sigma[n]);
        }
    }

    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }

        // prior
        for (t in 1:T){
            a[t] ~ normal(mu_a, sigma_a);
        }
        for (k in 1:K){
            b[k] ~ normal(mu_b[k], sigma_b[k]);
        }


        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma[n]);     
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma[n]); 
            yhat[n] = normal_rng( mu[n], sigma[n]); 
        }
    }
    '''

    return model,data

# Model 15
def time_varying_slope_var(df,dependent_var,exog_var):
    """
    y_ijt = a + b_t x_ijt + N(0,sigma)
    
    """
    
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'year'  : time_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a;
        real b[T,K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a;
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] + b[year[n],k] * x[n,k];
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25); 
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }

        // prior
        a ~ normal(mu_a, sigma_a);
        for (t in 1:T){
            for (k in 1:K){
                b[t,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma[n]);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma[n]); 
            yhat[n] = normal_rng( mu[n], sigma[n]); 
        }
    }
    '''

    return model,data

# Model 16
def time_varying_intercept_slope_var(df,dependent_var,exog_var):
    """
    y_ijt = a_t + b_t x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'year'  : time_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[T];
        real b[T,K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[year[n]];
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] + b[year[n],k] * x[n,k] ;
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }

        // prior
        for (t in 1:T){
            a[t] ~ normal(mu_a, sigma_a);
            for (k in 1:K){
                b[t,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma[n]);
            
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma[n]);
            yhat[n] = normal_rng( mu[n], sigma[n]);
        }
    }
    '''
    return model, data

#### Entity-varying models ####

# Model 17
def entity_varying_intercept_var(df,dependent_var,exog_var):
    """
    y_ijt = a_j + b x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'J'     : len(n_entities),
            'entity': entity_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of entity groups
        int entity[N];    // lookup table with entity ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[J];
        real b[K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[entity[n]];
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] + b[k] * x[n,k];
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }

    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }

        // prior
        for (j in 1:J){
            a[j] ~ normal(mu_a, sigma_a);
        }
        for (k in 1:K){
            b[k] ~ normal(mu_b[k], sigma_b[k]);
        }
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma[n]);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma[n]); 
            yhat[n] = normal_rng( mu[n], sigma[n]); 
        }
    }
    '''
    return model, data

# Model 18
def entity_varying_slope_var(df,dependent_var,exog_var):
    """
    y_ijt = a + b_j x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'J'     : len(n_entities),
            'entity': entity_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of entity groups
        int entity[N];    // lookup table with entity ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a;
        real b[J,K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a;
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] + b[entity[n],k] * x[n,k] ;
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }

        // prior
        a ~ normal(mu_a, sigma_a);
        for (j in 1:J){
            for (k in 1:K){
                b[j,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma[n]);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] | mu[n], sigma[n]); 
            yhat[n] = normal_rng( mu[n], sigma[n]); 
        }
    }
    '''
    return model, data

# Model 19
def entity_varying_intercept_slope_var(df,dependent_var,exog_var):
    """
    y_ijt = a_j + b_j x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'J'     : len(n_entities),
            'entity': entity_ids+1, # Stan counts starting at 1
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of entity groups
        int entity[N];    // lookup table with entity ids
        real y[N];        
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real mu_a;
        real mu_b[K];
        real a[J];
        real b[J,K];
        real g;
        real delta[K];
    }
    
    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[entity[n]];
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] + b[entity[n],k] * x[n,k] ;
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }

        // prior
        for (j in 1:J){
            a[j] ~ normal(mu_a, sigma_a);
            for (k in 1:K){
                b[j,k] ~ normal(mu_b[k], sigma_b[k]);
            }
        }

        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma[n]);
        }
    }

    generated quantities{
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf( y[n] |  mu[n], sigma[n]); 
            yhat[n] = normal_rng( mu[n], sigma[n]);
        }
    }
    '''
    return model, data

#### Entity- and time-varying models ####

# Model 20
def time_entity_varying_intercept_var(df,dependent_var,exog_var):
    """
    y_ijt = a_j + g_t + b x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real a[J,T];
        real b[K];
        real mu_a;
        real mu_b[K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[entity[n],year[n]];
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] + b[k] * x[n,k] ;
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }
            
        // prior
        for (j in 1:J){
            for (t in 1:T){
                a[j,t] ~ normal(mu_a, sigma_a);   
            }
        }
        for (k in 1:K){
            b[k] ~ normal(mu_b[k] , sigma_b[k]);
        }

        // likelihood
        for (n in 1:N){
            y[n] ~ normal( mu[n], sigma[n]);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
             log_lik[n] = normal_lpdf(y[n] |  mu[n], sigma[n]);
            yhat[n] = normal_rng( mu[n], sigma[n]);
        }
    }

    '''

    return model,data

# Model 21
def time_entity_varying_intercept_time_varying_slope_var(df,dependent_var,exog_var):
    """
    y_ijt = a_jt + b_t x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real a[J,T];
        real b[T,K];
        real mu_a;
        real mu_b[K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[entity[n],year[n]];
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] + b[year[n],k] * x[n,k];
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }
            
        // prior
        for (j in 1:J){
            for (t in 1:T){
                a[j,t] ~ normal(mu_a, sigma_a);
            }
        }        

        for (t in 1:T){
            for (k in 1:K){
                b[t,k] ~ normal(mu_b[k] , sigma_b[k]);
            }
        }
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma[n]);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
             log_lik[n] = normal_lpdf(y[n] | mu[n], sigma[n]);
            yhat[n] = normal_rng( mu[n], sigma[n]);
        }
    } 

    '''
    return model,data 

# Model 22
def time_entity_varying_intercept_slope_var(df,dependent_var,exog_var):
    """
    y_ijt = a_jt + b_jt x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real a[J,T];
        real b[J,T,K];
        real mu_a;
        real mu_b[K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[entity[n],year[n]];
            sigma[n]  = g;  
            for (k in 1:K){
                mu[n] = mu[n] +  b[entity[n],year[n],k] * x[n,k];
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    
    model{
        // hyperprior
        mu_a ~ normal(0, 10);
        sigma_a ~ gamma(0.25, 0.25);
        g ~ normal(0,10);

        for (k in 1:K){
            mu_b[k] ~ normal(0, 10);
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
        }
            
        // prior
        for (j in 1:J){
            for (t in 1:T){
                a[j,t] ~ normal(mu_a, sigma_a);
                for (k in 1:K){
                    b[j,t,k] ~ normal(mu_b[k], sigma_b[k]);
                }
            }
        }
        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma[n]);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf(y[n] | mu[n], sigma[n]);
            yhat[n] = normal_rng(mu[n], sigma[n]);
        }
             
    }

    '''

    return model,data

# Model 23
def random_walk_model_var(df,dependent_var,exog_var):
    """
    y_ijt = a_jt + b_jt x_ijt + N(0,sigma)
    
    """
    entity_ids,time_ids=generate_entity_time_ids(df)
    n_years=np.unique(time_ids)
    n_entities=np.unique(entity_ids)

    data = {
            'N'     : len(df),
            'K'     : len(exog_var),
            'T'     : len(n_years),
            'J'     : len(n_entities),
            'year'  : time_ids+1, # Stan counts starting at 1
            'entity': entity_ids+1,
            'x'     : df[exog_var].values,
            'y'     : df[dependent_var].values
           }

    model = '''
    data {
        int<lower=0> N;   // number of data items
        int<lower=0> K;   // number of predictors        
        int<lower=0> J;   // number of spatial groups
        int<lower=0> T;   // number of time observations
        int year[N];      // lookup table with year ids
        int entity[N];    // lookup table with entity ids
        real y[N];
        real x[N,K];
    }
    parameters {
        real<lower=0> sigma_a;
        real<lower=0> sigma_b[K];
        real a[J,T];
        real b[J,T,K];
        real mu_a[J];
        real mu_b[J,K];
        real g;
        real delta[K];
    }

    transformed parameters {
        vector[N] mu;
        vector[N] sigma;
        for (n in 1:N){
            mu[n] = a[entity[n],year[n]];
            sigma[n]  = g;
            for (k in 1:K){
                mu[n]=mu[n]+ b[entity[n],year[n],k] * x[n,k];
                sigma[n] = sigma[n] + delta[k] * x[n,k];
           }
           sigma[n] = exp(sigma[n]);
        }
    }
    
    model{
        // hyperprior
        sigma_a ~ gamma(0.25, 0.25);
        for (j in 1:J){
            mu_a[j] ~ normal(0, 10);
        }
        g ~ normal(0,10);

        for (k in 1:K){
            sigma_b[k] ~ gamma(0.25, 0.25);
            delta[k] ~ normal(0,10);
            for (j in 1:J){
                mu_b[j,k] ~ normal(0, 10);
            }
        }
            
        // prior
        for (j in 1:J){
            a[j,1] ~ normal(mu_a[j], sigma_a);
            for (t in 2:T){
                a[j,t] ~ normal(a[j,t-1], sigma_a);
            }
        }

        for (j in 1:J){
            for (k in 1:K){
                b[j,1,k] ~ normal(mu_b[j,k], sigma_b[k]);
                for (t in 2:T){
                    b[j,t,k] ~ normal(b[j,t-1,k], sigma_b[k]);
                }
            }
        }

        
        // likelihood
        for (n in 1:N){
            y[n] ~ normal(mu[n], sigma[n]);     
        }
    }

    generated quantities {
        vector[N] log_lik;
        vector[N] yhat;
        for (n in 1:N){
            log_lik[n] = normal_lpdf(y[n] | mu[n], sigma[n]);
            yhat[n] = normal_rng(mu[n], sigma[n]);
        }
             
    }

    '''

    return model,data


#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################

# Driver
def urban_scaling(df,dependent_var,model_key,save_path,loadtrace=False):
    exog_var =[ 'Population']
    # pooled model
    if  model_key=='model_1':
        model,data=pooled(df,dependent_var,exog_var)

    # time-varying models
    elif model_key=='model_2':
        model,data=time_varying_intercept(df,dependent_var,exog_var)
    
    elif model_key=='model_3':
        model,data=time_varying_slope(df,dependent_var,exog_var)
    
    elif model_key=='model_4':
        model,data=time_varying_intercept_slope(df,dependent_var,exog_var)

    # entity-varying models   
    elif model_key=='model_5':
        model,data=entity_varying_intercept(df,dependent_var,exog_var)
    
    elif model_key=='model_6':
        model,data=entity_varying_slope(df,dependent_var,exog_var)
    
    elif model_key=='model_7':
        model,data=entity_varying_intercept_slope(df,dependent_var,exog_var)

    # entity- and time-varying models
    elif model_key=='model_8':
        model,data=time_entity_varying_intercept(df,dependent_var,exog_var)
    
    elif model_key=='model_9':
        model,data=time_entity_varying_intercept_time_varying_slope(df,dependent_var,exog_var)
    
    elif model_key=='model_10':
        model,data=time_entity_varying_intercept_slope(df,dependent_var,exog_var)

    elif model_key=='model_11':
        model,data=random_walk_model(df,dependent_var,exog_var)
        
    # pooled model
    elif model_key=='model_12':
        model,data=pooled_var(df,dependent_var,exog_var)

    # time-varying models
    elif model_key=='model_13':
        model,data=time_varying_intercept_var(df,dependent_var,exog_var)
    
    elif model_key=='model_14':
        model,data=time_varying_slope_var(df,dependent_var,exog_var)
    
    elif model_key=='model_15':
        model,data=time_varying_intercept_slope_var(df,dependent_var,exog_var)

    # entity-varying models   
    elif model_key=='model_16':
        model,data=entity_varying_intercept_var(df,dependent_var,exog_var)
    
    elif model_key=='model_17':
        model,data=entity_varying_slope_var(df,dependent_var,exog_var)
    
    elif model_key=='model_18':
        model,data=entity_varying_intercept_slope_var(df,dependent_var,exog_var)

    # entity- and time-varying models
    elif model_key=='model_19':
        model,data=time_entity_varying_intercept_var(df,dependent_var,exog_var)
    
    elif model_key=='model_20':
        model,data=time_entity_varying_intercept_time_varying_slope_var(df,dependent_var,exog_var)
    
    elif model_key=='model_21':
        model,data=time_entity_varying_intercept_slope_var(df,dependent_var,exog_var)

    elif model_key=='model_22':
        model,data=random_walk_model_var(df,dependent_var,exog_var) 

    else:
        print('Model not found')
        return 0
    
    if loadtrace==True:
        trace=load_results(save_path)
    else:
        # Compile
        smodel=compile_stanmodel(model)

        # Inference
        fit=MCMCinference(smodel,data)

        # Save
        trace=save_results(fit,data,save_path)
    return model,trace

# Driver
def cobb_douglas(df,dependent_var,model_key,save_path,loadtrace=False):

    exog_var = [ 'Whites', 'Blacks', 'Minorities','Poverty','Unemployment','Education']
    data = df.copy()
    df = center_data(data,exog_var)

    # pooled model
    if  model_key=='model_1':
        model,data=pooled(df,dependent_var,exog_var)

    # time-varying models
    elif model_key=='model_2':
        model,data=time_varying_intercept(df,dependent_var,exog_var)
    
    elif model_key=='model_3':
        model,data=time_varying_slope(df,dependent_var,exog_var)
    
    elif model_key=='model_4':
        model,data=time_varying_intercept_slope(df,dependent_var,exog_var)

    # entity-varying models   
    elif model_key=='model_5':
        model,data=entity_varying_intercept(df,dependent_var,exog_var)
    
    elif model_key=='model_6':
        model,data=entity_varying_slope(df,dependent_var,exog_var)
    
    elif model_key=='model_7':
        model,data=entity_varying_intercept_slope(df,dependent_var,exog_var)

    # entity- and time-varying models
    elif model_key=='model_8':
        model,data=time_entity_varying_intercept(df,dependent_var,exog_var)
    
    elif model_key=='model_9':
        model,data=time_entity_varying_intercept_time_varying_slope(df,dependent_var,exog_var)
    
    elif model_key=='model_10':
        model,data=time_entity_varying_intercept_slope(df,dependent_var,exog_var)

    elif model_key=='model_11':
        model,data=random_walk_model(df,dependent_var,exog_var)

    elif model_key=='model_12':
        model,data=lasso_model(df,dependent_var,exog_var)       

    # pooled model
    elif  model_key=='model_13':
        model,data=pooled_var(df,dependent_var,exog_var)

    # time-varying models
    elif model_key=='model_14':
        model,data=time_varying_intercept_var(df,dependent_var,exog_var)
    
    elif model_key=='model_15':
        model,data=time_varying_slope_var(df,dependent_var,exog_var)
    
    elif model_key=='model_16':
        model,data=time_varying_intercept_slope_var(df,dependent_var,exog_var)

    # entity-varying models   
    elif model_key=='model_17':
        model,data=entity_varying_intercept_var(df,dependent_var,exog_var)
    
    elif model_key=='model_18':
        model,data=entity_varying_slope_var(df,dependent_var,exog_var)
    
    elif model_key=='model_19':
        model,data=entity_varying_intercept_slope_var(df,dependent_var,exog_var)

    # entity- and time-varying models
    elif model_key=='model_20':
        model,data=time_entity_varying_intercept_var(df,dependent_var,exog_var)
    
    elif model_key=='model_21':
        model,data=time_entity_varying_intercept_time_varying_slope_var(df,dependent_var,exog_var)
    
    elif model_key=='model_22':
        model,data=time_entity_varying_intercept_slope_var(df,dependent_var,exog_var)

    elif model_key=='model_23':
        model,data=random_walk_model_var(df,dependent_var,exog_var)

    # Random walk with one less variable    
    elif model_key=='model_24':
        exog_var2 =[ 'Blacks', 'Minorities','Poverty','Unemployment','Education']
        model,data=random_walk_model_var(df,dependent_var,exog_var2)

    elif model_key=='model_25':
        exog_var2 =[ 'Whites', 'Minorities','Poverty','Unemployment','Education']
        model,data=random_walk_model_var(df,dependent_var,exog_var2)

    elif model_key=='model_26':
        exog_var2 =[ 'Whites', 'Blacks','Poverty','Unemployment','Education']
        model,data=random_walk_model_var(df,dependent_var,exog_var2)

    elif model_key=='model_27':
        exog_var2 =[ 'Whites', 'Blacks', 'Minorities','Unemployment','Education']
        model,data=random_walk_model_var(df,dependent_var,exog_var2)

    elif model_key=='model_28':
        exog_var2 =[ 'Whites', 'Blacks', 'Minorities','Poverty','Education']
        model,data=random_walk_model_var(df,dependent_var,exog_var2)

    elif model_key=='model_29':
        exog_var2 =[ 'Whites', 'Blacks', 'Minorities','Poverty','Unemployment']
        model,data=random_walk_model_var(df,dependent_var,exog_var2)

    ### model variations ### 
    elif model_key=='model_30':
        exog_var2 = [ 'Whites', 'Blacks', 'Minorities','Poverty','Unemployment']

        df2 = data[data.Year>=1990]
        df2 = center_data(df2,exog_var2)
        model,data=random_walk_model_var(df2,dependent_var,exog_var2)

    elif model_key=='model_31':
        exog_var2 = [ 'Whites', 'Blacks', 'Minorities','Poverty','Unemployment', 'Education']
        df2 = data[data.Year>=1990]
        df2 = center_data(df2,exog_var2)
        model,data=random_walk_model_var(df2,dependent_var,exog_var2)

    elif model_key=='model_32':
        exog_var2 = [ 'Whites','Blacks', 'Minorities','Poverty','Unemployment','Education']
        df2 = data[data.Year>=1970]
        df2 = center_data(df2,exog_var2)
        model,data=random_walk_model_var(df2,dependent_var,exog_var2)

    elif model_key=='model_33':
        exog_var2 = [ 'Whites', 'Blacks', 'Minorities','Poverty','Unemployment']
        df2 = data[data.Year>=1970]
        df2 = center_data(df2,exog_var2)
        model,data=random_walk_model_var(df2,dependent_var,exog_var2)

    elif model_key=='model_34':
        exog_var2 = [ 'Whites', 'Blacks', 'Minorities']
        df2 = data[data.Year>=1970]
        df2 = center_data(df2,exog_var2)
        model,data=random_walk_model_var(df2,dependent_var,exog_var2)

    elif model_key=='model_35':
        exog_var2 = [ 'Whites', 'Non-White','Poverty','Unemployment','Education']
        df = center_data(data,exog_var2)
        model,data=random_walk_model_var(df,dependent_var,exog_var2)

    elif model_key=='model_36':
        exog_var2 = [ 'Whites', 'Non-White','Poverty','Unemployment']
        df = center_data(data,exog_var2)
        model,data=random_walk_model_var(data,dependent_var,exog_var2)

    elif model_key=='model_37':
        exog_var2 = [ 'Whites', 'Non-White']
        df = center_data(data,exog_var2)
        model,data=random_walk_model_var(df,dependent_var,exog_var2)


    else:
        print('Model not found')
        return 0
    
    if loadtrace==True:
        trace=load_results(save_path)
    else:
        # Compile
        smodel=compile_stanmodel(model)

        # Inference
        fit=MCMCinference(smodel,data)

        # Save
        trace=save_results(fit,data,save_path)
    return model,trace
   


   
   

