import numpy as np
# import pymc3 as pm
import pandas as pd
from scipy import optimize

######### Statistics #########

def pearson(x,y):

    return np.corrcoef(x,y)[0][1]

def bootstrapping(x,y,func,samples,alpha):

    r=func(x,y)
    index=np.arange(0,len(x))
    results=np.zeros(samples)

    for i  in range(0,samples):
        rand_ind=np.random.choice(index,size=len(index))
        xr,yr=x[rand_ind],y[rand_ind]
        results[i]=func(xr,yr)
    rinf=np.percentile(results,alpha/2*100)
    rsup=np.percentile(results,(1-alpha/2)*100)

    return r,rinf,rsup

def linear(x, a,b):
    return a+b*x

def bayes_r2(y_true,y_pred):
    
    """ Compute the Bayesian R-square according to the formula proposed by 
    Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019). 
    R-squared for Bayesian regression models. American Statistician, 1-6. 
    https://doi.org/10.1080/00031305.2018.1549100"""
    var_y_est = np.var( y_pred.mean(0))
    var_e = np.var((y_true - y_pred),axis=0)
    r_squared = var_y_est / (var_y_est + var_e)
    return r_squared

def get_param_CI(trace,param):
    return np.median(trace[param]),np.percentile(trace[param],2.5),np.percentile(trace[param],97.5)

# def compute_summary(trace,path):
#     df=pm.summary(trace,hdi_prob=0.95)
#     df.to_csv(path)
    
def read_summary(path):
    df=pd.read_csv(path,index_col=0)
    return df

def generate_path_for_summary(model_type,model_name,dependent_var):
    path='../results_quest/summary/{}_{}_{}.csv'.format(model_type,model_name,dependent_var)
    return path

def model_summary(trace,model_type='cobb_douglas',model_name='model_1',dependent_var='Murder',compute=True):
    path=generate_path_for_summary(model_type,model_name,dependent_var)
    if compute==True:
        compute_summary(trace,path)
    df=read_summary(path)
    return df

def spearmanr_corr(x,y):
    corr=stats.pearsonr(x,y)[0]
    return corr

def pearson(x,y):
    return np.corrcoef(x,y)[0][1]

def cdf(data, nbins=15, log=False, surv=True):
    """
    Estimates the survival distribution from a np.array.

    Parameters
    ----------
    data: np.array
    nbins: number of bins.
    log: True for log-spaced bins (Default: False).
    surv: False for cumulative distribution function (Default: True).
    ----------
    Returns [bins,pdf].
    """
    size = len(data)
    if log:
        bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), nbins)
    else:
        bins = np.linspace(np.min(data), np.max(data),nbins)        
    densities = np.array([len(data[data>=binvalue])/size for binvalue in bins])
    if not surv:
        densities = 1 - densities
    return bins, densities 

def moving_average(a, n=3):
    """
    Calculates the moving average over an array.

    Parameters
    ----------
    a: array   
    n: windows size (default = 3)
    ----------
    Returns the moving average of a.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def pdf(data, nbins=15, log=False, symmetrical=False):
    """
    Estimates the pdf from a np.array.

    Parameters
    ----------
    data: np.array
    nbins: number of bins or np.histogram method name (Default: 15).
    log: True for log-spaced bins (Default: False).
    symmetrical: True for symmetrical bin interval (Default: False).
    ----------
    Returns [bins,pdf].
    """
    if not isinstance(nbins, str):
        if log:
            if symmetrical:
                maxdata = max(np.abs(np.min(data)), np.abs(np.max(data)))
                bins = np.logspace(-maxdata, maxdata, nbins)
            else:
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), nbins)
        else:
            if symmetrical:
                maxdata = max(np.abs(np.min(data)), np.abs(np.max(data)))
                bins = np.linspace(-maxdata, maxdata, nbins)
            else:
                bins = np.linspace(np.min(data), np.max(data), nbins)
    else:
        bins = nbins
    densities, bbins = np.histogram(data, bins=bins, density=True)
    return moving_average(bbins, 2), densities


def bootstrapping(x,y,func,samples,alpha):
    r=func(x,y)
    index=np.arange(0,len(x))
    results=np.zeros(samples)
    for i  in range(0,samples):
        rand_ind=np.random.choice(index,size=len(index))
        xr,yr=x[rand_ind],y[rand_ind]
        results[i]=func(xr,yr)
    rinf=np.percentile(results,alpha/2*100)
    rsup=np.percentile(results,(1-alpha/2)*100)
    return r,rinf,rsup

def bootstrapping_samples(x,y,func,samples):
    r=func(x,y)
    index=np.arange(0,len(x))
    results=np.zeros(samples)
    for i  in range(0,samples):
        rand_ind=np.random.choice(index,size=len(index))
        xr,yr=x[rand_ind],y[rand_ind]
        results[i]=func(xr,yr)
    return results

def linmodel_slope(x,y):
    param_bounds=([-np.inf,0.1],[np.inf,3])
    p_linear, cov_linear = optimize.curve_fit(linear, x, y,bounds=param_bounds)
    return p_linear[1]

def linmodel_ci(df,x_label,y_label,n_samples=10,alpha=0.05):
    a_bootstrap=[]
    b_bootstrap=[]
    for i in range(0,n_samples):
        sample=np.random.choice(df.index,size=len(df.index),replace=True)
        y=df[y_label][sample]       
        x=df[x_label][sample]
        x,y=np.array(x),np.array(y)
        param_bounds=([-np.inf,0.1],[np.inf,3])
        p_linear , pcov_linear = optimize.curve_fit(linear, x, y,bounds=param_bounds)
        a_bootstrap.append(p_linear[0])
        b_bootstrap.append(p_linear[1]) 

    ainf,asup=np.percentile(a_bootstrap,alpha*100/2),np.percentile(a_bootstrap,100-alpha*100/2)
    binf,bsup=np.percentile(b_bootstrap,alpha*100/2),np.percentile(b_bootstrap,100-alpha*100/2)
    return ainf,asup, binf,bsup

