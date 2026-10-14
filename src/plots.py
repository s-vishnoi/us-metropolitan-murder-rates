# figures
import matplotlib.mlab as ml
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.ticker as plticker
import palettable
import seaborn as sns
import numpy as np
import pandas as pd 
import arviz as az
# import pymc3 as pm

def stdfigsize(scale=1, nx=1, ny=1, ratio=1.3):
    """
    Returns a tuple to be used as figure size.
    -------
    returns (7*ratio*scale*nx, 7.*scale*ny)
    By default: ratio=1.3
    If ratio<0 them ratio = golden ratio
    """
    if ratio < 0:
        ratio = 1.61803398875
    return((7*ratio*scale*nx, 7*scale*ny))

def stdrcparams(usetex=True):
    """
    Set several mpl.rcParams and sns.set_style for my taste.
    ----
    usetex = True
    ----
    """
    sns.set_style("white")
    sns.set_style({"xtick.direction": "out",
                 "ytick.direction": "out"})
    rcparams = {
    'text.usetex': usetex,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times'],
    'axes.labelsize': 35,
    'axes.titlesize': 35,
    'legend.fontsize': 20,
    'ytick.right': 'off',
    'xtick.top': 'off',
    'ytick.left': 'on',
    'xtick.bottom': 'on',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'axes.linewidth': 2.5,
    'xtick.major.width': 1.8,
    'xtick.minor.width': 1.8,
    'xtick.major.size': 14,
    'xtick.minor.size': 7,
    'xtick.major.pad': 10,
    'xtick.minor.pad': 10,
    'ytick.major.width': 1.8,
    'ytick.minor.width': 1.8,
    'ytick.major.size': 14,
    'ytick.minor.size': 7,
    'ytick.major.pad': 10,
    'ytick.minor.pad': 10,
    'axes.labelpad': 15,
    'axes.titlepad': 15,
    'axes.spines.right': False,
    'axes.spines.top': False}

    mpl.rcParams.update(rcparams) 
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['pdf.fonttype'] = 42 

stdrcparams(usetex=True)
figsize=stdfigsize(ratio=-1)
xs,ys=figsize

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Parameters
    ----------
    color: the color variable (e.g. 'g', '#F034A3', (.3,.55,.1)).
    amount: the lighten amount factor (Default: 0.5)
    ----------
    returns modified color
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return mpl.colors.to_hex(colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2]))
    
def get_axis_limits(ax, scale=1.1):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

def plot_model_selection(df_cobb_douglas_WAIC,metric,color,ax):
    df=df_cobb_douglas_WAIC[metric].sort_values(by='waic',ascending=True)
    waic=df['waic'].values
    se=df['se'].values

    x=np.arange(0,len(df))
    ax.errorbar(waic,
                x, 
                xerr=se, 
                yerr=None, 
                fmt='',
                ecolor=color, 
                elinewidth=5, 
                linewidth=0,
                capsize=10, 
                barsabove=False, lolims=False, uplims=False, xlolims=False, 
                xuplims=False, errorevery=1, capthick=3, data=None)
    ax.plot(waic,x,'o',color=color,markersize=12)
    ax.axvline(waic[-1],color='gray',ls='solid',zorder=-1)
    ax.set_yticks(np.arange(0,len(df)))
    ax.set_yticklabels(labels=df.index,rotation=0,ha='right')
    ax.set_title('Model selection')
    ax.set_xlabel('$\log$ score')
    return ax

def sort_params(df_params,params_order):
    df_params=df_params.set_index('Parameter')
    df_params=df_params.loc[params_order]
    df_params=df_params.reset_index()
    return df_params

def plot_parameters(df_results,ax=None,color='#377eb8',offset=0):
    df=df_results.copy()
    n_parameters=len(df['Parameter'])
    if ax==None:
        fig, ax = plt.subplots(figsize=(xs,ys))

    v=df['Mean'].values
    vinf=df['2.5%'].values
    vsup=df['97.5%'].values 

    x=np.arange(0,n_parameters)
    ax.errorbar(v,
                x+offset, 
                xerr=[v-vinf,vsup-v], 
                yerr=None, 
                fmt='',
                ecolor=color, 
                elinewidth=5, 
                linewidth=0,
                capsize=10, 
                barsabove=False, lolims=False, uplims=False, xlolims=False, 
                xuplims=False, errorevery=1, capthick=3, data=None)
    ax.plot(v,x+offset,'o',color=color,markersize=12)
    ax.set_yticks(np.arange(0,n_parameters))
    ax.set_yticklabels(labels=df['Parameter'].values,rotation=0,ha='right')
    return ax

def plot_obs_pred(data_info,data,dependent_var,f,ax,cmap='Blues',cbar=True):
    yhat=np.mean(data_info['yhat'],axis=0)
    y=data[dependent_var]
    hb=ax.hexbin(yhat,y, gridsize=50, cmap=cmap,bins='log')
    if cbar:
        cbar = f.colorbar(hb, ax=ax)
        cbar.set_label(r'$\log (\#$ of observations$)$',rotation=-90,labelpad=40)
    ax.plot([-0.3,3.5],[-0.3,3.5],"--",color='k',linewidth=5)  
    # ax.set( xlim=(min(yhat),max(yhat)),ylim=(min(y),max(yhat)))  
    # ax.set_ylabel(r'Observed crime, $\log~H$')
    # ax.set_xlabel(r'Predicted crime, $\log~H$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    return ax

def temporal_obs_pred(data_info,data,dependent_var,city,color,ax):
    df=data[['MSA','State','Year','Murder']]
    selection=df['MSA']==city
    df_city=df[selection]


    mhat=np.mean(data_info['yhat'],axis=0)
    ax.plot(df_city['Year'],mhat[selection],label='Avg. prediction',color=color,lw=5)
    mhat=az.hdi(data_info['yhat'])
    yhdi=az.hdi(data_info['yhat'],hdi_prob=.95)
    ax.fill_between(df_city['Year'],yhdi[:,0][selection],yhdi[:,1][selection],color=color,alpha=0.3)


    ax.plot(df_city['Year'],
            df_city[dependent_var],'o',
            label='Data',
            color=color,
            markeredgecolor='k',
            markeredgewidth=1,
            markersize=10,alpha=0.8)

    return ax
