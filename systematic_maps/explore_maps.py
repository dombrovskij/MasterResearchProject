import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
plt.switch_backend("Agg")

data_dir = utils.dat_dir()
graph_dir = utils.fig_dir()

with open(data_dir+'/pixel_data.pickle', 'rb') as handle:
    pixel_data = pickle.load(handle)

with open(data_dir+'/ngal_norm.pickle', 'rb') as handle:
    ngal_norm = pickle.load(handle)

def plot_hist(pixel_data):
    
    cols = pixel_data.columns
    ncols = len(cols)
    
    nr, nc = len(cols)/3, 3
    fig , axs = plt.subplots(nrows= nr, ncols= nc , sharex= False, sharey= False, figsize= (15,10))
    cnt = 0
    for i in range(nr):
        for j in range(nc):

       	    hist = axs[i,j].hist(pixel_data[cols[cnt]], bins= 20, histtype= "step")
            axs[i,j].set_xlabel(cols[cnt])
            axs[i,j].set_ylabel("counts")
            cnt+=1
    plt.tight_layout()
    plt.savefig(graph_dir+"sys_hist.png")	    
    
    return None

def plot_corr(pixel_data):
    
    df = pd.DataFrame(pixel_data, columns = pixel_data.columns) #convert to a panda dataframe
    corr = df.corr() #compute the correlation matrix
    
    plt.figure(figsize=(10, 10))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.savefig(graph_dir+"sys_corr.png")	    

    return None

plot_hist(pixel_data)  
plot_corr(pixel_data)  
