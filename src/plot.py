from turtle import circle
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

def set_plot_param(ax,xlabel,ylabel,title,size=20):
    ax.set_xlabel(xlabel, fontsize = size)
    ax.set_ylabel(ylabel, fontsize = size)  
    ax.tick_params(axis="x", labelsize=size)
    ax.tick_params(axis="y", labelsize=size)
    ax.set_title(title, fontsize=size)

def plot_circle_with_labels (coords, label, title):
    x_cor, y_cor = coords[:,0],coords[:,1]    
    _,r_in = torch.max(label, 1)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x_cor[r_in==1],y_cor[r_in==1],'.' ,color='blue')
    ax.plot(x_cor[r_in==0],y_cor[r_in==0],'.' ,color='red')
    circle = plt.Circle((0, 0), np.sqrt(2/np.pi), 
                        color='black', fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    set_plot_param(ax,'X axis','Y axis',title)
    fig.savefig(title+'.png')




