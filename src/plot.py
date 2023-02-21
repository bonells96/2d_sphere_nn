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

def plot_circle_with_labels (coords, label, title, color_1='green', color_0='blue'):
    x_cor, y_cor = coords[:,0],coords[:,1]    
    _,r_in = torch.max(label, 1)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.rcParams["legend.loc"] ='upper right'
    ax.plot(x_cor[r_in==1],y_cor[r_in==1],'.' ,color=color_1, label= 'class 1')
    ax.plot(x_cor[r_in==0],y_cor[r_in==0],'.' ,color=color_0, label= 'class 0')
    circle = plt.Circle((0, 0), 1, 
                        color='black', fill=False, linewidth=2)
    ax.add_artist(circle)
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.legend()
    set_plot_param(ax,'X axis','Y axis',title)
    fig.savefig(title+'.png')



