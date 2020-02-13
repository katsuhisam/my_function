import matplotlib.pyplot as plt
from matplotlib.pyplot import draw
import numpy as np
import seaborn
import copy
import pandas as pd
import scipy.cluster.hierarchy as sch

def CreateExpandMatrixForPcolor(xgrid, ygrid):
    """
    Create expand matrix so that a grid is located at the center of the expanded grid matrix

    input:
    ------
    xgrid (axis[0] order axis[1] has the same value)
    ygrid (axis[1] order axis[0] has the same value)
    """

    diff_x = xgrid[1,0] - xgrid[0,0]
    if diff_x == 0:
        ValueError('Order is different. Switch xgrid and ygrid')
        exit(1)

    ex_x = (xgrid + xgrid + diff_x)/2
    row0 = xgrid[0,:] - diff_x/2
    ex_x = np.vstack([row0.reshape(1,-1), ex_x])
    col0 = ex_x[:,0]
    ex_x = np.hstack([col0.reshape(-1,1), ex_x])

    diff_y = ygrid[0,1] - ygrid[0,0]
    ex_y = (ygrid + ygrid + diff_y)/2
    col0 = ygrid[:,0] - diff_y/2
    ex_y = np.hstack([col0.reshape(-1,1), ex_y])
    row0 = ex_y[0,:]
    ex_y = np.vstack([row0, ex_y])

    return ex_x, ex_y



def makeROCFromDict(tp_fp_au, save_fig_name=None):
    """
    Make ROCurves in a plot (input is the result of makeROCinfo from evaluation)

    input:
    ------
    tp_fp_au:      Dictionary {Index: TPR, FPR, AUC} information
    """
    fig = plt.figure()
    ax  = fig.add_subplot(111, aspect='equal')
    color_pattern = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
    line_pattern = ['-', ':', '--', '-.']

    c_l_patt = [[c, l] for l in line_pattern for c in color_pattern] # Color and line patterns
    
    if len(c_l_patt) < len(tp_fp_au):
        print("Add the color or line pattern in order to avoid duplicate")
    
    # expand dictionary 
    i = 0
    for key, value in tp_fp_au.items():
        fpr = value['FPR']
        tpr = value['TPR']
        auc = value['AUC']
        i +=1
        ax.plot(fpr, tpr, c=c_l_patt[i][0], linestyle=c_l_patt[i][1], lw=2, label='{}: AUC={:.1%}'.format(key, auc))

    
    plt.plot([0,1], [0,1], color='gray', lw=2, label='Random')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=15)
    plt.xlabel("False positive ratio", fontsize=20)
    plt.ylabel("True positive ratio", fontsize=20)
    plt.axis([0, 1, 0, 1])
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight')
        plt.close()




def makeHeatMap(x, cname, colormap=None, mask=None, annotation=False, vmin=None, vmax=None, save_fig_name=None, label_names=None, use_diagonal=False, title=""):
    seaborn.set(font_scale=1.5)
    
    a4_dims = (11.7, 8.27) # inches
    margin  = 0.15 # bottom margin
    
    fig, ax = plt.subplots(figsize=a4_dims)
    
    if not mask is None:
        mask = np.zeros_like(x)
        if use_diagonal:
            mask[np.triu_indices_from(mask, 1)] = True
        else:
            mask[np.triu_indices_from(mask)] = True
    if colormap is None:
        #colormap = seaborn.cubehelix_palette(light=1, as_cmap=True)
        colormap = 'RdBu'
    ax = seaborn.heatmap(x, ax=ax, mask=mask, vmin=vmin, vmax=vmax, annot=annotation, cmap=colormap, cbar_kws={'label': cname}, fmt='.2g',
                        linecolor='white', linewidths=1, square=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set(title=title)
    if label_names is not None:
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        #ax.tick_params(axis='x', labelsize=8)
        #ax.tick_params(axis='y', labelsize=8)
         
    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight')  
        plt.close() 
    else:
        draw()

def makeBoxPlots(x, xnames, ylab, save_fig_name=None):
    fig = plt.figure()
    bp = plt.boxplot(x)
    plt.xticks(list(range(1,len(xnames)+1)), xnames, fontsize=15, rotation=45)
    plt.yticks(fontsize=20)
    plt.ylabel(ylab, fontsize=20)

    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight')  
        plt.close() 
    else:
        draw()

def MakeBarPlotsSeaborn(table, xname, yname, hue=None, save_fig_name=None, title='', fscale=2.0,
                        xlab_name=None, ylab_name=None, yrange=None, legend_list=None, palette='Set2', logy=False,
                        manual_label=None, dpi=200, use_swarm=False, show_legend=True, rotate_x=False, hue_order=None, fsize=(11.7, 8.27)):
    """
    Set the seaborn context and make a box plot
    """
    seaborn.set(font_scale=fscale, context='paper', style='whitegrid', palette=palette)
    
    margin  = 0.15 # bottom margin
    fig, ax = plt.subplots(figsize=fsize)
    
    ax      = seaborn.barplot(x=xname, y=yname, hue=hue, data=table, ax=ax, ci='sd')
    
    if xlab_name is None:
        xlab_name = xname
    if ylab_name is None:
        ylab_name = yname
    
    if logy:
        ax.set_yscale('log')
    
    seaborn.utils.axlabel(xlabel=xlab_name, ylabel=ylab_name)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.set_title(title)
    handles, _ = ax.get_legend_handles_labels()
    if legend_list is not None:
        ax.legend(handles, legend_list)
    
    if show_legend:
        ax.legend(bbox_to_anchor=(1.15, 1), borderaxespad=0.)
    else:
        ax.legend_.remove()
    
    if manual_label is not None:
        plt.xticks(plt.xticks()[0], manual_label)
    if rotate_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight', dpi=dpi)  
        plt.close() 
    else:
        plt.draw()

def MakeLinePlotsSeaborn(table, xname, yname, hue=None, fscale=3.0, save_fig_name=None, title='', 
                        xlab_name=None, ylab_name=None, yrange=None, legend_list=None, palette='Set2',logy=False,
                        manual_label=None, dpi=200,  show_legend=True, rotate_x=False, hue_order=None, reverse_x=False,
                        fsize=(11.7, 8.27)):
    """
    Make line plots with seaborns
    """
    seaborn.set(font_scale=fscale, context='paper', style='whitegrid', palette=palette)
    
    margin  = 0.15 # bottom margin
    fig, ax = plt.subplots(figsize=fsize)
    ax      = seaborn.lineplot(x=xname, y=yname, lw=3, hue=hue, 
                            hue_order=hue_order , data=table, ax=ax, ci='sd', style=hue)
    
    if reverse_x:
        ax.invert_xaxis()

    if xlab_name is None:
        xlab_name = xname
    if ylab_name is None:
        ylab_name = yname
    
    seaborn.utils.axlabel(xlabel=xlab_name, ylabel=ylab_name)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    
    handles = [copy.copy(ha) for ha in handles]
    [ha.set_linewidth(5) for ha in handles] # change the linewidth of the handle
    
    if legend_list is None:
        ax.legend(handles=handles[1:], labels=labels[1:])
        #ax.legend().set_title('')
    
    if show_legend:
        ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.15, 1), borderaxespad=0.)
    else:
        ax.legend_.remove()
    
    if logy:
        ax.set_yscale('log')
    
    #ax.grid(False)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    if manual_label is not None:
        plt.xticks(plt.xticks()[0], manual_label)
    if rotate_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight', dpi=dpi)  
        plt.close() 
    else:
        plt.draw()


def MakeBoxPlotsSeaborn(table, xname, yname, hue=None, fscale=2.0, save_fig_name=None, title='', 
                        xlab_name=None, ylab_name=None, yrange=None, legend_list=None, palette='Set2',
                        manual_label=None, dpi=200, use_swarm=False, show_legend=True, rotate_x=False, hue_order=None,
                        fsize=(11.7, 8.27)):
    """
    Set the seaborn context and make a box plot
    """
    seaborn.set(font_scale=fscale, context='paper', style='whitegrid', palette=palette)
    
    margin  = 0.15 # bottom margin
    fig, ax = plt.subplots(figsize=fsize)
    if use_swarm:
        ax      = seaborn.swarmplot(x=xname, y=yname, hue=hue, data=table, ax=ax)
    else:
        ax      = seaborn.boxplot(x=xname, y=yname, hue=hue, data=table, ax=ax, hue_order=hue_order)
    if xlab_name is None:
        xlab_name = xname
    if ylab_name is None:
        ylab_name = yname
    
    seaborn.utils.axlabel(xlabel=xlab_name, ylabel=ylab_name)
    if yrange is not None:
        ax.set_ylim(yrange)
    ax.set_title(title)
    handles, _ = ax.get_legend_handles_labels()
    if legend_list is not None:
        ax.legend(handles, legend_list)
    if show_legend:
        ax.legend(bbox_to_anchor=(1.15, 1), borderaxespad=0.)
    else:
        ax.legend_.remove()
    
    #ax.grid(False)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    if manual_label is not None:
        plt.xticks(plt.xticks()[0], manual_label)
    if rotate_x:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight', dpi=dpi)  
        plt.close() 
    else:
        plt.draw()


def makeOnePlotWithColor(x1, y1, c1, xlab, ylab,v1=None,v2=None, show_color_bar=False, cbarlabel='',
                        cm='RdYlGn_r',  title_name="", save_fig_name=None, x_range=None, y_range=None, 
                        non_lenged=False, annotate=False):
    a4_dims = (11.7, 8.27) # inches
    margin  = 0.15 # bottom margin
    
    fig, ax = plt.subplots(figsize=a4_dims)
    
    sc = ax.scatter(x1, y1, c=c1, cmap=cm, vmin=v1, vmax=v2, s=70)
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_xlim(calculate_xlim(x1))
    ax.set_ylim(calculate_xlim(y1))
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_title(title_name, fontsize=16)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    
    plt.rc('axes', titlesize=15)
    
    if annotate:
        if not isinstance(x1, pd.Series):
            print('cannot annotate witout using pandas series')
        else:
            for idx in x1.index:
                ax.annotate(idx, xy=(x1[idx],y1[idx]),  xytext=(0, 20), textcoords='offset points', ha='center', size=20)
    
    if show_color_bar:
        cbar=ax.figure.colorbar(sc, ax=ax)
        cbar.ax.set_ylabel(cbarlabel, fontsize=15)
        cbar.ax.tick_params(labelsize=15)
    if save_fig_name is not None:
        save_fig_name = save_fig_name.replace('*', '-')
        fig.savefig(save_fig_name)  
        plt.close() 
    else:
        draw()


def makePlotCategorical(x1, y1, category1, xlab, ylab, title_name='', save_fig_name=None, x_range=None, y_range=None, non_legend=False, rand_seed=1):
    """
    zColor and marker pattern plot defined by category1 cells
    """
    color_pattern = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
    marker_pattern = ['o', 'v', '>', '<', '*', 'h', '+', 'D', 's', '1','d']
    
    c_m_tmp = [[c, m] for c in color_pattern for m in marker_pattern]
    c_m =[]
    # reshuffle 
    for j in range(len(marker_pattern)):
        for i in range(len(color_pattern)-j):
            c_m.append(c_m_tmp[i*len(marker_pattern)+i+j])
    
    max_pattern = len(c_m)
    
    uni_cate = np.unique(category1)
    n_pattern = len(uni_cate)
    if n_pattern > max_pattern:
        print("# Categories exceed {}".format(max_pattern))
        return
    
    fig = plt.figure()

    # each color plotting
    for idx, cate in enumerate(uni_cate):
        plt.plot(x1[category1 == cate], y1[category1 == cate], c=c_m[idx][0], marker=c_m[idx][1], label=cate, linestyle='', ms=10, mew=1.3)
    
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.title(title_name, fontsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rc('axes', titlesize=15)
    if ~non_legend:
        leg = plt.legend(numpoints=1, loc='center left', fontsize=10, frameon=True, bbox_to_anchor=(1, 0.5))
    
    axes = plt.gca()

    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight')  
        plt.close() 
    else:
        draw()


def calculate_xlim(x1, x2=None, buff=0.1):
    """
    Calculate the xlim information based on two  data sets
    """
    rg_x = np.max(x1) - np.min(x1) 
    
    if x2 is not None:
        rg_x2 = np.max(x2) - np.min(x2)
        rg_x = max([rg_x2, rg_x])    
        axis_range = [min([min(x1), min(x2)])- buff*rg_x, max([max(x1), max(x2)])+buff*rg_x]
    else:
        axis_range = [np.min(x1) - buff*rg_x, np.max(x1) + buff*rg_x]
    
    return axis_range
    

def makeTwoPlots(x1, y1, x2, y2, label1, label2, xlab, ylab, title_name="", save_fig_name=None, x_range=None, y_range=None, non_legend=False, x2_marker=None, ms=10):
    """
    Fix the color schemes so that it should be visibl
    """
    
    a4_dims = (11.7, 11.7) # inches
    margin  = 0.15 # bottom margin
    
    fig, ax = plt.subplots(figsize=a4_dims)
    
    ax.plot(x1, y1 ,c="b", marker="o",label=label1, markersize=ms, markeredgecolor='none', linestyle='', alpha=0.8)
    
    if x2_marker is None:
        #ax.plot(x2, y2, c="m", marker="s", label=label2, ms=10, alpha=0.4, markeredgecolor='none', linestyle='')
        ax.plot(x2, y2, c="r", marker="s", label=label2, ms=ms, markeredgecolor='none', linestyle='')
    else:
        ax.plot(x2, y2, c="k", marker=x2_marker, label=label2, ms=ms, mew=1.5, linestyle='')
    
    
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_title(title_name, fontsize=16)

    if non_legend == False:
        leg = ax.legend(numpoints=1, loc='center left', fontsize=15, frameon=True, bbox_to_anchor=(1, 0.5))
        #leg = plt.legend(numpoints=1, loc='upper right', fontsize=15, frameon=True)
        leg.get_frame().set_edgecolor('k')
    
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    
    rg_x2 = np.max(x2) - np.min(x2)
    rg_x1 = np.max(x1) - np.min(x1) 
    rg_x = max([rg_x2, rg_x1])
    rg_y = max( [max(y1) - min(y1), max(y2) - min(y2)]) 
    
    ax.set_xlim([min([min(x1), min(x2)])-0.1*rg_x, max([max(x1), max(x2)])+0.1*rg_x])
    ax.set_ylim([min([min(y1), min(y2)])-0.1*rg_y, max([max(y1), max(y2)])+0.1*rg_y])
    if x_range is not None:
        ax.set_xlim(x_range)
    
    if save_fig_name is not None:
        fig.savefig(save_fig_name, bbox_inches='tight')  
        plt.close(fig) 
    else:
        draw()


def makeTwoPlots_shade(x1, y1, x2, y2, label1, label2, xlab, ylab, x3=None, 
                    y3_u=None, y3_l=None, title_name="", save_fig_name=None, x_range=None, y_range=None, non_legend=False, x2_marker=None):
    """
    With addition line with shades for making GP prediction curves
    """

    a4_dims = (11.7, 11.7) # inches
    margin  = 0.15 # bottom margin
    
    fig, ax = plt.subplots(figsize=a4_dims)
    
    ax.plot(x1, y1 ,c="b", marker="o",label=label1, ms=6, markeredgecolor='none', linestyle='', alpha=0.8)
    
    if x2_marker is None:
        #ax.plot(x2, y2, c="k", marker="s", label=label2, ms=10, alpha=0.4, markeredgecolor='none', linestyle='')
        ax.plot(x2, y2, c="r", marker="s", label=label2, ms=10, markeredgecolor='none', linestyle='')
    else:
        ax.plot(x2, y2, c="k", marker=x2_marker, label=label2, ms=10, mew=1.5, linestyle='')
    
    
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_title(title_name, fontsize=16)

    if non_legend == False:
        leg = ax.legend(numpoints=1, loc='center left', fontsize=15, frameon=True, bbox_to_anchor=(1, 0.5))
        #leg = plt.legend(numpoints=1, loc='upper right', fontsize=15, frameon=True)
        leg.get_frame().set_edgecolor('k')
    
    if x3 is not None:
        ax.fill_between(x3, y3_l.ravel(), y3_u.ravel(), facecolor='b', alpha=0.3)


    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.rc('axes', titlesize=15)
    
    rg_x2 = np.max(x2) - np.min(x2)
    rg_x1 = np.max(x1) - np.min(x1) 
    rg_x = max([rg_x2, rg_x1])
    rg_y = max( [max(y1) - min(y1), max(y2) - min(y2)]) 
    
    ax.set_xlim([min([min(x1), min(x2)])-0.1*rg_x, max([max(x1), max(x2)])+0.1*rg_x])
    ax.set_ylim([min([min(y1), min(y2)])-0.1*rg_y, max([max(y1), max(y2)])+0.1*rg_y])
    if x_range is not None:
        ax.set_xlim(x_range)
    
    if save_fig_name is not None:
        fig.savefig(save_fig_name, bbox_inches='tight')  
        plt.close(fig)
    else:
        draw()


def makeDendrogram(cluster, title_name="", save_fig_name=None, x_labels=None, y_label=None):
    """
    Cluster: scipy linkage objects to make dendrogram
    """

    
    a4_dims = (11.7, 8.27) # inches
    margin  = 0.15 # bottom margin
    
    fig, ax = plt.subplots(figsize=a4_dims)
    if x_labels is not None:
        sch.dendrogram(cluster, labels=x_labels, ax=ax)
    else:
        sch.dendrogram(cluster, ax=ax)
    ax.set_title(title_name, fontsize=16)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=15)

    ax.yaxis.set_tick_params(labelsize=15)
    ax.xaxis.set_tick_params(labelsize=15)
    if save_fig_name is not None:
        fig.savefig(save_fig_name)  
        plt.close(fig) 
    else:
        draw()



def makeFourPlots(x1, y1, x2, y2, label1, label2, xlab, ylab, x3=None, y3=None, x4=None, y4=None, 
                 title_name="", save_fig_name=None, x_range=None,
                 y_range=None, non_legend=False, label3=None, label4=None, ms=10):

    a4_dims = (11.7, 11.7) # inches
    margin  = 0.15 # bottom margin
    
    fig, ax = plt.subplots(figsize=a4_dims)
    
    ax.plot(x1, y1 ,c="b", marker="o",label=label1, markersize=ms, markeredgecolor='none', linestyle='', alpha=0.2)
    
    
    #ax.plot(x2, y2, c="m", marker="s", label=label2, ms=10, alpha=0.4, markeredgecolor='none', linestyle='')
    ax.plot(x2, y2, c="r", marker="s", label=label2, ms=ms, markeredgecolor='none', linestyle='', alpha=0.2)
    
    
    if (x3 is not None) and (y3 is not None):
        ax.plot(x3, y3, c="b", marker='o', label=label3, ms=ms+1, mew=1.5, linestyle='')
    
    if (x4 is not None) and (y4 is not None):
        ax.plot(x4, y4, c="r", marker="s", label=label4, ms=ms+1, mew=1.5, linestyle='')

    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    ax.set_title(title_name, fontsize=16)

    if non_legend == False:
        leg = ax.legend(numpoints=1, loc='center left', fontsize=15, frameon=True, bbox_to_anchor=(1, 0.5))
        #leg = plt.legend(numpoints=1, loc='upper right', fontsize=15, frameon=True)
        leg.get_frame().set_edgecolor('k')
    
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    
    rg_x2 = np.max(x2) - np.min(x2)
    rg_x1 = np.max(x1) - np.min(x1) 
    rg_x = max([rg_x2, rg_x1])
    rg_y = max( [max(y1) - min(y1), max(y2) - min(y2)]) 
    
    ax.set_xlim([min([min(x1), min(x2)])-0.1*rg_x, max([max(x1), max(x2)])+0.1*rg_x])
    ax.set_ylim([min([min(y1), min(y2)])-0.1*rg_y, max([max(y1), max(y2)])+0.1*rg_y])
    
    if x_range is not None:
        ax.set_xlim(x_range)
    
    if save_fig_name is not None:
        fig.savefig(save_fig_name, bbox_inches='tight')  
        plt.close(fig) 
    else:
        draw()

    

def makeThreePlots(x1, y1, x2, y2, label1, label2, xlab, ylab, x3=None, y3=None, x1_col="darkturquoise", x1_marker="o", title_name="", save_fig_name=None, x_range=None, y_range=None, non_legend=False, xticks=None, y3_based=False):
    plt.figure()
    
    if (x3 is not None) and (y3 is not None):
        plt.plot(x3, y3, color='gray', marker='o', ms=10, mec='gray',  alpha=0.3, linestyle='')
    
    plt.plot(x1, y1 ,c=x1_col,marker=x1_marker, 
    label=label1, ms=10, markeredgecolor='none', linestyle='',alpha=0.7)
    plt.plot(x2, y2, c="m", marker="s", 
    label=label2, ms=10, alpha=0.6, markeredgecolor='none', linestyle='')
    
    
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.title(title_name, fontsize=20)

    if non_legend == False:
        leg = plt.legend(numpoints=1, loc='upper right', fontsize=15, frameon=True)
        leg.get_frame().set_edgecolor('b')
    
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rc('axes', titlesize=15)
    axes = plt.gca()
    if xticks is not None:
        axes.set_xticks(xticks)
    
    if y3_based: # if y3 is the maximized point
        axes.set_ylim([y1.min() - 0.5, y3 + 0.5])
    
    if x_range is not None:
        axes.set_xlim(x_range)
    if y_range is not None:
        axes.set_ylim(y_range)

    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)
    if save_fig_name is not None:
        plt.savefig(save_fig_name, dpi=600)  
        plt.close()
    else:
        plt.show()
    
        
def makeOnePlot(x1, y1, label1, xlab, ylab, x_range=None, y_range=None, title_name="", save_fig_name=None, ms=5):
    plt.plot(x1, y1 ,c="darkturquoise",marker="o", 
    label=label1, ms=ms, markeredgecolor='none', linestyle='')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.rc('axes', titlesize=15)
    plt.xlabel(xlab, fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.title(title_name, fontsize=16)

    axes = plt.gca()
    figure = plt.gcf()
    figure.set_size_inches(10, 8)
    if x_range is not None:
        axes.set_xlim(x_range)
    if y_range is not None:
        axes.set_ylim(y_range)
    if save_fig_name is not None:
        plt.savefig(save_fig_name)  
        plt.close() 
    else:
        plt.show()

def MakeYYPlot(ytr, ytr_pred, yts=None, yts_pred=None, y3d_obs=None, y3d_pred=None, label_name=None, label_name2nd='Test',
               label_name3rd='others', xlab_name ='Observed', ylab_name='Predicted', save_fig_name=None, fig_title=None, 
               col_train=None, col_test=None, col_label='Dist2model', tobe_closed=True, minimum_val=None, write_outliers=False):
    """
    Making yyplot and save the plots in a file
    
    input:
    -------------------
    ytr: y training,
    ytr_pred: y training predction
    yts: ytest, 
    y3d: 3rd party additional data if necessary

    
    """
    a4_dims = (11.7, 11.7) # inches
    margin  = 0.15 # bottom margin
    
    if ytr.ndim == 2:
        ytr = ytr.squeeze()
    if ytr_pred.ndim == 2:
        ytr_pred = ytr_pred.squeeze()
        
    fig, ax = plt.subplots(figsize=a4_dims)
    buff = 0.1*ytr.std()
    if (yts is not None) and (yts_pred is not None):
        min_val = min([ytr.min(), ytr_pred.min(), yts.min(), yts_pred.min()]) - buff
        max_val = max([ytr.max(), ytr_pred.max(), yts.max(), yts_pred.max()]) + buff
    else:
        min_val = min([ytr.min(), ytr_pred.min()])- buff
        max_val = max([ytr.max(), ytr_pred.max()])+ buff
    
    if minimum_val is not None:
        min_val = minimum_val - buff

    ax.set_aspect('equal')

    ax.plot([min_val, 11], [min_val, 11], "k-", linewidth=2.0)
    
    if label_name is None:
        label_name  =''

    if col_train is None:
        ax.plot(ytr, ytr_pred, "b.", label=label_name, ms=15)
    else:
        ax.scatter(ytr, ytr_pred, label=label_name, c=col_train, s=15)

    if yts is not None and col_test is None:
        ax.plot(yts, yts_pred, "rx", label=label_name2nd, mew=2, ms=12)
    elif yts is not None:
        ax.scatter(yts, yts_pred, marker="s", label=label_name2nd, c=col_test, s=15)
        ax.colorbar(label=col_label)

    # outlier label annotation (based on yts index)
    if yts is not None:
        annotate_outliers(ax, yts, yts_pred, write_outliers)
    else:
        annotate_outliers(ax, ytr, ytr_pred, write_outliers)

    if y3d_obs is not None:
        ax.scatter(y3d_obs, y3d_pred, marker='s', color='k', label=label_name3rd, s=60)

    if fig_title is not None: ax.set_title(fig_title, fontsize=20)
    ax.set_xlim(min_val, 11)
    ax.set_ylim(min_val, 11)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_xlabel(xlab_name, fontsize=20)
    ax.set_ylabel(ylab_name, fontsize=20)
    
    if label_name is not None:
       #ax.legend(loc="upper left", fontsize=20, scatterpoints=1)
        ax.legend(loc="upper left", numpoints=1, fontsize=20)
    
    if tobe_closed == False: return plt

    if save_fig_name is not None:
        fig.savefig(save_fig_name, bbox_inches='tight')
        plt.close()
    return

def annotate_outliers(ax, yobs, ypred, noutliers):
    """
    Put annotated outliers on the axis
    """
    if not isinstance(yobs, pd.Series):
        print("need pandas series for creating outlier labels")
        return
    
    diff = pd.Series(np.abs(yobs - ypred.ravel()), index=yobs.index)
    diff.sort_values(ascending=False, inplace=True)
    if noutliers == 0:
        return
    if noutliers == 'all':
        noutliers = len(yobs)

    yobs_outlier = yobs.loc[diff.iloc[:noutliers].index]
    ypred = pd.Series(ypred.ravel(), index=yobs.index)
    ypred_outlier = ypred.loc[diff.iloc[:noutliers].index]

    for idx in range(noutliers):
        x = yobs_outlier.iloc[idx]
        y = ypred_outlier.iloc[idx]
        s = yobs_outlier.index[idx]
        if y > x :
            ax.annotate(s, xy=(x,y),  xytext=(0, 20), textcoords='offset points', ha='center', size=20)
        else:
            ax.annotate(s, xy=(x,y),  xytext=(0, -20), textcoords='offset points', ha='center', size=20)



def make_d2model_absplot(dist_tr, abs_tr, dist_ts=None, abs_ts=None,
                         filename=None, figtitle=None, xlab='Distance to the model (std)', tobe_closed=True):
    """
    make a plot of absolute error of the predicted values against the  distance to model
    dist_tr: distance to the model (training data)
    abs_tr: absolute error between predicted and measured values (training)
    dist_ts: test
    abs_ts: test
    """
    plt.figure()
    if (dist_ts is not None) and (abs_ts is not None):
        minx = min([dist_tr.min(), dist_ts.min()]) - 0.1
        maxx = max([dist_tr.max(), dist_ts.max()]) + 0.1
        miny = min([abs_tr.min(), abs_ts.min()]) - 0.1
        maxy = max([abs_tr.max(), abs_ts.max()]) + 0.1

    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
    plt.plot(dist_tr, abs_tr, "b.", label='Training', ms=16)
    if dist_ts is not None: plt.plot(dist_ts, abs_ts, "rx", label='Test', mew=2, ms=10)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(xlab, fontsize=17)
    plt.ylabel('Absolute error', fontsize=17)
    plt.legend(loc="upper left", fontsize=13)
    if figtitle is not None: plt.title(figtitle, fontsize=17)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    if tobe_closed:
        plt.close()
    return


def makeHistogramThree(y1, y2, y3, n_bins=20, xlab_name='Value', legend1='Data 1', legend2='Data 2', legend3='Data 3', y_title='Histogram', save_fig_name=None, is_scale=False, range_x=None, leg_loc='upper left'):
    """
    Make the two histograms and then save it on the file if specified
    """
    plt.figure()
    if range_x is None:
        range_x = [np.min([np.min(y1),np.min(y2),np.min(y3)]), np.max([np.max(y1),np.max(y2),np.max(y3)])]

    h1 = plt.hist(y1, bins=n_bins, color='b', label=legend1, normed=is_scale, range=range_x)
    h2 = plt.hist(y2, bins=n_bins, color='r', alpha=0.5, label=legend2, normed=is_scale, range=range_x)
    h3 = plt.hist(y3, bins=n_bins, color='g', alpha=0.5, label=legend3, normed=is_scale, range=range_x)
    
    plt.xlabel(xlab_name)
    if is_scale:
        ylab_name = 'Probability'
    else:
        ylab_name = 'Frequency'

    plt.title(y_title)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(xlab_name, fontsize=17)
    plt.ylabel(ylab_name, fontsize=17)
    plt.legend(loc=leg_loc, fontsize=13)
    
    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return h1, h2, h3

def makeHistogramTwo(y1, y2, n_bins=20, xlab_name='Value', legend1='Data 1', legend2='Data 2',
                     y_title='Histogram', save_fig_name=None, is_scale=False, range_x=None, leg_loc='upper left'):
    """
    Make the two histograms and then save it on the file if specified
    """
    plt.figure()
    if range_x is None:
        range_x = [np.min([np.min(y1),np.min(y2)]), np.max([np.max(y1),np.max(y2)])]

    h1 = plt.hist(y1, bins=n_bins, color='b', label=legend1, normed=is_scale, range=range_x)
    h2 = plt.hist(y2, bins=n_bins, color='r', alpha=0.5, label=legend2, normed=is_scale, range=range_x)
    plt.xlabel(xlab_name)
    if is_scale:
        ylab_name = 'Probability'
    else:
        ylab_name = 'Frequency'

    plt.title(y_title)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(xlab_name, fontsize=17)
    plt.ylabel(ylab_name, fontsize=17)
    plt.legend(loc=leg_loc, fontsize=13)
    
    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return h1, h2


def makeHistogramOne(y1, n_bins=20, xlab_name='Value', legend1='Data 1', y_title='', save_fig_name=None, show_legend=True):
    """
    Make one histograms and then save it on the file if specified
    """
    plt.figure()
    h1 = plt.hist(y1, bins=n_bins, color='b', label=legend1)
    plt.xlabel(xlab_name, fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title(y_title)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if show_legend:
        plt.legend(loc='upper right', fontsize=18)
    if save_fig_name is not None:
        plt.savefig(save_fig_name, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return h1

def makeSurfWithScatter(z, x_minmax, y_minmax, c_map='RdYlGn', xlab='x1', ylab='x2', clab='value',
 x1=None, y1=None, x2=None, y2=None, x3=None, y3=None, save_fig_name=None, col2='bs', show_cbar=True, xticks=None, yticks=None):
    """
    Make a surface plot with scatter points
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    surf = ax.imshow(z, extent=[np.min(x_minmax), np.max(x_minmax), np.min(y_minmax), np.max(y_minmax)], origin='lower', cmap=c_map)
    if show_cbar:
        cbar= fig.colorbar(surf)
        cbar.ax.tick_params(labelsize=25)
        cbar.set_label(clab, fontsize=25)
    # ax.grid(b=False)
    ax.tick_params(labelsize=25)
    
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    if x1 is not None:
        ax.plot(x1, y1, 'k.', markersize=16)
    
    if x2 is not None:
        ax.plot(x2, y2, col2, alpha=0.4, markersize=10)
    if x3 is not None:
        ax.plot(x3, y3, 'r*', markersize=15)

    ax.set_xlabel(xlab, fontsize=25)
    ax.set_ylabel(ylab, fontsize=25)

    if save_fig_name is not None:
        fig.savefig(save_fig_name)

if __name__ == '__main__':
    import pandas as pd
    test_file = '/Users/miyao/work/fromBit/work/de/analysis/application/script/analysis_for_factorX/after_paper_submission_nn_detection/t1.txt'
    save_file = '/Users/miyao/work/fromBit/work/de/analysis/application/script/analysis_for_factorX/after_paper_submission_nn_detection/s1.pdf'
    x = pd.read_csv(test_file, sep=' ')
    x = x.as_matrix()
    print(x)
    #makePlotCategorical(x.iloc[:,0], x.iloc[:,1],x.iloc[:,2],'test1', 'test2') 
    makePlotCategorical(x[:,0], x[:,1],x[:,2],'test1', 'test2', save_fig_name=save_file) 
