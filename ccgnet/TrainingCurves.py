import numpy as np
import matplotlib.pyplot as plt
import glob

def GetTrainingHistory(history_file):
    history = eval(open(history_file).read())
    return history

def GetCVHistory(path):
    files = glob.glob(path+'/*/*history*')
    History = {}
    for f in files:
        his = GetTrainingHistory(f)
        for key in his:
            if key not in History:
                History[key] = []
            else:
                History[key].append(his[key])
    for k in History:
        History[k] = np.array(History[k])
    return History

def PlotSingleCVCurve(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    plt.figure(figsize=(16,10), dpi= 100)
    #plt.ylabel("# Daily Orders", fontsize=16)  
    x = list(range(0,100))
    plt.plot(x, mean, color="white", lw=2) 
    plt.fill_between(x, mean-std, mean+std, color="#3F5D7D", alpha=0.5)
    
    Min = np.min(mean-std)
    Max = np.max(mean+std)
    plt.ylim(Min, Max)
    plt.show()


import random
colors_library = ['#0395AE', '#F2DD66', '#7E2E0B', '#14B09B', '#EBE5D9', '#CC8A56', '#DB5A59', '#DEBC7A', '#8A54A2', '#D92139', '#33C7F7', '#FFB21A']
def PlotMultiCVCurve(his_dic, color_lib=colors_library, legend_loc=None, shuffle_color_lib=True, dpi=100, figsize=(16,10), savename=None):
    '''
    hisdic: A dictionary, the key is the name of the model, and the value is an array of the history of each fold
    colors_lib: colors lib
    legend_loc: Location of legend：'upper left', 'upper right', 'lower left', 'lower right', 'upper center', 'lower center', 'center left', 'center right'
    '''
    his_draw = {}
    MinMax = []
    for model in his_dic:
        his_draw[model] = {}
        his_draw[model]['mean'] = his_dic[model].mean(axis=0) * 100
        his_draw[model]['std'] = his_dic[model].std(axis=0) * 100
        MinMax.append(min(his_draw[model]['mean']-his_draw[model]['std']))
        MinMax.append(max(his_draw[model]['mean']+his_draw[model]['std']))
    limMin = min(MinMax)
    limMax = max(MinMax)
    
    ##### draw #####
    plt.figure(figsize=figsize, dpi=dpi)
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    if shuffle_color_lib:
        random.shuffle(color_lib)
    x = list(range(1,101))
    n = 0
    for k in his_draw:
        plt.plot(x, his_draw[k]['mean'], color=color_lib[n], lw=2, label=k)
        low = his_draw[k]['mean'] - his_draw[k]['std']
        high = his_draw[k]['mean'] + his_draw[k]['std']
        plt.fill_between(x, low, high, color=color_lib[n], alpha=0.3, linewidth=1)
        n += 1
    plt.ylim(limMin, limMax)
    if legend_loc:
        plt.legend(loc=legend_loc, fontsize='xx-large')
    else:
        plt.legend()
    if savename:
        plt.savefig(savename, dpi=dpi)
    plt.show()