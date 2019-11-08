import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef
from matplotlib import pyplot as plt
from figure.plots import MakeBarPlotsSeaborn

def ScoreMatrix(trueY, predY):
    scores = np.full(6,np.nan)
    scores[0] = accuracy_score   (trueY, predY)
    scores[1] = precision_score  (trueY, predY)
    scores[2] = recall_score     (trueY, predY)
    scores[3] = f1_score         (trueY, predY)
    scores[4] = roc_auc_score    (trueY, predY)
    scores[5] = matthews_corrcoef(trueY, predY)
    
    return list(map(lambda x:round(x,2), scores))

def ScoreList(trueY, predY):
    scores = list()
    scores.append(accuracy_score(trueY, predY))
    scores.append(precision_score(trueY, predY))
    scores.append(recall_score(trueY, predY))
    scores.append(f1_score(trueY, predY))
    scores.append(roc_auc_score(trueY, predY))
    scores.append(matthews_corrcoef(trueY, predY))
    
    return list(map(lambda x:round(x,2), scores))

def AddIndex(df):
    
    if type(df) != type(pd.DataFrame()):
        table = pd.DataFrame(df)
    else:
        table = df
    
    table.index = ["Accuracy", "Precision", "Recall", "f1", "AUC_ROC", "Matthews_coeff"]
    
    return table 

def ScoreMatrixBasic(trueY,predY):
    scores = np.full(4,np.nan)
    scores[0] = accuracy_score   (trueY, predY)
    scores[1] = precision_score  (trueY, predY)
    scores[2] = recall_score     (trueY, predY)
    scores[3] = f1_score         (trueY, predY)
    
    return scores

class ScoreTable():
    
    def __init__(self):
        self.scores = dict()
        self.targets = list()
        self.score_name = ["Accuracy", "Precision", "Recall", "f1", "AUC_ROC", "Matthews_coeff"]
        self.unused_score = ["Accuracy","Precision","f1"]
        self.add_columns = False
        
    def calc_scores(self, target, df_res, count=True):
        
        self.targets.append(target)
        score = ScoreList(df_res["trueY"], df_res["predY"])
        
        if count:
            n_predicted = int(df_res.shape[0])
            n_ac = int(df_res["trueY"].value_counts()[1])
            score += [n_predicted, n_ac]
            
            if not self.add_columns:
                self.score_name += ["#predicted","#AC"]
                self.add_columns = True
        
        self.scores[target] = score
        
        return self
    
    def MakeTable(self, index="score"):
        
        if index == "score":
            df_score = pd.DataFrame.from_dict(self.scores, orient="columns")
            df_score = df_score.loc[:,self.targets]
            df_score.index = self.score_name
            
        elif index == "target":
            df_score = pd.DataFrame.from_dict(self.scores, orient="index", columns=self.score_name)
            use_col = [idx for idx in self.score_name if idx not in self.unused_score]
            
            df_score = df_score.loc[self.targets, use_col]
            df_score.index = ["Target - %d" %(i+1) for i in range(df_score.shape[0])]
            
        else:
            raise ValueError("only recoginize score or target for index.\n")
        
        return df_score
    
    def SetUnuseScore(self, score_names):
        self.unused_score = score_names
        return self
            
MakeScoreTable = ScoreTable()

def RenderDF2Fig(data, col_width=3.0, row_height=0.625, font_size=20, header_color='#40466e',
                 row_colors=['#f1f1f2', 'w'], edge_color='w', bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
    
    plt.rcParams["font.family"] = "Arial"
    
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    
    cmap = np.full([data.shape[0],data.shape[1]], "a", dtype="U10")
    c1 = "#d2deef"
    c2 = "#eaeff7"
    
    for i in range(cmap.shape[0]):
        if i%2==0:
            cmap[i,:]=c1
        else:
            cmap[i,:]=c2
    
    mpl_table = ax.table(cellText=data.values, cellLoc="center", cellColours=cmap,
                         rowLabels=data.index, rowColours=cmap[:,0],
                         colColours=["#5b9bd5"]*len(data.columns), colLabels=data.columns,
                         bbox=bbox,edges="BRTL",**kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    return ax