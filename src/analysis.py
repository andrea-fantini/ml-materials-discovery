import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.5)
from sklearn.model_selection import cross_val_score


def plot_corr(data) :
    """
    Plots the correlation matrix for the dataframe.
    
    Args:
        data: this is a Pandas DataFrame with the data

    """
    
    corr = data.corr()
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    palette = sns.diverging_palette(20, 220, n=256)
    
    fig = plt.figure(figsize=(16,16))
    sns.heatmap(corr,mask=mask, cmap=palette,annot=True)
    plt.tight_layout()



def compare_models(models, X, y, scoring='neg_mean_squared_error'):
    results = []
    names = []
    for name, m in models.items():
        m.fit(X, y)
        cv_scores = cross_val_score(m, X, y, cv= 10, scoring=scoring)
        results.append(cv_scores)
        names.append(name)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle(scoring)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names, rotation=0)
    plt.show()


def compare_predictions(y_pred, y_axis, subtitles, title):
    """
    Generate adjacent plots to compare prediction results.
    
    Args:
        y_pred: dataframe containig the predicted y
        y_axis: dataframe containig what to plot on the y axis (Ex. y_true, residuals, etc..)
        labels: list of titles for each subplot
        title: string containing the figure title

    """
    fig, axes = plt.subplots(1,len(subtitles),figsize=(15,5))
    fig.suptitle(title)
    for i in range(3):
        axes[i].scatter(y_pred[subtitles[i]], y_axis[subtitles[i]])
        axes[i].title.set_text(subtitles[i])
        extremes = [y_axis[subtitles[i]].min(),y_axis[subtitles[i]].max()]
        axes[i].plot(extremes, extremes)

    # Set common labels
    _ = fig.text(0.5, 0.04, 'Predicted', ha='center', va='center')
    _ = fig.text(0.06, 0.5, 'True', ha='center', va='center', rotation='vertical')

    
    
def compare_residuals(y_pred, y_axis, subtitles, title):
    """
    Generate adjacent plots to compare the residuals.
    
    Args:
        y_pred: dataframe containig the predicted y
        y_axis: dataframe containig what to plot on the y axis (Ex. y_true, residuals, etc..)
        labels: list of titles for each subplot
        title: string containing the figure title

    """
    fig, axes = plt.subplots(1,len(subtitles),figsize=(15,5))
    fig.suptitle(title)
    for i in range(3):
        axes[i].scatter(y_pred[subtitles[i]], y_axis[subtitles[i]])
        axes[i].title.set_text(subtitles[i])
        extremes = [y_pred[subtitles[i]].min(),y_pred[subtitles[i]].max()]
        axes[i].plot(extremes, [0,0])
    # Set common labels
    _ = fig.text(0.5, 0.04, 'Predicted', ha='center', va='center')
    _ = fig.text(0.06, 0.5, 'True', ha='center', va='center', rotation='vertical')
    


def compare_generated_samples(Y_new, Y_ref, title):
    """
    Generate adjacent plots to compare the the genearted samples with the reference samples.
    
    Args:
        Y_new: dataframe containig the new measurement data
        Y_ref: dataframe containig the new measurement data
        title: string containing the figure title

    """
    f, (ax1, ax2)  = plt.subplots(1, 2, sharey=True, figsize=(15, 5))
    f.suptitle(title)
    ax1.scatter(Y_new['Measurement-1'],Y_new['Measurement-3'])
    ax1.scatter(Y_ref['Measurement-1'],Y_ref['Measurement-3'])
    ax1.set(xlabel='Measurement-1')
    ax1.axhline(y = 400, color = 'r', linestyle = '-')

    ax2.scatter(Y_new['Measurement-2'],Y_new['Measurement-3'])
    ax2.scatter(Y_ref['Measurement-2'],Y_ref['Measurement-3'])
    ax2.set(xlabel='Measurement-2')
    ax2.axhline(y = 400, color = 'r', linestyle = '-')

    _ = f.text(0.06, 0.5, 'Measurement-3', ha='center', va='center', rotation='vertical')
