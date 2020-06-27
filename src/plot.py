"""
Library containing plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(
    cm,
    y_true,
    y_pred,
    classes,
    figsize=(50, 50),
    path=None,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
):
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    ax.set_ylim(len(cm) - 0.5, -0.5)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    if path is not None:
        plt.savefig(path)
    return ax
