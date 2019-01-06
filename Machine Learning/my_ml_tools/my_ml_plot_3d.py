import numpy as np
import matplotlib.pyplot as plt


def Plot_3d_reg(x1, x2, y, model=None):
    """ Function to plot a 3D graph
    [Parameters]
        x1: (1d array) first feature
        x2: (1d array) second feature
        y: (1d array) target
        model: (Regression model, optional) If not provided, scatter plot only
    [Returns]
        None
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    cmap = plt.get_cmap('jet')
    ax = plt.gca(projection='3d')    
    ax.scatter(x1, x2, y, c=y, cmap=cmap)
    ax.set_xlabel('Feature-1')
    ax.set_ylabel('Feature-2')
    ax.set_zlabel('y')
    ax.view_init(elev=30, azim=225)
    
    if model is None:
        return
    else:
        xxmin = min(x1.min() * 0.95, x1.min() * 1.05)
        yymin = min(x2.min() * 0.95, x2.min() * 1.05)
        xxmax = max(x1.max() * 0.95, x1.max() * 1.05)
        yymax = max(x2.max() * 0.95, x2.max() * 1.05)

        xx, yy = np.meshgrid(
            np.linspace(xxmin, xxmax, 100),
            np.linspace(yymin, yymax, 100)
        )
        zz = np.c_[xx.ravel(), yy.ravel()]
        zz = model.predict(zz)
        surf = ax.plot_surface(xx, yy, zz.reshape(xx.shape), cmap=cmap, alpha=0.5)
        cbar = plt.colorbar(surf, shrink=0.7)
        cbar.ax.set_ylabel('Prediction')