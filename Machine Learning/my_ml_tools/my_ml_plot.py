import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


def Plot_2d(X, y, ax=None):
    """ Function to plot a 2D scatter graph
    [Parameters]
        X: (2d array) two features
        y: (1d array) target
        ax: matplotlib Axes, optional
    [Returns]
        None
    """    
    if ax is None:
        ax = plt.gca()
    
    colors = ['indianred', 'royalblue', 'yellowgreen', 'darkorange', 'mediumpurple', 'gray']
        
    for i in np.unique(y):
        ax.scatter(X[y==i, 0], X[y==i, 1], c=colors[i], label=f'Class-{i}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()


def Plot_2d_decision(model, X, y, ax=None, fillcolor=True, scaler=None):
    """ Function to plot a 2D Decision Boundary
    [Parameters]
        model: a trained model for plotting decision boundary
        X: (2d array) two features
        y: (1d array) target
        ax: matplotlib Axes, optional
        fillcolor: (bool) By default, a contour plot filled with colors
    [Returns]
        None
    """
    if ax is None:
        ax = plt.gca()
    from matplotlib.colors import ListedColormap
    colors = ['indianred', 'royalblue', 'yellowgreen', 'darkorange', 'mediumpurple', 'gray']
    cmap = ListedColormap(colors[: len(np.unique(y))])
    
    x1 = X[:, 0]
    x2 = X[:, 1]
    xxmin = min(x1.min() * 0.95, x1.min() * 1.05)
    yymin = min(x2.min() * 0.95, x2.min() * 1.05)
    xxmax = max(x1.max() * 0.95, x1.max() * 1.05)
    yymax = max(x2.max() * 0.95, x2.max() * 1.05)
    
    xx, yy = np.meshgrid(
        np.linspace(xxmin, xxmax, 100),
        np.linspace(yymin, yymax, 100)
    )
    zz = np.c_[xx.ravel(), yy.ravel()]
    
    if scaler is not None:
        zz = scaler.transform(zz)
        
    zz = model.predict(zz)
    if fillcolor:
        ax.contourf(xx, yy, zz.reshape(xx.shape), cmap=cmap, alpha=0.4)
    else:
        ax.contour(xx, yy, zz.reshape(xx.shape), colors='k', linewidths=0.5)

def Plot_decision_regions(model, X, y, ax=None):
    """ Function to plot a 2D Decision Boundary (prediction errors marked 'x')
    [Parameters]
        model: a trained model for plotting decision boundary
        X: (2d array)
        y: (1d array) target
        ax: matplotlib Axes, optional
    [Returns]
        None
    """
    from matplotlib.colors import ListedColormap
    colors = ['indianred', 'royalblue', 'yellowgreen', 'darkorange', 'mediumpurple', 'gray']
    cmap = ListedColormap(colors[: len(np.unique(y))])
    
    if ax is None:
        ax = plt.gca()
        
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() * 0.9, X[:, 0].max() * 1.1, 100),
        np.linspace(X[:, 1].min() * 0.9, X[:, 1].max() * 1.1, 100),
    )
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)
    if len(np.unique(zz)) == 1:
        ax.contour(xx, yy, zz)
    else:
        ax.contourf(xx, yy, zz, cmap=cmap, alpha=0.5)
    for idx, i in enumerate(np.unique(y)):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=colors[idx], label=i)
        
    y_hat = model.predict(X)
    ax.scatter(X[y_hat != y, 0], X[y_hat != y, 1], marker='x', color='k')
    ax.legend()
    
def Plot_decision_multi_class(model, X_train, X_test, y_train, y_test, ax=None, poly=None):
    """ Function to plot a 2D Decision Boundary distinguishing Train & Test sets (prediction errors marked 'x')
    [Parameters]
        model: a trained model for plotting decision boundary
        X_train: (2d array)
        X_test: (2d array)
        y_train: (1d array) target
        y_test: (1d array) target
        ax: matplotlib Axes, optional
        poly: a trained PolynomialFeatures()
    [Returns]
        None
    """
    if ax is None:
        ax = plt.gca()
    if poly is not None:
        X_test_poly = poly.transform(X_test)
        y_pred = model.predict(X_test_poly)
        print(f'Test score = {model.score(X_test_poly, y_test):.2f}')
    else:
        y_pred = model.predict(X_test)
        print(f'Test score = {model.score(X_test, y_test):.2f}')
        
    markers = ['s', '^', 'v', 'D', 's', '^', 'v', 'D']
    for i in np.unique(y_train):
        ax.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], marker=markers[i], s=25, label=str(i))
    ax.scatter(X_test[:, 0], X_test[:, 1], marker='o', facecolor='w', edgecolor='k', s=60, label='Test', alpha=1)
    ax.scatter(X_test[y_pred != y_test, 0], X_test[y_pred != y_test, 1], marker='x', c='k', s=40, label='Error')

    xx, yy = np.meshgrid(
        np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
        np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100),
    )
    zz = np.c_[xx.ravel(), yy.ravel()]
    if poly is not None:
        zz = poly.transform(zz)
    zz = model.predict(zz).reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=plt.cm.RdYlGn, alpha=0.5)
    ax.set_xlabel('Feature-1')
    ax.set_ylabel('Feature-2')
    ax.legend(framealpha=0.2)
    
def Plot_1d_lr(model, X, y, kwargs=None, ax=None):
    """ Function to plot a Linear Regression line (1-feature)
    [Parameters]
        model: a trained Linear Regressor
        X: (1d array)
        y: (1d array) target
        kwargs: matplotlib (ax.plot) parameters fmt = '[color][marker][line]'
        ax: matplotlib Axes, optional
    [Returns]
        None
    """
    if ax is None:
        ax = plt.gca()
    if kwargs is None:
        kwargs = dict()
    ax.scatter(X, y, marker='o', c='royalblue', edgecolor='lightgray')
    line = np.linspace(X.min(), X.max(), 100)
    ax.plot(line, model.predict(line.reshape(-1, 1)), kwargs.get('fmt', 'r-'), linewidth=2)
    