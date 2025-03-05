import matplotlib as mpl
import matplotlib.pyplot as plt

def change_mpl_default_colours():
    """
    Change the default colours of matplotlib plots
    """
    mpl.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.RdYlBu([0., 0.8, 1]))
