import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def oc(k, m, plot=0):
    if plot == 1:
        plt.plot(k, oc(k, m))
        plt.grid()
        plt.xlabel("k")
        plt.ylabel("P(Fehler 2. Art)")
        plt.title("Operationscharakteristik für m = 5")

    return st.norm.cdf(3 - k * np.sqrt(m)) - st.norm.cdf(-3 - k * np.sqrt(m))


def aoq(k, m):
    return 1 - oc(k, m)
