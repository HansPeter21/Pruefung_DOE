import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def QQ_plot(X):
    X = X.reshape(-1)
    st.probplot(X, dist="norm", plot=plt)


def shapiro(X):
    res = st.shapiro(X)
    print(f"Teststatistik = {res[0]}")
    print(f"p-Wert = {res[1]}")
