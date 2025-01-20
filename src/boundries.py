import numpy as np
from scipy.special import gamma
import scipy.stats as st

from src.constants import d
from src.table import table
import src.error as error


def limits_R(X):
    error.error_Type1(X)
    error.error_Shape1(X, 2, 2)

    R = X.max(axis=1) - X.min(axis=1)  # Array der Spannweite
    R_bar = R.mean()  # Mittelwert der Spannweite
    A = np.shape(X)  # Dimensionen der Daten
    m = A[1]  # Anzahl der Spalten
    d2, d3 = d(m, 100000)

    L3 = (1 - 3 * d3 / d2) * R_bar  # Tiefere Kontrollgrenze 3
    L2 = (1 - 2 * d3 / d2) * R_bar  # Tiefere Kontrollgrenze 2
    L1 = (1 - d3 / d2) * R_bar  # Tiefere Kontrollgrenze 1
    C = R_bar  # Mittelwert
    U1 = (1 + d3 / d2) * R_bar  # Obere Kontrollgrenze 1
    U2 = (1 + 2 * d3 / d2) * R_bar  # Obere Kontrollgrenze 2
    U3 = (1 + 3 * d3 / d2) * R_bar  # Obere Kontrollgrenze 3

    table(L3, L2, L1, C, U1, U2, U3)


def limits_xr(X):
    error.error_Type1(X)
    error.error_Shape1(X, 2, 2)

    X_bar = X.mean(axis=1)  # Mittlwert jeder Zeile
    X_bar_bar = X_bar.mean()  # Mittelwert der Mittelwerte
    R = X.max(axis=1) - X.min(axis=1)  # Spannweite
    R_bar = R.mean()  # Mittelwert der Spannweite

    A = np.shape(X)  # Dimensionen der Daten
    m = A[1]  # Anzahl der Spalten
    d2, d3 = d(m, 100000)

    L3 = X_bar_bar - 3 / (d2 * np.sqrt(m)) * R_bar  # Tiefere Kontrollgrenze 3
    L2 = X_bar_bar - 2 / (d2 * np.sqrt(m)) * R_bar  # Tiefere Kontrollgrenze 2
    L1 = X_bar_bar - 1 / (d2 * np.sqrt(m)) * R_bar  # Tiefere Kontrollgrenze 1
    C = X_bar_bar  # Mittelwert
    U1 = X_bar_bar + 1 / (d2 * np.sqrt(m)) * R_bar  # Obere Kontrollgrenze 1
    U2 = X_bar_bar + 2 / (d2 * np.sqrt(m)) * R_bar  # Obere Kontrollgrenze 2
    U3 = X_bar_bar + 3 / (d2 * np.sqrt(m)) * R_bar  # Obere Kontrollgrenze 3

    table(L3, L2, L1, C, U1, U2, U3)


def limits_xs(X):
    # Mittlwert jeder Zeile
    X_bar = X.mean(axis=1)
    # Mittelwert der Mittelwerte
    X_bar_bar = X_bar.mean()
    # Standartabweichung der einzelnen Spalten
    s = X.std(axis=1, ddof=1)
    # Mittelwert der Standartabweichung
    s_bar = s.mean()
    # Dimensionen der Daten
    A = np.shape(X)
    # Anzahl der Spalten
    m = A[1]
    # Anzahl der Zeilen
    n = A[0]
    d2, d3 = d(m, 100000)
    C4 = gamma(m / 2) / gamma((m - 1) / 2) * np.sqrt(2 / (m - 1))
    # Tiefere Kontrollgrenze
    L3 = X_bar_bar - 3 / (C4 * np.sqrt(m)) * s_bar
    L2 = X_bar_bar - 2 / (C4 * np.sqrt(m)) * s_bar
    L1 = X_bar_bar - 1 / (C4 * np.sqrt(m)) * s_bar
    # Mittelwert
    C = X_bar_bar
    # Obere Kontrollgrenze
    U1 = X_bar_bar + 1 / (C4 * np.sqrt(m)) * s_bar
    U2 = X_bar_bar + 2 / (C4 * np.sqrt(m)) * s_bar
    U3 = X_bar_bar + 3 / (C4 * np.sqrt(m)) * s_bar

    table(L3, L2, L1, C, U1, U2, U3)


def limits_s(X):
    error.error_Type1(X)
    error.error_Shape1(X, 2, 2)

    s = X.std(axis=1, ddof=1)  # Standartabweichung der einzelnen Spalten
    s_bar = s.mean()  # Mittelwert der Standartabweichung

    A = np.shape(X)  # Dimensionen der Daten
    m = A[1]  # Anzahl der Spalten

    C4 = gamma(m / 2) / gamma((m - 1) / 2) * np.sqrt(2 / (m - 1))  # Konstante

    L3 = s_bar - 3 * np.sqrt(1 - C4**2) / C4 * s_bar  # Tiefere Kontrollgrenze 3
    L2 = s_bar - 2 * np.sqrt(1 - C4**2) / C4 * s_bar  # Tiefere Kontrollgrenze 2
    L1 = s_bar - 1 * np.sqrt(1 - C4**2) / C4 * s_bar  # Tiefere Kontrollgrenze 1
    C = s_bar  # Mittelwert
    U1 = s_bar + 1 * np.sqrt(1 - C4**2) / C4 * s_bar  # Obere Kontrollgrenze 1
    U2 = s_bar + 2 * np.sqrt(1 - C4**2) / C4 * s_bar  # Obere Kontrollgrenze 2
    U3 = s_bar + 3 * np.sqrt(1 - C4**2) / C4 * s_bar  # Obere Kontrollgrenze 3

    table(L3, L2, L1, C, U1, U2, U3)


# Berechnet die Wahrscheinlichkeit das der Wert unter LSL liegt.
def to_LSL(mu=None, sigma=None, LSL=None, to="to"):
    if to == "to":
        p = st.norm(loc=mu, scale=sigma).cdf(LSL)

    if to == "from":
        p = 1 - st.norm(loc=mu, scale=sigma).cdf(LSL)
    p = round(p, 12)
    return p


# Berechnet die Wahrscheinlichkeit das der Wert über LSL liegt.
def from_USL(mu=None, sigma=None, USL=None, to="from"):
    if to == "to":
        p = st.norm(loc=mu, scale=sigma).cdf(USL)

    if to == "from":
        p = 1 - st.norm(loc=mu, scale=sigma).cdf(USL)
    p = round(p, 12)
    return p
