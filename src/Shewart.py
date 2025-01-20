import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.special import gamma

import src.error as error
from src.constants import c4, d


# Shewart_R
def shewart_R(X, X_new=None):
    error.error_Type1(X)
    error.error_Shape1(X, 2, 2)
    # Array der Spannweite
    R = X.max(axis=1) - X.min(axis=1)
    # Mittelwert der Spannweite
    R_bar = R.mean()
    # Dimensionen der Daten
    A = np.shape(X)
    # Anzahl der Spalten
    m = A[1]
    # Anzahl der Zeilen
    n = A[0]
    d2, d3 = d(m, sim_size=100000)
    # Tiefere Kontrollgrenze
    L = (1 - 3 * d3 / d2) * R_bar
    # Mittelwert
    C = R_bar
    # Obere Kontrollgrenze
    U = (1 + 3 * d3 / d2) * R_bar

    # Erstelle eine neue Figur
    fig = plt.figure(figsize=(10, 6))

    # Eingriffsgrenzen
    plt.hlines([U, C, L], 0, n + n + 1, linestyles="dashed", colors="gray")
    # Macht Linien bei den Kontrollgrenzen und Mittelwert
    plt.hlines(
        [
            (1 + d3 / d2) * R_bar,
            (1 + 2 * d3 / d2) * R_bar,
            (1 - d3 / d2) * R_bar,
            (1 - 2 * d3 / d2) * R_bar,
        ],
        0,
        n + n + 1,
        linestyles="dotted",
        colors="lightgray",
    )
    # Macht Linien bei 1 und 2 Sigma

    x = np.linspace(1, n, n)
    plt.plot(x, R, "o-")
    # Wenn für X_new ein Array angegeben wird er geplottet
    if X_new is not None:
        # Neue Daten
        # Dimension der neuen Daten
        A_new = np.shape(X_new)
        # Anzahl Zeilen der neuen Daten
        n_new = A_new[0]
        # Array der Spannweiten
        R_new = X_new.max(axis=1) - X_new.min(axis=1)
        # Neuer Intervall der Messwerte
        x_new = np.linspace(1, n_new, n_new)

        plt.vlines(n + 0.5, L, U, colors="red")
        # Plot der neuen Daten
        plt.plot(x_new + n, R_new, "*-")
    else:
        plt.xlim((0, n + 1))

    plt.title("$R$-Karte")
    plt.xlabel("Sample")
    plt.ylabel("$R$")
    plt.show()

    # return fig


def shewart_xr(X, X_new=None):
    error.error_Type1(X)
    error.error_Shape1(X, 2, 2)
    # Mittlwert jeder Zeile
    X_bar = X.mean(axis=1)
    # Mittelwert der Mittelwerte
    X_bar_bar = X_bar.mean()
    # Spannweite
    R = X.max(axis=1) - X.min(axis=1)
    # Mittelwert der Spannweite
    R_bar = R.mean()
    # Dimensionen der Daten
    A = np.shape(X)
    # Anzahl der Spalten
    m = A[1]
    # Anzahl der Zeilen
    n = A[0]
    d2, d3 = d(m, 100000)
    # Tiefere Kontrollgrenze
    L = X_bar_bar - 3 / (d2 * np.sqrt(m)) * R_bar
    # Mittelwert
    C = X_bar_bar
    # Obere Kontrollgrenze
    U = X_bar_bar + 3 / (d2 * np.sqrt(m)) * R_bar

    # Erstelle eine neue Figur
    fig = plt.figure(figsize=(10, 6))

    # Eingriffsgrenzen
    plt.hlines([U, C, L], 0, n + n + 1, linestyles="dashed", colors="gray")
    plt.hlines(
        [
            (X_bar_bar + 1 / (d2 * np.sqrt(m)) * R_bar),
            (X_bar_bar + 2 / (d2 * np.sqrt(m)) * R_bar),
            (X_bar_bar - 1 / (d2 * np.sqrt(m)) * R_bar),
            (X_bar_bar - 2 / (d2 * np.sqrt(m)) * R_bar),
        ],
        0,
        n + n + 1,
        linestyles="dotted",
        colors="lightgray",
    )

    x = np.linspace(1, n, n)
    plt.plot(x, X_bar, "o-")

    if X_new is not None:

        plt.vlines(n + 0.5, L, U, colors="red")

        A_new = np.shape(X_new)
        n_new = A_new[0]
        X_bar_new = X_new.mean(axis=1)
        x_new = np.linspace(1, n_new, n_new)
        plt.plot(x_new + n, X_bar_new, "*-")
    else:
        plt.xlim((0, n + 1))

    plt.title("$xr$-Karte")
    plt.xlabel("Sample")
    plt.ylabel("$x_{i_{mean}}$")
    plt.show()
    # return fig


def shewart_s(X, X_new=None):
    error.error_Type1(X)
    error.error_Shape1(X, 2, 2)
    # Standartabweichung der einzelnen Spalten
    s = X.std(axis=1, ddof=1)
    # Mittelwert der Standartabweichung
    s_bar = s.mean()
    # Spannweite
    R = X.max(axis=1) - X.min(axis=1)
    # Dimensionen der Daten
    A = np.shape(X)
    # Anzahl der Spalten
    m = A[1]
    # Anzahl der Zeilen
    n = A[0]
    d2, d3 = d(m, 100000)

    C4 = gamma(m / 2) / gamma((m - 1) / 2) * np.sqrt(2 / (m - 1))

    L = s_bar - 3 * np.sqrt(1 - C4**2) / C4 * s_bar

    C = s_bar

    U = s_bar + 3 * np.sqrt(1 - C4**2) / C4 * s_bar

    # Erstelle eine neue Figur
    fig = plt.figure(figsize=(10, 6))

    plt.hlines([U, C, L], 0, n + n + 1, linestyles="dashed", colors="gray")
    plt.hlines(
        [
            s_bar + np.sqrt(1 - C4**2) / C4 * s_bar,
            s_bar + 2 * np.sqrt(1 - C4**2) / C4 * s_bar,
            s_bar - np.sqrt(1 - C4**2) / C4 * s_bar,
            s_bar - 2 * np.sqrt(1 - C4**2) / C4 * s_bar,
        ],
        0,
        n + n + 1,
        linestyles="dotted",
        colors="lightgray",
    )

    x = np.linspace(1, n, n)
    plt.plot(x, s, "o-")

    if X_new is not None:
        plt.vlines(n + 0.5, L, U, colors="red")

        A_new = np.shape(X_new)
        n_new = A_new[0]
        s_new = X_new.std(axis=1, ddof=1)
        x_new = np.linspace(1, n_new, n_new)
        plt.plot(x_new + n, s_new, "*-")
    else:
        plt.xlim((0, n + 1))

    plt.title("$s$-Karte")
    plt.xlabel("Sample")
    plt.ylabel("$s$")
    plt.show()
    # return fig


def shewart_xs(X, X_new=None):
    error.error_Type1(X)
    error.error_Shape1(X, 2, 2)
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
    L = X_bar_bar - 3 / (C4 * np.sqrt(m)) * s_bar
    # Mittelwert
    C = X_bar_bar
    # Obere Kontrollgrenze
    U = X_bar_bar + 3 / (C4 * np.sqrt(m)) * s_bar

    # Erstelle eine neue Figur
    fig = plt.figure(figsize=(10, 6))

    # Eingriffsgrenzen
    plt.hlines([U, C, L], 0, n + n + 1, linestyles="dashed", colors="gray")
    plt.hlines(
        [
            (X_bar_bar + 1 / (C4 * np.sqrt(m)) * s_bar),
            (X_bar_bar + 2 / (C4 * np.sqrt(m)) * s_bar),
            (X_bar_bar - 1 / (C4 * np.sqrt(m)) * s_bar),
            (X_bar_bar - 2 / (C4 * np.sqrt(m)) * s_bar),
        ],
        0,
        n + n + 1,
        linestyles="dotted",
        colors="lightgray",
    )

    x = np.linspace(1, n, n)
    plt.plot(x, X_bar, "o-")

    if X_new is not None:

        plt.vlines(n + 0.5, L, U, colors="red")

        A_new = np.shape(X_new)
        n_new = A_new[0]
        X_bar_new = X_new.mean(axis=1)
        x_new = np.linspace(1, n_new, n_new)
        plt.plot(x_new + n, X_bar_new, "*-")
    else:
        plt.xlim((0, n + 1))

    plt.title("$xs$-Karte")
    plt.xlabel("Sample")
    plt.ylabel("$x_{i_{mean}}$")
    plt.show()
    # return fig


# m wird Standartmässig als 2 angesehen
def shewart_ind(X, m):
    error.error_Type1(X)

    MR = rolling(X, m)
    MR_i = np.abs(MR)
    MR_bar = MR_i.mean()

    # R-Teil:
    # Array der Spannweite
    R = np.max(X) - np.min(X)
    # Mittelwert der Spannweite
    R_bar = R.mean()

    # x-Teil:
    X_bar = np.mean(X)
    X_bar_bar = X_bar.mean()

    A = np.shape(X)
    n = A[0]

    d2, d3 = d(m, 10000000)

    RL = (1 - 3 * d3 / d2) * MR_bar

    RC = MR_bar

    RU = (1 + 3 * d3 / d2) * MR_bar

    # Erstelle eine neue Figur
    fig = plt.figure(figsize=(10, 6))

    # Eingriffsgrenzen
    # R-Karte
    plt.figure()
    plt.hlines([RU, RC, RL], 0, n + n + 1, linestyles="dashed", colors="gray")
    plt.hlines(
        [
            (1 + d3 / d2) * MR_bar,
            (1 + 2 * d3 / d2) * MR_bar,
            (1 - d3 / d2) * MR_bar,
            (1 - 2 * d3 / d2) * MR_bar,
        ],
        0,
        n + n + 1,
        linestyles="dotted",
        colors="lightgray",
    )

    x = np.linspace(1, n - m + 1, n - m + 1)
    plt.plot(x, MR_i, "o-")

    plt.title("$R$-Karte")
    plt.xlabel("Sample")
    plt.ylabel("$R$")
    plt.grid()
    plt.xlim([0, n - m + 2])
    plt.show()

    # x-Karte
    XL = X_bar_bar - 3 / (d2 * np.sqrt(m)) * MR_bar

    XC = X_bar_bar

    XU = X_bar_bar + 3 / (d2 * np.sqrt(m)) * MR_bar

    # Eingriffsgrenzen
    plt.hlines([XU, XC, XL], 0, n + n + 1, linestyles="dashed", colors="gray")
    plt.hlines(
        [
            (X_bar_bar + 1 / (d2 * np.sqrt(m)) * MR_bar),
            (X_bar_bar + 2 / (d2 * np.sqrt(m)) * MR_bar),
            (X_bar_bar - 1 / (d2 * np.sqrt(m)) * MR_bar),
            (X_bar_bar - 2 / (d2 * np.sqrt(m)) * MR_bar),
        ],
        0,
        n + n + 1,
        linestyles="dotted",
        colors="lightgray",
    )

    x = np.linspace(1, n, n)
    plt.plot(x, X, "o-")

    plt.title("$x$-Karte")
    plt.xlabel("Sample")
    plt.ylabel("$x_{i_{mean}}$")
    plt.xlim([0, n + 1])
    plt.grid()
    plt.show()
    # return fig


def rolling(B, m):
    # Macht einen rolling average über eine Gruppe der grösse m im Array B
    error.error_Type1(B)
    A = np.shape(B)
    rolling_diff = []
    for i in range(A[0] - m + 1):
        rolling_diff.append(np.max(B[i : i + m]) - np.min(B[i : i + m]))
    return rolling_diff


def rollingx(B, m):
    # Macht einen rolling average über eine Gruppe der grösse m im Array B
    error.error_Type1(B)
    A = np.shape(B)
    rolling_diff = []
    for i in range(A[0] - m + 1):
        rolling_diff.append(np.mean(B[i : i + m]))
    return rolling_diff
