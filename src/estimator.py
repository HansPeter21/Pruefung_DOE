from scipy.special import gamma
import numpy as np


def Schätzer_mu(X):
    X_bar = X.mean(axis=1)  # Mittlwert jeder Zeile
    X_bar_bar = X_bar.mean()  # Mittelwert der Mittelwerte
    print(f"Der Erwartungwert mu0 ist {round(X_bar_bar,4)}")
    return X_bar_bar


def Schätzer_sig(X):
    s = X.std(axis=1, ddof=1)  # Standartabweichung der einzelnen Spalten
    s_bar = s.mean()  # Mittelwert der Standartabweichung

    A = np.shape(X)  # Dimensionen der Daten
    m = A[1]  # Anzahl der Spalten

    C4 = gamma(m / 2) / gamma((m - 1) / 2) * np.sqrt(2 / (m - 1))  # Konstante

    sigma = s_bar / C4

    print(f"Der Schätzer Sigma_Dach ist {round(sigma,4)} ")
    return sigma
