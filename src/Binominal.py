import scipy.stats as st
from scipy.special import binom
from scipy.stats import poisson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def p_card(m, n, X, norm=True):
    if norm == True:
        Xp = X / m
        p_hat = np.mean(Xp)
        sig = np.sqrt((p_hat * (1 - p_hat)) / m)
        C = p_hat
        U = C + 3 * sig
        L = C - 3 * sig
        print("Grenzen der p-Karte: ", np.round(L, 3), np.round(C, 3), np.round(U, 3))
        plt.plot(Xp, "-o")
        plt.hlines(L, xmin=0, xmax=n, linestyles="--", color="black")
        plt.hlines(C, xmin=0, xmax=n, linestyle="solid", colors="black")
        plt.hlines(U, xmin=0, xmax=n, linestyles="--", color="black")
        plt.title(r"$p$-Karte")
        plt.grid()
        plt.show()

    if norm == False:
        # error probability
        alpha = 0.0027

        p_hat = np.mean(Xp)
        process_model = binom(m, p_hat)

        # possible number of failures: 0 .. 50
        numbers = np.arange(m + 1)

        # indicator for left side (possibly 0)
        if process_model.pmf(0) > alpha / 2:
            L_star = 0
        else:
            mask_low = process_model.cdf(numbers) <= alpha / 2
            L_star = np.max(numbers[mask_low])

        # indicator for right side
        mask_high = (1 - process_model.cdf(numbers)) <= alpha / 2
        U_star = np.min(numbers[mask_high])

        # bounds on the relative frequency
        L = L_star / m
        U = U_star / m


def c_card(data_file, alpha=0.0027):
    # Einlesen der Daten
    df = pd.read_csv(data_file)
    n = df.shape[0]

    # Berechnung von lambda_hat (Durchschnitt der Fehler)
    lambda_hat = df["Fehler"].mean()

    # Poisson-Verteilungs-Grenzen
    process_model = poisson(mu=lambda_hat)
    L = process_model.ppf(alpha / 2)
    U = process_model.ppf(1 - alpha / 2)

    print("\nGrenzen der p-Karte (Poissonverteilung):")
    print("L= ", np.round(L, 3), "C= ", np.round(lambda_hat, 3), "U= ", np.round(U, 3))

    # Wahrscheinlichkeiten für echte Fehler
    alpha_true_low = process_model.cdf(L - 1)
    alpha_true_high = 1 - process_model.cdf(U)
    print("Niveau: ", alpha_true_low + alpha_true_high)

    # Plot erstellen
    plt.plot([0, n], [U, U], "--", color="blue", label="Obere Grenze (U)")
    plt.plot([0, n], [L, L], "--", color="blue", label="Untere Grenze (L)")

    # Mittelwertlinie
    plt.plot(
        [0, n], [lambda_hat, lambda_hat], "--", color="black", label="Mittelwert (C)"
    )

    # Fehlerwerte plotten
    plt.plot(df["Fehler"], "o-", label="Fehleranzahl")

    # Plot anpassen
    plt.grid()
    plt.title("Poisson-Kontrollkarte für Fehleranzahl")
    plt.xlabel("Probe")
    plt.ylabel("Fehleranzahl")
    plt.legend()
