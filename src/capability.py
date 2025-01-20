import src.estimator as est
import numpy as np


def cal_Cp(USL, LSL, n, sigma):
    cp = (USL - LSL) / (n * sigma)
    cp = round(cp, 2)
    return cp


# Prozessfähigkeits-Index
def Cp(USL=None, LSL=None, sigma=None, X=None):
    # Wenn Sigma_Dach gegeben ist.
    if sigma != None:
        # cp = (USL - LSL) / (6 * sigma)
        cp = cal_Cp(USL, LSL, 6, sigma)
        print(f"Der Prozessfähigkeits-Index berägt {cp}")

    # Wenn Sigma_Dach berechnet werden muss.
    else:
        sig_hat = est.Schätzer_sig(X)
        # cp = (USL - LSL) / (6 * sig_hat)
        cp = cal_Cp(USL, LSL, 6, sig_hat)
        print(f"Der Prozessfähigkeits-Index berägt {cp}")


def side_Cp(USL=None, LSL=None, sigma=None, X=None, mu=None):
    if sigma != None:
        # cpu = (USL - mu) / (3 * sigma)
        cpu = cal_Cp(USL, mu, 3, sigma)
        print(f"C_pu für USL relevant ist {cpu}")

        # cpl = (mu - LSL) / (3 * sigma)
        cpl = cal_Cp(mu, LSL, 3, sigma)
        print(f"C_pl für LSL relevant ist {cpl}")

    else:
        sig_hat = est.Schätzer_sig(X)

        # Für das obere Spezifikationslimit
        cpu = cal_Cp(USL, mu, 3, sig_hat)
        print(f"C_pu für USL relevant ist {cpu}")

        # Für das untere Spezifikationslimit
        cpl = cal_Cp(mu, LSL, 3, sig_hat)
        print(f"C_pl für LSL relevant ist {cpl}")


def Cpk(USL=None, LSL=None, sigma=None, X=None, mu=None):
    if sigma != None:
        cp = (USL - LSL) / (6 * sigma)
    else:
        sig_hat = est.Schätzer_sig(X)
        cp = (USL - LSL) / (6 * sig_hat)

    k_hat = abs(((USL + LSL) / 2) - mu) / ((USL - LSL) / 2)
    cpk = cp * (1 - k_hat)
    cpk = round(cpk, 2)
    print(f"C_pk ist {cpk}")


# Prozessdurchschnitt
def y_bar_bar(USL, LSL, m, Cp1, Cpk1):
    sigma_hat = (USL - LSL) / (Cp1 * 6)
    k_hat = 1 - Cpk1 / Cp1
    x = k_hat * (USL - LSL) / 2
    mu = m + x
    return mu
