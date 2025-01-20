import src.error as error

def ranges(X):
    error.error_Type1(X)
    error.error_Shape1(X,1,1)
    R = X.max(axis=1)-X.min(axis=1) # Array der Spannweite
    R_bar = R.mean() # Mittelwert der Spannweite
    print(f"Spannweiten: {R}, Mittelwert der Spannweiten: {R_bar}")

def mean(X):
    error.error_Type1(X)
    error.error_Shape1(X,1,1)
    X_bar = X.mean(axis=1) # Mittlwert jeder Zeile
    X_bar_bar  = X_bar.mean() # Mittelwert der Mittelwerte
    print(f"Mittelwerte: {X_bar}, Mittelwert der Mittelwerte: {X_bar_bar}")

def std(X):
    error.error_Type1(X)
    error.error_Shape1(X,2,2) # Muss zwei sein ansonsten Division durch 0! n-1 unter dem Bruch!
    s = X.std(axis=1,ddof=1) # Standartabweichung der einzelnen Spalten  
    s_bar = s.mean() # Mittelwert der Standartabweichung
    print(f"Standartabweichungen: {s}, Mittelwert der Standartabweichungen: {s_bar}")  