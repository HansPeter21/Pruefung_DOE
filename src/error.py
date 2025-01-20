import numpy as np
import requests

def error_Type1(X): # Diese Funktion gibt einen Error aus wenn X == None oder X != Numpy Array or List ist.
    if not isinstance(X, np.ndarray):
        raise TypeError("Der Parameter muss ein NumPy-Array (np.ndarray) sein.")
    
def error_Shape1(X, min_Zeilen, min_Spalten):
    if X.shape[0] < min_Zeilen:
        raise ValueError(f"Das Array muss mindestens {min_Zeilen} Zeilen haben. Das Array hat {X.shape[0]} Zeilen.")
    elif X.shape[0] < min_Zeilen  or X.shape[1] < min_Spalten:
        raise ValueError(f"Das Array muss mindestens {min_Zeilen} Zeilen und {min_Spalten} Spalten haben. Das Array hat {X.shape[0]} Zeilen und {X.shape[1]} Spalten.")
    
def addressReachable(URL):
    try: 
        r = requests.get(f'{URL}')
        if r.status_code != 200: # Überprüft ob die Seite erreicht werden konnte
            raise ValueError(f"Die URL {URL} konnte nicht erreicht werden.")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Website {URL} konnte nicht gefunden werden")