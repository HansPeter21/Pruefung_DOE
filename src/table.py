

def table(L3,L2,L1,C,U1,U2,U3):
# Tabellarische Ausgabe mit f-Strings
# : Beginnt Formatstring, . steht für Nachkommastellen, 2f steht für 2 Nachkommastellen
    table = f"""
    +---------------------+---------+
    | Kontrollgrenze      | Wert    |
    +---------------------+---------+
    | L3                  | {L3:.2f} |
    | L2                  | {L2:.2f} |
    | L1                  | {L1:.2f} |
    | C                   | {C:.2f}  |
    | U1                  | {U1:.2f} |
    | U2                  | {U2:.2f} |
    | U3                  | {U3:.2f} |
    +---------------------+---------+
    """
    print(table)