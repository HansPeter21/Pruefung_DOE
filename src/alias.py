import numpy as np
import pandas as pd
from itertools import combinations


def get_interaction_table(design):
    factor_count = design.shape[1]
    column_names = [chr(ord('A') + i) for i in range(factor_count)]

    # Get all possible combinations
    combinations_list = []
    for r in range(1, len(column_names) + 1):
        combinations_list.extend(combinations(column_names, r))
    combinations_strings = [''.join(combination) for combination in combinations_list]

    combinations_index = []
    for r in range(1, len(column_names) + 1):
        combinations_index.extend(combinations(list(range(factor_count)), r))

    # Calculate all combinations
    data = []
    for i in range(len(combinations_index)):
        col_data = design[:, combinations_index[i][0]].copy()
        for comb in combinations_index[i][1:]:
            col_data *= design[:, comb]
        data.append(col_data)

    df = pd.DataFrame(np.array(data).T, columns=combinations_strings)

    return df


def get_alias_duos(design):
    df = get_interaction_table(design)

    aliases = []
    for col in range(len(df.columns) - 1):
        for other_col in range(col + 1, len(df.columns)):
            if np.array_equal(df.iloc[:,col].values, df.iloc[:,other_col].values):
                aliases.append(f"{df.columns[col]} + {df.columns[other_col]}")
    return aliases


def get_aliases(design):
    aliases = get_alias_duos(design)
    
    combined_final = []
    while len(aliases) > 0:
        combined = aliases[0].split(" + ")
        unused_aliases = []
        for i in range(1, len(aliases)):
            used = False
            splitted = aliases[i].split(" + ")
            for s in splitted:
                if s in combined:
                    combined.extend(splitted)
                    used = True
                    break
            if not used:
                unused_aliases.append(aliases[i])
        combined = list(set(combined))
        combined = sorted(combined, key=lambda x: (len(x), x))
        combined_final.append(" + ".join(combined))
        aliases = unused_aliases
    return sorted(combined_final, key=lambda x: (len(x.split(" + ")[0]), len(x.split(" + ")[1]), x))

