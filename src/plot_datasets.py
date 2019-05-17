import pandas as pd
import numpy as np
import csv

def load_data():
    print("Loading data...")
    with open('./data/cov.csv', 'r') as f:
        df = list(csv.reader(f, delimiter=','))
    #print(df)
    with open('./data/cov_norm.csv', 'r') as f:
        df_norm = list(csv.reader(f, delimiter=','))
    #print(df_norm)
    maxRows = 50
    df = df[0:maxRows]
    df_norm = df_norm[0:maxRows]
    df = [val for sublist in df for val in sublist]
    df_norm = [val for sublist in df_norm for val in sublist]
    return df, df_norm

def plot(df, df_norm):
    print("Ploting data...")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(df, df_norm, 'bo')
    plt.show()
    fig.savefig('dataset.png', dpi=fig.dpi)
    return df, df_norm

if __name__ == "__main__":
    df, df_norm = load_data()
    df, df_norm = plot(df, df_norm)

