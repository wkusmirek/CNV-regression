import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import csv

def load_data():
    print("Loading data...")
    with open('./data/1_cov_NA_gt.csv', 'r') as f:
        gt = list(csv.reader(f, delimiter=','))
    with open('./data/1_cov_NA.csv', 'r') as f:
        df = list(csv.reader(f, delimiter=','))
    with open('./data/1_cov_NA_norm.csv', 'r') as f:
        df_norm = list(csv.reader(f, delimiter=','))
    return gt, df, df_norm

def quality_control(rowId, gt, df, df_norm):
    print("Quality control...")
    mask = np.percentile(df, axis=0, q=80) >= 20
    if(mask == False):
        return [], [], []
    else:
        return gt, df, df_norm

def cnv_control(rowId, gt, df, df_norm):
    print("CNV control...")
    mask = sum(gt) > 0
    if(mask == False):
        return [], [], []
    else:
        return gt, df, df_norm

def regression(rowId, gt, df, df_norm):
    if len(df) == 0:
        return
    print("Linear regression...")
    df = np.array(df).reshape(-1, 1)
    df_norm = np.array(df_norm).reshape(-1, 1)
    regr = linear_model.LinearRegression()
    regr.fit(df, df_norm)
    df_pred = regr.predict(df)
    print('Coefficients: \n', regr.coef_)

    print("Polynominal regression...")
    df = np.array(df).reshape(1, -1)
    df_norm = np.array(df_norm).reshape(1, -1)
    z = np.poly1d(np.polyfit(df[0,:],df_norm[0,:],5))
    df_pred = z(df)

    #print("Ploting data...")
    #df = df[0]
    #df_norm = df_norm[0]
    #plt.show()
    #fig = plt.figure()
    #indices = [i for i, x in enumerate(gt) if x == 0]
    #plt.scatter(df[indices], df_norm[indices], color='black')
    #indices = [i for i, x in enumerate(gt) if x == 1]
    #plt.scatter(df[indices], df_norm[indices], color='red')
    #plt.scatter(df, df_pred, color='blue')
    #plt.xticks(np.arange(df.min(), df.max(), 50))
    #plt.yticks(np.arange(df_norm.min(), df_norm.max(), 50))
    #fig.savefig('dataset' + str(rowId) + '.png', dpi=fig.dpi)

if __name__ == "__main__":
    gt_all, df_all, df_norm_all = load_data()
    for rowId in range(1,len(gt_all)):
        gt = gt_all[rowId]
        df = df_all[rowId]
        df_norm = df_norm_all[rowId]
        gt = list(map(int, gt))
        df = list(map(int, df))
        df_norm = list(map(int, df_norm))
        
        gt, df, df_norm = quality_control(rowId, gt, df, df_norm)
        #gt, df, df_norm = cnv_control(rowId, gt, df, df_norm)
        regression(rowId, gt, df, df_norm)
