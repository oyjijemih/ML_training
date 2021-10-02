import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, xtrain, ytrain):
        self.x = xtrain
        self.y = ytrain
        self.num_x = xtrain.shape[1]
        self.num_obs = xtrain.shape[0]
    
    def train(self):
        z = np.concatenate([self.x, np.ones([self.num_obs, 1])], axis=1)
        zz = 1/self.num_obs * (z.T @ z)
        zy = 1/self.num_obs * (z.T @ self.y)
        coef = np.linalg.inv(zz) @ zy
        self.w = coef[:-1]
        self.b = coef[-1]
        
    def trainRegularlized(self, lamb = 0.1):
        z = np.concatenate([self.x, np.ones([self.num_obs, 1])], axis=1)
        zz = 1/self.num_obs * (z.T @ z)
        zy = 1/self.num_obs * (z.T @ self.y)
        one = np.identity(z.shape[1])
        v = np.linalg.inv(zz + lamb*one) @ zy
        self.w = v[:-1]
        self.b = v[-1]
        
    def predict(self, xdata):
        pre = xdata @ self.w + self.b
        return pre
    
    def RMSE(self, x, y):
        residual = y - self.predict(x)
        rmse = np.sqrt(np.mean(np.square(residual)))
        return rmse
    
    def R2(self, x, y):
        residual = y - self.predict(x)
        ssr = residual.T @ residual
        ydemean = y - np.mean(y, axis=0)
        yvar  = ydemean.T @ ydemean
        return 1 - ssr/yvar
    
    def PlotResult(self, x=[], y=[], xlabel="", ylabel=""):
        if x.shape[1] != 1:
            return
        
        xlin = np.array([[0], [np.max(x)]])
        ylin = self.predict(xlin)
        
        plt.plot(x, y, '.', label="data")
        plt.plot(xlin, ylin, 'r', label="regression_line")
        plt.legend()
        
        plt.xlim([0, np.max(x)])
        plt.ylim([0, np.max(y)])
        plt.xlabel(xlabel, fontSize=14)
        plt.ylabel(ylabel, fontSize=14)
        
        plt.show()
        
mydata = pd.read_csv("C:/Users/oyjij/python　練習/MLBook/data/train.csv")
X_s = mydata[mydata['MSSubClass']==60][['GrLivArea']].values
Y_s = mydata[mydata['MSSubClass']==60][['SalePrice']].values

num_tr = int(len(X_s)*0.9)
Xtr_s = X_s[:num_tr]
Ytr_s = Y_s[:num_tr]
Xte_s = X_s[num_tr:]
Yte_s = Y_s[num_tr:]

mymodel = LinearRegression(Xtr_s, Ytr_s)
mymodel.train()

mymodel.RMSE(Xte_s, Yte_s)
mymodel.R2(Xte_s, Yte_s)
mymodel.PlotResult(Xtr_s, Ytr_s)

X_mm = mydata[mydata['MSSubClass']==60][['GrLivArea','GarageArea','PoolArea','BedroomAbvGr','TotRmsAbvGrd']].values
Y_mm = mydata[mydata['MSSubClass']==60][['SalePrice']].values

#標準化すると係数が [-1,1] になる
X_m = (X_mm - np.mean(X_mm, axis=0)) / np.std(X_mm, axis=0)
Y_m = (Y_mm - np.mean(Y_mm, axis=0)) / np.std(Y_mm, axis=0)

Xtr_m = X_m[:num_tr]
Ytr_m = Y_m[:num_tr]
Xte_m = X_m[num_tr:]
Yte_m = Y_m[num_tr:]

mymodel_m = LinearRegression(Xtr_m, Ytr_m)
mymodel_m.train()

#パラメータの絶対値の大きさで重要度を比較できる
mymodel_m.RMSE(Xte_m, Yte_m)
mymodel_m.R2(Xte_m, Yte_m)
print(f"モデルパラメータ:\nw={mymodel_m.w}, \nb={mymodel_m.b}")

Xtr_out = Xtr_s
Ytr_out = Ytr_s
Xte_out = Xte_s
Yte_out = Yte_s
Ytr_out[np.argsort(Ytr_out, axis=0)[-2:]] = Ytr_out[np.argsort(Ytr_out, axis=0)[-2:]] - 700000

#外れ値があると回帰直線の傾きが大きく変わる
mymodel_out = LinearRegression(Xtr_out, Ytr_out)
mymodel_out.train()
mymodel_out.PlotResult(Xtr_out, Ytr_out)
print(f"モデルパラメータ:\nw={mymodel.w}, \nb={mymodel.b}")
print(f"モデルパラメータ:\nw={mymodel_out.w}, \nb={mymodel_out.b}")

#L2ノルム正則化を行うと係数のずれが小さい
mymodel_reg = LinearRegression(Xtr_out, Ytr_out)
mymodel_reg.trainRegularlized()
mymodel_reg.PlotResult(Xtr_out, Ytr_out)
print(f"モデルパラメータ:\nw={mymodel_reg.w}, \nb={mymodel_reg.b}")
