import pandas_datareader.data as reader
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import math
import ast



path = str(r'C:\Users\tranchinake\Desktop\personale\ScriptPython\momentum')
path = path.replace("\\","/")
os.chdir(path)

########## SCARICO DATI e salvataggio in csv

#start = dt.datetime(2019,3,1)
#end = dt.datetime(2021,3,31)
#
##start = dt.date(2019,3,29)
##end = dt.date(2021,4,30)
#
#df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
## per avere una lista di titoli
#tickers = df.Symbol.to_list()
### conto i titoli
#print(len(tickers))
#df = yf.download(tickers,start=start,end=end)
#stocks = df['Adj Close']
#
#stocks= stocks[1:]
#
#stocks = stocks[:-2]
#
#stocks.to_csv("stocks.csv")
#
#stocks_pct = stocks.pct_change() ## vengono trasformati in percentuale 
#stocks_pct
#stocks_pct = stocks_pct[1:]
#
#funds = ['SPY']
#fundsret = reader.get_data_yahoo(funds,start,end)['Adj Close'].pct_change()
#fundsret = fundsret[1:-1]
## fundsret = fundsret.resample('M').agg(lambda x: (x+1).prod()-1) # dà il rendimento mensile
##fundsretPrice = reader.get_data_yahoo(funds,start,end)['Adj Close']
#
#fundsret.to_csv("fundsret.csv")
#
#fundsret = fundsret[1:]
#fundsret.index = stocks_pct.index
#


#### 

# 5 anni precedenti di prezzi degli stocks



#- Lista di stocs
#- I prezzi di chiusura degli stocks 5 anni precedenti ad oggi, con granularità giornaliera
#- Valore dell'indice di riferimento sp500 per 5 anni
#- Fare una regressione lineare tra i rendimenti dei prezzi di un titolo  e rendimenti dell'indice SP500 al variare dei giorni in un anno
#        Fama french
#- Estrapolare i rendimenti residuali giornalieri
##(prezzoFinale - prezzoINiziale / prezzoIniziale) ---> rendimento mensile
#- Calcolo i residui mensili
#- Momentum residuale
#- Vedere chi ha il momentum più alto con decili (50 stocks), percetile (5)
#- Dei winners tornare ai rendimenti reali e fare una media equiponderata e plottare insieme al rendimento del'sp550


#### LETTURA FILE
stocks = pd.read_csv("stocks.csv")
stocks = stocks.set_index('Date')
stocks.index = pd.to_datetime(stocks.index)

fundsret = pd.read_csv("fundsret.csv")
fundsret = fundsret.set_index('Date')
fundsret.index = pd.to_datetime(fundsret.index)

stocks_pct = stocks.pct_change() ## vengono trasformati in percentuale 
stocks_pct
stocks_pct = stocks_pct[1:]
stocks_pct_monthly = stocks_pct.resample('M').agg(lambda x: (x+1).prod()-1)
stocks_pct_monthly.index = pd.to_datetime(stocks_pct_monthly.index)

factors = reader.DataReader('F-F_Research_Data_Factors_daily','famafrench',start,end)[0]
factors = factors[1:-3]
factors = factors[['SMB','HML']]

#factors.merge(fundsret)

#Vo=275.68560791015625
#Vf=283.58489990234375
#
#
##
##
#rendimento = (Vf-Vo)/Vo
#rendimento

fundsret_monthly = fundsret.resample('M').agg(lambda x: (x+1).prod()-1)
fundsret_monthly.index = pd.to_datetime(fundsret_monthly.index)


fundsret3 = factors.join(fundsret)

#fundsret = fundsret[1:]

start = dt.datetime(2019,3,1)
end = dt.datetime(2021,3,31)

# Inizio finestra
startWindow = start
# Fine finestra
endWindow = start + relativedelta(months=+12) 

df = pd.DataFrame()

while(endWindow<=end):
    
    filtroData = (fundsret3.index <= endWindow) & (fundsret3.index >= startWindow)
    filtroStocksData = (stocks_pct.index <= endWindow) & (stocks_pct.index >= startWindow)
    fundsret2 = fundsret3[filtroData]
    stocks_pct2 = stocks_pct[filtroStocksData]
    print("\n")
    print("startWindow : " + str(startWindow))
    print("endWindow : " + str(endWindow))
    print("\n")
    
    # inizio nuova finestra
    startWindow = startWindow + relativedelta(months=+1)
    
    # fine nuova finestra
    endWindow = startWindow + relativedelta(months=+12)
    
    stocks_pct2.index = fundsret2.index
    
    x = fundsret2
    x
    
    y = stocks_pct2
    y
    
#    x = sm.add_constant(x)
#    x
    
    OLS_model = sm.OLS(y,x).fit() # training the model
    predicted_values = OLS_model.predict()  # predicted values
    residual_values = OLS_model.resid # residual values
    coefficient = OLS_model.params # residual values
    coefficient
    residual_values.index = pd.to_datetime(residual_values.index)
    
    residuals = residual_values.resample('12M').agg(lambda x: (x+1).prod()-1) # dà il rendimento mensile
    
    residualMomentum = residuals.iloc[0]
    residualMomentum
    
    residualMomentum = pd.DataFrame(residualMomentum)
    
    residualMomentum['cinquantile'] = pd.qcut(residualMomentum.iloc[:,0],50,labels=False,duplicates='drop')
    
    winners = residualMomentum[residualMomentum['cinquantile'] == 49]
    losers = residualMomentum[residualMomentum['cinquantile'] == 1]
    winners.index
    
    investedMonth = endWindow #+relativedelta(months=+1)
    investedMonth = investedMonth +relativedelta(days=-1)
    investedMonth = pd.to_datetime(investedMonth)
    
    stocks_winners = stocks_pct_monthly[winners.index]
    stocks_winners = stocks_winners[stocks_winners.index == investedMonth]
    stocks_winners['portafoglio'] = stocks_winners.iloc[-1].mean()
    
    stocks_winners.loc[:,'winners'] = str(list(winners.index))
    stocks_winners.loc[:,'winners']
    stocks_winners = stocks_winners[['winners','portafoglio']]
    
    df = df.append(stocks_winners)
    df
#    rendimentoMensile = rendimentoMensile.append(pd.Series(stocks_winners.iloc[-1].mean()), ignore_index=True)


#
#def internally_studentized_residual(X,Y):
#    X = np.array(X, dtype=float)
#    Y = np.array(Y, dtype=float)
#    mean_X = np.mean(X)
#    mean_Y = np.mean(Y)
#    n = len(X)
#    diff_mean_sqr = np.dot((X - mean_X), (X - mean_X))
#    beta1 = np.dot((X - mean_X), (Y - mean_Y)) / diff_mean_sqr
#    beta0 = mean_Y - beta1 * mean_X
#    y_hat = beta0 + beta1 * X
#    residuals = Y - y_hat
#    h_ii = (X - mean_X)*2 / diff_mean_sqr + (1 / n)
#    Var_e = math.sqrt(sum((Y - y_hat) *2)/(n-2))
#    SE_regression = Var_e*((1-h_ii) ** 0.5)
#    studentized_residuals = residuals/SE_regression
#    return studentized_residuals

# Momentum video per togliere il mese

########################## fine for
# Normalizzazione
df.index = pd.to_datetime(df.index)
df['SPY'] = fundsret_monthly[fundsret_monthly.index.isin(df.index)]


#filtroData = (fundsret_monthly.index <='2020-12-31') & (fundsret_monthly.index >= '2020-01-31')
#z = fundsret_monthly[filtroData]
#rendimentoIndice = z['SPY'].agg(lambda x: (x+1).prod()-1) # d
#rendimentoIndice
#
#df['SPY'].agg(lambda x: (x+1).prod()-1)
#
##################
df.index = pd.to_datetime(df.index)
df['SPY'] = fundsret_monthly[fundsret_monthly.index.isin(df.index)]

rendimentoIndice = df['SPY'].agg(lambda x: (x+1).prod()-1) # d
rendimentoIndice
rendimentoPortafoglio = df['portafoglio'].agg(lambda x: (x+1).prod()-1) # d
rendimentoPortafoglio

## investimento
#Vo=100
#
#Vf=Vo*rendimentoPortafoglio +Vo
#Vf
#
#Vf=Vo*rendimentoIndice +Vo
#Vf

normalized_df=(df-df.min())/(df.max()-df.min())

# normalizzazione
plt.figure(figsize=(15,10))

time = np.array(df.index)
plt.plot(time,df[['portafoglio','SPY']], linewidth=1)
# plt.scatter(time,df_price_tweets.loc[:,'high_median'], color='darkblue',linewidth=None,edgecolors=None , marker='o')
# plt.scatter(df_price_tweets.loc[:,'time'],df_price_tweets.loc[:,'high_mean'], color='aqua',linewidth=None,edgecolors=None , marker='x')
#plt.plot(df_price_tweets.loc[:,'date'],df_price_tweets.loc[:,'price_eur_median'], color='aqua',linewidth=3)
plt.xticks(rotation=70)
plt.legend(('portafoglio', 'SPY'))
plt.show()


### RISCHIO
type(ast.literal_eval(df.iloc[1,1]))

initial_weight = np.array([0.166,0.166,0.166,0.166,0.166,0.166,0.166,0.166,0.166,0.166,0.166])

start = dt.datetime(2020,3,1)
end = dt.datetime(2021,3,31)

# Inizio finestra
startWindow = start
# Fine finestra
endWindow = start + relativedelta(months=+12) 

seriesRisk = pd.DataFrame()

for index, row in df.iterrows():
    
    s = stocks_pct[ast.literal_eval(row['winners'])]
    filtroData = (s.index <= endWindow) & (s.index >= startWindow)
    
    s = s[filtroData]
    s
    
    matrix_covariance_portfolio = s #.iloc[:,:-1]
    matrix_covariance_portfolio = (matrix_covariance_portfolio.cov())
    
    matrix_covariance_portfolio
    
    portfolio_variance = np.dot(initial_weight.T,np.dot(matrix_covariance_portfolio, initial_weight))
    
    #standard deviation (risk of portfolio)
    portfolio_risk = np.sqrt(portfolio_variance)*np.sqrt(22)
    portfolio_risk
    
    d = {'Date': [s.index[0]], 'riskMonthly': [portfolio_risk]}
    d = pd.DataFrame(data=d,index=None)
    d
    
    seriesRisk = seriesRisk.append(d)
    
        # inizio nuova finestra
    startWindow = startWindow + relativedelta(months=+1)
    
    # fine nuova finestra
    endWindow = startWindow + relativedelta(months=+12)



seriesRisk['riskMonthlyAnnual'] = seriesRisk['riskMonthly']*(np.sqrt(12))
m = seriesRisk['riskMonthly'].mean() # .agg(lambda x: (x+1).prod()-1)
m #  0.07128853484365637
m = seriesRisk['riskMonthlyAnnual'].mean() # .agg(lambda x: (x+1).prod()-1)
m # 0.24695072869271412


# Giacomo
# Jupiter


# Kevin
# Sistemo un po' il codice
# Fare l'elenco dei titolo con la frequenza di apparizione
# Grafico da sistemare i colori, le label





