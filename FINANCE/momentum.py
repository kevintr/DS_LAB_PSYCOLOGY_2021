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



path = str(r'C:\Users\tranchinake\Desktop\personale\ScriptPython\momentum')
path = path.replace("\\","/")
os.chdir(path)

########## SCARICO DATI e salvataggio in csv

#start = dt.date(2019,4,1)
#end = dt.date(2021,4,30)
#df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
## per avere una lista di titoli
#tickers = df.Symbol.to_list()
### conto i titoli
#print(len(tickers))
#df = yf.download(tickers,start=start,end=end)
#stocks = df['Adj Close']
#
#stocks.to_csv("stocks.csv")
#
#stocks_pct = stocks.pct_change() ## vengono trasformati in percentuale 
#stocks_pct
#stocks_pct = stocks_pct[1:]
#
#funds = ['SPY']
#fundsret = reader.get_data_yahoo(funds,start,end)['Adj Close'].pct_change()
## fundsret = fundsret.resample('M').agg(lambda x: (x+1).prod()-1) # dà il rendimento mensile
#
#fundsret.to_csv("fundsret.csv")
#
#fundsret = fundsret[1:]
#fundsret.index = stocks_pct.index


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
fundsret = fundsret[:-1]

stocks_pct = stocks.pct_change() ## vengono trasformati in percentuale 
stocks_pct
stocks_pct = stocks_pct[1:]
stocks_pct_monthly = stocks_pct.resample('M').agg(lambda x: (x+1).prod()-1)
stocks_pct_monthly.index = pd.to_datetime(stocks_pct_monthly.index)

fundsret_monthly = fundsret.resample('M').agg(lambda x: (x+1).prod()-1)
fundsret_monthly.index = pd.to_datetime(fundsret_monthly.index)

start = dt.datetime(2019,4,1)
end = dt.datetime(2021,4,30)

# Inizio finestra
startWindow = start
# Fine finestra
endWindow = start + relativedelta(months=+12) 

df = pd.DataFrame()

while(endWindow<=end):
    
    filtroData = (fundsret.index < endWindow) & (fundsret.index > startWindow)
    filtroStocksData = (stocks_pct.index < endWindow) & (stocks_pct.index > startWindow)
    fundsret2 = fundsret[filtroData]
    stocks_pct2 = stocks_pct[filtroStocksData]
    print("startWindow" + str(startWindow))
    print("endWindow" + str(endWindow))
    
    # inizio nuova finestra
    startWindow = startWindow + relativedelta(months=+1)
    # fine nuova finestra
    endWindow = startWindow + relativedelta(months=+12)
    
    stocks_pct.index = fundsret.index
    
    x = fundsret2
    x
    
    y = stocks_pct2
    y
    
    x = sm.add_constant(x)
    x
    
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
    
    residualMomentum['percentile'] = pd.qcut(residualMomentum.iloc[:,0],100,labels=False,duplicates='drop')
    
    winners = residualMomentum[residualMomentum['percentile'] == 99]
    losers = residualMomentum[residualMomentum['percentile'] == 1]
    winners.index
    
    investedMonth = endWindow #+relativedelta(months=+1)
    investedMonth = investedMonth +relativedelta(days=-1)
    investedMonth = pd.to_datetime(investedMonth)
    
    stocks_winners = stocks_pct_monthly[winners.index]
    stocks_winners = stocks_winners[stocks_winners.index == investedMonth]
    stocks_winners['portafoglio'] = stocks_winners.iloc[-1].mean()
    
    stocks_winners.loc[:,'winners'] = str(list(winners.index))
    stocks_winners = stocks_winners[['winners','portafoglio']]
    
    df = df.append(stocks_winners)
    df
#    rendimentoMensile = rendimentoMensile.append(pd.Series(stocks_winners.iloc[-1].mean()), ignore_index=True)

# Kevin
# Fare un metodo per fare una finestra manualmente

# Giacomo
# Momentum video
# dividere i residuals per lo standar standarError

########################## fine for
# Normalizzazione
df.index = pd.to_datetime(df.index)
df['SPY'] = fundsret_monthly[fundsret_monthly.index.isin(df.index)]

df
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

