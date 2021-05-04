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
fundsret = pd.read_csv("fundsret.csv")
fundsret = fundsret.set_index('Date')

stocks_pct = stocks.pct_change() ## vengono trasformati in percentuale 
stocks_pct
stocks_pct = stocks_pct[1:]
#
#portFolioAnnuale = pd.DataFrame
#rendimentoMensile = pd.Series
#
#
#six_months = datetime.strptime('2019-04-01', '%y-%m-%d') + relativedelta(months=-6)
#
#datetime.strptime('2019-04-01', '%m-%d-%Y').date()
#
#date_str = '2019-04-01'
#
#date_object = datetime.strptime(date_str, '%Y-%m-%d').date()
#
#
#stocks_pct['']
#
#for()
#fundsret.loc['2019-04-02',:]
#
#
#w = fundsret.rolling(1).SPY
#w


# Kevin
# Fare un metodo per fare una finestra manualmente


# Giacomo
# MOmentum video
# RollingOLS con calcolo dei Giorni di finestra varibili da mese in mese, considerando che è importante regredire con tutti i 500 stocks
# RollingOLS sui mesi prendendo un finestra di 10 anni, regredendo di 2 anni ogni volta
# dividere i residuals per lo standar standarError


fundsret = fundsret[:-1]

#x = fundsret
#y =  stocks_pct['AAL']
fundsret.index = stocks_pct.index

fundsret = stocks_pct['ZION']
fundsret['AAL'] = stocks_pct['AAL']
x = fundsret
x
y = stocks_pct['A']

#OLS_model = RollingOLS(endog =y,exog=x,window=20)
#rres = model.fit()

#finestra di 12

#
x = sm.add_constant(x)
x
#fundsret['SPY2'] =  stocks_pct['AAL']
OLS_model = sm.OLS(y,x).fit() # training the model
predicted_values = OLS_model.predict()  # predicted values
residual_values = OLS_model.resid # residual values
coefficient = OLS_model.params # residual values
coefficient
rsquared = OLS_model.rsquared # residual values
rsquared
residual_values.index = pd.to_datetime(residual_values.index)



residuals = residual_values.resample('M').agg(lambda x: (x+1).prod()-1) # dà il rendimento mensile

residuals1 = residuals
residuals


# Escludere l'ultimo mese- Rolling


residualMomentum = residuals.iloc[-1] -residuals.iloc[0]
residualMomentum

residualMomentum = pd.DataFrame(residualMomentum)

residualMomentum['cinquenatile'] = pd.qcut(residualMomentum.iloc[:,0],50,labels=False,duplicates='drop')

winners = residualMomentum[residualMomentum['cinquenatile'] == 49]
losers = residualMomentum[residualMomentum['cinquenatile'] == 1]
winners.index
stocks_winners = stocks_pct[winners.index]
stocks_winners.index = pd.to_datetime(stocks_winners.index)
stocks_winners = stocks_winners.resample('M').agg(lambda x: (x+1).prod()-1)

portFolio['MAY'] = stocks_winners.iloc[-1].mean()



rendimentoMensile = pd.Series(stocks_winners.iloc[-1].mean())

rendimentoMensile = rendimentoMensile.append(pd.Series(stocks_winners.iloc[-1].mean()), ignore_index=True)



########################## fine for
# Normalizzazione
df = stocks_winners.mean()
normalized_df=(df-df.min())/(df.max()-df.min())





# normalizzazione
plt.figure(figsize=(15,10))

time = np.array(normalized_df.index)
plt.plot(time,normalized_df, linewidth=1)
# plt.scatter(time,df_price_tweets.loc[:,'high_median'], color='darkblue',linewidth=None,edgecolors=None , marker='o')
# plt.scatter(df_price_tweets.loc[:,'time'],df_price_tweets.loc[:,'high_mean'], color='aqua',linewidth=None,edgecolors=None , marker='x')
#plt.plot(df_price_tweets.loc[:,'date'],df_price_tweets.loc[:,'price_eur_median'], color='aqua',linewidth=3)
plt.xticks(rotation=70)
# plt.legend(('compound_media', 'high_median', 'high_mean'))
plt.show()

