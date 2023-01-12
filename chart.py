#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
import mplfinance as mpf
import time
import requests
import urllib.request
from bs4 import BeautifulSoup
import re

c1, c2, c3 = st.columns([1,1,1])

with c1:
    if st.checkbox('Custom symbol', False):
        ticker = st.text_input('Custom stock symbol').upper()
    else:
        ticker = st.selectbox('Choose stock symbol', options=['^GSPC', '^DJI', '^IXIC'], index=0)
with c2:
    st.write('&nbsp;')
    default_date = dt.date.today() - dt.timedelta(days=180)
    start = st.date_input('Show data from', default_date)
with c3:
    st.write('&nbsp;')
    st.write('&nbsp;')
    show_data = st.checkbox('Show data table', False)
    
st.markdown('---')

st.sidebar.subheader('Settings')
st.sidebar.caption('Adjust charts settings and then press apply')

with st.sidebar.form('settings_form'):
    show_nontrading_days = st.checkbox('Show non-trading days', False)
    weekly = st.checkbox('Weekly', False)
    # https://github.com/matplotlib/mplfinance/blob/master/examples/styles.ipynb
    chart_styles = [
        'default', 'binance', 'blueskies', 'brasil', 
        'charles', 'checkers', 'classic', 'yahoo',
        'mike', 'nightclouds', 'sas', 'starsandstripes'
    ]
    marketcolors = [
        'default', 'binance', 'blueskies', 'brasil', 
        'charles', 'checkers', 'classic', 'yahoo',
        'mike', 'nightclouds', 'sas', 'starsandstripes'        
    ]
    chart_style = st.selectbox('Chart style', options=chart_styles, index=chart_styles.index('classic'))
    market_color = st.selectbox('Market colors', options=marketcolors, index=chart_styles.index('yahoo')) 
    mc = mpf.make_marketcolors(base_mpf_style=market_color)
    s  = mpf.make_mpf_style(base_mpf_style=chart_style,marketcolors=mc)
    
    chart_types = [
        'candle', 'ohlc', 'line', 'renko', 'pnf'
    ]
    chart_type = st.selectbox('Chart type', options=chart_types, index=chart_types.index('candle'))

    indicators = [
        'Moving Average','Ichimoku Clouds'
    ]

    indicator = st.selectbox('Indicator', options=indicators, index=indicators.index('Moving Average'))


    st.form_submit_button('Apply')

def get_historical_data(ticker, start = None):
    
    end = dt.date.today()
    start2 = start - dt.timedelta(days=365)
    df = yf.download(ticker, start=start2, end=end)
    
    return df

def get_moving_average(df):
    df['Volume'] = df['Volume']/1000000
    if weekly:
        df['10ma'] = df['Close'].rolling(window=10).mean()
        df['30ma'] = df['Close'].rolling(window=30).mean()
        df['40ma'] = df['Close'].rolling(window=40).mean()
        df["30ma_vol"] = df['Volume'].rolling(window=30).mean()

    else:
        df['10ma'] = df['Adj Close'].rolling(window=10).mean()
        df['21ema'] = df['Adj Close'].ewm(span=21).mean()
        df['50ma'] = df['Adj Close'].rolling(window=50).mean()
        df['150ma'] = df['Adj Close'].rolling(window=150).mean()
        df['200ma'] = df['Adj Close'].rolling(window=200).mean()
        df["50ma_vol"] = df['Volume'].rolling(window=50).mean()        

def ichimoku_cloud(df):
    max_9 = df['High'].rolling(window=9).max()
    min_9 = df['Low'].rolling(window=9).min()

    max_26 = df['High'].rolling(window=26).max()
    min_26 = df['Low'].rolling(window=26).min()

    max_52 = df['High'].rolling(window=52).max()
    min_52 = df['Low'].rolling(window=52).min()

    df['tenkan_sen'] = (max_9 + min_9)/2
    df['kijun_sen'] = (max_26 + min_26)/2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen'])/2).shift(25)
    df['senkou_span_b'] = ((max_52 + min_52)/2).shift(25)
    df['chikou_span'] = df['Close'].shift(-25)   

def plot_data(df, ticker):
    fig, axlist = mpf.plot(
        df,
        title=f'{ticker}, {start}',
        type=chart_type,
        show_nontrading=show_nontrading_days,
        addplot = addplot_ma,
        fill_between = fill_betweens,
        style=s,
        figsize=(20,10),

        returnfig=True,)

    st.pyplot(fig)

def get_fundamentals_quarter(ticker):
    #Code33情報の取得
    #取得するWEBページのURL(zaq)
    url = 'https://www.zacks.com/stock/research/{}/earnings-calendar'.format(ticker)
    data = None

    # HTTPリクエストヘッダにユーザーエージェントを設定
    headers = {'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36'}

    request = urllib.request.Request(url, data, headers)
    response = urllib.request.urlopen(request)
    html = response.read()

    soup = BeautifulSoup(html, "html.parser")

    deal = soup.find_all("script")
    a = str(deal[25])

    b = a.replace('<div class=\\"right pos positive pos_icon showinline up\\">',"")
    c = b.replace('<div class=\\"right neg negative neg_icon showinline down\\">',"")
    d = c.replace('</div>',"")
    e = d.replace(" ","")
    f = e.replace("[","")
    g = f.replace("]","")
    e = g.replace("$","")
    f = e.replace('<divclass=\\"rightpos_nashowinline\\">',"")
    p = f.split("\"")
    earn = p.index('earnings_announcements_sales_table')+8

    PE = [p[i] for i in range(5,p.index('earnings_announcements_sales_table'),14)]
    if len(PE) < 8:
        for i in range(8-len(PE)):
            PE.append('no_data')
    else:
        PE = [p[i] for i in range(5,117,14)]
        

    EPS = []
    if len(range(9,p.index('earnings_announcements_sales_table'),14)) > 7:
        try:
            for i in range(9,121,14):
                EPS.append(float(p[i]))
        except:
            for i in range(8-len(EPS)):
                EPS.append(0)
    else:
        for i in range(9,p.index('earnings_announcements_sales_table'),14):
            try:
                EPS.append(float(p[i]))
            except:
                EPS.append(0)
        for i in range(8-len(EPS)):
            EPS.append(0)
            
    EARN = []
    if len(range(earn,p.index('earnings_announcements_guidance_table'),14)) > 7:
        try:
            for i in range(earn,earn + 112,14):
                EARN.append(float(p[i].replace(',', '')))
        except:
            for i in range(8-len(EARN)):
                EARN.append(0)
    else:
        for i in range(earn,p.index('earnings_announcements_guidance_table'),14):
            try:
                EARN.append(float(p[i].replace(',', '')))
            except:
                EARN.append(0)
        for i in range(8-len(EARN)):
            EARN.append(0)
            
    df_q = pd.DataFrame(data = [EARN], columns = PE,index = ['Revenue'])

    df_q.loc[''] = PE
    try:
        EARNstr = [p[i].replace('.00', '') for i in range(earn,earn + 112,14)]
    except:
        EARNstr = [p[i].replace('.00', '') for i in range(earn,len(p),14)]
    df_q.loc['Revenue(Mil.)'] = EARN

    list_perchg3 = list(df_q.loc['Revenue(Mil.)',:].pct_change(-4))

    list_perchg2 = [f'+{i:.1%}' if i > 0 else f'{i:.1%}' for i in list_perchg3]
    list_perchg = list_perchg2[:4]

    list_perchg.extend([0,0,0,0])
    df_q.loc['%Chg'] = list_perchg

    df_q.loc['EPS'] = EPS

    list_perchgeps3 = list(df_q.loc['EPS',:].pct_change(-4))
    list_perchgeps2 = ["+" + '{:.1f}%'.format((df_q.iloc[4,i] - df_q.iloc[4,i + 4]) * 100 / np.array(abs(df_q.iloc[4,i + 4]))) if (df_q.iloc[4,i] - df_q.iloc[4,i + 4]) * 100 / np.array(abs(df_q.iloc[4,i + 4]))>0
                            else '{:.1f}%'.format((df_q.iloc[4,i] - df_q.iloc[4,i + 4]) * 100 / np.array(abs(df_q.iloc[4,i + 4])))  for i in range(0,4)]
    list_perchgeps = list_perchgeps2[:4]

    list_perchgeps.extend([0,0,0,0])
    df_q.loc['%Chgeps'] = list_perchgeps
    df_q.set_axis(df_q.iloc[0,:], axis='columns')
    df_q2 = df_q.rename(index={'%Chgeps': '%Chg'})

    df_quarter = df_q2.iloc[2:,0:4]

    return df_quarter

def get_fundamentals_annual(ticker):
    #Annualデータの取得
    url = 'https://www.zacks.com/stock/quote/{}/income-statement'.format(ticker)
    # WEBページデータを取得
    headers = {'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36'}
    data = None
    request = urllib.request.Request(url, data, headers)
    response = requests.get(url=url, headers=headers)
    response.encoding = response.apparent_encoding

    # HTMLテーブルを取得
    table = pd.read_html(response.text)
    df_yrev = pd.DataFrame(table[2]) #salesの表

    columns = df_yrev.columns.to_list()[1:]
    columns_new = [columns[i].replace('12/31/','20') for i in range(len(columns))] #12/31/21を2021表示に

    df2_yrev = pd.DataFrame(table[4]) #EPSの表

    df3_yrev = pd.DataFrame(data = [df_yrev.loc[0][1:].to_list(),df2_yrev.loc[1][1:].to_list()] , columns = columns_new)

    df3_yrev.loc[''] = columns_new

    df3_yrev.loc['Revenue(Mil.)'] = df3_yrev.loc[0].astype(float).apply('{:,}'.format)
    list_perchgrev = ["+" + '{:.1f}%'.format((df3_yrev.iloc[0,i] - df3_yrev.iloc[0,i + 1]) * 100 / np.array(abs(df3_yrev.iloc[0,i + 1]))) if (df3_yrev.iloc[0,i] - df3_yrev.iloc[0,i + 1]) * 100 / np.array(abs(df3_yrev.iloc[0,i + 1]))>0
                else '{:.1f}%'.format((df3_yrev.iloc[0,i] - df3_yrev.iloc[0,i + 1]) * 100 / np.array(abs(df3_yrev.iloc[0,i + 1])))  for i in range(0,4)
                ]
    list_perchgrev.extend([0])
    df3_yrev.loc['%Chg'] = list_perchgrev
    df3_yrev.loc['EPS'] = df3_yrev.loc[1]
    list_perchgeps2 = ["+" + '{:.1f}%'.format((df3_yrev.iloc[1,i] - df3_yrev.iloc[1,i + 1]) * 100 / np.array(abs(df3_yrev.iloc[1,i + 1]))) if (df3_yrev.iloc[1,i] - df3_yrev.iloc[1,i + 1]) * 100 / np.array(abs(df3_yrev.iloc[1,i + 1]))>0
                else '{:.1f}%'.format((df3_yrev.iloc[1,i] - df3_yrev.iloc[1,i + 1]) * 100 / np.array(abs(df3_yrev.iloc[1,i + 1])))  for i in range(0,4)
                ]
    list_perchgeps2.extend([0])
    df3_yrev.loc['%Chgeps'] = list_perchgeps2
    df3_yrev2 = df3_yrev.rename(index={'%Chgeps': '%Chg'})
    df_annual = df3_yrev2.iloc[3:,:4]

    return df_annual

    
if ticker:
    df = get_historical_data(ticker, start)
    df2 = df.copy()
    if weekly:
        d_ohlcv = {'Open': 'first',
                   'High': 'max',
                   'Low': 'min',
                   'Close': 'last',
                   'Volume': 'sum'}

        df = df.resample('W-MON', closed='left', label='left').agg(d_ohlcv)
        get_moving_average(df)
        
    #vol_color
        df['up_flag'] = df['Close'].diff() #up_flagは9列
        colors = []
        #'#12B56B'緑, '#FE3E40'赤
        #終値が前日以上
        #かつ出来高が平均以上かつ値幅の上40%以上なら緑
        #それ以外なら薄灰
        #終値が前日以下
        #かつ出来高が平均以上かつ値幅の上40%以下なら赤
        #それ以外なら薄灰

        for i in range(len(df)):
            if df.iloc[i, 9] >= 0: #終値が前日以上
                if all([df.iloc[i, 4] > df.iloc[i, 8],
                         df.iloc[i, 3] >= df.iloc[i, 2] + (df.iloc[i, 1] - df.iloc[i, 2]) * .6]): #値幅の上40%以上なら緑
                    colors.append('#12B56B')
                else:
                    colors.append('lightgrey')
            else: #終値が前日未満
                if all([df.iloc[i, 4] > df.iloc[i, 8], #出来高が平均以上
                         df.iloc[i, 3] < df.iloc[i, 2] + (df.iloc[i, 1] - df.iloc[i, 2]) * .6,]): #値幅の上40%未満なら赤
                    colors.append('#FE3E40')
                else:
                    colors.append('dimgrey')        
                    
        df = df.loc[start:]
        
        period = len(df) * -1
        colors = colors[period:]
        
        addplot_ma = [
                    mpf.make_addplot(df[['10ma','30ma','40ma']],panel=0,width=0.6),
                    mpf.make_addplot(df['Volume'],panel=1,type='bar',ylabel = 'Volume(M)',color=colors),
                    mpf.make_addplot(df["30ma_vol"],width=0.6,panel=1,color='orange',secondary_y=False),
                     ]
        
        fill_dummy = dict(y1 = df['10ma'].values, y2 = df['30ma'].values,
                        where = df['10ma'] >= df['30ma'], alpha = 0, color = 'w')
        fill_betweens = fill_dummy
        
        plot_data(df, ticker)
        
    else:
        get_moving_average(df)

    #vol_color
        df['up_flag'] = df['Adj Close'].diff() #up_flagは12列
        colors = []
        for i in range(1,11):
            if df.loc[df.index[i],'Adj Close'] > df.loc[df.index[i - 1],'Adj Close']:
                colors.append('lightgrey') #通常の緑#12B56B
            else:
                colors.append('dimgrey') #通常の赤#FE3E40

        for i in range(11,len(df)):
            if df.iloc[i, 12] >= 0: #終値が前日以上
                if len(df.iloc[i - 10:i, 5][df['up_flag'] < 0]) == 0: #過去10日に前日より終値が低いときがない
                    colors.append('#12B56B')
                elif df.iloc[i, 5] > max(df.iloc[i - 10:i, 5][df['up_flag'] < 0]): #過去10日の前日より終値が低いときの出来高の最大より今の出来高が高い
                    colors.append('#12B56B')
                else:
                    colors.append('lightgrey')
            elif all([df.iloc[i, 5] > df.iloc[i, 11], #出来高が50日平均以上
                        df.iloc[i, 4] < df.iloc[i, 2] + (df.iloc[i, 1] - df.iloc[i, 2]) * .6,]): #終値が値幅の下60%で引ける
                colors.append('#FE3E40')
            else:
                colors.append('dimgrey')

    #dry-up
        df['dryup'] = df.loc[df['Volume'] < df['50ma_vol'] * 0.55, '50ma_vol']

        ichimoku_cloud(df)

        df = df.loc[start:]
        period = len(df) * -1
        colors = colors[period:]

        if indicator == 'Moving Average':
            addplot_ma = [
                        mpf.make_addplot(df['10ma'],panel=0,width=0.6,color='navy'),
                        mpf.make_addplot(df['21ema'],panel=0,width=0.6,color='deeppink'),
                        mpf.make_addplot(df['50ma'],panel=0,width=0.6,color='orange'),
                        mpf.make_addplot(df['150ma'],panel=0,width=0.6,color='grey'),
                        mpf.make_addplot(df['200ma'],panel=0,width=0.6,color='green'),
                        mpf.make_addplot(df['Volume'],panel=1,type='bar',ylabel = 'Volume(M)',color=colors),
                        mpf.make_addplot(df["50ma_vol"],width=0.6,panel=1,color='orange',secondary_y=False),
                        ]
            fill_dummy = dict(y1 = df['10ma'].values, y2 = df['21ema'].values,
                           where = df['10ma'] >= df['21ema'], alpha = 0, color = 'w')
            fill_betweens = fill_dummy
        elif indicator == 'Ichimoku Clouds':
            addplot_ma = [
                        mpf.make_addplot(df['tenkan_sen'],panel=0, color = 'blue', width = 1),
                        mpf.make_addplot(df['kijun_sen'],panel=0, color = 'red', width = 1),
                        mpf.make_addplot(df['senkou_span_a'],panel=0, color = 'lightgreen', width = 0.5),
                        mpf.make_addplot(df['senkou_span_b'],panel=0, color = 'lightcoral', width = 0.5),
                        mpf.make_addplot(df['chikou_span'],panel=0, color = 'green', width = 1),
                        mpf.make_addplot(df['Volume'],panel=1,type='bar',ylabel = 'Volume(M)',color=colors),
                        mpf.make_addplot(df["50ma_vol"],width=0.6,panel=1,color='orange',secondary_y=False),
            ]
            fill_up = dict(y1 = df['senkou_span_a'].values, y2 = df['senkou_span_b'].values,
                           where = df['senkou_span_a'] >= df['senkou_span_b'], alpha = 0.5, color = 'honeydew', interpolate=True)
            fill_down = dict(y1 = df['senkou_span_a'].values, y2 = df['senkou_span_b'].values,
                             where = df['senkou_span_a'] < df['senkou_span_b'], alpha = 0.5, color = 'mistyrose', interpolate=True)
            fill_betweens = [fill_up, fill_down]


        plot_data(df, ticker)

    if show_data:
        
        st.markdown('---')
        try:
            get_fundamentals_quarter(ticker)
            st.subheader(f'Fundamentals ({ticker})')
            st.write('Quarterly')
            st.dataframe(get_fundamentals_quarter(ticker))
            st.write('Annual')
            st.dataframe(get_fundamentals_annual(ticker))
        except:
            st.subheader(f'Historical Data ({ticker})')
            st.dataframe(df2)



# In[ ]:
