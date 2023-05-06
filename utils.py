import re
import time
from sqlalchemy import create_engine

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import akshare as ak
import baostock as bs

import logging
logging.getLogger('cmdstanpy').disabled=True
import warnings
warnings.filterwarnings('ignore')


"""
命名：驼峰转下划线
"""
def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

"""
兼容nan的加权平均
"""
def nan_wm(vec, weights=None):
    if weights is None:
        return np.nanmean(vec)
    else:
        return np.nansum(np.multiply(vec,weights))/np.sum(np.multiply(weights,~np.isnan(vec)))


"""
获取指数的成份股
    -- by akshare & baostock
"""
def get_index_components(index_code='399102'):
    stock_list = ak.index_stock_cons(index_code)["品种代码"].apply(ak.stock_a_code_to_symbol).apply(lambda s:s[:2]+'.'+s[2:]).values
    return np.unique(stock_list)


"""
获取股票列表的历史K线
    -- by baostock
"""
def get_history_k(stock_list, stt_date, end_date, freq='d', adjust='1'):
    bs.login()
    output_list = []
    len_ = len(stock_list)
    t0 = time.time()
    for i, stk in enumerate(stock_list):
        rs = bs.query_history_k_data_plus(stk, "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
                start_date=stt_date, end_date=end_date,
                frequency=freq, adjustflag=adjust)
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        output_list.append(pd.DataFrame(data_list, columns=rs.fields))
    df_output = pd.concat(output_list)
    # 格式转换
    df_output['date'] = df_output['date'].astype('datetime64[ns]')
    for col in np.setdiff1d(df_output.columns, ['code', 'date']):
        df_output[col] = pd.to_numeric(df_output[col])
    # 字段名转换
    df_output.columns = df_output.columns.map(camel_to_snake)
    bs.logout()
    return df_output


"""
获取指定股票的表数据
    -- by tencent cloud 110.40.223.131
"""
def get_remote_table(table, stk_tup, date_stt, date_end, code_col="code", date_col="date"):
    # 初始化数据库连接，使用pymysql模块
    user = "lostlau"
    pwd = "Liuwn_0717"
    ip = "110.40.223.131"
    port = "3306"
    db_name = "db_quant"
    engine = create_engine(f"mysql+pymysql://{user}:{pwd}@{ip}:{port}/{db_name}")

    return pd.read_sql(f"select * from {table} where {code_col} in {stk_tup} and {date_col}>='{date_stt}' and {date_col}<='{date_end}'", con=engine)



"""
依赖func
"""
## 生成财务季度的增量经营性cashflow的依赖func
def functmp_cfo_quarter(df):
    return df['net_profit']*df['cfo_to_np'] - (df['net_profit']*df['cfo_to_np']).shift(1).fillna(0)

## 生成每股净收益的未来一年预测值的依赖func
def functmp_eps_ttm_forecast(vec):
    df_prophet = pd.DataFrame({"ds":np.arange(np.datetime64("2022-01-01"),np.datetime64("2022-01-01")+np.timedelta64(len(vec),'D')),
                               "y":vec})
    m = Prophet(uncertainty_samples=False)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=4, include_history=False)
    forecast = m.predict(future)['yhat']
    return forecast.mean()

## 生成过去3年盈利增长率的依赖func
def functmp_e_growth(df):
    return df['eps_ttm'].rolling(12,min_periods=4).apply(lambda vec:np.polyfit(x=np.arange(len(vec)),y=vec,deg=1)[0],raw=True)/df['eps_ttm'].rolling(12,min_periods=4).mean().abs()

## 生成过去3年营收增长率的依赖func
def functmp_s_growth(df):
    df['mbr_q'] = (df['mb_revenue']/df['stat_quarter']).fillna(method='ffill').fillna(method='bfill')
    return df['mbr_q'].rolling(12,min_periods=4).apply(lambda vec:np.polyfit(x=np.arange(len(vec)),y=vec,deg=1)[0],raw=True)/df['mbr_q'].rolling(12,min_periods=4).mean().abs()

## 生成momentum的依赖func
def functmp_mtm(df,shift_d,ewm_d,halflife,min_p):
    W_tmp = np.exp(-np.log(2)/halflife*np.arange(ewm_d)[::-1])
    return df['pct_chg'].shift(shift_d).rolling(ewm_d,min_periods=min_p).apply(lambda vec:nan_wm(vec, W_tmp[-len(vec):]),raw=True)

## 生成日收益标准差的依赖func
def functmp_daily_sd(df,ewm_d,halflife,min_p):
    W_tmp = np.exp(-np.log(2)/halflife*np.arange(ewm_d)[::-1])
    return (df['pct_chg']-df['pct_chg'].mean()).pow(2).rolling(ewm_d,min_periods=min_p).apply(lambda vec:nan_wm(vec, W_tmp[-len(vec):]),raw=True).pow(0.5)

## 生成离差历史波动率的依赖func
def functmp_cum_range(df):
    return df['pct_chg'].rolling(252,min_periods=126).apply(lambda vec:vec[-(len(vec)//21*21):].reshape(21,-1).mean(axis=1).max()-vec[-(len(vec)//21*21):].reshape(21,-1).mean(axis=1).min(),raw=True)

## 生成size_nonlin的依赖func
def functmp_size_nonlin(df):
    size_expo = ((df['size']-df['size'].mean())/df['size'].std()).fillna(0)
    size_cube = size_expo**3
    reg = LinearRegression().fit(size_expo.values.reshape(-1,1),size_cube.values)
    return reg.predict(size_expo.values.reshape(-1,1))-size_cube

## 生成平均月换手率的依赖func
def functmp_stom_avg(df,period,min_p):
    return np.log(df['turn'].rolling(21*period,min_periods=int(21*min_p)).apply(nan_wm)*21)
