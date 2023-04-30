import time
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine
import pymysql

import pandas as pd
import numpy as np
import baostock as bs

from utils import *

import logging
logging.getLogger('cmdstanpy').disabled=True
import warnings
warnings.filterwarnings('ignore')


class factory(object):
    def __init__(self, df_dk, df_fi, today, overwrite_begin, cols_keep):
        self.today = today                      ## 今天
        self.overwrite_begin = overwrite_begin  ## 复写的起始日
        self.cols_keep = cols_keep              ## 最终需要留下的cols
        self.df_dk = df_dk                      ## daily_k 数据
        self.df_fi = df_fi                      ## finance_indicator 数据
        self.df_merge = None                    ## dk和fi的合并，也作为df_factors的中间备份
        self.df_factors = None                  ## 最终因子表

    """
    df_dk的预处理
    """

    def dk_prec(self):
        self.df_dk = self.df_dk.sort_values('date').reset_index(drop=True)
        self.df_dk['pct_chg'] = self.df_dk['pct_chg'].fillna(0)

    """
    df_fi的预处理
    """

    def fi_prec(self):
        self.df_fi = self.df_fi.sort_values('stat_date').reset_index(drop=True)
        ## 生成统计年和季度
        self.df_fi['stat_year'] = self.df_fi['stat_date'].dt.year
        self.df_fi['stat_quarter'] = self.df_fi['stat_date'].dt.quarter
        ## 生成财务季度的增量经营性cashflow
        self.df_fi['cfo_quarter'] = self.df_fi.groupby(['code', 'stat_year']).apply(functmp_cfo_quarter).reset_index(
            level=['code', 'stat_year'], drop=True)
        ## 生成按年滚动的年度经营性cashflow
        self.df_fi['cfo_annual_rolling'] = self.df_fi.groupby("code").apply(
            lambda df: df['cfo_quarter'].rolling(4, min_periods=1).mean() * 4).reset_index(level=['code'], drop=True)
        ## 生成每股净收益的未来一年预测值（采取prophet）
        self.df_fi['eps_ttm_forecast'] = self.df_fi.groupby("code").apply(
            lambda df: df['eps_ttm'].rolling(8, min_periods=4).apply(functmp_eps_ttm_forecast, raw=True)).reset_index(
            level=['code'], drop=True)
        ## 生成过去3年盈利增长率
        self.df_fi['e_growth'] = self.df_fi.groupby("code").apply(functmp_e_growth).reset_index(level=['code'],
                                                                                                drop=True)
        ## 生成过去3年营收增长率
        self.df_fi['s_growth'] = self.df_fi.groupby("code").apply(functmp_s_growth).reset_index(level=['code'],
                                                                                                drop=True)

    """
    merge dk和fi
    """

    def merge(self):
        self.df_merge = pd.merge_asof(self.df_dk.drop(columns=['id', 'create_time']),
                                      self.df_fi.drop(columns=['id', 'create_time']),
                                      left_on="date", right_on="stat_date", by="code", direction="backward")
        self.df_factors = self.df_merge.copy()

    """ 
    生成size
    """

    def gen_size(self):
        self.df_factors['size'] = np.log(self.df_factors['total_share'] * self.df_factors['close'])

    """
    生成beta
    """

    def gen_beta(self):
        l_ratio = np.log(2) / 61

        date_min = self.df_merge['date'].dt.strftime("%Y-%m-%d").min()
        date_max = self.df_merge['date'].dt.strftime("%Y-%m-%d").max()

        df_ashare = get_history_k(['sh.000002'], date_min, date_max).sort_values('date')
        df_self = get_history_k(['sz.399102'], date_min, date_max).sort_values('date')

        df_beta_list = []
        # len_flush = len(
        #     self.df_merge.query(f"date>='{self.overwrite_begin}' and date<='{self.today}'")['date'].unique())
        for i, thisday in enumerate(
                self.df_merge.query(f"date>='{self.overwrite_begin}' and date<='{self.today}'")['date'].dt.strftime(
                        "%Y-%m-%d").unique()):
            # print("\r", end="")
            # print(f"已完成: {100 * (1 + i) / len_flush:>0.2f}%", "▋" * (((i + 1) * 100) // len_flush), end="")
            # print(f"剩余: {(time.time() - t0) / (i + 1) * (len_flush - i - 1) / 60:>0.2f}mins", end="")
            # sys.stdout.flush()
            date_startfrom = (datetime.strptime(thisday, "%Y-%m-%d") - timedelta(365)).strftime("%Y-%m-%d")

            X_ashare = df_ashare.query(f"date>='{date_startfrom}' and date<='{thisday}'")['pct_chg'].values.reshape(-1,
                                                                                                                    1)
            X_self = df_self.query(f"date>='{date_startfrom}' and date<='{thisday}'")['pct_chg'].values.reshape(-1, 1)

            len_ = X_self.shape[0]
            groupby_ = self.df_merge.query(f"date<='{thisday}' and date>'{date_startfrom}'").sort_values('date')[
                ['code', 'pct_chg']].groupby('code')
            stocks = [grp for (grp, df_tmp) in groupby_]
            Y = np.array([np.pad(df_tmp['pct_chg'].fillna(0).values, (len_ - df_tmp.shape[0], 0)) for (grp, df_tmp) in
                          self.df_merge.query(f"date<='{thisday}' and date>'{date_startfrom}'").sort_values('date')[
                              ['code', 'pct_chg']].groupby('code')]).T
            W = np.exp(-l_ratio * np.arange(len_)[::-1])

            reg_ashare = LinearRegression(fit_intercept=False).fit(X_ashare, Y, sample_weight=W)
            reg_self = LinearRegression(fit_intercept=False).fit(X_self, Y, sample_weight=W)

            beta_ashare = reg_ashare.coef_.reshape(-1)
            beta_self = reg_self.coef_.reshape(-1)
            hsigma_ashare = (reg_ashare.predict(X_ashare) - Y).std(axis=0)
            hsigma_self = (reg_self.predict(X_self) - Y).std(axis=0)

            df_beta_tmp = pd.DataFrame({"code": stocks, "beta_ashare": beta_ashare, "beta_self": beta_self,
                                        "hsigma_ashare": hsigma_ashare, "hsigma_self": hsigma_self})
            df_beta_tmp['date'] = np.datetime64(thisday)
            df_beta_list.append(df_beta_tmp)

        df_beta = pd.concat(df_beta_list, ignore_index=True)

        self.df_factors = pd.merge(self.df_factors, df_beta, on=['date', 'code'], how='left')

    """
    生成momentum
    """

    def gen_mtm(self):
        self.df_factors['momentum_long'] = self.df_factors.groupby("code").apply(functmp_mtm, shift_d=20, ewm_d=504,
                                                                                 halflife=126, min_p=21).reset_index(
            level='code', drop=True).sort_index().values
        self.df_factors['momentum_short'] = self.df_factors.groupby("code").apply(functmp_mtm, shift_d=0, ewm_d=21,
                                                                                  halflife=7, min_p=21).reset_index(
            level='code', drop=True).sort_index().values

    """
    生成residual_volatility
    """

    def gen_resvol(self):
        self.df_factors['daily_sd'] = self.df_factors.groupby("code").apply(functmp_daily_sd, ewm_d=252, halflife=42,
                                                                            min_p=126).reset_index(level='code',
                                                                                                   drop=True).sort_index().values
        self.df_factors['cum_range'] = self.df_factors.groupby("code").apply(functmp_cum_range).reset_index(
            level='code', drop=True).sort_index().values
        self.df_factors['res_vol_ashare'] = (
                    0.74 * self.df_factors['daily_sd'] + 0.16 * self.df_factors['cum_range'] + 0.1 * self.df_factors[
                'hsigma_ashare'])
        self.df_factors['res_vol_self'] = (
                    0.74 * self.df_factors['daily_sd'] + 0.16 * self.df_factors['cum_range'] + 0.1 * self.df_factors[
                'hsigma_self'])

    """
    生成non_linear_size
    """

    def gen_sizenonlin(self):
        self.df_factors['size_nonlin'] = self.df_factors.groupby('date').apply(functmp_size_nonlin).reset_index(
            level='date', drop=True).sort_index().values

    """
    生成book_to_price_ratio
    """

    def gen_bpvalue(self):
        self.df_factors['bp_value'] = (1 / self.df_factors['pb_mrq']).clip(upper=10)

    """
    生成liquidity
    """

    def gen_liquidity(self):
        self.df_factors['stom'] = self.df_factors.groupby('code').apply(functmp_stom_avg, period=1,
                                                                        min_p=10 / 21).reset_index(level='code',
                                                                                                   drop=True).sort_index().values
        self.df_factors['stom_q_avg'] = self.df_factors.groupby('code').apply(functmp_stom_avg, period=3,
                                                                              min_p=1).reset_index(level='code',
                                                                                                   drop=True).sort_index().values
        self.df_factors['stom_a_avg'] = self.df_factors.groupby('code').apply(functmp_stom_avg, period=12,
                                                                              min_p=6).reset_index(level='code',
                                                                                                   drop=True).sort_index().values
        self.df_factors['liquidity'] = 0.35 * self.df_factors['stom'] + 0.35 * self.df_factors['stom_q_avg'] + 0.3 * \
                                       self.df_factors['stom_a_avg']

    """
    生成earnings_yield
    """

    def gen_earnyield(self):
        self.df_factors['pe_to_p'] = self.df_factors['eps_ttm_forecast'] / self.df_factors['close']
        self.df_factors['ce_to_p'] = self.df_factors['cfo_annual_rolling'] / np.exp(self.df_factors['size'])
        self.df_factors['ep'] = 1 / self.df_factors['pe_ttm']
        self.df_factors['earnings_yield'] = 0.68 * self.df_factors['pe_to_p'] + 0.21 * self.df_factors[
            'ce_to_p'] + 0.11 * self.df_factors['ep']

    """
    生成growth
    """

    def gen_growth(self):
        self.df_factors['pe_growth'] = self.df_factors['eps_ttm_forecast'] / self.df_factors['eps_ttm'] - 1
        self.df_factors['growth'] = 0.29 * self.df_factors['pe_growth'] + 0.24 * self.df_factors['e_growth'] + 0.47 * \
                                    self.df_factors['s_growth']

    """
    生成leverage
    """

    def gen_leverage(self):
        self.df_factors['leverage'] = self.df_factors['dupont_asset_sto_equity']

    """
    大一统
    """
    def transform(self):
        self.dk_prec()
        self.fi_prec()
        self.merge()
        self.gen_size()
        self.gen_beta()
        self.gen_mtm()
        self.gen_resvol()
        self.gen_sizenonlin()
        self.gen_bpvalue()
        self.gen_liquidity()
        self.gen_earnyield()
        self.gen_growth()
        self.gen_leverage()
        return self.df_factors.query(f"date>='{self.overwrite_begin}' and date<='{self.today}'")[self.cols_keep]




def job():
    user = "lostlau"
    pwd = "Liuwn_0717"
    ip = "110.40.223.131"
    port = "3306"
    db_name = "db_quant"

    cols_keep = [
        'code', 'date', 'pub_date', 'stat_date', 'stat_year', 'stat_quarter',
        'daily_sd', 'cum_range', 'hsigma_ashare', 'hsigma_self', 'stom', 'stom_q_avg', 'stom_a_avg',
        'eps_ttm_forecast', 'pe_to_p', 'cfo_quarter', 'cfo_annual_rolling', 'ce_to_p', 'ep',
        'pe_growth', 'e_growth', 's_growth',
        'size', 'beta_ashare', 'beta_self', 'momentum_long', 'momentum_short', 'res_vol_ashare', 'res_vol_self',
        'size_nonlin', 'bp_value', 'liquidity', 'earnings_yield', 'growth', 'leverage'
    ]

    overwrite_days = 183
    lookback_years_dk = 2
    lookback_years_fi = 3
    today = date.today().strftime('%Y-%m-%d')
    print(f"{today}")
    overwrite_begin = (date.today() - timedelta(days=overwrite_days)).strftime('%Y-%m-%d')
    d_stt_dk = (date.today() - timedelta(days=int(lookback_years_dk*365+overwrite_days+1))).strftime('%Y-%m-%d')
    d_stt_fi = (date.today() - timedelta(days=int(lookback_years_fi*365+overwrite_days+1))).strftime('%Y-%m-%d')

    ## 取数
    t0 = time.time()
    bs.login()
    stock_list = get_index_components(index_code='399102')
    df_dk = get_remote_table("stock_daily_k", tuple(stock_list), d_stt_dk, today)
    df_fi = get_remote_table("stock_finance_indicator", tuple(stock_list), d_stt_fi, today, date_col='stat_date')
    engine = create_engine(f"mysql+pymysql://{user}:{pwd}@{ip}:{port}/{db_name}")
    df_uid = pd.read_sql(f"select code, date, stat_date from stock_factors where date>='{overwrite_begin}'", con=engine)
    bs.logout()
    print(f"-- 取数{(time.time()-t0)/60:>0.2f}mins")

    ## 加工生成啰！
    t0=time.time()
    ff = factory(df_dk,df_fi,today,overwrite_begin,cols_keep)
    df_output = ff.transform()
    print(f"-- 加工{(time.time()-t0)/60:>0.2f}mins")

    ## 筛选需要写进数据库的数据（减轻复合）
    t0=time.time()
    df_filter = pd.merge(df_output[['code', 'date', 'stat_date']],
                         df_uid.rename(columns={"stat_date": "stat_date_old"}),
                         on=['code', 'date'], how='left')
    df_filter_final = df_filter.query("stat_date!=stat_date_old")[['code', 'date']]
    df_output_final = pd.merge(df_filter_final, df_output, on=['code', 'date'], how='left')
    df_output_final = df_output_final.astype(object).where(df_output_final.notnull(), None)
    print(f"-- 筛选{(time.time()-t0)/60:>0.2f}mins，最终需复写行数：{df_output_final.shape[0]}")

    ## 回写入数据库
    conn = pymysql.connect(
        host=ip,
        user=user,
        password=pwd,
        db=db_name
    )
    cursor = conn.cursor()
    sql_query = """
        INSERT INTO stock_factors (
            code, `date`, pub_date, stat_date, stat_year, stat_quarter, daily_sd, cum_range, hsigma_ashare, hsigma_self,
            stom, stom_q_avg, stom_a_avg, eps_ttm_forecast, pe_to_p, cfo_quarter, cfo_annual_rolling, ce_to_p, ep, 
            pe_growth, e_growth, s_growth, size, beta_ashare, beta_self, momentum_long, momentum_short, res_vol_ashare,
            res_vol_self, size_nonlin, bp_value, liquidity, earnings_yield, growth, leverage
        ) 
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE 
            code=VALUES(code), 
            `date`=VALUES(`date`), 
            pub_date=VALUES(pub_date), 
            stat_date=VALUES(stat_date),
            stat_year=VALUES(stat_year),
            stat_quarter=VALUES(stat_quarter),
            daily_sd=VALUES(daily_sd),
            cum_range=VALUES(cum_range),
            hsigma_ashare=VALUES(hsigma_ashare),
            hsigma_self=VALUES(hsigma_self),
            stom=VALUES(stom),
            stom_q_avg=VALUES(stom_q_avg),
            stom_a_avg=VALUES(stom_a_avg),
            eps_ttm_forecast=VALUES(eps_ttm_forecast),
            pe_to_p=VALUES(pe_to_p),
            cfo_quarter=VALUES(cfo_quarter),
            cfo_annual_rolling=VALUES(cfo_annual_rolling),
            ce_to_p=VALUES(ce_to_p),
            ep=VALUES(ep),
            pe_growth=VALUES(pe_growth),
            e_growth=VALUES(e_growth),
            s_growth=VALUES(s_growth),
            size=VALUES(size),
            beta_ashare=VALUES(beta_ashare),
            beta_self=VALUES(beta_self),
            momentum_long=VALUES(momentum_long),
            momentum_short=VALUES(momentum_short),
            res_vol_ashare=VALUES(res_vol_ashare),
            res_vol_self=VALUES(res_vol_self),
            size_nonlin=VALUES(size_nonlin),
            bp_value=VALUES(bp_value),
            liquidity=VALUES(liquidity),
            earnings_yield=VALUES(earnings_yield),
            growth=VALUES(growth),
            leverage=VALUES(leverage)
    """
    cursor.executemany(sql_query, df_output_final.values.tolist())
    conn.commit()
    conn.close()
    print(f"-- 写入数据库：{(time.time()-t0)/60:>0.2f}mins")
