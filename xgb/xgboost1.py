#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy.stats import mode
import csv
import matplotlib.dates
import matplotlib.pyplot as plt
from datetime import *
import json, random, math

from sklearn.preprocessing import *
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.externals import joblib

# 导入数据
def importDf(url, sep=',', na_values='-1', header='infer', index_col=None, colNames=None):
    df = pd.read_csv(url, sep=sep, na_values='-1', header=header, index_col=index_col, names=colNames)
    return df

# 缩放字段至0-1
def scalerFea(df, cols):
    df.dropna(inplace=True, subset=cols)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols].values)
    return df,scaler

# 矩估计法计算贝叶斯平滑参数
def countBetaParamByMME(inputArr):
    EX = inputArr.mean()
    EX2 = (inputArr ** 2).mean()
    alpha = (EX*(EX-EX2)) / (EX2 - EX**2)
    beta = alpha * (1/EX - 1)
    return alpha,beta

# 对numpy数组进行贝叶斯平滑处理
def biasSmooth(aArr, bArr, method='MME', alpha=None, beta=None):
    ratioArr = aArr / bArr
    if method=='MME':
        alpha,beta = countBetaParamByMME(ratioArr[ratioArr==ratioArr])
    resultArr = (aArr+alpha) / (bArr+alpha+beta)
    return resultArr

# 导出预测结果
def exportResult(df, fileName, header=True, index=False, sep=','):
    df.to_csv('./%s' % fileName, sep=sep, header=header, index=index)

# 格式化数据集
def timeTransform(df, col):
    df.loc[:,col] = pd.to_datetime(df[col])
    return df

# 按日期统计过去几天的数目，总和
def statYearMonthLenSum(df, index, values, statLen=None, skipLen=1):
    tempDf = pd.pivot_table(df, index=index, columns='year_month', values=values, aggfunc=[len,np.sum])
    targetDate = pd.to_datetime('2017-05-01')
    if targetDate not in tempDf.columns.levels[-1]:
        tempDf.loc[:, pd.IndexSlice['len',targetDate]] = tempDf.loc[:, pd.IndexSlice['sum',targetDate]] = np.nan
    if statLen==None:
        for i,dt in enumerate(tempDf.columns.levels[-1][skipLen:]):
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,:i+skipLen].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,:i+skipLen].sum(axis=1)
    else:
        for i,dt in enumerate(tempDf.columns.levels[-1][statLen:]):
            tempDf.loc[:,pd.IndexSlice['addup_len',dt]] = tempDf['len'].iloc[:,i:i+statLen].sum(axis=1)
            tempDf.loc[:,pd.IndexSlice['addup_sum',dt]] = tempDf['sum'].iloc[:,i:i+statLen].sum(axis=1)
    tempDf = tempDf.stack()
    return tempDf[['addup_len','addup_sum']]

class FeaFactory():
    def __init__(self, dfs):
        startTime = datetime.now()
        dfs = self.dataFormatter(dfs)

        dfs['order_df']['year'] = dfs['order_df'].o_date.dt.year
        dfs['order_df']['month'] = dfs['order_df'].o_date.dt.month
        dfs['order_df']['day'] = dfs['order_df'].o_date.dt.day
        dfs['order_df']['year_month'] = dfs['order_df'].o_date.dt.year.astype(str) + '-' + dfs['order_df'].o_date.dt.month.astype(str) + '-1'
        dfs['order_df']['year_month'] = pd.to_datetime(dfs['order_df']['year_month'])
        dfs['order_df'] = dfs['order_df'].merge(dfs['sku_df'][['sku_id','cate']], how='left', on='sku_id')
        dfs['order_df']['day_of_month_end'] = pd.Index(dfs['order_df']['year_month']).shift(1,freq='MS')
        dfs['order_df']['day_of_month_end'] = (dfs['order_df']['day_of_month_end'] - dfs['order_df']['o_date']).dt.days
        dfs['action_df']['year'] = dfs['action_df'].a_date.dt.year
        dfs['action_df']['month'] = dfs['action_df'].a_date.dt.month
        dfs['action_df']['day'] = dfs['action_df'].a_date.dt.day
        dfs['action_df']['year_month'] = dfs['action_df'].a_date.dt.year.astype(str) + '-' + dfs['action_df'].a_date.dt.month.astype(str) + '-1'
        dfs['action_df']['year_month'] = pd.to_datetime(dfs['action_df']['year_month'])
        dfs['action_df'] = dfs['action_df'].merge(dfs['sku_df'][['sku_id','cate']], how='left', on='sku_id')
        dfs['action_df']['day_of_month_end'] = pd.Index(dfs['action_df']['year_month']).shift(1,freq='MS')
        dfs['action_df']['day_of_month_end'] = (dfs['action_df']['day_of_month_end'] - dfs['action_df']['a_date']).dt.days

        self.userDf = dfs['user_df']
        self.skuDf = dfs['sku_df']
        self.actionDf = dfs['action_df']
        self.orderDf = dfs['order_df']
        self.commDf = dfs['comm_df']
        print('init fea featory:', datetime.now() - startTime)

    # 数据格式化
    def dataFormatter(self, dfs):
        dfs['action_df'] = timeTransform(dfs['action_df'], 'a_date')
        dfs['order_df'] = timeTransform(dfs['order_df'], 'o_date')
        dfs['comm_df'] = timeTransform(dfs['comm_df'], 'comment_create_tm')
        dfs['user_df'].loc[dfs['user_df'].sex==2, 'sex'] = np.nan
        return dfs

    # 初始化数据集
    def initDf(self, startDate, endDate):
        startTime = datetime.now()
        # 构建index
        userList = self.userDf['user_id'].values
        dateList = pd.date_range(start=startDate, end=endDate, freq='MS')
        tempIdx = pd.MultiIndex.from_product([userList,dateList], names=['user_id', 'year_month'])
        df = pd.DataFrame(index=tempIdx)
        df.reset_index(inplace=True)

        # 剔除历史未出现的用户
        tempDf = pd.pivot_table(self.actionDf, index='user_id', values='a_date', aggfunc=np.min)
        tempDf.columns = ['user_first_date']
        df = df.merge(tempDf, how='left', left_on='user_id', right_index=True)
        tempDf = pd.pivot_table(self.orderDf, index='user_id', values='o_date', aggfunc=np.min)
        tempDf.columns = ['user_first_order']
        df = df.merge(tempDf, how='left', left_on='user_id', right_index=True)
        df.loc[df.user_first_date.isnull(),'user_first_date'] = df.loc[df.user_first_date.isnull(),'user_first_order']
        tempIdx = df[df.year_month<=df.user_first_date].index
        df.drop(tempIdx, inplace=True)

        # 计算数据集label
        skuList = self.skuDf[self.skuDf.cate.isin([101,30])]['sku_id'].values
        tempDf = pd.pivot_table(self.orderDf[self.orderDf.sku_id.isin(skuList)], index=['user_id','year_month'], values=['o_date'], aggfunc=np.min)
        tempDf.columns=['date_label']
        df = df.merge(tempDf, how='left', left_on=['user_id', 'year_month'], right_index=True)
        df['day_label'] = df['date_label'].dt.day
        df['buy_label'] = df['date_label'].notnull().astype(int)
        print('init df:', datetime.now() - startTime)
        return df

    def addUserFea(self, df, **params):
        cateList = list(set(self.skuDf.cate))
        # 用户基础信息
        df = df.merge(self.userDf, how='left', on='user_id')
        df['user_his_month'] = (df['year_month'] - df['user_first_date']).dt.days // 30

        # 历史行为统计
        startTime = datetime.now()
        tempDf = statYearMonthLenSum(self.actionDf, index=['user_id','cate','a_type'], values='a_num', skipLen=params['skipLen'])
        del tempDf['addup_len']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.set_levels(['view','follow'], level='a_type')
        tempDf.columns = ['user_cate%s_his_%s'%(x[0],x[1]) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)

        tempDf = statYearMonthLenSum(self.orderDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='o_sku_num', skipLen=params['skipLen'])
        del tempDf['addup_sum']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_his_order'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        for x in cateList+['all','task','other']:
            df['user_cate%s_his_order_permonth'%x] = df['user_cate%s_his_order'%x] / (df['user_his_month']+1)

        tempDf = statYearMonthLenSum(self.orderDf.drop_duplicates(subset=['user_id','cate','o_date']), index=['user_id','cate'], values='o_sku_num', skipLen=params['skipLen'])
        del tempDf['addup_sum']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_his_orderday'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        for x in cateList+['all','task','other']:
            df['user_cate%s_his_orderday_permonth'%x] = df['user_cate%s_his_orderday'%x] / (df['user_his_month']+1)
        print('user his stat:', datetime.now() - startTime)

        # 过去一个月行为统计
        startTime = datetime.now()
        tempMonth = pd.to_datetime(params['startDate']) - timedelta(days=31)
        tempDf = self.actionDf[self.actionDf.year_month >= tempMonth]
        tempDf = statYearMonthLenSum(tempDf, index=['user_id','cate','a_type'], values='a_num', statLen=1)
        del tempDf['addup_len']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.set_levels(['view','follow'], level='a_type')
        tempDf.columns = ['user_cate%s_lastmonth_%s'%(x[0],x[1]) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)

        tempDf = self.orderDf[self.orderDf.year_month >= tempMonth]
        tempDf = statYearMonthLenSum(tempDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='o_sku_num', statLen=1)
        del tempDf['addup_sum']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_lastmonth_order'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)

        tempDf = self.orderDf[self.orderDf.year_month >= tempMonth]
        tempDf = statYearMonthLenSum(tempDf.drop_duplicates(subset=['user_id','cate','o_date']), index=['user_id','cate'], values='o_sku_num', statLen=1)
        del tempDf['addup_sum']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_lastmonth_orderday'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        print('user last month stat:', datetime.now() - startTime)

        # 过去3个月行为统计
        startTime = datetime.now()
        tempMonth = pd.to_datetime(params['startDate']) - timedelta(days=31*3)
        tempDf = self.actionDf[self.actionDf.year_month >= tempMonth]
        tempDf = statYearMonthLenSum(tempDf, index=['user_id','cate','a_type'], values='a_num', statLen=3)
        del tempDf['addup_len']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.set_levels(['view','follow'], level='a_type')
        tempDf.columns = ['user_cate%s_last3month_%s'%(x[0],x[1]) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)

        tempDf = self.orderDf[self.orderDf.year_month >= tempMonth]
        tempDf = statYearMonthLenSum(tempDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='o_sku_num', statLen=3)
        del tempDf['addup_sum']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_last3month_order'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)

        tempDf = self.orderDf[self.orderDf.year_month >= tempMonth]
        tempDf = statYearMonthLenSum(tempDf.drop_duplicates(subset=['user_id','cate','o_date']), index=['user_id','cate'], values='o_sku_num', statLen=3)
        del tempDf['addup_sum']
        tempDf = tempDf.unstack(level=1)
        tempDf.columns = tempDf.columns.droplevel()
        tempDf['all'] = tempDf.sum(axis=1)
        tempDf['task'] = tempDf[101] + tempDf[30]
        tempDf['other'] = tempDf['all'] - tempDf['task']
        tempDf.columns = ['user_cate%s_last3month_orderday'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        print('user last 3month stat:', datetime.now() - startTime)

        # # 月末1天行为统计
        # startTime = datetime.now()
        # tempDf = self.actionDf[(self.actionDf.year_month >= tempMonth)&(self.actionDf.day_of_month_end <= 1)]
        # tempDf = statYearMonthLenSum(tempDf, index=['user_id','cate','a_type'], values='a_num', statLen=1)
        # del tempDf['addup_len']
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.droplevel()
        # tempDf['all'] = tempDf.sum(axis=1)
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.set_levels(['view','follow'], level='a_type')
        # tempDf.columns = ['user_cate%s_endday1_%s'%(x[0],x[1]) for x in tempDf.columns]
        # df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        # df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        #
        # tempDf = self.orderDf[(self.orderDf.year_month >= tempMonth)&(self.orderDf.day_of_month_end <= 1)]
        # tempDf = statYearMonthLenSum(tempDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='o_sku_num', statLen=1)
        # del tempDf['addup_sum']
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.droplevel()
        # tempDf['all'] = tempDf.sum(axis=1)
        # tempDf.columns = ['user_cate%s_endday1_order'%x for x in tempDf.columns]
        # df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        # df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        # print('user last 1days stat:', datetime.now() - startTime)
        #
        # # 月末3天行为统计
        # startTime = datetime.now()
        # tempDf = self.actionDf[(self.actionDf.year_month >= tempMonth)&(self.actionDf.day_of_month_end <= 3)]
        # tempDf = statYearMonthLenSum(tempDf, index=['user_id','cate','a_type'], values='a_num', statLen=1)
        # del tempDf['addup_len']
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.droplevel()
        # tempDf['all'] = tempDf.sum(axis=1)
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.set_levels(['view','follow'], level='a_type')
        # tempDf.columns = ['user_cate%s_endday3_%s'%(x[0],x[1]) for x in tempDf.columns]
        # df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        # df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        #
        # tempDf = self.orderDf[(self.orderDf.year_month >= tempMonth)&(self.orderDf.day_of_month_end <= 3)]
        # tempDf = statYearMonthLenSum(tempDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='o_sku_num', statLen=1)
        # del tempDf['addup_sum']
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.droplevel()
        # tempDf['all'] = tempDf.sum(axis=1)
        # tempDf.columns = ['user_cate%s_endday3_order'%x for x in tempDf.columns]
        # df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        # df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        # print('user last 3days stat:', datetime.now() - startTime)

        # 月周期性行为统计
        # startTime = datetime.now()
        # tempDf = statYearMonthLenSum(self.actionDf, index=['user_id','cate','a_type'], values='day', skipLen=params['skipLen'])
        # print(tempDf.head())
        # del tempDf['addup_len']
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.droplevel()
        # tempDf['all'] = tempDf.sum(axis=1)
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.set_levels(['view','follow'], level='a_type')
        # tempDf.columns = ['user_cate%s_endday3_%s'%(x[0],x[1]) for x in tempDf.columns]
        # df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        # df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        #
        # tempDf = self.orderDf[(self.orderDf.year_month >= tempMonth)&(self.orderDf.day_of_month_end <= 3)]
        # tempDf = statYearMonthLenSum(tempDf.drop_duplicates(subset=['o_id','cate']), index=['user_id','cate'], values='o_sku_num', skipLen=params['skipLen'])
        # del tempDf['addup_sum']
        # tempDf = tempDf.unstack(level=1)
        # tempDf.columns = tempDf.columns.droplevel()
        # tempDf['all'] = tempDf.sum(axis=1)
        # tempDf.columns = ['user_cate%s_endday3_order'%x for x in tempDf.columns]
        # df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        # df.fillna({k:0 for k in tempDf.columns.values}, inplace=True)
        # print('user period stat:', datetime.now() - startTime)

        # 距离上次行为时间
        startTime = datetime.now()
        tempDf = pd.pivot_table(self.actionDf, index=['year_month'], columns=['user_id','cate','a_type'], values='a_date', aggfunc=np.max)
        tempDf.index = tempDf.index.shift(1,freq='MS')
        tempDf.fillna(method='ffill', inplace=True)
        tempDf = tempDf.stack(level=['user_id','a_type'])
        tempDf['all'] = tempDf.max(axis=1)
        tempDf = tempDf.stack(level=['cate'])
        tempDf = (tempDf.index.get_level_values('year_month').values - tempDf).dt.days
        tempDf.index.set_levels(['view','follow'], level='a_type', inplace=True)
        tempDf = tempDf.unstack(level=['cate','a_type'])
        tempDf.columns = ['user_cate%s_last_%s_timedelta'%(x[0],x[1]) for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['year_month','user_id'], right_index=True)
        df.fillna({k:999 for k in tempDf.columns.values}, inplace=True)

        tempDf = pd.pivot_table(self.orderDf, index=['year_month'], columns=['user_id','cate'], values='o_date', aggfunc=np.max)
        tempDf.index = tempDf.index.shift(1,freq='MS')
        tempDf.fillna(method='ffill', inplace=True)
        tempDf = tempDf.stack(level=['user_id'])
        tempDf['all'] = tempDf.max(axis=1)
        tempDf = tempDf.stack(level=['cate'])
        tempDf = (tempDf.index.get_level_values('year_month').values - tempDf).dt.days
        tempDf = tempDf.unstack(level=['cate'])
        tempDf.columns = ['user_cate%s_last_order_timedelta'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['year_month','user_id'], right_index=True)
        df.fillna({k:999 for k in tempDf.columns.values}, inplace=True)
        for x in cateList+['all']:
            df['user_cate%s_last_order_view_timedelta'%x] = df['user_cate%s_last_order_timedelta'%x] - df['user_cate%s_last_view_timedelta'%x]
            df['user_cate%s_last_order_follow_timedelta'%x] = df['user_cate%s_last_order_timedelta'%x] - df['user_cate%s_last_follow_timedelta'%x]
        print('user last timedelta:', datetime.now() - startTime)

        # 计算用户各类别下单间隔
        startTime = datetime.now()
        orderDf = self.orderDf.drop_duplicates(subset=['o_id','cate']).sort_index(by=['user_id','cate','o_date'])
        orderDf['last_user'] = orderDf['user_id'].shift(1) == orderDf['user_id']
        orderDf['last_cate'] = orderDf['cate'].shift(1) == orderDf['cate']
        orderDf['last_o_date'] = orderDf['o_date'].shift(1)
        orderDf['last_cate_o_daydelta'] = (orderDf['o_date'] - orderDf['last_o_date']).dt.days
        orderDf.loc[~(orderDf.last_user&orderDf.last_cate), 'last_cate_o_daydelta'] = np.nan
        self.orderDf = self.orderDf.merge(orderDf[['o_id','last_cate_o_daydelta']], how='left')
        tempDf = pd.pivot_table(orderDf, index=['user_id'], columns='cate', values='last_cate_o_daydelta', aggfunc=np.mean)
        cateDayDelta = tempDf.apply(lambda x: (x[x>0]//1).value_counts().index[0])
        tempDf = statYearMonthLenSum(orderDf, ['user_id','cate'], 'last_cate_o_daydelta')
        tempDf['addup_mean'] = tempDf['addup_sum'] / (tempDf['addup_len'] - 1)
        tempDf.loc[tempDf.addup_mean==0,'addup_mean'] = np.nan
        tempDf.drop(['addup_len','addup_sum'], axis=1, inplace=True)
        tempDf = tempDf.unstack(level='cate')
        tempDf.columns = tempDf.columns.droplevel()
        tempDf.columns = ['user_cate%s_order_daydelta_mean'%x for x in tempDf.columns]
        df = df.merge(tempDf, how='left', left_on=['user_id','year_month'], right_index=True)
        print('user cate daydelta mean:', datetime.now() - startTime)
        # 根据平均间隔推算各类目下次购买时间
        startTime = datetime.now()
        for x in set(self.skuDf.cate):
            temp = df['user_cate%s_order_daydelta_mean'%x].fillna(cateDayDelta[x])
            df['cate%s_next_order_pred'%x] = temp - df['user_cate%s_last_order_timedelta'%x]
        print('user next cate order predict:', datetime.now() - startTime)
        return df

    def getFeaDf(self, startDate='2016-08-01', endDate='2017-05-01'):
        params = {
            'startDate': startDate,
            'endDate': endDate,
            'skipLen': (int(startDate[5:7]) - 5) % 12
            }
        df = self.initDf(startDate, endDate)
        df = self.addUserFea(df, **params)
        return df

class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            # 'objective': 'reg:linear',
            # 'eval_metric':['rmse','auc'],
            'objective': 'binary:logistic',
            'eval_metric':'auc',
            'silent': True,
            'eta': 0.1,
            # 'max_depth': 4,
            # 'gamma': 10,
            # 'subsample': 0.95,
            # 'colsample_bytree': 1,
            # 'min_child_weight': 9,
            'scale_pos_weight': 1.2,
            # 'lambda': 250,
            # 'nthread': 15,
            }
        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
        X = X.astype(float)
        if train_size==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feaNames)
        dval = xgb.DMatrix(X_test, label=y_test, feature_names=self.feaNames)
        watchlist = [(dtrain,'train'),(dval,'val')]
        clf = xgb.train(
            self.params, dtrain,
            num_boost_round = num_boost_round,
            evals = watchlist,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        self.clf = clf

    def trainCV(self, X, y, nFold=3, verbose=True, num_boost_round=1500, early_stopping_rounds=30, weight=None):
        X = X.astype(float)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        if weight!=None:
            dtrain.set_weight(weight)
        cvResult = xgb.cv(
            self.params, dtrain,
            num_boost_round = num_boost_round,
            nfold = nFold,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        clf = xgb.train(
            self.params, dtrain,
            num_boost_round = cvResult.shape[0],
        )
        self.clf = clf

    def gridSearch(self, X, y, nFold=3, verbose=1, num_boost_round=150):
        paramsGrids = {
            # 'n_estimators': [50+5*i for i in range(0,30)],
            'gamma': [0,0.05,0.1,0.5,1,5,10,50,100],
            'max_depth': list(range(3,8)),
            'min_child_weight': list(range(0,10)),
            'subsample': [1-0.05*i for i in range(0,6)],
            # 'colsample_bytree': [1-0.05*i for i in range(0,6)],
            # 'reg_alpha': [0+2*i for i in range(0,10)],
            'reg_lambda': [0,5,10,15,20,30,50,70,100,150,200,250,300,500],
            'scale_pos_weight': [1+0.2*i for i in range(10)],
            # 'max_delta_step': [0+1*i for i in range(0,8)],
        }
        for k,v in paramsGrids.items():
            gsearch = GridSearchCV(
                estimator = xgb.XGBClassifier(
                    max_depth = self.params['max_depth'],
                    gamma = self.params['gamma'],
                    learning_rate = self.params['eta'],
                    # max_delta_step = self.params['max_delta_step'],
                    min_child_weight = self.params['min_child_weight'],
                    subsample = self.params['subsample'],
                    colsample_bytree = self.params['colsample_bytree'],
                    scale_pos_weight = self.params['scale_pos_weight'],
                    silent = self.params['silent'],
                    reg_lambda = self.params['lambda'],
                    n_estimators = num_boost_round,
                ),
                # param_grid = paramsGrids,
                param_grid = {k:v},
                scoring = 'roc_auc',
                cv = nFold,
                verbose = verbose,
                n_jobs = 8
            )
            gsearch.fit(X, y)
            print(pd.DataFrame(gsearch.cv_results_))
            print(gsearch.best_params_)
        exit()

    def predict(self, X):
        X = X.astype(float)
        return self.clf.predict(xgb.DMatrix(X, feature_names=self.feaNames))

    def getFeaScore(self, show=False):
        fscore = self.clf.get_score()
        feaNames = fscore.keys()
        scoreDf = pd.DataFrame(index=feaNames, columns=['importance'])
        for k,v in fscore.items():
            scoreDf.loc[k, 'importance'] = v
        if show:
            print(scoreDf.sort_index(by=['importance'], ascending=False))
        return scoreDf

# 获取stacking下一层数据集
def getOof(clf, trainX, trainY, testX, nFold=5, stratify=False):
    oofTrain = np.zeros(trainX.shape[0])
    oofTest = np.zeros(testX.shape[0])
    oofTestSkf = np.zeros((testX.shape[0], nFold))
    if stratify:
        kf = StratifiedKFold(n_splits=nFold, shuffle=True)
    else:
        kf = KFold(n_splits=nFold, shuffle=True)
    for i, (trainIdx, testIdx) in enumerate(kf.split(trainX, trainY)):
        kfTrainX = trainX[trainIdx]
        kfTrainY = trainY[trainIdx]
        kfTestX = trainX[testIdx]
        clf.trainCV(kfTrainX, kfTrainY, verbose=False)
        oofTrain[testIdx] = clf.predict(kfTestX)
        oofTestSkf[:,i] = clf.predict(testX)
    oofTest[:] = oofTestSkf.mean(axis=1)
    return oofTrain, oofTest

def getRankDf(df, rankCol='buy_predict'):
    resultDf = df.sort_index(by=[rankCol], ascending=False).iloc[:50000]
    return resultDf

def score1(labelList):
    weightList = [1/(1+math.log(i)) for i in range(1,len(labelList)+1)]
    s1 = np.sum(np.array(weightList) * np.array(labelList))
    s1 /= np.sum(weightList)
    return s1

def score2(labelList, predList):
    fu = [10 / (10 + (x1-x2)**2) for x1,x2 in np.column_stack([predList,labelList]) if x2==x2]
    userLen = len(labelList[labelList==labelList])
    s2 = np.sum(fu) / userLen
    return s2

# print(score2(np.array([1,2,3,4,5]), np.array([2,3,np.nan,6,3])))

def main():
    # 数据导入
    startTime = datetime.now()
    userDf = importDf('./data/jdata_user_basic_info.csv')
    skuDf = importDf('./data/jdata_sku_basic_info.csv')
    actionDf = importDf('./data/jdata_user_action.csv')
    orderDf = importDf('./data/jdata_user_order.csv')
    commDf = importDf('./data/jdata_user_comment_score.csv')
    # userList = userDf.sample(frac=0.5)['user_id'].values
    # dfs = {
    #     'user_df': userDf[userDf.user_id.isin(userList)],
    #     'sku_df': skuDf,
    #     'action_df': actionDf[actionDf.user_id.isin(userList)],
    #     'order_df': orderDf[orderDf.user_id.isin(userList)],
    #     'comm_df': commDf[commDf.user_id.isin(userList)]
    #     }
    dfs = {
        'user_df': userDf,
        'sku_df': skuDf,
        'action_df': actionDf,
        'order_df': orderDf,
        'comm_df': commDf
        }
    print('import dataset:', datetime.now() - startTime)

    # 特征工程
    feaFactory = FeaFactory(dfs)
    df = feaFactory.getFeaDf()
    print('train user num:', df.year_month.value_counts())
    fea = [
        'age','sex','user_lv_cd','user_his_month',
        ]
    cateList = list(set(skuDf.cate))
    fea.extend(['user_cate%s_his_view'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_his_follow'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_his_order'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_his_order_permonth'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_his_orderday'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_his_orderday_permonth'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_lastmonth_view'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_lastmonth_follow'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_lastmonth_order'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_lastmonth_orderday'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_last3month_view'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_last3month_follow'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_last3month_order'%x for x in cateList+['all','task','other']])
    fea.extend(['user_cate%s_last3month_orderday'%x for x in cateList+['all','task','other']])
    # fea.extend(['user_cate%s_endday1_view'%x for x in cateList+['all']])
    # fea.extend(['user_cate%s_endday1_follow'%x for x in cateList+['all']])
    # fea.extend(['user_cate%s_endday1_order'%x for x in cateList+['all']])
    # fea.extend(['user_cate%s_endday3_view'%x for x in cateList+['all']])
    # fea.extend(['user_cate%s_endday3_follow'%x for x in cateList+['all']])
    # fea.extend(['user_cate%s_endday3_order'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_view_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_follow_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_view_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_last_order_follow_timedelta'%x for x in cateList+['all']])
    fea.extend(['user_cate%s_order_daydelta_mean'%x for x in cateList])
    fea.extend(['cate%s_next_order_pred'%x for x in cateList])
    print(df[fea].info())
    # exit()

    # 模型构建
    buyModel = XgbModel(feaNames=fea)
    dateParams = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        }
    dateModel = XgbModel(feaNames=fea, params=dateParams)
    feaScoreDf = pd.DataFrame(index=fea)
    costDf = pd.DataFrame(index=['auc','score1','rmse','score2','score'])
    for dt in pd.date_range(start='2017-02-01', end='2017-04-01', freq='MS'):
        trainDf = df[(df.year_month<dt)&(df.year_month>=dt-timedelta(days=31))]
        # trainDf = df[(df.year_month<dt)]
        testDf = df[df.year_month==dt]
        print('train num:', len(trainDf))
        # buyModel.gridSearch(trainDf[fea].values, trainDf['buy_label'].values)
        buyModel.trainCV(trainDf[fea].values, trainDf['buy_label'].values)
        testDf.loc[:,'buy_predict'] = buyModel.predict(testDf[fea].values)
        # trainDf.loc[:,'buy_predict'],testDf.loc[:,'buy_predict'] = getOof(buyModel, trainDf[fea].values, trainDf['buy_label'].values, testDf[fea].values, stratify=True)
        scoreDf = buyModel.getFeaScore()
        scoreDf.columns = [dt.strftime('%Y-%m-%d')+'_buy']
        feaScoreDf = feaScoreDf.merge(scoreDf, how='left', left_index=True, right_index=True)
        auc = metrics.roc_auc_score(testDf['buy_label'].values, testDf['buy_predict'].values)
        costDf.loc['auc',dt.strftime('%Y-%m-%d')] = auc
        costDf.loc['score1',dt.strftime('%Y-%m-%d')] = score1(getRankDf(testDf, 'buy_predict')['buy_label'].values)
        # costDf.loc['oof_cost',dt.strftime('%Y-%m-%d')] = metrics.log_loss(testDf['is_trade'].values, testDf['oof_predict'].values)

        # hasBuyIdx = df[df.date_label.notnull()].index
        dateModel.trainCV(trainDf[trainDf.day_label.notnull()][fea].values, trainDf[trainDf.day_label.notnull()]['day_label'].values)
        testDf.loc[:,'day_predict'] = dateModel.predict(testDf[fea].values)
        # trainDf.loc[:,'buy_predict'],testDf.loc[:,'buy_predict'] = getOof(dateModel, trainDf[fea].values, trainDf['buy_label'].values, testDf[fea].values, stratify=True)
        scoreDf = dateModel.getFeaScore()
        scoreDf.columns = [dt.strftime('%Y-%m-%d')+'_date']
        feaScoreDf = feaScoreDf.merge(scoreDf, how='left', left_index=True, right_index=True)
        rmse = metrics.mean_squared_error(testDf[testDf.day_label.notnull()]['day_label'].values, testDf[testDf.day_label.notnull()]['day_predict'].values)
        costDf.loc['rmse',dt.strftime('%Y-%m-%d')] = rmse
        resultDf = getRankDf(testDf, 'buy_predict')
        print(len(resultDf[resultDf.day_predict<1]), len(resultDf[resultDf.day_predict>31]))
        print(resultDf[(resultDf.day_predict<1)|(resultDf.day_predict>31)][['user_id','buy_predict','day_predict','day_label']])
        costDf.loc['score2',dt.strftime('%Y-%m-%d')] = score2(resultDf['day_label'].values, resultDf['day_predict'].values)
        costDf.loc['score',dt.strftime('%Y-%m-%d')] = 0.4*costDf.loc['score1',dt.strftime('%Y-%m-%d')] + 0.6*costDf.loc['score2',dt.strftime('%Y-%m-%d')]
        # costDf.loc['oof_cost',dt.strftime('%Y-%m-%d')] = metrics.log_loss(testDf['is_trade'].values, testDf['oof_predict'].values)
    print(feaScoreDf.iloc[:60], feaScoreDf.iloc[60:])
    print(costDf)
    exit()

    # 正式模型
    modelName = 'buyModel1_single_month'
    trainDf = df[df.year_month<date(2017,5,1)]
    print(len(trainDf))
    testDf = df[df.year_month==date(2017,5,1)]
    buyModel.trainCV(trainDf[fea].values, trainDf['buy_label'].values)
    buyModel.getFeaScore(show=True)
    testDf.loc[:,'buy_predict'] = buyModel.predict(testDf[fea].values)
    dateModel.trainCV(trainDf[trainDf.day_label.notnull()][fea].values, trainDf[trainDf.day_label.notnull()]['day_label'].values)
    dateModel.getFeaScore(show=True)
    testDf.loc[:,'day_predict'] = dateModel.predict(testDf[fea].values)
    print('invalid:', resultDf[(resultDf.day_predict<1)|(resultDf.day_predict>31)][['user_id','buy_predict','day_predict']])
    resultDf = getRankDf(testDf[(testDf.day_predict>0)&(testDf.day_predict<32)], rankCol='buy_predict')
    resultDf['pred_date'] = resultDf['day_predict'].map(lambda x: '2017-05-%02d'%x)
    print(resultDf.head(20)[['user_id','buy_predict','day_predict']])
    print(resultDf['pred_date'].value_counts())

    # 导出模型
    exportResult(resultDf[['user_id','pred_date']], '%s.csv'%modelName)

if __name__ == '__main__':
    main()
