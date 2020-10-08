"""
In-Home Install Future Work Force - Service Line Orders Forecasting

This script conatins the common reusable functions

"""

import pandas as pd
import os
import logging
import numpy as np
import pickle
import shutil


def get_date(fisc_calender, start, end, num_test):
    """
    Function to get fiscal calender date corresponding to testing and validation dates based on the current date and number of weeks of lag.
    Input:
        fisc_calender: Pandas Dataframe - Fiscal Calendar Data
        start: int - Start week
        end: int - End week
        num_test: int - Number of weeks to go back
    Output:
        pred_start: int - Pred start
        pred_end: int - Pred end
        mod_start: int - Model start
        fb_val: int - Fb validation start
        lasso_val: int - Lasso validation start
        summ_end: int - summary end
        summ_start: int - summary start
        lst_mth: int - last month start
        lst_qtr: int - last qaurter start
        lst_yr_start: int - last eyar start
    """
    
    fiscal_calendar = fisc_calender[['FISC_WK_OF_MTH_ID']].drop_duplicates()
    fiscal_calendar = fiscal_calendar.drop_duplicates().sort_values('FISC_WK_OF_MTH_ID', ascending = True).reset_index(drop = True)
    pred_end_ix = fiscal_calendar.index[fiscal_calendar['FISC_WK_OF_MTH_ID'] == end][0]
    pred_start_ix = pred_end_ix - num_test + 1 - 4
    pred_start = fiscal_calendar.iloc[pred_start_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    pred_end = fiscal_calendar.iloc[pred_end_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    start_ix = fiscal_calendar.index[fiscal_calendar['FISC_WK_OF_MTH_ID'] == start][0]
    mod_start_ix = start_ix + 0
    mod_start = fiscal_calendar.iloc[mod_start_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    fb_val_ix = pred_start_ix - 35
    fb_val = fiscal_calendar.iloc[fb_val_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    lasso_val_ix = pred_start_ix - 13
    lasso_val = fiscal_calendar.iloc[lasso_val_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    summ_end_ix = pred_start_ix + 16
    summ_end = fiscal_calendar.iloc[summ_end_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    summ_start_ix = summ_end_ix - 51
    summ_start = fiscal_calendar.iloc[summ_start_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    lst_mth_ix = summ_end_ix - 3
    lst_mth = fiscal_calendar.iloc[lst_mth_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    lst_qtr_ix = summ_end_ix - 12
    lst_qtr = fiscal_calendar.iloc[lst_qtr_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    lst_yr_start_ix = summ_start_ix - 52
    lst_yr_start = fiscal_calendar.iloc[lst_yr_start_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    start_ix = fiscal_calendar.index[fiscal_calendar['FISC_WK_OF_MTH_ID'] == pred_start][0]
    start_ix = start_ix - 118
    start_rule = fiscal_calendar.iloc[start_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    
    return pred_start, pred_end, mod_start, fb_val, lasso_val, summ_end, summ_start, lst_mth, lst_qtr, lst_yr_start, start_rule


def score_get_date(base_dir, lkp_dir, fiscal_params, pred_start, num_pred, act_end):
    """
    Function to get fiscal calendar date corresponding to testing and validation dates based on the current date and number of weeks of lag.
    Input:
        base_dir: str - Base directory
        lkp_dir: str - Lookup directory
        fiscal_params: Pandas dataframe - Fiscal calender parameters
        pred_start: int - Pred start
        num_pred: int - Number of weeks for prediction
        act_end: int - Actual end week
    Output:
        pred_end: int - Pred end
        start: int - Data start
        pred_start_new: int - Pred start where actual is not present
    """
    fisc_calender_all = pd.read_csv(os.path.join(base_dir, lkp_dir, fiscal_params['filename']))
    fiscal_calendar = fisc_calender_all[["FISC_WK_OF_MTH_ID", "FISC_YR_NBR"]].drop_duplicates().sort_values('FISC_WK_OF_MTH_ID', ascending = True).reset_index(drop = True)
    pred_start_ix = fiscal_calendar.index[fiscal_calendar['FISC_WK_OF_MTH_ID'] == pred_start][0]
    pred_end_ix = pred_start_ix + num_pred-1
    pred_end = fiscal_calendar.iloc[pred_end_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    start_ix = fiscal_calendar.index[fiscal_calendar['FISC_WK_OF_MTH_ID'] == pred_start][0]
    start_ix = start_ix - 118
    start = fiscal_calendar.iloc[start_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    pred_start_new_ix = fiscal_calendar.index[fiscal_calendar.FISC_WK_OF_MTH_ID == act_end][0]+1
    pred_start_new = fiscal_calendar.iloc[pred_start_new_ix]['FISC_WK_OF_MTH_ID'].astype(int)
    return pred_end, start, pred_start_new



def gen_prv_yr_obs(tt, agg_col, level):
    """
    Utility to create previous year same week variable
    Input:
        tt: Pandas dataframe - Model data
        agg_col: DV columns name
        level: Week or Month
    Output:
        tt: Pandas dataframe - Model data with previous year week
    """
    if level == "wk":
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_WK_OF_YR_NBR'])
        tt["prev_yr_wk"] = tt.groupby(['FISC_WK_OF_YR_NBR'])[agg_col].shift()
    elif level == "mth":
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_MTH_NBR'])
        tttt = tt.groupby(['FISC_YR_NBR', 'FISC_MTH_NBR']).mean()[agg_col].reset_index()
        tttt["prev_yr_mth_avg"] = tttt[agg_col].shift(12)
        tt = pd.merge(tt, tttt[['FISC_YR_NBR', 'FISC_MTH_NBR', "prev_yr_mth_avg"]] , how = "left")
    return tt


def gen_prv_obs(tt, agg_col, obs, level, pred_start):
    """
    Utility to create previous week variable
    Input:
        tt: Pandas dataframe - Model data
        agg_col: DV column name
        obs: Previous week count
        level: Week or Month
        pred_start: Train test split Fiscal week
    Output:
        tt: Pandas dataframe - Model data with previous week variable
    """
    if level == "wk":
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_WK_OF_YR_NBR'])
    elif level == "mth":
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_MTH_NBR'])
    col_name = f'prev_{obs}_{level}'
    tt_train = tt[(tt.FISC_WK_OF_MTH_ID < pred_start)].copy()
    tt_test = tt[((tt.FISC_WK_OF_MTH_ID >= pred_start) | (tt.FISC_WK_OF_MTH_ID < pred_start))].copy()
    tt_train[col_name] = tt_train[agg_col].shift(obs)
    tt_test[col_name] = tt_test[agg_col].shift(obs)
    tt_test = tt_test[(tt_test.FISC_WK_OF_MTH_ID >= pred_start)]
    tt = pd.concat([tt_train, tt_test])
    return tt


def gen_rolling_feats(tt, window, agg_col, level, pred_start):
    """
    Utility to create previous rolling week variable
    Input:
        tt: Pandas dataframe - Model data
        window: Rolling week count
        agg_col: DV column name
        level: Week or Month
        pred_start: Train test split Fiscal week
    Output:
        tt: Pandas dataframe - Model data with previous rolling week variable
    """
    if level == "wk":
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_WK_OF_YR_NBR'])
    elif level == "mth":
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_MTH_NBR'])
    col_name = f'prev_{window}_{level}_avg'
    tt_train = tt[(tt.FISC_WK_OF_MTH_ID < pred_start)].copy()
    tt_test = tt[((tt.FISC_WK_OF_MTH_ID >= pred_start) | (tt.FISC_WK_OF_MTH_ID < pred_start))].copy()
    tt_train[col_name] = tt_train[agg_col].shift().rolling(min_periods=1, window=window).mean()
    tt_test[col_name] = tt_test[agg_col].shift().rolling(min_periods=1, window=window).mean()
    tt_test = tt_test[(tt_test.FISC_WK_OF_MTH_ID >= pred_start)]
    tt = pd.concat([tt_train, tt_test])
    return tt


def merge_cal_tt(tt, cal_df, level):
    """
    Utility to merge model data with calendar data
    Input:
        tt: Pandas dataframe - Model data
        cal_df: Pandas dataframe - Calendar data
        level: Week
    Output:
        tt: Pandas dataframe - Model data merged with calendar dataframe
    """
    tt = pd.merge(cal_df, tt, how='left')
    if level == "wk":
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_WK_OF_YR_NBR'])
    
    return tt


def gen_prv_yr_var(tt):
    """
    Utility to create previous year rolling week variable
    Input:
        tt: Pandas dataframe - Model data
    Output:
        tt: Pandas dataframe - Model data with previous year rolling week variable
    """
    level = "wk"
    for window in [2, 4]:
        col_name = f'prev_{window}_{level}_avg'
        tt = tt.sort_values(['FISC_YR_NBR', 'FISC_WK_OF_YR_NBR'])
        tt["prev_yr_"+ col_name] = tt.groupby(['FISC_WK_OF_YR_NBR'])[col_name].shift()
    return tt


def gen_velocity_var(data, agg_col, pred_start):
    """
    Utility to create velocity variable
    Input:
        data: Pandas dataframe - Model data
        agg_col: DV column name
        pred_start: Train test split Fiscal week
    Output:
        tt: Pandas dataframe - Model data with velocity variable
    """
    data["vr_prev_1_2_wk"] = data["prev_1_wk"]/data["prev_2_wk"]
    data["vr_prev_12_34_wk"] = (data["prev_1_wk"].values + data["prev_2_wk"].values)/(data["prev_3_wk"].values + data["prev_4_wk"].values)
    mean_2_weeks = data[["prev_2_wk", "prev_3_wk"]].mean(axis=1)
    data["vr_2_wk"] = data["prev_1_wk"]/ mean_2_weeks
    mean_3_weeks = data[["prev_2_wk", "prev_3_wk", "prev_4_wk"]].mean(axis=1)
    data["vr_3_wk"] = data["prev_1_wk"]/ mean_3_weeks
    tt_train = data[(data.FISC_WK_OF_MTH_ID < pred_start)].copy()
    tt_test = data[((data.FISC_WK_OF_MTH_ID >= pred_start) | (data.FISC_WK_OF_MTH_ID < pred_start))].copy()
    tt_train["vr_prev_4_8_ravg_wk"] = tt_train[agg_col].shift().rolling(min_periods=1, window=4).mean()/tt_train[agg_col].shift().rolling(min_periods=1, window=8).mean()
    tt_train["vr_pre_months"] = tt_train[agg_col].shift().rolling(min_periods=1, window=4).mean()/tt_train[agg_col].shift(5).rolling(min_periods=1, window=4).mean()
    tt_train["prev_4_wk_min"] = tt_train[agg_col].shift().rolling(min_periods=1, window=4).min()
    tt_train["prev_4_wk_max"] = tt_train[agg_col].shift().rolling(min_periods=1, window=4).max()
    tt_test["vr_prev_4_8_ravg_wk"] = tt_test[agg_col].shift().rolling(min_periods=1, window=4).mean()/tt_test[agg_col].shift().rolling(min_periods=1, window=8).mean()
    tt_test["vr_pre_months"] = tt_test[agg_col].shift().rolling(min_periods=1, window=4).mean()/tt_test[agg_col].shift(5).rolling(min_periods=1, window=4).mean()
    tt_test["prev_4_wk_min"] = tt_test[agg_col].shift().rolling(min_periods=1, window=4).min()
    tt_test["prev_4_wk_max"] = tt_test[agg_col].shift().rolling(min_periods=1, window=4).max()
    tt_test = tt_test[(tt_test.FISC_WK_OF_MTH_ID >= pred_start)]
    tt = pd.concat([tt_train, tt_test])
    return tt


def gen_prv_yr_wk(tt, prop_col, pred_col, level, prop_type, fisc_calender, pred_start):
    """
    Utility to create previous year same week variable for all the Minor Markets
    Input:
        tt: Pandas dataframe - Model data
        prop_col: List - Minor market list
        pred_col: DV columns name
        level: Week or Month
        prop_type: MM Proportion type
        fisc_calender: Pandas dataframe - Calendar data
        pred_start: Train test split Fiscal week
    Output:
        tt_values: Pandas dataframe - Model data with previous year week variable
    """
    prop_col_str = [str(s) for s in prop_col]
    tt = tt.sort_values(level)
    tt = pd.merge(fisc_calender, tt, left_on = "LY_FISC_WK_OF_MTH_ID", right_on = "FISC_WK_OF_MTH_ID", how = "left")
    tt.columns = tt.columns.str.replace("_x", "")
    tt_values = tt.copy()
    tt_values = fillna_values(tt_values.copy(), pred_start)
    tt_values[prop_col_str] = tt_values[prop_col_str].fillna(method='ffill')
    tt_values = tt_values[level + list(prop_col_str)]
    tt_values.rename(columns=dict(zip(prop_col_str, [s + prop_type for s in prop_col_str])),inplace=True)
    return tt_values


def gen_prv_prop(tt_all, prop_col, pred_col, level, window, prop_type, pred_start):
    """
    Utility to create previous year variable for all the Minor Markets
    Input:
        tt_all: Pandas dataframe - Model data
        prop_col: List - Minor market list
        pred_col: DV columns name
        level: Week or Month
        window: Weeks shift
        prop_type: MM Proportion type
        pred_start: Train test split Fiscal week
    Output:
        tt: Pandas dataframe - Model data with previous year variable
    """
    tt = tt_all.copy()
    if window == 1:
        tt = tt[tt.FISC_WK_OF_MTH_ID < pred_start]
        tt_ts = tt_all[tt_all.FISC_WK_OF_MTH_ID >= pred_start]
        tt = tt.groupby(level).sum().reset_index()
        tt_ts = tt_ts.groupby(level).sum().reset_index()
        yr_list = tt["FISC_YR_NBR"].unique()
        tt = pd.concat([tt, tt_ts[~tt_ts.FISC_YR_NBR.isin(yr_list)]], axis = 0)
    else:
        tt = tt.groupby(level).sum().reset_index()
    prop_col_str = [str(s) for s in prop_col]
    tt = tt.sort_values(level)
    tt[prop_col_str] = tt[prop_col_str].shift(window)
    tt[prop_col_str] = tt[prop_col_str].fillna(method='ffill')
    tt[prop_col_str] = tt[prop_col_str].fillna(method='bfill')
    tt = tt[level + list(prop_col_str)]
    tt.rename(columns=dict(zip(prop_col_str, [s + prop_type for s in prop_col_str])),inplace=True)
    return tt


def gen_prv_roll_prop_ls_yr(tt, prop_col, pred_col, level, window, prop_type, sf, fisc_calender, pred_start):
    """
    Utility to create previous year rolling weeks variable for all the Minor Markets
    Input:
        tt: Pandas dataframe - Model data
        prop_col: List - Minor market list
        pred_col: DV columns name
        level: Week or Month
        window: Rolling window
        prop_type: MM Proportion type
        sf : Weeks shift
        fisc_calender: Pandas dataframe - Fiscal calendar
        pred_start: Train test split Fiscal week
    Output:
        tt_values: Pandas dataframe - Model data with previous year rolling week variable
    """
    prop_col_str = [str(s) for s in prop_col]
    tt = tt.sort_values(level)
    tt[prop_col_str] = tt[prop_col_str].rolling(min_periods = 1, window = window).sum().shift(sf)
    tt[pred_col] = tt[pred_col].rolling(min_periods = 1, window = window).sum().shift(sf)
    tt_values = tt.copy()
    tt_values = pd.merge(fisc_calender, tt_values, left_on = "LY_FISC_WK_OF_MTH_ID", right_on = "FISC_WK_OF_MTH_ID", how = "left")
    tt_values.columns = tt_values.columns.str.replace("_x", "")
    tt_values = fillna_values(tt_values.copy(), pred_start)
    tt_values[prop_col_str] = tt_values[prop_col_str].fillna(method='ffill')
    tt_values = tt_values[level + list(prop_col_str)]
    tt_values.rename(columns=dict(zip(prop_col_str, [s + prop_type for s in prop_col_str])),inplace=True)
    return tt_values


def prev_avg_scoring(mm_list, train_df, test_df, prop_type):
    """
    Utility to create moving avergae variable
    Input:
        mm_list: List - Minor market
        train_df: Pandas dataframe - Train data
        test_df: Pandas dataframe - Train data
        prop_type: Proportion type
    Output:
        train_df_all: Pandas dataframe - Train & test data with moving avergae variable
    """
    train_df_all = train_df.copy()
    train_df_all[mm_list] = train_df_all[mm_list].shift(5).rolling(min_periods=1, window=5).mean()
    train_df_all[mm_list] = train_df_all[mm_list].fillna(method='ffill')
    train_df_all[mm_list] = train_df_all[mm_list].fillna(method='bfill')
    train_df_prop = train_df.copy()
    for i in range(0, test_df.shape[0]):
        train_df_prop = train_df.append(test_df.iloc[i])
        train_df_prop[mm_list] = train_df_prop[mm_list].shift(5).rolling(min_periods=1, window=5).mean()
        train_df = train_df.append(train_df_prop.iloc[-1])
    train_df_all = train_df_all.append(train_df.tail(n=test_df.shape[0]))
    train_df_all[mm_list] = train_df_all[mm_list].fillna(method='ffill')
    train_df_all[mm_list] = train_df_all[mm_list].fillna(method='bfill')
    train_df_all = train_df_all[["FISC_WK_OF_MTH_ID"]+mm_list]
    train_df_all.rename(columns=dict(zip(mm_list, [s + prop_type for s in mm_list])),inplace=True)
    return train_df_all


def prev_mth_pv_mth_lyr(tt, prop_col, pred_col, level, prop_type, pred_start, cal_df):
    """
    Utility to create (month-1/last year same month -1)*last year same month value
    Input:
        tt: Pandas dataframe - Model data
        prop_col: List - Minor market list
        pred_col: DV columns name
        level: Week or Month
        prop_type: MM Proportion type
        pred_start: Train test split Fiscal week
    Output:
        model_data_pivot_mth_rf: Pandas dataframe - Model data with previous year week variable
    """
    prop_col_str = [str(s) for s in prop_col]
    tt = tt.sort_values(level)
    model_data_pivot_mth = tt.groupby(level).sum()[prop_col_str].reset_index()
    model_data_pivot_mth[prop_col_str] = model_data_pivot_mth[prop_col_str].replace({0 : np.nan})
    model_data_pivot_mth_r1 = model_data_pivot_mth.copy()
    model_data_pivot_mth_r2 = model_data_pivot_mth.copy()
    model_data_pivot_mth_r3 = model_data_pivot_mth.copy()
    model_data_pivot_mth_rf = model_data_pivot_mth.copy()
    model_data_pivot_mth_r1[prop_col_str] = model_data_pivot_mth_r1[prop_col_str].shift(2)
    model_data_pivot_mth_r2[prop_col_str] = model_data_pivot_mth_r2[prop_col_str].shift(14)
    model_data_pivot_mth_r3[prop_col_str] = model_data_pivot_mth_r3[prop_col_str].shift(12)
    model_data_pivot_mth_r1 = merge_cal_tt(model_data_pivot_mth_r1, cal_df, "wk")
    model_data_pivot_mth_r2 = merge_cal_tt(model_data_pivot_mth_r2, cal_df, "wk")
    model_data_pivot_mth_r3 = merge_cal_tt(model_data_pivot_mth_r3, cal_df, "wk")
    model_data_pivot_mth_r1 = fillna_values(model_data_pivot_mth_r1.copy(), pred_start)
    model_data_pivot_mth_r2 = fillna_values(model_data_pivot_mth_r2.copy(), pred_start)
    model_data_pivot_mth_r3 = fillna_values(model_data_pivot_mth_r3.copy(), pred_start)
    model_data_pivot_mth_r1[prop_col_str] = model_data_pivot_mth_r1[prop_col_str].fillna(method='ffill')
    model_data_pivot_mth_r2[prop_col_str] = model_data_pivot_mth_r2[prop_col_str].fillna(method='ffill')
    model_data_pivot_mth_r3[prop_col_str] = model_data_pivot_mth_r3[prop_col_str].fillna(method='ffill')
    model_data_pivot_mth_rf[prop_col_str] = model_data_pivot_mth_r3[prop_col_str]*(model_data_pivot_mth_r1[prop_col_str]/model_data_pivot_mth_r2[prop_col_str])
    model_data_pivot_mth_rf = model_data_pivot_mth_rf[level + list(prop_col_str)]
    model_data_pivot_mth_rf.rename(columns=dict(zip(prop_col_str, [s + prop_type for s in prop_col_str])),inplace=True)
    return model_data_pivot_mth_rf


def train_test_stats(train_data, test_data, agg_col):
    """
    Utility to find train and test data stats
    Input:
        train_data: Pandas dataframe - Train data
        test_data: Pandas dataframe - Test data
        agg_col: DV column name
    Output:
        stats_tr: Pandas dataframe - Train data stats
        stats_ts: Pandas dataframe - Test data stats
    """
    # Collecting stats on train & test data
    stats_tr = train_data[agg_col].describe().to_dict()
    stats_tr = {your_key: stats_tr[your_key] for your_key in ["mean", "std"]}
    stats_tr["std_2plus"] = train_data[(train_data[agg_col] > (stats_tr["mean"]+(2*stats_tr["std"]))) | (train_data[agg_col] < (stats_tr["mean"]-(2*stats_tr["std"])))].shape[0]
    for k in list(stats_tr.keys()):
        stats_tr[k + "_train"] = stats_tr.pop(k)
    stats_ts = test_data[agg_col].describe().to_dict()
    stats_ts = {your_key: stats_ts[your_key] for your_key in ["mean", "std"]}
    stats_ts["std_2plus"] = test_data[(test_data[agg_col] > (stats_ts["mean"]+(2*stats_ts["std"]))) | (test_data[agg_col] < (stats_ts["mean"]-(2*stats_ts["std"])))].shape[0]
    for k in list(stats_ts.keys()):
        stats_ts[k + "_test"] = stats_ts.pop(k)
    return stats_tr, stats_ts


def fillna_values(data_df, pred_start):
    """
    Utility to fill na
    Input:
        data_df: Pandas dataframe - Train Test data
        pred_start: Train Test split
    Output:
        model_data_df: Pandas dataframe - Train Test data
    """
    train = data_df[(data_df.FISC_WK_OF_MTH_ID < pred_start) ]
    test = data_df[(data_df.FISC_WK_OF_MTH_ID >= pred_start) ]
    train = train.fillna(method='ffill')
    train = train.fillna(method='bfill')
    model_data_df = pd.concat([train, test], axis = 0, sort = "False")
    return model_data_df


def read_pkl(file):
    """
    Utility to read pickle file
    Input:
        file: Name of the file
    Output:
       Data loaded from pickle file
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def res_return(file_name):
    """
    Utility to parse pickle object
    Input:
        file_name: File name
    Output:
        results_f_df: Pandas dataframe - Results summary
        train_all: Pandas dataframe - Train data
        test_all: Pandas dataframe - Test data
        feat_imp_f_all_df: Pandas dataframe - Feature importance
    """
    if os.path.exists(file_name):
        pkl_files = os.listdir(file_name)

        results = []
        for file in pkl_files:
            try:
                results.append(read_pkl(os.path.join(file_name, file)))
            except:
                pass
        
        results_list_f = []
        train_list_f = []
        test_list_f = []
        err_loc_list = []
        feat_imp_f_all = []

        for thread_res in results:
            results_list_f.extend(thread_res[0])
            train_list_f.extend(thread_res[1])
            test_list_f.extend(thread_res[2])
            err_loc_list.extend(thread_res[3])
            try:
                feat_imp_f_all.append(thread_res[4])
            except:
                pass

        results_f_df = pd.DataFrame(results_list_f)
        train_all = pd.concat(train_list_f)
        test_all = pd.concat(test_list_f)
        feat_imp_f_all = sum(feat_imp_f_all, [])
        try:
            feat_imp_f_all_df = pd.concat(feat_imp_f_all)
        except:
            feat_imp_f_all_df = 0

    else:
        print("Pickle Object folder doesn't exists.")
    return results_f_df, train_all, test_all, feat_imp_f_all_df


def my_custom_logger(base_dir, log_path, log_filename, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(log_filename+ ".log")
    logger.setLevel(level)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(os.path.join(base_dir, log_path, log_filename+ ".log"), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

        
def calc_mape(true, preds):
    """
    Utility to calculate MAPE & MAD metrics
    Input:
        true: Pandas dataframe - Actuals
        preds: Pandas dataframe - Predictions
    Output:
        mape_val: MAPE metric
        mean_abs_error: MAD metric
        median_abs_error: Median deviation metric
        wmape: Weighted MAPE metric
    """
    true.reset_index(drop=True, inplace=True)
    preds.reset_index(drop=True, inplace=True)
    mape_val = ((((true - preds)/true)*100).abs()).mean()
    eror = (true-preds)
    abs_eror = eror.abs()
    mean_abs_error = abs_eror.mean()
    median_abs_error = abs_eror.median()
    act_sum = true.sum()
    eror_sum = abs_eror.sum()
    wmape = (eror_sum/act_sum)*100
    return mape_val, mean_abs_error, median_abs_error, wmape


def lag_variable_melt(ntnl_pred_col, model_data_sales, var_list):
    """
    Utility to melt dataframe which has proportion variables by Week X MM
    Input:
        ntnl_pred_col: DV column
        model_data_sales: Pandas dataframe - Train Test data with proportion variables
        var_list: List - Minor market
    Output:
        model_data_sales_melt: Pandas dataframe - Melted dataframe which has proportion variables by Week X MM
    """
    model_data_sales_melt = pd.melt(model_data_sales, id_vars=['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_QTR_NBR'], 
            value_vars=model_data_sales.columns[model_data_sales.columns.str.contains("salesMAPP")])
    model_data_sales_melt["Variable_Type"] = model_data_sales_melt.variable.str.replace(".*salesMAPP_", "")
    model_data_sales_melt["variable"] = model_data_sales_melt.variable.str.replace("_salesMAPP_.*", "")
    model_data_sales_melt = model_data_sales_melt[model_data_sales_melt.Variable_Type.isin(var_list)]
    model_data_sales_melt.columns = ['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_QTR_NBR',
           'variable', ntnl_pred_col+'_VALUE', 'Variable_Type']
    return model_data_sales_melt


def mape_stability(df_sample, comm, grp, agg1, agg2, df_sample_list):
    """
    Utiltiy to generate model stability summary
    Input:
        df_sample: Pandas dataframe - Model results
        comm: Identifier column
        grp: Category columns
        agg1: Metric 1
        agg2: Metric 2
        df_sample_list: List - Result
    Output:
        df_sample_list: List - Result
    """
    mape = df_sample[agg1].mean()
    df_sample["WEIGHTED MAPE"] = df_sample[agg2]/df_sample[agg2].sum()
    df_sample["WEIGHTED MAPE"] = df_sample["WEIGHTED MAPE"]*df_sample[agg1]
    weigths = df_sample["WEIGHTED MAPE"].sum()
    df_sample = df_sample.groupby(grp).agg({agg1:"mean", agg2:"sum"}).reset_index()
    df_sample[agg2+"_PER"] = df_sample[agg2]/df_sample[agg2].sum()
    df_sample = df_sample.drop(agg2, axis = 1)
    df_sample["IDENTIFIER"] = comm
    for col in [agg2+"_PER", agg1]:
        df_sample[agg1].fillna(0, inplace = True)
        df_sample_col = df_sample.pivot_table([col], 
                                              ['IDENTIFIER'], 
                                                  grp)
        if len(df_sample_col) > 0:
            df_sample_col.columns = pd.Index(['_'.join([str(_) for _ in v]) for v in df_sample_col.columns])
            cols_new = list(df_sample_col.columns.str.replace(col+"_", ""))
            df_sample_col.columns = cols_new
            df_sample_col["MAPE"] = mape
            df_sample_col["WEIGHTED MAPE"] = weigths
            df_sample_col["METRIC"] = col
            df_sample_list.append(df_sample_col)
    return df_sample_list


def cov_summary(df_cov, comm, df_summary):
    """
    Utiltiy to generate covariance summary on Line orders
    Input:
        df_cov: Pandas dataframe - FPG X MM by Covariance buckets
        comm: Metric column
        df_summary: List
    Output:
        df_summary: List - Result
    """
    sumary_grp_mm = df_cov[df_cov.CAT>0].groupby(["AVG_BIN", "COV_BIN"]).agg({"LO_SUM":"sum", comm: "count"}).reset_index()
    for col in ["LO_SUM", comm]:
        sumary_grp_mm_pivot = sumary_grp_mm.pivot_table([col], 
                                          ['COV_BIN'], 
                                          "AVG_BIN")
        sumary_grp_mm_pivot = sumary_grp_mm_pivot/sumary_grp_mm[col].sum()    
        sumary_grp_mm_pivot.columns = pd.Index(['_'.join([str(_) for _ in v]) for v in sumary_grp_mm_pivot.columns])
        cols_new = list(sumary_grp_mm_pivot.columns.str.replace(col+"_", ""))
        sumary_grp_mm_pivot.columns = cols_new
        sumary_grp_mm_pivot["SUMMARY_COL"] = col
        sumary_grp_mm_pivot["IDENTIFIER"] = comm
        df_summary.append(sumary_grp_mm_pivot)
    return df_summary


def mape_stability_2d(df_sample, col1, col2):
    """
    Utiltiy to generate model stability summary by two dimensions
    Input:
        df_sample: Pandas dataframe - Model results
        col1: Metric 1
        col2: Metric 2
    Output:
        df_sample_res: Pandas dataframe - Result
    """
    mape = df_sample.groupby(col1)["MAPE"].mean().reset_index()
    lo_sum = df_sample.groupby(col1)["LINE_ORDERS_ACT"].sum().reset_index()
    lo_sum.columns = [col1, "LINE_ORDERS_ACT_SUM"]
    df_sample = pd.merge(df_sample, lo_sum)
    df_sample["WEIGHTED MAPE"] = df_sample["LINE_ORDERS_ACT"]/df_sample["LINE_ORDERS_ACT_SUM"]
    df_sample["WEIGHTED MAPE"] = df_sample["WEIGHTED MAPE"]*df_sample["MAPE"]
    weigths = df_sample.groupby(col1)["WEIGHTED MAPE"].sum().reset_index()
    df_sample = df_sample.groupby([col1, col2]).agg({"MAPE":"mean", "LINE_ORDERS_ACT":"sum"}).reset_index()
    df_sample_mth = df_sample.groupby([col1]).agg({"LINE_ORDERS_ACT":"sum"}).reset_index()
    df_sample_mth.columns = [col1, 'LINE_ORDERS_ACT_MTH']
    df_sample = pd.merge(df_sample, df_sample_mth)
    df_sample["LINE_ORDERS_ACT_PER"] = df_sample.LINE_ORDERS_ACT/df_sample.LINE_ORDERS_ACT_MTH
    df_sample = df_sample.drop("LINE_ORDERS_ACT", axis = 1)
    df_sample_list = []
    for col in ["LINE_ORDERS_ACT_PER", "MAPE"]:
        df_sample_col = df_sample.pivot_table([col], 
                                              [col1], 
                                                  col2)
        df_sample_col.columns = pd.Index(['_'.join([str(_) for _ in v]) for v in df_sample_col.columns])
        cols_new = list(df_sample_col.columns.str.replace(col+"_", ""))
        df_sample_col.columns = cols_new
        df_sample_col["METRIC"] = col
        df_sample_col.reset_index(inplace = True)
        df_sample_col = pd.merge(df_sample_col, mape)
        df_sample_col = pd.merge(df_sample_col, weigths)
        df_sample_list.append(df_sample_col)
    df_sample_res = pd.concat(df_sample_list)
    return df_sample_res


def sh_mape(sample_df):
    """
    Utiltiy to calculate Service hour MAPE
    Input:
        sample_df: Pandas dataframe - Model results
    Output:
        sample_df: Pandas dataframe - Model results
    """
    sample_df["TOTAL_LEAD_ACT"] = ((sample_df["LEAD_MNTS_ACT"].fillna(0)+sample_df["LD_UNCLAIMED_ACT"].fillna(0)+sample_df["LD_DRIVE_ACT"].fillna(0)))/60
    sample_df["TOTAL_LEAD_PRED"] = ((sample_df["IDV_LEAD_MNTS_Pred"].fillna(0)+sample_df["IDV_LD_UNCLAIMED_Pred"].fillna(0)+sample_df["IDV_LD_DRIVE_Pred"].fillna(0)))/60
    sample_df["TOTAL_HELPER_ACT"] = (sample_df["HELPER_MNTS_ACT"].fillna(0)+sample_df["HP_UNCLAIMED_ACT"].fillna(0)+sample_df["HP_DRIVE_ACT"].fillna(0))/60
    sample_df["TOTAL_HELPER_PRED"] = (sample_df["IDV_HELPER_MNTS_Pred"].fillna(0)+sample_df["IDV_HP_UNCLAIMED_Pred"].fillna(0)+sample_df["IDV_HP_DRIVE_Pred"].fillna(0))/60
    sample_df["TOTAL_HELPERN_ACT"] = (sample_df["HELPER_OVR_ACT"].fillna(0)+sample_df["HPN_UNCLAIMED_ACT"].fillna(0)+sample_df["HPN_DRIVE_ACT"].fillna(0))/60
    sample_df["TOTAL_HELPERN_PRED"] = (sample_df["IDV_HELPER_OVR_Pred"].fillna(0)+sample_df["IDV_HPN_UNCLAIMED_Pred"].fillna(0)+sample_df["IDV_HPN_DRIVE_Pred"].fillna(0))/60
    sample_df["TOTAL_LEAD_PRED_ACT"] = (sample_df["IDV_LEAD_MNTS_ACT"].fillna(0)+sample_df["IDV_LD_UNCLAIMED_ACT"].fillna(0)+sample_df["IDV_LD_DRIVE_ACT"].fillna(0))/60
    sample_df["TOTAL_HELPER_PRED_ACT"] = (sample_df["IDV_HELPER_MNTS_ACT"].fillna(0)+sample_df["IDV_HP_UNCLAIMED_ACT"].fillna(0)+sample_df["IDV_HP_DRIVE_ACT"].fillna(0))/60
    sample_df["TOTAL_HELPERN_PRED_ACT"] = (sample_df["IDV_HELPER_OVR_ACT"].fillna(0)+sample_df["IDV_HPN_UNCLAIMED_ACT"].fillna(0)+sample_df["IDV_HPN_DRIVE_ACT"].fillna(0))/60
    sample_df[["TOTAL_LEAD_ACT", 
                    "TOTAL_HELPER_ACT", 
                   "TOTAL_HELPERN_ACT", ]] = sample_df[[
                           "TOTAL_LEAD_ACT", 
                    "TOTAL_HELPER_ACT", 
                   "TOTAL_HELPERN_ACT"]].replace({0:np.nan})
    
    sample_df["MAPE_LD"] = abs(sample_df["TOTAL_LEAD_ACT"]-sample_df["TOTAL_LEAD_PRED"])/sample_df["TOTAL_LEAD_ACT"]
    sample_df["MAPE_HP"] = abs(sample_df["TOTAL_HELPER_ACT"]-sample_df["TOTAL_HELPER_PRED"])/sample_df["TOTAL_HELPER_ACT"]
    sample_df["MAPE_HPN"] = abs(sample_df["TOTAL_HELPERN_ACT"]-sample_df["TOTAL_HELPERN_PRED"])/sample_df["TOTAL_HELPERN_ACT"]
    sample_df["MAPE_LD_ACT_PRED"] = abs(sample_df["TOTAL_LEAD_ACT"]-sample_df["TOTAL_LEAD_PRED_ACT"])/sample_df["TOTAL_LEAD_ACT"]
    sample_df["MAPE_HP_ACT_PRED"] = abs(sample_df["TOTAL_HELPER_ACT"]-sample_df["TOTAL_HELPER_PRED_ACT"])/sample_df["TOTAL_HELPER_ACT"]
    sample_df["MAPE_HPN_ACT_PRED"] = abs(sample_df["TOTAL_HELPERN_ACT"]-sample_df["TOTAL_HELPERN_PRED_ACT"])/sample_df["TOTAL_HELPERN_ACT"]
    sample_df[["TOTAL_LEAD_PRED_ACT", 
                    "TOTAL_HELPER_PRED_ACT", 
                   "TOTAL_HELPERN_PRED_ACT", ]] = sample_df[[
                           "TOTAL_LEAD_PRED_ACT", 
                    "TOTAL_HELPER_PRED_ACT", 
                   "TOTAL_HELPERN_PRED_ACT"]].replace({0:np.nan})
    sample_df["MAPE_LD_ACT"] = abs(sample_df["TOTAL_LEAD_PRED_ACT"]-sample_df["TOTAL_LEAD_PRED"])/sample_df["TOTAL_LEAD_PRED_ACT"]
    sample_df["MAPE_HP_ACT"] = abs(sample_df["TOTAL_HELPER_PRED_ACT"]-sample_df["TOTAL_HELPER_PRED"])/sample_df["TOTAL_HELPER_PRED_ACT"]
    sample_df["MAPE_HPN_ACT"] = abs(sample_df["TOTAL_HELPERN_PRED_ACT"]-sample_df["TOTAL_HELPERN_PRED"])/sample_df["TOTAL_HELPERN_PRED_ACT"]
    bins= [-1, 0.1, 0.15, 0.20, 0.30, 0.50, 1.00, np.inf]
    labels = [1, 2, 3, 4, 5, 6, 7]
    sample_df["MAPE_LD_BIN"] = pd.cut(sample_df['MAPE_LD'], bins=bins, labels=labels)
    sample_df["MAPE_HP_BIN"] = pd.cut(sample_df['MAPE_HP'], bins=bins, labels=labels)
    sample_df["MAPE_HPN_BIN"] = pd.cut(sample_df['MAPE_HPN'], bins=bins, labels=labels)
    sample_df["MAPE_LDAP_BIN"] = pd.cut(sample_df['MAPE_LD_ACT_PRED'], bins=bins, labels=labels)
    sample_df["MAPE_HPAP_BIN"] = pd.cut(sample_df['MAPE_HP_ACT_PRED'], bins=bins, labels=labels)
    sample_df["MAPE_HPNAP_BIN"] = pd.cut(sample_df['MAPE_HPN_ACT_PRED'], bins=bins, labels=labels)
    sample_df["MAPE_LDA_BIN"] = pd.cut(sample_df['MAPE_LD_ACT'], bins=bins, labels=labels)
    sample_df["MAPE_HPA_BIN"] = pd.cut(sample_df['MAPE_HP_ACT'], bins=bins, labels=labels)
    sample_df["MAPE_HPNA_BIN"] = pd.cut(sample_df['MAPE_HPN_ACT'], bins=bins, labels=labels)
    return sample_df


def remove_dir(version, base_dir, *del_dir):
    """
    Utility to delete folders created in case of incomplete run of the framwork
    Input:
        version : Name of current version
        *del dir: Comma separated list of directories from which the version folder has to be removed
    """
    print(del_dir)
    for d in del_dir:
        
        if os.path.isdir(os.path.join(base_dir, d, version)):
            shutil.rmtree(os.path.join(base_dir, d, version))
            print('Removed {version} folder from {d}')
        
            
