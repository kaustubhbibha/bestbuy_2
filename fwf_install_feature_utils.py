"""
In-Home Install Future Work Force - Service Line Orders Forecasting

This script conatins the helper functions used for feature engineering

This file requires following scripts to execute
    * fwf_install_common_utils.py - functions containing common utilities
"""

import pandas as pd
import numpy as np
import fwf_install_common_utils as cu


def model_data_prep_all_ntnl(grp_cols, agg_col, base_var, base, model_data, cal_df, pred_start):
    """
    Utility for higher level model FPG X National, FPG X Time Zone & FPG X State data preparation
    Input:
        grp_cols: List - Group column list
        agg_col: DV column name
        base_var: FPG category column name
        base: FPG value
        model_data: Pandas dataframe - Model data
        cal_df: Pandas dataframe - Calendar data
        pred_start: Train test split Fiscal week
    Output:
        model_df: Pandas dataframe - Train data
        test_df: Pandas dataframe - Test data
    """
    level = "wk"
    # Filter for particular granularity
    tt = model_data.copy()
    tt.rename(columns={"ds": "FISC_EOW_DT"}, inplace = True)
    # Rearrange date by fiscal weeks
    tt[base_var] = base
    # Preserve the actual
    tt["act_"+agg_col] = tt[agg_col]
    # Fetch previous year week
    tt = cu.gen_prv_yr_obs(tt, agg_col, level)
    for wk in [1, 2]:
        tt = cu.gen_prv_obs(tt, agg_col, wk, level, pred_start)
    # Compute rolling averages
    for window in [2, 4, 6, 8, 12, 52]:
        tt = cu.gen_rolling_feats(tt, window, agg_col, level, pred_start)
    tt["prev_yr_wk"] = tt["prev_yr_wk"].fillna(method='bfill')
    model_df = tt[(tt.FISC_WK_OF_MTH_ID < pred_start) ]
    test_df = tt[(tt.FISC_WK_OF_MTH_ID >= pred_start) ]
    model_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return model_df, test_df


def model_data_prep(grp_cols, agg_col, base_var, base, model_data, cal_df, ntnl_rs, state_rs, timezone_rs, fisc_calender, sales_data, pred_start, mod_start):
    """
    Utility for FPG X MM model data preparation
    Input:
        grp_cols: List - Group column list
        agg_col: DV column name
        base_var: FPG category column name
        base: FPG value
        model_data: Pandas dataframe - Line orders data
        cal_df: Pandas dataframe - Calendar data with holiday variables
        ntnl_rs: Pandas dataframe - National Line orders model results
        state_rs: Pandas dataframe - State Line orders model results
        timezone_rs: Pandas dataframe - Time Zone Line orders model results
        fisc_calender: Pandas dataframe - Fiscal calendar data
        sales_data: Pandas dataframe - MAPP variables
        pred_start: Train test split Fiscal week
        mod_start: Train data start Fiscal week
    Output:
        model_df: Pandas dataframe - Train data
        test_df: Pandas dataframe - Test data
        [train_avlbty_flg, tst_avlbty_flg]: List - Train & Test weeks present
    """
    level = "wk"
    # Filter for particular granularity
    tt = model_data[model_data[base_var] == base]
    # Rearrange date by fiscal weeks
    tt = cu.merge_cal_tt(tt, cal_df, level)
    tt_fisc_dummy = pd.get_dummies(tt["FISC_MTH_NBR"].astype(int), prefix = "FISC_MTH_NBR")
    tt = pd.concat([tt, tt_fisc_dummy], axis = 1)
    tt_fisc_dummy = pd.get_dummies(tt["FISC_QTR_NBR"].astype(int), prefix = "FISC_QTR_NBR")
    tt = pd.concat([tt, tt_fisc_dummy], axis = 1)
    
    
    tt[base_var] = base
    tt[agg_col+"_ACTUAL"] = tt[agg_col]
    tt['FISC_EOW_DT'] = pd.to_datetime(tt['FISC_EOW_DT'],infer_datetime_format=True)
    tt['FISC_EOW_DT'] = tt['FISC_EOW_DT'].dt.date
    fill_col = ["RSS", "MM", "TIME_ZONE", "STATE"]
    for fcol in fill_col:
        tt[fcol] = tt[fcol].fillna(method="ffill")
        tt[fcol] = tt[fcol].fillna(method="bfill")
    tt[agg_col].replace(0, np.nan, inplace=True)

    tt_train = tt[(tt.FISC_WK_OF_MTH_ID < pred_start) ]
    train_avlbty_flg = tt_train[agg_col].isna().sum()
    tt_train[agg_col] = tt_train[agg_col].fillna(method='ffill')
    tt_train[agg_col] = tt_train[agg_col].fillna(method='bfill')
    # Not imputing test dataset
    tt_test = tt[(tt.FISC_WK_OF_MTH_ID >= pred_start) ]
    tt = pd.concat([tt_train, tt_test], axis=0)

    # recomputing prev obs & prev year obs with capped data
    tt = cu.gen_prv_yr_obs(tt, agg_col, "mth")
    for wk in [1, 2, 3, 4, 5, 6]:
        tt = cu.gen_prv_obs(tt, agg_col, wk, level, pred_start)
    # Compute rolling averages
    for window in [2, 4, 6, 8, 12, 52]:
        tt = cu.gen_rolling_feats(tt, window, agg_col, level, pred_start)
    # previous year lag and rolling average variables
    tt = cu.gen_prv_yr_var(tt)
    tt_values = tt[["FISC_WK_OF_MTH_ID", agg_col]].copy()
    tt_values["prev_yr_wk"] = tt_values[agg_col].shift(0)
    tt_values["prev_yr_prev_1_wk"] = tt_values[agg_col].shift(1)
    tt_values["prev_yr_prev_2_wk"] = tt_values[agg_col].shift(2)
    tt_values["prev_yr_nx_1_wk"] = tt_values[agg_col].shift(-1)
    tt_values["prev_yr_nx_2_wk"] = tt_values[agg_col].shift(-2)
    tt_values = pd.merge(fisc_calender, tt_values, left_on = "LY_FISC_WK_OF_MTH_ID", right_on = "FISC_WK_OF_MTH_ID", how = "left")
    tt_values.columns = tt_values.columns.str.replace("_x", "")
    tt = pd.merge(tt, tt_values[["FISC_WK_OF_MTH_ID", "prev_yr_wk", "prev_yr_prev_1_wk", "prev_yr_prev_2_wk", "prev_yr_nx_1_wk", "prev_yr_nx_2_wk"]], how ="left")
    tt["prev_yr_mid_3_wk_avg"] = (tt["prev_yr_prev_1_wk"] + tt["prev_yr_wk"] + tt["prev_yr_nx_1_wk"])/3
    tt["prev_yr_mid_5_wk_avg"] = (tt["prev_yr_prev_2_wk"] + tt["prev_yr_prev_1_wk"] + tt["prev_yr_wk"] + tt["prev_yr_nx_1_wk"] + tt["prev_yr_nx_2_wk"])/5

    fill = ['prev_yr_wk', 'prev_yr_mth_avg',
            'prev_yr_prev_1_wk', 'prev_yr_prev_2_wk', 
            'prev_yr_nx_1_wk', 'prev_yr_nx_2_wk',
            'prev_yr_prev_2_wk_avg', 'prev_yr_prev_4_wk_avg',
            'prev_yr_mid_3_wk_avg', 'prev_yr_mid_5_wk_avg']
    for var in fill:
        tt[var] = tt[var].fillna(method="bfill")
        tt[var] = tt[var].fillna(method="ffill")
    tt = cu.gen_velocity_var(tt, agg_col, pred_start)
    # Using Fiscal year 2018 and 2019 for training model
    model_df = tt[(tt.FISC_WK_OF_MTH_ID < pred_start)]
    model_df.reset_index(drop = True, inplace =True)
    # computing variables with respect to national model
    hr_model_agg_col = "Fb_Prediction"
    model_df = model_df.merge(ntnl_rs, on = ["FISC_WK_OF_MTH_ID", "RSS"], how="left")
    model_df = model_df.merge(state_rs, on = ["FISC_WK_OF_MTH_ID", "STATE", "RSS"], how="left")
    model_df = model_df.merge(timezone_rs, on = ["FISC_WK_OF_MTH_ID", "TIME_ZONE", "RSS"], how="left")
    model_df["prv_yr_vr_1_wk"] = (model_df["prev_yr_prev_1_wk"].values/model_df["prev_yr_prev_2_wk"].values)*model_df["prev_1_wk"].values
    model_df["prev_1wk_ntnl_val"] = (model_df["prev_1_wk"].values/model_df["prev_1_wk_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_2wk_ntnl_val"] = (model_df["prev_2_wk"].values/model_df["prev_2_wk_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_wk_yr_ntnl_val"] = (model_df["prev_1_wk"].values/model_df[agg_col+"_FY_Avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_2wk_avg_ntnl_val"] = (model_df["prev_2_wk_avg"].values/model_df["prev_2_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_4wk_avg_ntnl_val"] = (model_df["prev_4_wk_avg"].values/model_df["prev_4_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_6wk_avg_ntnl_val"] = (model_df["prev_6_wk_avg"].values/model_df["prev_6_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_8wk_avg_ntnl_val"] = (model_df["prev_8_wk_avg"].values/model_df["prev_8_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_12wk_avg_ntnl_val"] = (model_df["prev_12_wk_avg"].values/model_df["prev_12_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_52wk_avg_ntnl_val"] = (model_df["prev_52_wk_avg"].values/model_df["prev_52_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_wk_2wk_avg_ntnl_val"] = (model_df["prev_1_wk"].values/model_df["prev_2_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_wk_4wk_avg_ntnl_val"] = (model_df["prev_1_wk"].values/model_df["prev_4_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_wk_52wk_avg_ntnl_val"] = (model_df["prev_1_wk"].values/model_df["prev_52_wk_avg_ntnl"].values)*model_df[hr_model_agg_col+"_ntnl"].values
    model_df["prev_1wk_state_val"] = (model_df["prev_1_wk"].values/model_df["prev_1_wk_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_2wk_state_val"] = (model_df["prev_2_wk"].values/model_df["prev_2_wk_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_wk_yr_state_val"] = (model_df["prev_1_wk"].values/model_df[agg_col+"_FY_Avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_2wk_avg_state_val"] = (model_df["prev_2_wk_avg"].values/model_df["prev_2_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_4wk_avg_state_val"] = (model_df["prev_4_wk_avg"].values/model_df["prev_4_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_6wk_avg_state_val"] = (model_df["prev_6_wk_avg"].values/model_df["prev_6_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_8wk_avg_state_val"] = (model_df["prev_8_wk_avg"].values/model_df["prev_8_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_12wk_avg_state_val"] = (model_df["prev_12_wk_avg"].values/model_df["prev_12_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_52wk_avg_state_val"] = (model_df["prev_52_wk_avg"].values/model_df["prev_52_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_wk_2wk_avg_state_val"] = (model_df["prev_1_wk"].values/model_df["prev_2_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_wk_4wk_avg_state_val"] = (model_df["prev_1_wk"].values/model_df["prev_4_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_wk_52wk_avg_state_val"] = (model_df["prev_1_wk"].values/model_df["prev_52_wk_avg_stt"].values)*model_df[hr_model_agg_col+"_stt"].values
    model_df["prev_1wk_time_zone_val"] = (model_df["prev_1_wk"].values/model_df["prev_1_wk_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_2wk_time_zone_val"] = (model_df["prev_2_wk"].values/model_df["prev_2_wk_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_wk_yr_time_zone_val"] = (model_df["prev_1_wk"].values/model_df[agg_col+"_FY_Avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_2wk_avg_time_zone_val"] = (model_df["prev_2_wk_avg"].values/model_df["prev_2_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_4wk_avg_time_zone_val"] = (model_df["prev_4_wk_avg"].values/model_df["prev_4_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_6wk_avg_time_zone_val"] = (model_df["prev_6_wk_avg"].values/model_df["prev_6_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_8wk_avg_time_zone_val"] = (model_df["prev_8_wk_avg"].values/model_df["prev_8_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_12wk_avg_time_zone_val"] = (model_df["prev_12_wk_avg"].values/model_df["prev_12_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_52wk_avg_time_zone_val"] = (model_df["prev_52_wk_avg"].values/model_df["prev_52_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_wk_2wk_avg_time_zone_val"] = (model_df["prev_1_wk"].values/model_df["prev_2_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_wk_4wk_avg_time_zone_val"] = (model_df["prev_1_wk"].values/model_df["prev_4_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["prev_wk_52wk_avg_time_zone_val"] = (model_df["prev_1_wk"].values/model_df["prev_52_wk_avg_tz"].values)*model_df[hr_model_agg_col+"_tz"].values
    model_df["cagr_prev_wk_5wk_ntnl_val"] = model_df[agg_col].shift()*(1+(model_df[hr_model_agg_col+"_ntnl"].shift() - model_df[hr_model_agg_col+"_ntnl"].shift(5))/model_df[hr_model_agg_col+"_ntnl"].shift(5))
    model_df["cagr_prev_mt_2mt_rol_ntnl_val"] = model_df[agg_col].shift()*(model_df[hr_model_agg_col+"_ntnl"].shift().rolling(min_periods=1, window=4).mean()/model_df[hr_model_agg_col+"_ntnl"].shift(5).rolling(min_periods=1, window=4).mean())
    model_df["cagr_prev_wk_5wk_ntnl_ratio"] = (model_df[hr_model_agg_col+"_ntnl"].shift() - model_df[hr_model_agg_col+"_ntnl"].shift(5))/model_df[hr_model_agg_col+"_ntnl"].shift(5)
    temp_ = model_df[hr_model_agg_col+"_ntnl"].shift(5).rolling(min_periods=1, window=4).mean()
    model_df["cagr_prev_mt_2mt_rol_ntnl_ratio"] = (model_df[hr_model_agg_col+"_ntnl"].shift().rolling(min_periods=1, window=4).mean() - temp_)/temp_
    fill_cagr =["cagr_prev_wk_5wk_ntnl_val", "cagr_prev_mt_2mt_rol_ntnl_val", "cagr_prev_wk_5wk_ntnl_ratio", "cagr_prev_mt_2mt_rol_ntnl_ratio"]

    model_df = model_df[(model_df.FISC_WK_OF_MTH_ID >= mod_start)]
    for var in fill_cagr:
        model_df[var] = model_df[var].fillna(method="bfill")
        model_df[var] = model_df[var].fillna(method="ffill")
    test_df = tt[(tt.FISC_WK_OF_MTH_ID >= pred_start) ]
    test_df = test_df.merge(ntnl_rs, how="left")
    test_df = test_df.merge(state_rs, how="left")
    test_df = test_df.merge(timezone_rs, how="left")
    tst_avlbty_flg = test_df[agg_col].isna().sum()
    model_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    model_df = model_df.merge(sales_data, on = ["FISC_WK_OF_MTH_ID", "MM"], how="inner")
    test_df = test_df.merge(sales_data, on = ["FISC_WK_OF_MTH_ID", "MM"], how="left")
    model_df['FISC_EOW_DT'] = model_df['FISC_EOW_DT'].astype(str)
    test_df['FISC_EOW_DT'] = test_df['FISC_EOW_DT'].astype(str)
    return model_df, test_df, [train_avlbty_flg, tst_avlbty_flg]


def lag_variable_calculation(agg_col, grp_var, model_data, cal_df, mm_list, fisc_calender, pred_start):
    """
    Utility to calculate lag variables based on MM proportion
    Input:
        agg_col: DV column name
        grp_var: MM column name
        model_data: Pandas dataframe - Train Test data
        cal_df: Pandas dataframe - Fiscal calendar data with holiday variables
        mm_list: List - Minor market
        fisc_calender: Pandas dataframe - Fiscal calendar data
        pred_start: Train Test split
    Output:
        tt_concat: Pandas dataframe - MM Proportion variables
    """
    tt = model_data
    level = "wk"
    # Rearrange date by fiscal weeks
    tt = cu.merge_cal_tt(tt, cal_df.drop('FISC_EOW_DT', axis = 1), level)
    tt[agg_col+"_ACT"] = tt[agg_col]
    tt[agg_col].replace(0, np.nan, inplace=True)
    model_data_pivot = tt.pivot_table([agg_col], 
                                      ['FISC_YR_NBR', 'FISC_MTH_NBR','FISC_WK_OF_MTH_ID', 'FISC_WK_OF_YR_NBR', 'FISC_QTR_NBR'], 
                                      grp_var)
    model_data_pivot.columns = pd.Index(['_'.join([str(_) for _ in v]) for v in model_data_pivot.columns])
    model_data_pivot.columns = model_data_pivot.columns.str.replace(agg_col+"_","")
    model_data_pivot = model_data_pivot.reset_index()
    model_data_pivot = cu.merge_cal_tt(model_data_pivot, cal_df, level)
    model_data_pivot = cu.fillna_values(model_data_pivot, pred_start)
    for i in mm_list:
        if str(i) not in model_data_pivot.columns:
            model_data_pivot[str(i)] = 0
    model_data_pivot[agg_col] = model_data_pivot[mm_list].sum(axis =1)
    tt = model_data_pivot.copy()
    train = tt[(tt.FISC_WK_OF_MTH_ID < pred_start) ]
    test = tt[(tt.FISC_WK_OF_MTH_ID >= pred_start) ]
    for i in mm_list:
        if str(i) not in train.columns:
            train[str(i)] = 0
        if str(i) not in test.columns:
            test[str(i)] = 0
    tt_wk_val = cu.gen_prv_yr_wk(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR","FISC_WK_OF_YR_NBR"], "_salesMAPP_pv_yr_wk", fisc_calender, pred_start)
    tt_yr_val = cu.gen_prv_prop(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR"], 1, "_salesMAPP_pv_yr", pred_start)
    tt_qtr_val = cu.gen_prv_prop(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR","FISC_QTR_NBR"], 4, "_salesMAPP_pv_yr_qtr", pred_start)
    tt_mth_val = cu.gen_prv_prop(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR","FISC_MTH_NBR"], 12, "_salesMAPP_pv_yr_mth", pred_start)
    tt_pv_mid_3wk_val = cu.gen_prv_roll_prop_ls_yr(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR","FISC_WK_OF_YR_NBR"], 3, "_salesMAPP_pv_yr_mid_3wk", -1, fisc_calender, pred_start)
    tt_pv_mid_5wk_val = cu.gen_prv_roll_prop_ls_yr(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR","FISC_WK_OF_YR_NBR"], 5, "_salesMAPP_pv_yr_mid_5wk", -2, fisc_calender, pred_start)
    tt_pv_2wk_Val = cu.gen_prv_roll_prop_ls_yr(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR","FISC_WK_OF_YR_NBR"], 2, "_salesMAPP_pv_2wk", 1, fisc_calender, pred_start)
    tt_pv_4wk_val = cu.gen_prv_roll_prop_ls_yr(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR","FISC_WK_OF_YR_NBR"], 4, "_salesMAPP_pv_4wk", 1, fisc_calender, pred_start)
    tt_pv_roll_5wk_val = cu.prev_avg_scoring(mm_list, train, test, "_salesMAPP_pv_roll_5wk")
    tt_prev_mth_pv_mth_lyr = cu.prev_mth_pv_mth_lyr(tt.copy(), mm_list, agg_col, ["FISC_YR_NBR", "FISC_MTH_NBR"], "_salesMAPP_pv_mth_pv_mth_ly", pred_start, cal_df)

    tt_concat = pd.concat([ tt_pv_mid_5wk_val, tt_pv_mid_3wk_val, tt_wk_val], axis = 1 )
    tt_concat = tt_concat.loc[:,~tt_concat.columns.duplicated()]
    tt_concat = pd.merge(tt_concat, cal_df[['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_WK_OF_YR_NBR',
       'FISC_QTR_NBR']], how = "inner" )
    tt_concat = pd.merge(tt_concat, tt_pv_2wk_Val, how = "left")
    tt_concat = pd.merge(tt_concat, tt_pv_4wk_val, how = "left")
    tt_concat = pd.merge(tt_concat, tt_yr_val, how = "left")
    tt_concat = pd.merge(tt_concat, tt_qtr_val, how = "left")
    tt_concat = pd.merge(tt_concat, tt_mth_val, how = "left")
    tt_concat = pd.merge(tt_concat, tt_pv_roll_5wk_val, how = "left")
    tt_concat = pd.merge(tt_concat, tt_prev_mth_pv_mth_lyr, how = "left")
    return tt_concat


def sales_data_prep(fpg, df, sales, fisc_calender, cal_df, forecast, pred_start, pred_col) :
    
    """
    Utility to execute proportion variable creation
    Input:
        fpg: FPG name
        df: Pandas dataframe - Train Test data
        sales: Pandas dataframe - Sales data
        fisc_calender: Pandas dataframe - Fiscal calendar data
        cal_df: Pandas dataframe - Fiscal calendar data with holiday variables
        forecast: Pandas dataframe - Forecast data
        pred_start: Train Test split
    Output:
        sales_variable_slc_pivot: Pandas dataframe - Proportion variables
    """
   
    model_base_var = "RSS"
    pivot_var = "MM"
    df_grp_all = df[df[model_base_var] == fpg]
    sales_grp = sales[sales[model_base_var] == fpg]
    level_list = list(df_grp_all[pivot_var].unique())
    sales_grp = sales_grp[sales_grp[pivot_var].isin(level_list)]
    ntnl_pred_col = pred_col
    ntnl_df = sales_grp.groupby(["FISC_YR_NBR", "FISC_WK_OF_MTH_ID", "FISC_WK_OF_YR_NBR"])[ntnl_pred_col].sum().reset_index()
    ntnl_df.sort_values("FISC_WK_OF_MTH_ID", inplace = True)
    ntnl_df.reset_index(inplace = True, drop = True)
    ntnl_mm_df = sales_grp.groupby(["FISC_YR_NBR","FISC_WK_OF_MTH_ID", "FISC_WK_OF_YR_NBR", pivot_var])[ntnl_pred_col].sum().reset_index()
    model_data_sales_df = lag_variable_calculation(ntnl_pred_col, pivot_var, ntnl_mm_df, cal_df, level_list, fisc_calender, pred_start)
    var_list = ['pv_yr_wk', 'pv_yr', 'pv_yr_qtr', 'pv_yr_mth', 'pv_yr_mid_3wk', 'pv_yr_mid_5wk', 'pv_2wk', 'pv_4wk']
    
    # "SALES_VALUE" - Lag values - MM x Week level
    model_data_sales_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_sales_df, var_list)
    # Lag variables national values
    model_data_sales_melt_df_grp = model_data_sales_melt_df.groupby(['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_QTR_NBR', 'Variable_Type']).sum()[ntnl_pred_col+"_VALUE"].reset_index()
    # "SALES_VALUE_ALL" - Lag variables national values
    model_data_sales_melt_df_grp.columns = ['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_QTR_NBR',
       'Variable_Type', ntnl_pred_col+'_VALUE_ALL']
    model_data_sales_melt_df = pd.merge(model_data_sales_melt_df, model_data_sales_melt_df_grp, how = "inner")
    model_data_melt_all = model_data_sales_melt_df.copy()
    model_data_melt_all.columns = model_data_melt_all.columns.str.replace("variable", "MM")
    
    forecast_grp = forecast[forecast[model_base_var] == fpg]
    forecast_grp = forecast_grp[["FISC_WK_OF_MTH_ID", "SALES"]].sort_values("FISC_WK_OF_MTH_ID")
    forecast_grp.rename({"SALES": "SALES_Grp"},axis=1, inplace=True)
    sales_variable = model_data_melt_all.copy()
    sales_variable_forcast = pd.merge(sales_variable, forecast_grp[["FISC_WK_OF_MTH_ID", "SALES_Grp"]], how = "inner")
    sales_variable_slc = sales_variable_forcast[['FISC_WK_OF_MTH_ID', 'MM', 'Variable_Type', ntnl_pred_col+'_VALUE', ntnl_pred_col+"_VALUE_ALL", 
       'SALES_Grp']].copy()
    
    
    sales_variable_slc["DIS"] = sales_variable_slc[ntnl_pred_col+"_VALUE"]/sales_variable_slc[ntnl_pred_col+"_VALUE_ALL"]
    sales_variable_slc["IDV"] = sales_variable_slc[f"{ntnl_pred_col}_Grp"]*sales_variable_slc["DIS"]
    
    sales_variable_slc_pivot = sales_variable_slc.pivot_table(['IDV'], 
                                      ['FISC_WK_OF_MTH_ID', 'MM'], 
                                      ["Variable_Type"]).reset_index()
    sales_variable_slc_pivot.columns = pd.Index(['_'.join([str(_) for _ in v]) for v in sales_variable_slc_pivot.columns])
    sales_variable_slc_pivot.columns = sales_variable_slc_pivot.columns.str.replace("FISC_WK_OF_MTH_ID_","FISC_WK_OF_MTH_ID")
    sales_variable_slc_pivot.columns = sales_variable_slc_pivot.columns.str.replace("MM_","MM")
    sales_variable_slc_pivot.columns = sales_variable_slc_pivot.columns.str.replace("IDV_","")
    sales_variable_slc_pivot.columns = sales_variable_slc_pivot.columns.str.replace("pv_","sales_MAPP_pv_")
    
    return sales_variable_slc_pivot