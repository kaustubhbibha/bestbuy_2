"""
In-Home Install Future Work Force - Service Line Orders Forecasting

This script conatins the helper functions used for model building/scoring Service Line orders

This file requires following scripts to execute
    * fwf_install_common_utils.py - functions containing common utilities
    * fwf_install_feature_utils.py - functions containing feature engineering utils
"""

import pandas as pd
import numpy as np
import os
os.environ["TMPDIR"] = "/opt/appdata/temp/python"
from fbprophet import Prophet
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import compress
import time
import pickle
import logging
from sklearn.model_selection import RandomizedSearchCV
import fwf_install_common_utils as cu
import fwf_install_feature_utils as fu
from functools import wraps
import errno
import signal
import os


class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL,seconds) #used timer instead of alarm
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    return decorator


@timeout(200)
def fb_prophet_evaluate(holidays, ntnl_train, ntnl_test, res, seas, change_pt, fcast_var):
    """
    Utility to evaluate fb prophet models
    Input:
        holidays: List - Holiday columns
        ntnl_train: Pandas dataframe - Train data
        ntnl_test: Pandas dataframe - Test data
        res: Dict - Result summary dictionary
        seas: Seasonality mode
        change_pt: Pandas dataframe - Change point
    Output:
        mod: Fb Prophet model
        res: Dict - Result summary dictionary
        fb_mv_train: Pandas dataframe - Train data
        fb_mv_preds: Pandas dataframe - Test data
    """

    if fcast_var:
        try:
            
            mod = Prophet(weekly_seasonality = True, 
                    yearly_seasonality = True, 
                    holidays = holidays,
                    seasonality_mode = seas,
                    changepoint_prior_scale=change_pt,
                    uncertainty_samples=False)
            idv_cols = ["ds"]
            mod.add_regressor('SALES')
            idv_cols.append("SALES")
            print(idv_cols)
            min_cap = ntnl_train.y.mean()
            ntnl_train_prophet = ntnl_train.copy()
            ntnl_train_prophet.y.fillna(0, inplace = True)
            #"""
            
            mod.fit(ntnl_train_prophet[idv_cols+["y"]])
            fb_mv_train = mod.predict(ntnl_train[idv_cols])
            fb_mv_preds = mod.predict(ntnl_test[idv_cols])
            
        except:
            mod = Prophet(weekly_seasonality = True, 
                    yearly_seasonality = True, 
                    holidays = holidays,
                    seasonality_mode = seas,
                    changepoint_prior_scale=change_pt,
                    uncertainty_samples=False)
            idv_cols = ["ds"]
            
            min_cap = ntnl_train.y.mean()
            ntnl_train_prophet = ntnl_train.copy()
            ntnl_train_prophet.y.fillna(0, inplace = True)
            
            mod.fit(ntnl_train_prophet[idv_cols+["y"]])
            fb_mv_train = mod.predict(ntnl_train[idv_cols])
            fb_mv_preds = mod.predict(ntnl_test[idv_cols])
    else:
        mod = Prophet(weekly_seasonality = True, 
                    yearly_seasonality = True, 
                    holidays = holidays,
                    seasonality_mode = seas,
                    changepoint_prior_scale=change_pt,
                    uncertainty_samples=False)
        idv_cols = ["ds"]
        
        min_cap = ntnl_train.y.mean()
        ntnl_train_prophet = ntnl_train.copy()
        ntnl_train_prophet.y.fillna(0, inplace = True)
        mod.fit(ntnl_train_prophet[idv_cols+["y"]])
        fb_mv_train = mod.predict(ntnl_train[idv_cols])
        fb_mv_preds = mod.predict(ntnl_test[idv_cols])

    fb_mv_train["yhat"] = np.where(fb_mv_train["yhat"]<=0, min_cap, fb_mv_train["yhat"])
    fb_mv_preds["yhat"] = np.where(fb_mv_preds["yhat"]<=0, min_cap, fb_mv_preds["yhat"])
    res["mape"], res["mae"], res["made"], res["wmape"]  = cu.calc_mape(ntnl_test.y, fb_mv_preds.yhat)
    res["mape_13w"], res["mae_13w"], res["made_13w"], res["wmape_13w"]  = cu.calc_mape(ntnl_test.y[4:17], fb_mv_preds.yhat[4:17])
    res["seas_mode"] = seas
    res["change_point"] = change_pt
    res['min_cap'] = min_cap
    res['idv'] = idv_cols
    
    fb_mv_train = fb_mv_train[idv_cols+["yhat"]]
    fb_mv_preds = fb_mv_preds[idv_cols+["yhat"]]

    return mod, res, fb_mv_train, fb_mv_preds


def fb_prophet_hyper(holidays, ntnl_train, ntnl_test, res, fcast_var):
    """
    Utility to execute hyper parameters for fb prophet models
    Input:
        holidays: List - Holiday columns
        ntnl_train: Pandas dataframe - Train data
        ntnl_test: Pandas dataframe - Test data
        res: Dict - Result summary dictionary
    Output:
        res_df: Pandas dataframe - Train data
        fb_mv_train: Pandas dataframe - Train data
        fb_mv_preds: Pandas dataframe - Test data
    """
    seas = ["multiplicative", "additive"]
    change_point = [ 0.1, 0.2, 0.3, 0.4, 0.5]
    res_all = []
    for s in seas:
        for c in change_point:
            try:
                start_time = time.time()
                mod, res, fb_mv_train, fb_mv_preds = fb_prophet_evaluate(holidays.copy(), ntnl_train.copy(), ntnl_test.copy(), res.copy(), s, c, fcast_var)
                res_all.append(res)
                tot_time = time.time()-start_time 
                print("Execution successful")
                print(tot_time, c, s)
            except:
                tot_time = time.time()-start_time 
                print("Failed execution")
                print(tot_time, c, s)
                continue
    res_df = pd.DataFrame(res_all)
    res_df = res_df.sort_values("mape")
    res_df.reset_index(inplace = True, drop = True)
    return res_df, fb_mv_train, fb_mv_preds


def fb_prophet(ntnl_df, agg_col, cal_df, grp_cols, base_var, base, pred_start, hol_cols, fpg, base_dir, model_dump_folder, model_filename, fb_val, version, ntnl_fcast, fcast_var):
    """
    Utility to build fb prophet model, executes Ridge model and prepares variables from model predictions which will be used by FPG X MM models
    Input:
        ntnl_df: Pandas dataframe - Model data
        agg_col: DV column name
        cal_df: Pandas dataframe - Calendar data
        grp_cols: List - Group column list
        base_var: FPG category column name
        base: FPG value
        pred_start: Train test split Fiscal week
        hol_cols: List - Holiday columns
        fpg: FPG name
        base_dir: Framework parent directory
        model_dump_folder: Directory to dump model
        model_filename: Model file name
        fb_val: Train evaluation split
        version: Model train version
        ntnl_fcast: Dataframe - forecast data
    Output:
        res: Fb & Ridge results summary
        train_test: Pandas dataframe - Fb results data
        train_test_df: Pandas dataframe - Fb & Ridge result
    """
    ntnl_df_fb = ntnl_df.copy()
    level = "wk"
    ntnl_df_fb = cu.merge_cal_tt(ntnl_df_fb, cal_df, level)
#    ntnl_df_fb = ntnl_df_fb.merge(ntnl_fcast, on=["FISC_WK_OF_MTH_ID"], how="left")
    ntnl_df_fb.sort_values("FISC_WK_OF_MTH_ID", inplace =True)
    ntnl_df_fb.rename(columns={"FISC_EOW_DT": "ds", agg_col: "y"}, inplace = True)
    ntnl_df_fb['ds'] = pd.to_datetime(ntnl_df_fb['ds'],infer_datetime_format=True)
    ntnl_df_fb['ds'] = ntnl_df_fb['ds'].dt.date
    res = {}
    res["na_cnt"] = len(ntnl_df_fb) - ntnl_df_fb["y"].count()
    ntnl_df_fb.reset_index(inplace = True, drop = True)
    ntnl_train = ntnl_df_fb[ntnl_df_fb.FISC_WK_OF_MTH_ID < pred_start]
    ntnl_test = ntnl_df_fb[ntnl_df_fb.FISC_WK_OF_MTH_ID >= pred_start]
    ntnl_test.reset_index(inplace = True, drop = True)
    playoffs = pd.DataFrame({
      'holiday': 'playoff',
      'ds': pd.to_datetime(cal_df[cal_df.Holiday == 1]["FISC_EOW_DT"]),
      'lower_window': -7,
      'upper_window': 7,
    })
    
    superbowls = pd.DataFrame({
      'holiday': 'superbowl',
      'ds': pd.to_datetime(cal_df[cal_df.HOLIDAY_SPIKE == 1]["FISC_EOW_DT"]),
      'lower_window': -7,
      'upper_window': 21,
    })
    
    holidays = pd.concat((playoffs, superbowls))
    tr_df = ntnl_train[ntnl_train.FISC_WK_OF_MTH_ID < fb_val].copy()
    val_df = ntnl_train[ntnl_train.FISC_WK_OF_MTH_ID >= fb_val].copy()

    res_df, fb_mv_train, fb_mv_preds = fb_prophet_hyper(holidays.copy(), tr_df.copy(), val_df.copy(), res.copy(), fcast_var)
    
    m, res, fb_mv_train, fb_mv_preds = fb_prophet_evaluate(holidays.copy(), ntnl_train.copy(), ntnl_test.copy(), res.copy(), res_df.ix[0]["seas_mode"], res_df.ix[0]["change_point"], fcast_var)

    train_dataf = pd.concat([ntnl_train, fb_mv_train[["ds", "yhat"]]], axis = 1 )
    test_dataf = pd.concat([ntnl_test, fb_mv_preds[["ds", "yhat"]]], axis = 1 )
    train_test = pd.concat([train_dataf, test_dataf], ignore_index=True, sort=False)
    train_test = train_test.loc[:,~train_test.columns.duplicated()]
    train_test.rename(columns={"y": agg_col, "yhat": "Fb_Prediction"}, inplace = True)
    
    ntnl_agg_col = "Fb_Prediction"
    res["RSS"] = fpg
    dum_cols = ['FISC_MTH_NBR']
    
    train_test, model_ridge, scaler, min_cap_rid = ridge(train_test, ntnl_agg_col, agg_col, hol_cols, dum_cols, pred_start)
    ntnl_df_train, ntnl_df_test = fu.model_data_prep_all_ntnl(grp_cols, ntnl_agg_col, base_var, base, train_test, cal_df, pred_start)
    
    
    
    res["mape_reg_13w"], res["mae_reg_13w"], res["made_reg_13w"], res["wmape_reg_13w"]  = cu.calc_mape(ntnl_df_test[agg_col][4:17], ntnl_df_test[ntnl_agg_col][4:17])
    stats_tr, stats_ts = cu.train_test_stats(ntnl_df_train, ntnl_df_test, agg_col)
    res.update(stats_tr)
    res.update(stats_ts)
    objects = {'model_fb':m, 'model_ridge':model_ridge, 'scaler':scaler, 'ridge_cols':hol_cols, 'ridge_dum_cols': dum_cols, 'min_cap_fb':res['min_cap'],'min_cap_rid' : min_cap_rid}

    with open(os.path.join(base_dir, model_dump_folder, version + "/", model_filename + '_model.sav'), 'wb') as f:
        pickle.dump(objects, f)
    train_test_df = pd.concat([ntnl_df_train, ntnl_df_test])
    train_test_df.reset_index(drop =True, inplace =True)
    train_test_df = train_test_df.fillna(method='ffill')
    train_test_df = train_test_df.fillna(method='bfill')
    train_test.drop(["ds"], inplace = True, axis = 1)
    train_test_df.drop(["FISC_EOW_DT"], inplace = True, axis = 1)
    return res, train_test, train_test_df


def ridge(df, pred_col, target_col, columns, col_dummies, pred_start):
    """
    Utility to build Ridge model
    Input:
        df: Pandas dataframe - Model data
        pred_col: IDV column name
        target_col: DV column name
        columns: List - Holiday column names
        col_dummies: List - Calendar column list for creating dummies
        pred_start: Train test split Fiscal week
    Output:
        df_temp: Pandas dataframe - Ridge results data
        ls_random.best_estimator_: Ridge best model parameter
        scalar: Scalar object
        min_cap: Minimum cap
    """
    df_temp = df.copy()
    dum_vars = []
    for col in col_dummies:
        if col == 'FISC_WK_OF_YR_NBR':
            print('k')
            dums = pd.get_dummies(df_temp[col], prefix = col)
            dum_vars += list(dums.columns)[4:35]
        else:
            dums = pd.get_dummies(df_temp[col], prefix = col)
            dum_vars += list(dums.columns)
        df_temp = pd.concat([df_temp, dums], axis = 1)
    alphas = {"alpha":[0.05, 0.1, 0.2, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]}
    lm = Ridge()
    # Random search of parameters, using 3 fold cross validation, 
    ls_random = RandomizedSearchCV(estimator = lm, cv =3, 
                                   param_distributions = alphas, 
                                   n_iter = 100, verbose=0, random_state=42, n_jobs = 1)
    scalar = MinMaxScaler()
    temp_cols = columns + dum_vars + [pred_col]
    df_temp_train = df_temp[df_temp['FISC_WK_OF_MTH_ID'] < pred_start]
    min_cap = df_temp_train[target_col].min()
    scalar.fit(df_temp_train[temp_cols + [target_col]])
    new = scalar.transform(df_temp[temp_cols + [target_col]])
    new = pd.DataFrame(new, columns = temp_cols + [target_col])
    df_temp_train = df_temp_train.dropna()
    new_train = scalar.transform(df_temp_train[temp_cols + [target_col]])
    new_train = pd.DataFrame(new_train, columns = temp_cols + [target_col])
    ls_random.fit(new_train[temp_cols], new_train[target_col])
    new['pred'] = ls_random.predict(new[temp_cols])
    inverse = scalar.inverse_transform(new[temp_cols + ['pred']])
    df_temp[pred_col+'_old'] = df_temp[pred_col].values
    df_temp[pred_col] = inverse[:,-1]
    df_temp[pred_col] = np.where(df_temp[pred_col]<=0, min_cap, df_temp[pred_col])
    return df_temp, ls_random.best_estimator_, scalar, min_cap


def predict_test_trf(train_data, test_df, model, model_cols, agg_col, cols_to_scale, mod_type, scaler, pred_start, min_cap):
    """
    Utility for FPG X MM model prediction. This logic involves prediction week on week and the each weeks predictions are fed back
    and used for creating variables for the upcoming weeks
    Input:
        train_data: Pandas dataframe - Train data
        test_df: Pandas dataframe - Test data
        model: Model object
        model_cols: List - Model columns
        agg_col: DV column name
        cols_to_scale: List - Columns to scale
        mod_type: Scalar type
        scaler: Scalar object
        pred_start: Train test split Fiscal Week
    Output:
        test_pred_lr: Pandas dataframe - Model predictions
    """
    level = "wk"
    # Scoring single week at time by updating features using previous week(s) predictions
    train_df = train_data.copy()
    train_df = train_df.tail(n=54)
    for i in range(0, test_df.shape[0]):
        train_df = train_df.append(test_df.iloc[i])
        for wk in [1, 2, 3, 4, 5, 6]:
            train_df = cu.gen_prv_obs(train_df, agg_col, wk, level, pred_start)
        for window in [2, 4, 6, 8, 12, 52]:
            train_df = cu.gen_rolling_feats(train_df, window, agg_col, level, pred_start)
        
        hr_model_agg_col = "Fb_Prediction"
        train_df = cu.gen_velocity_var(train_df, agg_col, pred_start)
        train_df["prv_yr_vr_1_wk"] = (train_df["prev_yr_prev_1_wk"].values/train_df["prev_yr_prev_2_wk"].values)*train_df["prev_1_wk"].values
        train_df["prev_1wk_ntnl_val"] = (train_df["prev_1_wk"].values/train_df["prev_1_wk_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_2wk_ntnl_val"] = (train_df["prev_2_wk"].values/train_df["prev_2_wk_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_wk_yr_ntnl_val"] = (train_df["prev_1_wk"].values/train_df[agg_col+"_FY_Avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_2wk_avg_ntnl_val"] = (train_df["prev_2_wk_avg"].values/train_df["prev_2_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_4wk_avg_ntnl_val"] = (train_df["prev_4_wk_avg"].values/train_df["prev_4_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_6wk_avg_ntnl_val"] = (train_df["prev_6_wk_avg"].values/train_df["prev_6_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_8wk_avg_ntnl_val"] = (train_df["prev_8_wk_avg"].values/train_df["prev_8_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_12wk_avg_ntnl_val"] = (train_df["prev_12_wk_avg"].values/train_df["prev_12_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_52wk_avg_ntnl_val"] = (train_df["prev_52_wk_avg"].values/train_df["prev_52_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_wk_2wk_avg_ntnl_val"] = (train_df["prev_1_wk"].values/train_df["prev_2_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_wk_4wk_avg_ntnl_val"] = (train_df["prev_1_wk"].values/train_df["prev_4_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_wk_52wk_avg_ntnl_val"] = (train_df["prev_1_wk"].values/train_df["prev_52_wk_avg_ntnl"].values)*train_df[hr_model_agg_col+"_ntnl"].values
        train_df["prev_1wk_state_val"] = (train_df["prev_1_wk"].values/train_df["prev_1_wk_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_2wk_state_val"] = (train_df["prev_2_wk"].values/train_df["prev_2_wk_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_wk_yr_state_val"] = (train_df["prev_1_wk"].values/train_df[agg_col+"_FY_Avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_2wk_avg_state_val"] = (train_df["prev_2_wk_avg"].values/train_df["prev_2_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_4wk_avg_state_val"] = (train_df["prev_4_wk_avg"].values/train_df["prev_4_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_6wk_avg_state_val"] = (train_df["prev_6_wk_avg"].values/train_df["prev_6_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_8wk_avg_state_val"] = (train_df["prev_8_wk_avg"].values/train_df["prev_8_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_12wk_avg_state_val"] = (train_df["prev_12_wk_avg"].values/train_df["prev_12_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_52wk_avg_state_val"] = (train_df["prev_52_wk_avg"].values/train_df["prev_52_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_wk_2wk_avg_state_val"] = (train_df["prev_1_wk"].values/train_df["prev_2_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_wk_4wk_avg_state_val"] = (train_df["prev_1_wk"].values/train_df["prev_4_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_wk_52wk_avg_state_val"] = (train_df["prev_1_wk"].values/train_df["prev_52_wk_avg_stt"].values)*train_df[hr_model_agg_col+"_stt"].values
        train_df["prev_1wk_time_zone_val"] = (train_df["prev_1_wk"].values/train_df["prev_1_wk_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_2wk_time_zone_val"] = (train_df["prev_2_wk"].values/train_df["prev_2_wk_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_wk_yr_time_zone_val"] = (train_df["prev_1_wk"].values/train_df[agg_col+"_FY_Avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_2wk_avg_time_zone_val"] = (train_df["prev_2_wk_avg"].values/train_df["prev_2_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_4wk_avg_time_zone_val"] = (train_df["prev_4_wk_avg"].values/train_df["prev_4_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_6wk_avg_time_zone_val"] = (train_df["prev_6_wk_avg"].values/train_df["prev_6_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_8wk_avg_time_zone_val"] = (train_df["prev_8_wk_avg"].values/train_df["prev_8_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_12wk_avg_time_zone_val"] = (train_df["prev_12_wk_avg"].values/train_df["prev_12_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_52wk_avg_time_zone_val"] = (train_df["prev_52_wk_avg"].values/train_df["prev_52_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_wk_2wk_avg_time_zone_val"] = (train_df["prev_1_wk"].values/train_df["prev_2_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_wk_4wk_avg_time_zone_val"] = (train_df["prev_1_wk"].values/train_df["prev_4_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["prev_wk_52wk_avg_time_zone_val"] = (train_df["prev_1_wk"].values/train_df["prev_52_wk_avg_tz"].values)*train_df[hr_model_agg_col+"_tz"].values
        train_df["cagr_prev_wk_5wk_ntnl_val"] = train_df[agg_col].shift()*(1+(train_df[hr_model_agg_col+"_ntnl"].shift() - train_df[hr_model_agg_col+"_ntnl"].shift(5))/train_df[hr_model_agg_col+"_ntnl"].shift(5))
        train_df["cagr_prev_mt_2mt_rol_ntnl_val"] = train_df[agg_col].shift()*(train_df[hr_model_agg_col+"_ntnl"].shift().rolling(min_periods=1, window=4).mean()/train_df[hr_model_agg_col+"_ntnl"].shift(5).rolling(min_periods=1, window=4).mean())
        train_df["cagr_prev_wk_5wk_ntnl_ratio"] = (train_df[hr_model_agg_col+"_ntnl"].shift() - train_df[hr_model_agg_col+"_ntnl"].shift(5))/train_df[hr_model_agg_col+"_ntnl"].shift(5)
        temp_ = train_df[hr_model_agg_col+"_ntnl"].shift(5).rolling(min_periods=1, window=4).mean()
        train_df["cagr_prev_mt_2mt_rol_ntnl_ratio"] = (train_df[hr_model_agg_col+"_ntnl"].shift().rolling(min_periods=1, window=4).mean() - temp_)/temp_

        if mod_type == "Scaler":
            # Scoring on train & test datasets. Compute metric
            train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
            lr_preds = model.predict(train_df[model_cols].tail(n=1))
            train_df[agg_col].iloc[-1] = lr_preds
            train_df[cols_to_scale] = scaler.inverse_transform(train_df[cols_to_scale])
            if train_df[agg_col].iloc[-1] < 0:
                train_df[agg_col].iloc[-1] = min_cap
        elif mod_type == "ScalerMM":
            # Scoring on train & test datasets. Compute metric
            train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
            lr_preds = model.predict(train_df[model_cols].tail(n=1))
            train_df[agg_col].iloc[-1] = lr_preds
            train_df[cols_to_scale] = scaler.inverse_transform(train_df[cols_to_scale])
            if train_df[agg_col].iloc[-1] < 0:
                train_df[agg_col].iloc[-1] = min_cap
    
    test_pred_lr = train_df.tail(n=test_df.shape[0])
    return test_pred_lr


def lasso_model_all(agg_col, model_cols, train_data, test_df, mod_type, ls, pred_start):
    """
    Utility for model building. It performs transformations and build linear model with lasso regularization
    and returns all the model artifacts
    Input:
        agg_col: DV column name
        model_cols: List - Model columns
        train_data: Pandas dataframe - Train data
        test_df: Pandas dataframe - Test data
        cols_to_scale: List - Columns to scale
        mod_type: Scaler type
        ls: Model object
        pred_start: Train test split Fiscal Week
    Output:
        res_trf: Dict: Result summary
        test_pred: Pandas dataframe - Model predictions
        train_data: Pandas dataframe - Model train data
        feature_importance: Pandas dataframe - Feature importance
        ls: Model object
        scaler: Scaler object
    """
    cols_to_scale = model_cols + [agg_col]
    res_trf = {}
    calc = "LINE_ORDERS_ACTUAL"
    min_cap = train_data[agg_col].mean()
    res_trf["min_cap"] = min_cap
    if mod_type == "Scaler":
        # Define scaler
        cols_to_scale = model_cols + [agg_col]
        train_data[agg_col+"_ACT"] = train_data[agg_col]
        # Standardize train dataset
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_data[cols_to_scale] = scaler.fit_transform(train_data[cols_to_scale])
        # Train the model using the training sets
        ls.fit(train_data[model_cols], train_data[agg_col])
        train_data[agg_col] = ls.predict(train_data[model_cols])
        # rescale data back
        train_data[cols_to_scale] = scaler.inverse_transform(train_data[cols_to_scale])
        train_data["Prediction_Trf"] = train_data[agg_col]
        train_data[agg_col] = train_data[agg_col+"_ACT"]
        train_data["Prediction_Trf"] = np.where(train_data["Prediction_Trf"] < 0, min_cap, train_data["Prediction_Trf"])
        # -------------------------------------------------------------------------
        test_pred = predict_test_trf(train_data.copy(), test_df.copy(), ls, model_cols, agg_col, cols_to_scale, mod_type, scaler, pred_start, min_cap)
        # Scoring on train & test datasets. Compute metric

        res_trf["mape"], res_trf["mae"], res_trf["made"], res_trf["wmape"] = cu.calc_mape(test_df[calc], test_pred[agg_col])
        res_trf["mape_13w"], res_trf["mae_13w"], res_trf["made_13w"], res_trf["wmape_13w"] = cu.calc_mape(test_df[calc][4:17], test_pred[agg_col][4:17])
        test_pred["Prediction_Trf"] = test_pred[agg_col]
        test_pred[agg_col] = test_df[agg_col]
    elif mod_type == "ScalerMM":
        # Define scaler
        cols_to_scale = model_cols + [agg_col]
        train_data[agg_col+"_ACT"] = train_data[agg_col]
        scaler = MinMaxScaler()
        train_data[cols_to_scale] = scaler.fit_transform(train_data[cols_to_scale])
        ls.fit(train_data[model_cols], train_data[agg_col])
        train_data[agg_col] = ls.predict(train_data[model_cols])
        # rescale data back
        train_data[cols_to_scale] = scaler.inverse_transform(train_data[cols_to_scale])
        train_data["Prediction_Trf"] = train_data[agg_col]
        train_data[agg_col] = train_data[agg_col+"_ACT"]
        train_data["Prediction_Trf"] = np.where(train_data["Prediction_Trf"] < 0, min_cap, train_data["Prediction_Trf"])
        # -------------------------------------------------------------------------
        test_pred = predict_test_trf(train_data.copy(), test_df.copy(), ls, model_cols, agg_col, cols_to_scale, mod_type, scaler, pred_start, min_cap)
        # Scoring on train & test datasets. Compute metric
        
        res_trf["mape"], res_trf["mae"], res_trf["made"], res_trf["wmape"] = cu.calc_mape(test_df[calc], test_pred[agg_col])
        res_trf["mape_13w"], res_trf["mae_13w"], res_trf["made_13w"], res_trf["wmape_13w"] = cu.calc_mape(test_df[calc][4:17], test_pred[agg_col][4:17])
        test_pred["Prediction_Trf"] = test_pred[agg_col]
        test_pred[agg_col] = test_df[agg_col]
    
    try:
        res_trf["slctd_feat_hyper"] = "; ".join(list(compress(model_cols, ls.coef_!=0)))
        res_trf["slctd_feat"] = "; ".join(model_cols)
        feature_importance = pd.DataFrame(ls.coef_,
                               index = model_cols,
                                columns=['importance']).sort_values('importance',                                                                 
                                        ascending=False)
    except:
        res_trf["slctd_feat_hyper"] = "; ".join(list(compress(model_cols, ls.best_estimator_.coef_!=0)))
        res_trf["slctd_feat"] = "; ".join(list(compress(model_cols,ls.best_estimator_.coef_)))
        feature_importance = pd.DataFrame(ls.best_estimator_.coef_,
                               index = model_cols,
                                columns=['importance']).sort_values('importance',                                                                 
                                        ascending=False)
    res_trf["feat_slctn"] = "LASSO"
    res_trf["mod_type"] = mod_type
    try:
        res_trf["alpha"] = ls.best_params_
    except:
        res_trf["alpha"] = ls.alpha
    try:
        res_trf["slctd_coeff"] = ls.coef_
    except:
        res_trf["slctd_coeff"] = ls.best_estimator_.coef_
    res_trf["train_mape"], res_trf["train_mae"], res_trf["train_made"], res_trf["train_wmape"] = cu.calc_mape(train_data[calc], train_data["Prediction_Trf"])
    return res_trf, test_pred, train_data, feature_importance, ls, scaler



def train_pred(agg_col, model_cols, train_data, mod_type, ls, scaler, min_cap):
    """
    Utility for FPG X MM Train data predictions. This is one shot predictions for the given number of weeks.
    Input:
        agg_col: DV column name
        model_cols: List - Model columns
        train_data: Pandas dataframe - Train data
        test_df: Pandas dataframe - Test data
        cols_to_scale: List - Columns to scale
        mod_type: Scaler type
        ls: Model object
        min_cap: Minimum cap for negative predictions
    Output:
        train_data: Pandas dataframe - Model train data predictions
    """
    cols_to_scale = model_cols + [agg_col]
    if mod_type == "Scaler":
        # Define scaler
        cols_to_scale = model_cols + [agg_col]
        train_data[cols_to_scale] = scaler.transform(train_data[cols_to_scale])
        train_data[agg_col] = ls.predict(train_data[model_cols])
        # rescale data back
        train_data[cols_to_scale] = scaler.inverse_transform(train_data[cols_to_scale])
        train_data[agg_col] = np.where(train_data[agg_col] < 0, min_cap, train_data[agg_col])
    elif mod_type == "ScalerMM":
        # Define scaler
        cols_to_scale = model_cols + [agg_col]
        train_data[cols_to_scale] = scaler.fit_transform(train_data[cols_to_scale])
        train_data[agg_col] = ls.predict(train_data[model_cols])
        # rescale data back
        train_data[cols_to_scale] = scaler.inverse_transform(train_data[cols_to_scale])
        train_data[agg_col] = np.where(train_data[agg_col] < 0, min_cap, train_data[agg_col])
    return train_data


def lasso_grid_model_all(agg_col, model_cols, train_data, test_df, mod_type_all, pred_start):
    """
    Utility for lasso model hyper parameter tuning
    Input:
        agg_col: DV column name
        model_cols: List - Model columns
        train_data: Pandas dataframe - Train data
        test_df: Pandas dataframe - Test data
        mod_type_all: List - Scaler type
        pred_start: Train test split Fiscal Week
    Output:
        res_trf: Pandas dataframe - Result summary
        ls: Model object
    """
    param = {"alpha":[0.001, 0.01, 0.02, 0.1, 0.2, 0.3, 0.5, 1]}

    out_dict_all = []
    test_df_all = []
    train_df_all = []
    feature_importance_all = []
    
    for mod_type in mod_type_all:

        try:
            # First create the base model to tune
            ls = Lasso()
            # Random search of parameters, using 3 fold cross validation, 
            ls_random = RandomizedSearchCV(estimator = ls, cv = 4, 
                                           param_distributions = param, 
                                           n_iter = 100, verbose=0, random_state=42, n_jobs = 1, scoring = 'r2')

            out_dict, test_res, train_res, feature_importance, ls, scaler = lasso_model_all(agg_col, model_cols, train_data.copy(), test_df.copy(), mod_type, ls_random, pred_start)
        except Exception as e:
            print(e)
            continue
        out_dict_all.append(out_dict)
        test_df_all.append(test_res)
        train_df_all.append(train_res)
        feature_importance_all.append(feature_importance)
    res = pd.DataFrame(out_dict_all)
    res = res.sort_values("mape")
    try:
        res_n = res[~(res["slctd_feat"] == "")] 
        res_n.reset_index(drop = True, inplace = True)
        ls = Lasso(**res_n.ix[0]["alpha"])
        res = res_n.copy()
    except:
        ls = Lasso(**res.ix[0]["alpha"])
    return res, ls


def inhome_model(multi_thread_parameters, file_name):
    """
    Utility for FPG X MM model building & evalaution. This function runs in multiple thread. Each FPG X MM goes through
    model data prep, model evaluation & model building.
    Input:
        multi_thread_parameters: List - Model building & evalaution parameters
        file_name: File name
    Output:
        Model results & model object dump
    """

    start_idx = multi_thread_parameters[0]
    end_idx = multi_thread_parameters[1]
    grp_cols = multi_thread_parameters[2]
    model_base = multi_thread_parameters[3]
    agg_col = multi_thread_parameters[4]
    base_list = multi_thread_parameters[5]
    model_data = multi_thread_parameters[6]
    cal_df = multi_thread_parameters[7]
    ntnl_model = multi_thread_parameters[8]
    model_cols_all = multi_thread_parameters[9]
    file_name = multi_thread_parameters[10]
    n = multi_thread_parameters[11]
    start_time = multi_thread_parameters[12]
    sales_prop = multi_thread_parameters[13]
    state_model = multi_thread_parameters[14]
    timezone_model = multi_thread_parameters[15]
    fisc_calender = multi_thread_parameters[16]
    model_dump_folder = multi_thread_parameters[17]
    pred_start = multi_thread_parameters[18]
    mod_start =  multi_thread_parameters[19]
    lasso_val = multi_thread_parameters[20]
    base_dir = multi_thread_parameters[21]
    version = multi_thread_parameters[22]
    log_path = multi_thread_parameters[23]
    tr_log_all = multi_thread_parameters[24]
    
    results_list = []
    train_df_bst = []
    test_df_bst = []
    feat_imp = []
    err_loc = []
    ###############################################################################
    num_loc = len(base_list)

    for i in range(start_idx, end_idx):

        results = {}
        model_dict = {}
        if i > num_loc-1:
            break
        
        comb_name = base_list[i].replace('/','-')
        tr_log = cu.my_custom_logger(base_dir, log_path+version, base_list[i].replace('/','-' ))
        tr_log.info("Model data preparation")
        
        try:
            train_data, test_data, avl_flg = fu.model_data_prep(grp_cols, agg_col, model_base, base_list[i], model_data, cal_df, ntnl_model, state_model, timezone_model, fisc_calender, sales_prop, pred_start, mod_start)
            train_data[model_cols_all] = train_data[model_cols_all].fillna(method="bfill")
            to_keep_model_cols = model_cols_all.copy()
            
            if test_data.shape[0] == test_data[agg_col].isna().sum(): # no volume in Fiscal year 2020
                tr_log.info("No volume in test data for " + str(base_list[i]))
                continue
            mod_type_all = ["Scaler", "ScalerMM"]
            tr_df = train_data[train_data.FISC_WK_OF_MTH_ID < lasso_val].copy()
            val_df = train_data[train_data.FISC_WK_OF_MTH_ID >= lasso_val].copy()
            train_data_act = train_data[train_data.FISC_WK_OF_MTH_ID < pred_start].copy()
            ############Run random cv - lasso ##########################################
            tr_log.info("Hyper parameter tuning and model evaluation")
            result_tr_val_lasso_n, ls_model_n = lasso_grid_model_all(agg_col, to_keep_model_cols.copy(), tr_df.copy(), val_df.copy(), mod_type_all, pred_start)
            eval_model_cols = result_tr_val_lasso_n.ix[0]["slctd_feat_hyper"].replace(" ","").split(";")
    
            tr_log.info("Lasso model building")
            if len(eval_model_cols[0]) == 0:
                result_lasso_n, test_df_f_n, train_df_f_n, feature_importance_ls, ls, scaler_ls = lasso_model_all(agg_col, to_keep_model_cols.copy(), train_data_act.copy(), test_data.copy(), result_tr_val_lasso_n.ix[0]["mod_type"], ls_model_n, pred_start)
                feature_importance_ls.reset_index(inplace = True)
                result_lasso_n["model_col"] = 0
                ls_slc_model_cols = result_lasso_n["slctd_feat_hyper"].replace(" ","").split(";")
                ls_slc_model_cols_f = result_lasso_n["slctd_feat"].replace(" ","").split(";")
            else:
                result_lasso_n, test_df_f_n, train_df_f_n, feature_importance_ls, ls, scaler_ls = lasso_model_all(agg_col, eval_model_cols.copy(), train_data_act.copy(), test_data.copy(), result_tr_val_lasso_n.ix[0]["mod_type"], ls_model_n, pred_start)
                feature_importance_ls.reset_index(inplace = True)
                result_lasso_n["model_col"] = 1
                ls_slc_model_cols = result_lasso_n["slctd_feat_hyper"].replace(" ","").split(";")
                ls_slc_model_cols_f = result_lasso_n["slctd_feat"].replace(" ","").split(";")
            if len(ls_slc_model_cols[0]) == 0:
                ls_slc_model_cols = to_keep_model_cols.copy()
    
            feature_importance_ls[model_base] = base_list[i]
            result_tr_val_lasso_n = result_tr_val_lasso_n.add_suffix('_tr_eval')
            result_lasso_n.update(result_tr_val_lasso_n.ix[0][2:])
            results.update(result_lasso_n)
            test_df_bst.append(test_df_f_n)
            train_df_bst.append(train_df_f_n)
            feat_imp.append(feature_importance_ls)
            tr_log.info("Dumping final model artifacts")
            model_dict = {'model_comb': base_list[i], 
                          'selected_fts': ls_slc_model_cols_f, 
                          'scale_model': scaler_ls, 
                          'forecast_model': ls, 
                          'trans_model': result_lasso_n["mod_type"],
                          'min_cap': result_lasso_n["min_cap"]}
            with open(os.path.join(base_dir, model_dump_folder, version+"/", base_list[i].replace('/','-') + '_model.sav'), 'wb') as f:
                pickle.dump(model_dict, f)
        ###########################################################################
            results.update({model_base: base_list[i]})
            results["train_data_flg"] = avl_flg[0]
            results["test_data_flg"] = avl_flg[1]
            stats_tr, stats_ts = cu.train_test_stats(train_data_act, test_data, agg_col)
            results.update(stats_tr)
            results.update(stats_ts)
            results_list.append(results)
            tr_log.info("[" + str(base_list[i]) + ": " + str(results["mape_13w"]) + " - " + str(results["feat_slctn"]) + "]")

        except:
            tr_log.exception(comb_name+".log - Error while model scoring")
            err_loc.append(base_list[i])
            tr_log_all.exception(comb_name+".log - Refer the log file to identify the issue")

        core_pct = ((i - start_idx + 1)/(end_idx - start_idx)) * 100
        print("combinations - %s => %.2f%% - %.2fs; Core %.0f => %.2f%%" % (base_list[i], results["mape_13w"], time.time()-start_time, n, core_pct))
    output_variables = [results_list, train_df_bst, test_df_bst, err_loc, feat_imp]
    if not os.path.isdir(file_name):
        os.mkdir(file_name)
    with open(os.path.join(file_name, "".join(["pkl_obj_", str(n), ".pkl"])), "wb") as file:
        pickle.dump(output_variables, file)
        

       
def inhome_tz_state_model(multi_thread_parameters, file_name):
    """
    Utility for FPG X Time zone & FPG X State model model building & evalaution. This function runs in multiple thread. Each 
    FPG X Time zone & FPG X State model goes through model data prep, model evaluation & model building.
    Input:
        multi_thread_parameters: List - Model building & evalaution parameters
        file_name: File name
    Output:
        Model results & model object dump
    """

    start_idx = multi_thread_parameters[0]
    end_idx = multi_thread_parameters[1]
    grp_cols = multi_thread_parameters[2]
    model_base = multi_thread_parameters[3]
    agg_col = multi_thread_parameters[4]
    base_list = multi_thread_parameters[5]
    model_data = multi_thread_parameters[6]
    cal_df = multi_thread_parameters[7]
    file_name = multi_thread_parameters[8]
    start_time = multi_thread_parameters[9]
    mod_type = multi_thread_parameters[10]
    n = multi_thread_parameters[11]
    pred_start = multi_thread_parameters[12]
    hol_cols = multi_thread_parameters[13]
    fpg = multi_thread_parameters[14]
    model_dump_folder = multi_thread_parameters[15]
    fb_val = multi_thread_parameters[16]
    base_dir = multi_thread_parameters[17]
    version = multi_thread_parameters[18]
    log_path = multi_thread_parameters[19]
    RSS_fcast = multi_thread_parameters[20]
    fcast_var = multi_thread_parameters[21]
    tr_log_all = multi_thread_parameters[22]
    
    results_list = []
    train_test_df_bst = []
    train_test_df_stt_bst = []

    
    err_loc = []
    
    ###############################################################################
    num_loc = len(base_list)
    for i in range(start_idx, end_idx):
        results = {}
        if i > num_loc-1:
            break
        try:
            comb_name = base_list[i]
            tr_log = cu.my_custom_logger(base_dir, os.path.join(log_path, version), (fpg+"_"+base_list[i]).replace('/', '-') )
            tr_log.info("Model building started for "+base_list[i])
            ntnl_df = model_data[model_data[model_base] == base_list[i]]
            ntnl_df["granularity"] = model_base
            base_var = "granularity"
            base = model_base
            model_filename = fpg+"_"+ base_list[i]+"_"+mod_type
    
            res_fb, train_test, train_test_df = fb_prophet(ntnl_df, agg_col, cal_df, grp_cols, base_var, base, pred_start, hol_cols, fpg, base_dir, model_dump_folder, model_filename, fb_val, version, None, fcast_var)
            tr_log.info("Model building ended for "+base_list[i])
            train_test[model_base] = base_list[i]
            train_test_df[model_base] = base_list[i]
            train_test_df.rename(columns={"prev_yr_wk": "prev_yr_wk_ntnl",
                                    "prev_1_wk": "prev_1_wk_ntnl",
                                    "prev_2_wk": "prev_2_wk_ntnl",
                                    "prev_2_wk_avg": "prev_2_wk_avg_ntnl",
                                    "prev_4_wk_avg": "prev_4_wk_avg_ntnl",
                                    "prev_6_wk_avg": "prev_6_wk_avg_ntnl",
                                    "prev_8_wk_avg": "prev_8_wk_avg_ntnl",
                                    "prev_12_wk_avg": "prev_12_wk_avg_ntnl",
                                    "prev_52_wk_avg": "prev_52_wk_avg_ntnl",
                                    "Fb_Prediction": "Fb_Prediction_ntnl",
                                    "Fb_Prediction_old": "Fb_wo_ridge_ntnl",
                                    agg_col: agg_col+"_ntnl"}, inplace=True)
            ntnl_avg = train_test_df.groupby("FISC_YR_NBR")["Fb_Prediction_ntnl"].mean().reset_index()
            ntnl_avg.columns = ["FISC_YR_NBR", agg_col+"_FY_Avg_ntnl"]
            train_test_df = train_test_df.merge(ntnl_avg, how="left")
            ntnl_rs_grp = train_test_df[["FISC_WK_OF_MTH_ID", model_base, "Fb_Prediction_ntnl", agg_col+"_ntnl", "Fb_wo_ridge_ntnl",
                                    "prev_yr_wk_ntnl", agg_col+"_FY_Avg_ntnl",
                                    "prev_1_wk_ntnl", "prev_2_wk_ntnl", 
                                    "prev_2_wk_avg_ntnl",
                                    "prev_4_wk_avg_ntnl", "prev_6_wk_avg_ntnl",
                                    "prev_8_wk_avg_ntnl", "prev_12_wk_avg_ntnl", "prev_52_wk_avg_ntnl"]]
            ntnl_rs_grp.columns = ntnl_rs_grp.columns.str.replace('ntnl', mod_type)
            results.update(res_fb)
            train_test_df_bst.append(train_test)
            train_test_df_stt_bst.append(ntnl_rs_grp)
            results.update({model_base: base_list[i]})
            results_list.append(results)
            logging.info("[" + str(base_list[i]) + ": " + str(results["mape_13w"]))
        
        except:
            tr_log.exception(comb_name+".log - Error while model scoring")
            err_loc.append(base_list[i])
            tr_log_all.exception(comb_name+".log - Refer the log file to identify the issue")
        
        core_pct = ((i - start_idx + 1)/(end_idx - start_idx)) * 100
        print("combinations - %s => %.2f%% - %.2fs; Core %.0f => %.2f%%" % (base_list[i], results["mape_13w"], time.time()-start_time, n, core_pct))
    
    output_variables = [results_list, train_test_df_bst, train_test_df_stt_bst, err_loc]
    
    if not os.path.isdir(file_name):
        os.mkdir(file_name)
        
    with open(os.path.join(file_name, "".join(["pkl_obj_", str(n), ".pkl"])), "wb") as file:
        pickle.dump(output_variables, file)
        

def service_hours_predictions(multi_thread_parameters, file_name):
    """
    Utility for RSS X MM Service hours prediction. This function runs in multiple thread.
    Input:
        multi_thread_parameters: List - Model building & evalaution parameters
        file_name: File name
    Output:
        Model results dump
    """
    logging.basicConfig(level=logging.INFO, filename=file_name+".log")
    start_idx = multi_thread_parameters[0]
    end_idx = multi_thread_parameters[1]
    file_name = multi_thread_parameters[2]
    n = multi_thread_parameters[3]
    start_time = multi_thread_parameters[4]
    df_all = multi_thread_parameters[5]
    cal_df = multi_thread_parameters[6]
    fisc_calender = multi_thread_parameters[7]
    pred_start = multi_thread_parameters[8]
    mnts_cols = multi_thread_parameters[9]
    tr_test = multi_thread_parameters[10]
    var_list = multi_thread_parameters[11]
    base_list = multi_thread_parameters[12]
    grp_mm_sparsity = multi_thread_parameters[13]
    log_path = multi_thread_parameters[14]
    tr_log_all = multi_thread_parameters[15]
    base_dir = multi_thread_parameters[16]
    version = multi_thread_parameters[17]
    
    results_list = []
    data_all = []
    err_loc = []

    
    num_loc = len(base_list)
    for i in range(start_idx, end_idx):

        if i > num_loc-1:
            
            break
        
        fpg_name = base_list[i].replace('/', '-')
        try:
            tr_log = cu.my_custom_logger(base_dir, log_path+version+'/', fpg_name+"_Service_Hour" )
            pivot_var = "MM"
            df_grp_fpg = df_all[df_all["RSS"] == base_list[i]].copy()
            df_grp_fpg_train = df_grp_fpg[df_grp_fpg.FISC_WK_OF_MTH_ID < pred_start]
            level_list = list(df_grp_fpg[pivot_var].unique())
            # -----------------------------------------------------------------------------
            ntnl_pred_col = "LINE_ORDERS"
            df_grp_act = df_grp_fpg.copy()
            df_grp_act.columns = df_grp_act.columns.str.replace(ntnl_pred_col, ntnl_pred_col+"_ACT")
        
            for mn_col in mnts_cols:
                df_grp_act.columns = df_grp_act.columns.str.replace(mn_col, mn_col+"_ACT")
                df_grp_act[mn_col+"_ACT"].replace({0: np.nan}, inplace=True)
        
            model_data_orders_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
            model_data_orders_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_orders_df, var_list)
            # -----------------------------------------------------------------------------
            
            tr_log.info("Lead service hours calculation for "+base_list[i])
            
            ntnl_pred_col = "LEAD_MNTS"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):            
            
                tr_log.error(fpg_name + " - Lead hours is not present during the training period")
                tr_log_all.error(fpg_name + " - Lead hours is not present during the training period")
                tr_log_all.error(fpg_name+"_Service_Hour.log - Refer the log file to identify the issue")
                
            else:
                model_data_ld_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_ld_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_ld_df, var_list)
                model_data_sh_melt_df = model_data_ld_melt_df.copy()
            
            tr_log.info("Helper service hours calculation for "+base_list[i])
            ntnl_pred_col = "HELPER_MNTS"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                pass
            else:
                model_data_hp_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_hp_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_hp_df, var_list)
                model_data_sh_melt_df = model_data_ld_melt_df.merge(model_data_hp_melt_df, how = "left")
            
            tr_log.info("Helper float service hours calculation for "+base_list[i])
            ntnl_pred_col = "HELPER_OVR"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_hpn_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_hpn_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_hpn_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_hpn_melt_df, how = "left")
            
            tr_log.info("Lead unclaimed service hours calculation for "+base_list[i])
            ntnl_pred_col = "LD_UNCLAIMED"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_lduc_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_lduc_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_lduc_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_lduc_melt_df, how = "left")
            
            tr_log.info("Helper unclaimed service hours calculation for "+base_list[i])
            ntnl_pred_col = "HP_UNCLAIMED"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_hpuc_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_hpuc_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_hpuc_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_hpuc_melt_df, how = "left")
            
            tr_log.info("Helper floating service hours calculation for "+base_list[i])
            ntnl_pred_col = "HPN_UNCLAIMED"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_hpnuc_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_hpnuc_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_hpnuc_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_hpnuc_melt_df, how = "left")
            
            tr_log.info("Lead drive hours calculation for "+base_list[i])
            ntnl_pred_col = "LD_DRIVE"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_lddt_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_lddt_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_lddt_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_lddt_melt_df, how = "left")
            
            tr_log.info("Helper drive hours calculation for "+base_list[i])
            ntnl_pred_col = "HP_DRIVE"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_hpdt_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_hpdt_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_hpdt_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_hpdt_melt_df, how = "left")
            
            tr_log.info("Helper float drive hours calculation for "+base_list[i])
            ntnl_pred_col = "HPN_DRIVE"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_hpndt_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_hpndt_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_hpndt_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_hpndt_melt_df, how = "left")
            
            tr_log.info("Appointments  calculation for "+base_list[i])
            ntnl_pred_col = "APPT_ID"
            if df_grp_fpg_train[ntnl_pred_col].sum() == 0 or sum(df_grp_fpg_train[ntnl_pred_col].isna()) == len(df_grp_fpg_train):
                print(f'{ntnl_pred_col} not calculated')
                
                pass
            else:
                model_data_hpndt_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp_fpg, cal_df, level_list, fisc_calender, pred_start)
                model_data_hpndt_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_hpndt_df, var_list)
                model_data_sh_melt_df = model_data_sh_melt_df.merge(model_data_hpndt_melt_df, how = "left")
            

           # -----------------------------------------------------------------------------
        
            model_data_melt_all = pd.merge(model_data_sh_melt_df, model_data_orders_melt_df,  how = "left")
            model_data_melt_all.columns = model_data_melt_all.columns.str.replace("variable", "MM")
            model_data_melt_all["RSS"] = base_list[i]
            # -----------------------------------------------------------------------------
            
            act_cols = df_grp_act.columns[df_grp_act.columns.str.contains("_ACT")].to_list()
            model_data_melt_all = pd.merge(model_data_melt_all, 
                                              df_grp_act[["FISC_WK_OF_MTH_ID", "MM"]+ act_cols], how ="left")
            # -----------------------------------------------------------------------------
            pred_col = "Prediction_Trf"
            test_grp = tr_test[tr_test["RSS"] == base_list[i]].copy()
            test_grp = test_grp[["FISC_WK_OF_MTH_ID", "MM", pred_col]].sort_values("FISC_WK_OF_MTH_ID")
            model_data_melt_all = pd.merge(model_data_melt_all, test_grp, how ="left")
            #----------------------------------------
        
            model_data_melt_all["RSS"] = base_list[i]
            model_data_melt_all["RSS_MM"] = model_data_melt_all["RSS"]+"_"+model_data_melt_all["MM"]
            model_data_melt_all = pd.merge(model_data_melt_all, grp_mm_sparsity)
            # -----------------------------------------------------------------------------
            for mn_col in mnts_cols:
                try:
                    model_data_melt_all["IDV_"+mn_col] = model_data_melt_all[mn_col+"_VALUE"]/model_data_melt_all["LINE_ORDERS_VALUE"]
                except:
                    continue
                model_data_melt_all["IDV_"+mn_col+"_Pred"] = model_data_melt_all["IDV_"+mn_col]*model_data_melt_all[pred_col]
                model_data_melt_all["IDV_"+mn_col+"_ACT"] = model_data_melt_all["IDV_"+mn_col]*model_data_melt_all["LINE_ORDERS_ACT"]
            print(model_data_melt_all["RSS_MM"].nunique(), base_list[i])
            data_all.append(model_data_melt_all)
            
        except:
            tr_log.exception(fpg_name+"_Service_Hour.log - Error while calculating service hours")
            tr_log_all.exception(fpg_name+"_Service_Hour.log - Refer the log file to identify the issue")
            
        core_pct = ((i - start_idx + 1)/(end_idx - start_idx)) * 100
        print("combinations - %s => - %.2fs; Core %.0f => %.2f%%" % (base_list[i], time.time()-start_time, n, core_pct))
    output_variables = [results_list, data_all, data_all, err_loc]
    if not os.path.isdir(file_name):
        os.mkdir(file_name)
    with open(os.path.join(file_name, "".join(["pkl_obj_", str(n), ".pkl"])), "wb") as file:
        pickle.dump(output_variables, file)


def sparse_predictions(df_grp, cal_df, fisc_calender, pred_start, fpg, res_all_sp):
    """
    Predictions for FPG X Minor Market sparser combinations. These are combinations where predictive models cannot be built
    mainly because of the data quality like number of weeks present, average line orders per week etc. This functions calls
    lag_variable_calculation() where lag values like previous year mid 5 week rolling average is calculated based on historical
    line orders and these will become the predictions for these sparser combinations.
    Input:
        df_grp: Pandas dataframe - dataframe with DV
        cal_df: Pandas dataframe - Holiday calendar
        fisc_calender: Pandas dataframe - Fiscal calendar lookup
        pred_start: Prediction start date
        fpg: FPG name
        res_all_sp: List - FPG X MM predictions
    Output:
        res_all_sp: List - FPG X CSS results appended to a list
    """
    pivot_var = "MM"
    ntnl_pred_col = "LINE_ORDERS"
    var_list_avg = ['pv_yr_wk', 'pv_yr', 'pv_yr_qtr', 'pv_yr_mth', 'pv_yr_mid_3wk', 'pv_yr_mid_5wk', 'pv_2wk', 'pv_4wk']
    level_list = list(df_grp[pivot_var].unique())
    model_data_orders_df = fu.lag_variable_calculation(ntnl_pred_col, pivot_var, df_grp, cal_df, level_list, fisc_calender, pred_start)
    model_data_orders_melt_df = cu.lag_variable_melt(ntnl_pred_col, model_data_orders_df, var_list_avg)
    model_data_orders_melt_df.columns = model_data_orders_melt_df.columns.str.replace("variable", pivot_var)
    model_data_orders_melt_df["RSS"] = fpg
    model_data_orders_melt_df["RSS_MM"] = model_data_orders_melt_df["RSS"]+"_"+model_data_orders_melt_df[pivot_var]
    model_data_orders_melt_df = pd.merge(model_data_orders_melt_df,
                                      df_grp[["FISC_WK_OF_MTH_ID", pivot_var, ntnl_pred_col, "SPARSITY"]], how ="left")
    var_list_div = {'pv_yr_wk': 1, 'pv_yr': 52, 'pv_yr_qtr': 13, 'pv_yr_mth': 4, 'pv_yr_mid_3wk': 3, 'pv_yr_mid_5wk': 5, 'pv_2wk': 2, 'pv_4wk': 4}
    for var in var_list_avg:
        model_data_orders_melt_df.loc[model_data_orders_melt_df["Variable_Type"]==var, "LINE_ORDERS_VALUE"] = model_data_orders_melt_df.loc[model_data_orders_melt_df["Variable_Type"]==var, "LINE_ORDERS_VALUE"]/var_list_div[var]
    model_data_orders_melt_df.columns = model_data_orders_melt_df.columns.str.replace("LINE_ORDERS_VALUE", "Prediction_Trf")
    model_data_orders_melt_df = model_data_orders_melt_df[['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID',
           pivot_var, 'Variable_Type', 'Prediction_Trf', 'RSS', 'RSS_MM']]
    res_all_sp.append(model_data_orders_melt_df)
    return res_all_sp
 

def prev_avg_one_shot(mm_list, train_df_rpt, test_df_rpt):
    """
    Function to re-score for the forecast period by including actual values
    for the period between the training period and the forecasting period.
    Prediction is based on the previous 5-week avg. with 4-week hold-out.
    Input:
        agg_col: String - Dependent variable
        train_df_rpt: Pandas Dataframe - Training data
        test_df_rpt: Pandas Dataframe - Test data
    Output:
        test_pred_lr_rpt: Pandas Dataframe - Test data with rescored prediction
    """
    temp_train = train_df_rpt.copy()
    for i in range(test_df_rpt.shape[0]):
        temp_train = train_df_rpt.append(test_df_rpt.iloc[i])
        temp_train[mm_list] = temp_train[mm_list].shift(5).rolling(min_periods=1, window=5).mean()
        train_df_rpt = train_df_rpt.append(temp_train.iloc[-1], ignore_index=True, sort=False)
    test_pred_lr_rpt = train_df_rpt.tail(n=test_df_rpt.shape[0])
    return test_pred_lr_rpt



def prev_avg_sparsity_scoring(mm_list, train_df, test_df, prop_type):
    
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
    test_df_prop = prev_avg_one_shot(mm_list, train_df.copy(), test_df.copy())
    train_df_all = train_df_all.append(test_df_prop, ignore_index=True, sort=False)
    train_df_all[mm_list] = train_df_all[mm_list].fillna(method='ffill')
    train_df_all[mm_list] = train_df_all[mm_list].fillna(method='bfill')
    train_df_all = train_df_all[["FISC_WK_OF_MTH_ID", 'FISC_MTH_NBR', 'FISC_QTR_NBR', 'FISC_YR_NBR']+mm_list]
    train_df_all.rename(columns=dict(zip(mm_list, [s + prop_type for s in mm_list])),inplace=True)
    return train_df_all


def avg_predictions(df_grp_2, cal_df, fisc_calender, pred_start, fpg, res_all_sp):
    """
    Predictions for FPG X Minor Market sparser combinations. These are combinations where predictive models cannot be built
    mainly because of the data quality like number of weeks present, average line orders per week etc. This functions calls
    lag_variable_calculation() where lag values like previous year mid 5 week rolling average is calculated based on historical
    line orders and these will become the predictions for these sparser combinations.
    Input:
        df_grp: Pandas dataframe - dataframe with DV
        cal_df: Pandas dataframe - Holiday calendar
        fisc_calender: Pandas dataframe - Fiscal calendar lookup
        pred_start: Prediction start date
        fpg: FPG name
        res_all_sp: List - FPG X MM predictions
    Output:
        res_all_sp: List - FPG X CSS results appended to a list
    """
    pivot_var = "MM"
    agg_col = "LINE_ORDERS"
    var_list = ['prev_5wk_avg_4wk_hld']
    level_list = list(df_grp_2[pivot_var].unique())
    
    tt = df_grp_2.copy()
    level = "wk"
    # Rearrange date by fiscal weeks
    tt = cu.merge_cal_tt(tt, cal_df.drop('FISC_EOW_DT', axis = 1), level)
    tt[agg_col+"_ACT"] = tt[agg_col]
    tt[agg_col].replace(np.nan, 0, inplace=True)
    
    model_data_pivot = pd.pivot_table(tt,
                                      index=['FISC_YR_NBR', 'FISC_MTH_NBR','FISC_WK_OF_MTH_ID', 'FISC_WK_OF_YR_NBR', 'FISC_QTR_NBR'],
                                      columns=pivot_var,
                                      values=agg_col).reset_index()
    model_data_pivot = cu.merge_cal_tt(model_data_pivot, cal_df, level)

    model_data_pivot = cu.fillna_values(model_data_pivot, pred_start)
    for i in level_list:
        if str(i) not in model_data_pivot.columns:
            model_data_pivot[str(i)] = 0
    tt = model_data_pivot.copy()
    train = tt[(tt.FISC_WK_OF_MTH_ID < pred_start) ]
    test = tt[(tt.FISC_WK_OF_MTH_ID >= pred_start) ]
    for i in level_list:
        if str(i) not in train.columns:
            train[str(i)] = 0
        if str(i) not in test.columns:
            test[str(i)] = 0
    
    model_data_orders_df = prev_avg_sparsity_scoring(level_list, train, test, "_salesMAPP_prev_5wk_avg_4wk_hld")
    model_data_orders_melt_df = cu.lag_variable_melt(agg_col, model_data_orders_df, var_list)
    model_data_orders_melt_df.columns = model_data_orders_melt_df.columns.str.replace("variable", pivot_var)
    model_data_orders_melt_df["RSS"] = fpg
    model_data_orders_melt_df["RSS_MM"] = model_data_orders_melt_df["RSS"]+"_"+model_data_orders_melt_df[pivot_var]
    model_data_orders_melt_df = pd.merge(model_data_orders_melt_df, 
                                      df_grp_2[["FISC_WK_OF_MTH_ID", pivot_var, agg_col, "SPARSITY"]], how ="left")
    
    model_data_orders_melt_df.columns = model_data_orders_melt_df.columns.str.replace("LINE_ORDERS_VALUE", "Prediction_Trf")
    model_data_orders_melt_df = model_data_orders_melt_df[['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID',
           pivot_var, 'Prediction_Trf', 'RSS', 'RSS_MM']]
    res_all_sp.append(model_data_orders_melt_df)
    
    return res_all_sp