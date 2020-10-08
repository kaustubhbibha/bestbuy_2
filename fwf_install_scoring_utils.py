"""
In-Home Install Future The Business - Service Line Orders Forecasting

This script conatins the helper functions used for scoring Service Line orders

This file requires following scripts to execute
    * fwf_install_common_utils.py - functions containing common utilities
    * fwf_install_feature_utils.py - functions containing feature engineering utils
    * fwf_install_model_utils.py - functions containing model building utils
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import logging
import fwf_install_common_utils as cu
import fwf_install_feature_utils as fu
import fwf_install_model_utils as mu


def score_national(grp_cols, base, base_var, pred_start, cal_df_score, base_dir, model_path, model_filename, fc, mod_type, version, sales_selec_grp):
    """
    Utility to score FC X Natioanal. It calls Fb prophet and Ridge model predictions
    and calls functions which creates rolling average and other variables from predictions to be used by FC X MM level models.
    Input:
        grp_cols: List - Group column list
        base: FC value
        base_var: FC category column name
        pred_start: Train test split Fiscal week
        cal_df_score: Pandas dataframe - Model scoring data
        base_dir: Framework parent directory
        model_path: Directory to dump model
        model_filename: Model file name
        fc: FC name
        mod_type: Model type
        version: Model trained version
        sales_selec_grp: Pandas dataframe - Sales data
    Output:
        score_df_grp_ntnl: Pandas dataframe - Ridge results
    """
    
    with open(os.path.join(base_dir, model_path, version, model_filename.replace('/','-') + '_model.sav'), 'rb') as f:
        model_pack = pickle.load(f)
    cal_df_dum = cal_df_score.copy()
    df = cal_df_score[["FISC_WK_OF_MTH_ID"]].drop_duplicates()
    m_fb = model_pack['model_fb']
    m_rid = model_pack['model_ridge']
    scaler = model_pack['scaler']
    min_cap_fb = model_pack['min_cap_fb']
    min_cap_rid = model_pack['min_cap_rid']
    print(min_cap_rid)
    df_score = cu.merge_cal_tt(df, cal_df_score, 'wk')
    sales_selec_grp["SALES"] = sales_selec_grp["SALES"]
    df_score = pd.merge(df_score, sales_selec_grp, how = "left")
    for_fb = df_score[['FISC_EOW_DT', 'SALES']]
    for_fb = for_fb.rename(columns = {'FISC_EOW_DT':'ds'})
    for_fb['ds'] = pd.to_datetime(for_fb['ds'])
    yhat = m_fb.predict(for_fb[['ds', 'SALES']])
    for_fb['yhat'] = yhat['yhat'].values
    for_fb['yhat'] = np.where(for_fb['yhat'].values <= 0, min_cap_fb, for_fb['yhat'].values)
    for_fb = for_fb.rename(columns = {'ds':'FISC_EOW_DT'})
    df_score = pd.concat([df_score, for_fb], axis = 1, join = 'inner')
    df_temp = df_score.copy()
    col_dummies = model_pack['ridge_dum_cols'][0]
    columns = model_pack['ridge_cols']
    dum_vars_all = []
    dums = pd.get_dummies(cal_df_dum[col_dummies].astype(int), prefix = col_dummies)
    dum_vars_all += list(dums.columns)
    dum_vars = []
    dums = pd.get_dummies(df_temp[col_dummies].astype(int), prefix = col_dummies)
    dum_vars += list(dums.columns)
    df_temp = pd.concat([df_temp, dums], axis = 1)
    for var in dum_vars_all:
        if var not in df_temp.columns:
            df_temp[var] = 0
    df_temp['filler'] = df_temp['yhat']
    temp_cols = columns + dum_vars_all + ['yhat']
    new = scaler.transform(df_temp[temp_cols + ['filler']])
    new = pd.DataFrame(new, columns = temp_cols + ['filler'])
    preds = m_rid.predict(new[temp_cols])
    new['filler'] = preds
    inverse = scaler.inverse_transform(new[temp_cols + ['filler']].values)
    df_temp['Fb_Prediction_old'] = df_temp['yhat'].values
    df_temp['yhat'] = inverse[:,-1]
    df_temp['yhat'] = np.where(df_temp['yhat'] <= 0, min_cap_rid, df_temp['yhat'])
    df_temp = df_temp.rename(columns = {'yhat' : 'Fb_Prediction'})
    df_temp = df_temp.drop(['FISC_EOW_DT', 'filler'], axis = 1)
    ntnl_df_train, ntnl_df_test = fu.model_data_prep_all_ntnl(grp_cols, 'Fb_Prediction', base_var, base,df_temp, cal_df_score, pred_start)
    train_test_df = pd.concat([ntnl_df_train, ntnl_df_test])
    train_test_df.reset_index(drop =True, inplace =True)
    train_test_df = train_test_df.fillna(method='ffill')
    train_test_df = train_test_df.fillna(method='bfill')
    agg_col = "LINE_ORDERS"
    train_test_df.rename(columns={"prev_yr_wk": "prev_yr_wk_"+mod_type,
                            "prev_1_wk": "prev_1_wk_"+mod_type,
                            "prev_2_wk": "prev_2_wk_"+mod_type,
                            "prev_2_wk_avg": "prev_2_wk_avg_"+mod_type,
                            "prev_4_wk_avg": "prev_4_wk_avg_"+mod_type,
                            "prev_6_wk_avg": "prev_6_wk_avg_"+mod_type,
                            "prev_8_wk_avg": "prev_8_wk_avg_"+mod_type,
                            "prev_12_wk_avg": "prev_12_wk_avg_"+mod_type,
                            "prev_52_wk_avg": "prev_52_wk_avg_"+mod_type,
                            "Fb_Prediction": "Fb_Prediction_"+mod_type}, inplace=True)
    score_df_avg = train_test_df.groupby("FISC_YR_NBR")["Fb_Prediction_"+mod_type].mean().reset_index()
    score_df_avg.columns = ["FISC_YR_NBR", agg_col+"_FY_Avg_"+mod_type]
    score_df_grp_ntnl = train_test_df.merge(score_df_avg, how="left")
    score_df_grp_ntnl["RSS"] = fc
    score_df_grp_ntnl = score_df_grp_ntnl[["FISC_WK_OF_MTH_ID", "RSS", "Fb_Prediction_"+mod_type,
                            "prev_yr_wk_"+mod_type, agg_col+"_FY_Avg_"+mod_type,
                            "prev_1_wk_"+mod_type, "prev_2_wk_"+mod_type, 
                            "prev_2_wk_avg_"+mod_type,
                            "prev_4_wk_avg_"+mod_type, "prev_6_wk_avg_"+mod_type,
                            "prev_8_wk_avg_"+mod_type, "prev_12_wk_avg_"+mod_type,
                            "prev_52_wk_avg_"+mod_type]]
    return score_df_grp_ntnl


def inhome_tz_state_model_score(multi_thread_parameters, file_name):
    """
    This utility run in multiple thread and scores for CSS X Time Zone & CSS X State level models. It calls Fb prophet and Ridge model predictions
    and calls functions which creates rolling average and other variables from predictions to be used by CSS X MM level models.
    Input:
        multi_thread_parameters: List - Model building & evalaution parameters
        file_name: File name
    Output:
        Model results dump
    """
    logging.basicConfig(level=logging.INFO, filename=file_name+".log")

    start_idx = multi_thread_parameters[0]
    end_idx = multi_thread_parameters[1]
    grp_cols = multi_thread_parameters[2]
    model_base = multi_thread_parameters[3]
    base_list = multi_thread_parameters[4]
    cal_df = multi_thread_parameters[5]
    file_name = multi_thread_parameters[6]
    start_time = multi_thread_parameters[7]
    mod_type = multi_thread_parameters[8]
    n = multi_thread_parameters[9]
    pred_start = multi_thread_parameters[10]
    model_path = multi_thread_parameters[11]
    css = multi_thread_parameters[12]
    base_dir = multi_thread_parameters[13]
    version = multi_thread_parameters[14]
    log_path = multi_thread_parameters[15]
    score_version = multi_thread_parameters[16]
    sales_selec_grp = multi_thread_parameters[17]
    sc_all_log = multi_thread_parameters[18]
    
    result_all = []
    train_test_df_bst = []
    train_test_df_stt_bst = []
    err_loc = []
    
    ###############################################################################
    num_loc = len(base_list)
    for i in range(start_idx, end_idx):
        if i > num_loc-1:
            break
        try:
            comb_name = base_list[i].replace("/", "-")
            css_name = css.replace("/", "-")
            sc_log = cu.my_custom_logger(base_dir, log_path+score_version+'/', css_name+"_"+comb_name )
            sc_log.info("Model scoring started for "+base_list[i])
            base_var = "granularity"
            base = model_base
            model_filename = css+"_"+ comb_name+"_"+mod_type
            print(model_filename)
            score_df = score_national(grp_cols, base, base_var, pred_start, cal_df, base_dir, model_path, model_filename, css, mod_type, version, sales_selec_grp)
            score_df[model_base] = base_list[i]
            train_test_df_bst.append(score_df)
            train_test_df_stt_bst.append(score_df)
            results = {"RSS":css}
            result_all.append(results)
            sc_log.info("Model scoring ended for "+base_list[i])
        except:
            sc_log.exception(comb_name+".log - Error while model scoring")
            err_loc.append(base_list[i])
            sc_all_log.exception(comb_name+".log - Refer the log file to identify the issue")
        
        core_pct = ((i - start_idx + 1)/(end_idx - start_idx)) * 100
        print("combinations - %s => - %.2fs; Core %.0f => %.2f%%" % (base_list[i], time.time()-start_time, n, core_pct))
    output_variables = [result_all, train_test_df_bst, train_test_df_stt_bst, err_loc]
    if not os.path.isdir(file_name):
        os.mkdir(file_name)
    with open(os.path.join(file_name, "".join(["pkl_obj_", str(n), ".pkl"])), "wb") as file:
        pickle.dump(output_variables, file)


def inhome_model_score(multi_thread_parameters, file_name):
    """
    This utility run in multiple thread and scores for CSS X MM models. It calls functions which collates all the variables created 
    in the higher level models and prepare data for model scoring. 
    Input:
        multi_thread_parameters: List - Model building & evalaution parameters
        file_name: File name
    Output:
        Model results dump
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
    stt_model = multi_thread_parameters[9]
    tz_model = multi_thread_parameters[10]
    file_name = multi_thread_parameters[11]
    n = multi_thread_parameters[12]
    start_time = multi_thread_parameters[13]
    sales_prop = multi_thread_parameters[14]
    fisc_calender = multi_thread_parameters[15]
    model_dump_folder = multi_thread_parameters[16]
    pred_start = multi_thread_parameters[17]
    base_dir = multi_thread_parameters[18]
    log_path = multi_thread_parameters[19]
    score_version = multi_thread_parameters[20]
    version = multi_thread_parameters[21]
    start = multi_thread_parameters[22]
    sc_all_log = multi_thread_parameters[23]
    results_list = []
    test_df_bst = []
    err_loc = []
    ###############################################################################
    num_loc = len(base_list)
    for i in range(start_idx, end_idx):
        model_dict = {}
        if i > num_loc-1:
            break
        try:
            
            comb_name = base_list[i].replace("/", "-")
            ins_log = cu.my_custom_logger(base_dir, log_path+score_version+"/", comb_name)
            ins_log.info("Scoring started for "+base_list[i])
            ins_log.info("Loding model dumps for scoring")
            with open(os.path.join(base_dir, model_dump_folder, version, comb_name + '_model.sav'), 'rb') as f:
                model_dict = pickle.load(f)
            
            ins_log.info("Model data preparation")
            train_data, test_data, avl_flg = fu.model_data_prep(grp_cols, agg_col, model_base, base_list[i], model_data.copy(), cal_df, ntnl_model, stt_model, tz_model, fisc_calender, sales_prop, pred_start, start)
            train_data = train_data.fillna(method='ffill')
            train_data = train_data.fillna(method='bfill')
            ins_log.info("Test prediction")
            result = mu.predict_test_trf(train_data.copy(), test_data.copy(), model_dict['forecast_model'],  model_dict['selected_fts'], agg_col, model_dict['selected_fts']+[agg_col], model_dict['trans_model'], model_dict['scale_model'], pred_start, model_dict['min_cap'])
            ins_log.info("Train prediction")
            result_train = mu.train_pred(agg_col, model_dict['selected_fts'], train_data.copy(), model_dict['trans_model'], model_dict['forecast_model'], model_dict['scale_model'], model_dict['min_cap'])
            result = pd.concat([result_train, result], axis = 0)
            result["Prediction_Trf"] = result["LINE_ORDERS"]
            test_df_bst.append(result)
        except Exception as e:
            ins_log.exception(comb_name+".log - Error while model scoring")
            err_loc.append(base_list[i])
            sc_all_log.exception(comb_name+".log - Refer the log file to identify the issue")
            
        core_pct = ((i - start_idx + 1)/(end_idx - start_idx)) * 100
        ins_log.info("Scoring ended for "+base_list[i])
        print("combinations - %s => - %.2fs; Core %.0f => %.2f%%" % (base_list[i], time.time()-start_time, n, core_pct))
    output_variables = [results_list, test_df_bst, test_df_bst, err_loc]
    comb_file_name = file_name.replace("/", "-")
    if not os.path.isdir(comb_file_name):
        os.mkdir(comb_file_name)
    with open(os.path.join(comb_file_name, "".join(["pkl_obj_", str(n), ".pkl"])), "wb") as file:
        pickle.dump(output_variables, file)

