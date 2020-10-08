"""In-Home Install Future Work Force - Service Line Orders Forecasting

This script allows the user to build service line orders forecasting
models for In-Home Install Future Work Force (IHI-FWF)

This script reads specifications as specified in the the config

This file requires following scripts to execute
    * fwf_install_common_utils.py - functions containing common utilities
    * fwf_install_feature_utils.py - functions containing feature engineering utils
    * fwf_install_model_utils.py - functions containing model building utils
    * fwf_install_data_prep_utils.py - functions containing data prep utils
    * fwf_install_post_processing.py - functions containing post processing utils
"""

import math
import os
import yaml
os.environ["TMPDIR"] = "/opt/appdata/temp/python"
main_dir = "/opt/appdata/development/labor_services/Install_FWF/svc_labor_inhome_install_sh_fcast/FWF"
os.chdir(main_dir)
stream = open(os.path.join(os.getcwd(), "config", "fwf_install_training_config.yaml"), 'r')
config = yaml.load(stream)

import sys
sys.path.append("./utilities")

import teradata
import time
import pandas as pd
import multiprocessing as mp
import logging
import fwf_install_model_utils as ut
import fwf_install_data_prep_utils as dp
import fwf_install_feature_utils as fu
import fwf_install_common_utils as cu
import fwf_install_post_processing_utils as pp
import warnings
import shutil
import pickle

warnings.filterwarnings("ignore")

os.environ["ODBCINI"] = "/opt/teradata/client/ODBC_64/odbc.ini"
os.environ["TMPDIR"] = "/opt/appdata/temp/python"

udaExec = teradata.UdaExec (appName="HelloWorld", version="1.0", logConsole=False, 
                            odbcLibPath="/opt/teradata/client/15.10/lib64/libodbc.so", configureLogging=False)

conn = udaExec.connect(method="odbc", dsn="Teradata_DSN")


thread_st = time.time()
version = config['version']
cores = int(config['cores'])
start = int(config['start'])
end = int(config['end'])
num_test = int(config['num_test'])
lkp_dir = config['lkp_dir']
log_path = config['log_path']
model_dir = config['model_dir']
base_dir = config['base_dir']
model_dump_folder = config['model_dump_folder']
data_flag = config["data_flag"]
data_dump = config['data_dump']
hol_cols = config['hol_cols']
model_cols = config['model_cols'].copy()
mnts_cols = config['mnts_cols']

cov_param = config["cov_buck"]
sparsity_params = config['grp_mm_sparsity']
loc_params = config['loc_state_all']
fisc_cal_params = config['fisc_calendar']
cal_param = config["cal_df"]

line_orders_df = config["line_orders_df"]
sales_df = config["sales_df"]

forecast_df = config["forecast_df"]

hlpr_time_params = config['helper_time']
svc_sku_params = config['service_sku']
dt_params = config['drive_time']
svc_cls_lkp = config['service_cls_bby']
location_params = config['location']
sales_loc = config['location_sales']


def check_dates(start, end, fisc_calender, num_test, tr_log):
    """
    Check validity of date parameters entered.
    Input:
        start: str - Start date of training period.
        end: str - End date of test period.
        fisc_calender: Pandas dataframe - Fiscal calendar lookup 
        num_test: Number of weeks for testing
        tr_log: Log object
    """
    if start >= end:
        tr_log.info("End date is prior to start date")
        sys.exit("End date is prior to start date") 
    if start not in fisc_calender.FISC_WK_OF_MTH_ID.unique():
        tr_log.info("Start date provided is not a Fiscal week")
        sys.exit("Start date provided is not a Fiscal week")
    if end not in fisc_calender.FISC_WK_OF_MTH_ID.unique():
        tr_log.info("Ennd date provided is not a Fiscal week")
        sys.exit("End date provided is not a Fiscal week")
     
    fisc_cal = fisc_calender[['FISC_WK_OF_MTH_ID']].drop_duplicates()
    fisc_cal = fisc_cal.drop_duplicates().sort_values('FISC_WK_OF_MTH_ID', ascending = True).reset_index(drop = True)
    pred_end_ix = fisc_cal.index[fisc_cal['FISC_WK_OF_MTH_ID'] == end][0]
    pred_start_ix = fisc_cal.index[fisc_cal['FISC_WK_OF_MTH_ID'] == start][0]
    ix_all = pred_end_ix - pred_start_ix + 1
    train_all = pred_end_ix - num_test + 1 - 4
    if ix_all < 135:
        tr_log.info("Duration between start and end date is less than 135 Weeks. Please provide the weeks which has duration greater than or equal to 135 weeks.")
        sys.exit("Duration between start and end date is less than 135 Weeks. Please provide the weeks which has duration greater than or equal to 135 weeks.")  
    if num_test < 13:
        tr_log.info("Number of weeks for testing is less than 13 Weeks. Please provide the duration greater than or equal to 13 weeks.")
        sys.exit("Number of weeks for testing is less than 13 Weeks. Please provide the duration greater than or equal to 13 weeks.")  
    if num_test > 52:
        tr_log.info("Number of weeks for testing is greater than 52 Weeks. Please provide the duration less than or equal to 52 weeks.")
        sys.exit("Number of weeks for testing is greater than 52 Weeks. Please provide the duration less than or equal to 52 weeks.")
    if train_all < 118:
        tr_log.info("Number of weeks for training is less than 118 Weeks. Out of total weeks between start and end, provide num_test such that atleast 118 weeks is used for training, 4 weeks for hold out and the rest for testing.")
        sys.exit("Number of weeks for training is less than 118 Weeks. Out of total weeks between start and end, provide num_test such that atleast 118 weeks is used for training, 4 weeks for hold out and the rest for testing.")  


def version_check(version, data_dump):
    """
    Function which checks if the folder with the version name already exists
    """
    if os.path.exists(os.path.join(log_path, version)):
    #     exits the program if version name already present
    #     ensures framework does not overwrite existing results
        sys.exit("Version name for iteration already exists. Specify different version name")
    if data_dump == 1:
        # Create folder with version name whenever data is pulled from DB or lookup is refreshed
        os.makedirs(os.path.join(base_dir,lkp_dir,version), exist_ok=True)
    os.makedirs(os.path.join(base_dir,model_dump_folder,version), exist_ok=True)
    os.makedirs(os.path.join(base_dir,model_dir,version), exist_ok=True)
    os.makedirs(os.path.join(base_dir,log_path,version), exist_ok=True)


def national_model_train(df_grp, cal_df, model_base_var, agg_col, grp_cols, hol_cols, model_dump_folder, pred_start, fpg, fb_val, base_dir, version, national_all, national_res_all, RSS_fcast, fcast_var):
    """
    For every FPG X National level, models are built & evaluated. This functions calls fb_prophet() where the national
    level Fb Prophet time series models are evaluated and built. Further it calls ridge() where the Fb prophet
    predictions are used to build a ridge model which better captures peaks. The predictions from Ridge is then used to
    create rolling averages and other variables which will be used at FPG X MM level models.
    Input:
        df_grp: Pandas dataframe - dataframe with DV
        cal_df: Calender data
        model_base_var: FPG column name
        agg_col: DV column name
        grp_cols: List - Group column list
        hol_cols: List - Ridge model column list
        model_dump_folder: Directory where model dump is present
        pred_start: Prediction start date
        fpg: FPG name
        fb_val: Fb model train eval split date in fiscal week format
        base_dir: Framework's parent directory
        version:  Model refresh date
        national_all: List - Natioanal model results
        national_res_all: List - NAtional model results summary
        RSS_fcast: Dataframe - forecast data
    Output:
        ntnl_rs_grp: Pandas dataframe - National model results
        national_all: List - Results appended to the list
        national_res_all: List - Results summary appended to the list
    """
    
    base_var = "granularity"
    base = "national"
    ntnl_df = df_grp.groupby(["FISC_YR_NBR", "FISC_EOW_DT", "FISC_WK_OF_MTH_ID", "FISC_WK_OF_YR_NBR"])[agg_col].sum().reset_index()
    ntnl_df["granularity"] = "national"
    
    ntnl_fcast = RSS_fcast.groupby("FISC_WK_OF_MTH_ID").agg({"SALES": "sum"}).reset_index()
    
    ntnl_df = pd.merge(ntnl_df, ntnl_fcast[['FISC_WK_OF_MTH_ID','SALES']].drop_duplicates(), on = ['FISC_WK_OF_MTH_ID'], how = 'left')
    # -----------------------------------------------------------------------------
    model_filename = fpg+"_National"
    
    try:
        res_fb, train_test_fb, train_test_df = ut.fb_prophet(ntnl_df, agg_col, cal_df, grp_cols, base_var, base, pred_start, hol_cols, fpg, base_dir, model_dump_folder, model_filename, fb_val, version, ntnl_fcast, fcast_var)
        
    except:
        cu.remove_dir(version, base_dir, 'output', 'input', 'pickles')
        sys.exit('Unable to train national level model for {fpg}. Removing all model files generated.')
        
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
                            "Fb_Prediction_old": "Fb_wo_ridge",
                            agg_col: agg_col+"_ntnl"}, inplace=True)
    ntnl_avg = train_test_df.groupby("FISC_YR_NBR")["Fb_Prediction_ntnl"].mean().reset_index()
    ntnl_avg.columns = ["FISC_YR_NBR", agg_col+"_FY_Avg_ntnl"]
    train_test_df = train_test_df.merge(ntnl_avg, how="left")
    ntnl_rs_grp = train_test_df[["FISC_WK_OF_MTH_ID", "Fb_Prediction_ntnl", agg_col+"_ntnl",
                                 "Fb_wo_ridge", "prev_yr_wk_ntnl", agg_col+"_FY_Avg_ntnl",
                                 "prev_1_wk_ntnl", "prev_2_wk_ntnl", 
                                 "prev_2_wk_avg_ntnl",
                                 "prev_4_wk_avg_ntnl", "prev_6_wk_avg_ntnl",
                                 "prev_8_wk_avg_ntnl", "prev_12_wk_avg_ntnl", "prev_52_wk_avg_ntnl"]]
    
    ntnl_rs_grp[model_base_var] = fpg
    res_fb[model_base_var] = fpg
    national_all.append(ntnl_rs_grp)
    national_res_all.append(res_fb)
    return ntnl_rs_grp, national_all, national_res_all


def state_timezone_model_train(df_grp, cal_df, model_base_var, agg_col, grp_cols, hol_cols, loc_mapping, cores, model_dump_folder, pred_start, fpg, fb_val, base_dir, version, state_all, state_res_all, log_path, RSS_fcast, model_base,tr_log, file_name, fcast_var):
    """
    For every FPG X State level, models are built & evaluated. Time Zone models are built concurrently which is 
    controled using this functions. This function calls inhome_tz_state_model() which further fb_prophet() where the State level
    Fb Prophet time series models are evaluated and built. Further it calls ridge() where the Fb prophet predictions are used to build a 
    ridge model which better captures peaks. The predictions from Ridge is then used to create rolling averages and
    other variables which will be used at FPG X MM level models. The results are then collated into one dataframe from multiple cores.
    Input:
        df_grp: Pandas dataframe - dataframe with DV
        cal_df: Calender data
        model_base_var: FPG column name
        agg_col: DV column name
        grp_cols: List - Group column list
        hol_cols: List - Ridge model column list
        loc_mapping: Pandas dataframe - Minor Market to Time Zone lookup
        cores: Number of cores to be used to build the models 
        model_dump_folder: Directory where model dump is present
        pred_start: Prediction start date
        fpg: FPG name
        fb_val: Fb model train eval split date in fiscal week format
        base_dir: Framework's parent directory
        version:  Model training date
        state_all: List - State model results
        state_res_all: List - State model results summary
        log_path: Log path
    Output:
        state_df: Pandas dataframe - FPG X State model results
        state_all: List - Results appended to the list
        state_res_all: List - Results summary appended to the list
    """
    mod_type = 'stt' if model_base == 'STATE' else 'tz'
    loc_state = loc_mapping[["MM", model_base]].drop_duplicates()
    loc_state_cnt =loc_state.groupby(model_base).nunique()["MM"].reset_index()
    loc_state_cnt.columns = [model_base, "MM_CNT"]
    df_grp_state = pd.merge(df_grp, loc_state, how = "left")
    RSS_fcast = RSS_fcast.groupby('FISC_WK_OF_MTH_ID')['SALES'].sum().reset_index()
    ntnl_df_level = df_grp_state.groupby(["FISC_YR_NBR", "FISC_EOW_DT", "FISC_WK_OF_MTH_ID", "FISC_WK_OF_YR_NBR", model_base])[agg_col].sum().reset_index()
    ntnl_df_level = pd.merge(ntnl_df_level, RSS_fcast[['FISC_WK_OF_MTH_ID', 'SALES']], on = 'FISC_WK_OF_MTH_ID', how = 'left')
    if fpg == 'Video Setup':
        fcast_var = False
    
    
    model_comb = list(ntnl_df_level[model_base].unique())
    
    # -----------------------------------------------------------------------------
    
    
    iter_per_core = math.ceil(len(model_comb)/(cores-1))
    threads = [None] * cores
    start_thread = time.time()
    for n in range(1, cores):
#        print(n)
        try:
            start_idx = (n-1)*iter_per_core
            end_idx = n*iter_per_core
            print("Indexes: ", str(start_idx), str(end_idx))

            multi_thread_parameters = [start_idx, end_idx, grp_cols, model_base, agg_col, model_comb, ntnl_df_level, cal_df, file_name, start_thread, mod_type, n, pred_start, hol_cols, fpg, model_dump_folder, fb_val, base_dir, version, log_path, RSS_fcast, fcast_var, tr_log]
            threads[n] = mp.Process(target = ut.inhome_tz_state_model, args=(multi_thread_parameters, file_name))
            threads[n].start()
        except Exception as e:
            cu.remove_dir(version, base_dir, 'output', 'input', 'pickle')
            logging.exception(str(e)+str(n))
            sys.exit('Error in training the {model_base} model. Removing all generated model files for {version}')            
            
            
    for m in range(1,cores):
        try:
            if threads[m]:
                threads[m].join()
        except Exception as e:
            cu.remove_dir(version, base_dir, 'output', 'input', 'pickle')
            logging.exception(str(e)+str(n))
            sys.exit('Error in collating the {model_base} model output. Removing all generated model files for {version}')
            
    
    print("time_taken for process: ", round(time.time()-start_thread), "sec")
    print("Overall time taken:", str(round(time.time()-thread_st)))
    
    results_f_df, train_test_fb_all, state_df, dummy = cu.res_return(file_name)
    results_f_df = pd.merge(results_f_df, loc_state_cnt, how = "left")
    state_df[model_base_var] = fpg
    results_f_df[model_base_var] = fpg
    state_all.append(state_df)
    state_res_all.append(results_f_df)
    shutil.rmtree(file_name)
    
    return state_df, state_all, state_res_all, len(model_comb)


def fpg_mm_model_train(df_grp, model_cols, loc_mapping, cal_df, ntnl_rs_grp, time_zone_df, state_df, model_data_sales_all, grp_mm_sparsity, fisc_calender, model_dump_folder, pred_start, mod_start, lasso_val, base_dir, cores, version, res_fpg_all, feat_imp, tr_test, log_path, model_base_var, fpg, agg_col, file_name, tr_log):
    """
    For every FPG X Minor Market level, models are built & evaluated. MM models are built concurrently which is 
    controled using this functions. This function calls inhome_model() which further calls functions which prepares the data for modelling,
    hyper parameter tuning and building linear model with lasso regularization.The results are then collated into one dataframe from multiple cores.
    Input:
        df_grp: Pandas dataframe - dataframe with DV
        model_cols: List - Model columns
        loc_mapping: Pandas dataframe - Minor Market to Time Zone lookup
        cal_df: Pandas dataframe - Holiday calendar
        ntnl_rs_grp: Pandas dataframe - National model results
        time_zone_df: Pandas dataframe - FPG X Time Zone model results
        state_df: Pandas dataframe - FPG X State model results
        model_data_sales_all: Pandas dataframe - Forecast variables
        grp_mm_sparsity: Pandas dataframe - Sparsity lookup
        fisc_calender: Pandas dataframe - Fiscal calendar lookup
        model_dump_folder: Directory where model dump is present
        pred_start: Prediction start date
        mod_start: Model start date
        lasso_val: Lasso model train eval split date in fiscal week format
        base_dir: Framework's parent directory
        cores: Number of cores to be used to build the models
        version:  Model training date
        res_fpg_all: List - FPG X MM results summary
        feat_imp: List - FPG X MM model selected features
        tr_test: List - FPG X MM model results
        log_path: Log path
        model_base_var: FPG column
        fpg: FPG name
        agg_col: DV column
    Output:
        res_fpg_all: List - FPG X GRP model results summary appended to a list
        feat_imp: List - FPG X GRP model selected features appended to a list
        tr_test: List - FPG X GRP model results appended to a list
    """
    pivot_var = "MM"
    grp_cols = [model_base_var+"_"+pivot_var, "FISC_WK_OF_MTH_ID"]
    model_base = model_base_var+"_"+pivot_var
    # -----------------------------------------------------------------------------
    model_data = pd.merge(df_grp, loc_mapping, on = "MM", how = "left")
    model_data["RSS"] = model_data[model_base].str.replace("_.*", "")
    model_data["MM"] = model_data[model_base].str.replace(".*_", "")
    # -----------------------------------------------------------------------------
    
    model_comb = list(model_data[model_data["SPARSITY"].str.contains("Prediction Model")][model_base].unique())

    threads = [None] * (cores+1)
    start_thread = time.time()
    iter_per_core = math.ceil(len(model_comb)/(cores))
    start_thread = time.time()
    for n in range(1, cores+1):

        try:
            start_idx = (n-1)*iter_per_core
            end_idx = n*iter_per_core
            print("Indexes: ", str(start_idx), str(end_idx))            
            multi_thread_parameters = [start_idx, end_idx, grp_cols, model_base, agg_col, model_comb, model_data, cal_df, ntnl_rs_grp, model_cols, file_name, n, start_thread, model_data_sales_all, state_df, time_zone_df, fisc_calender, model_dump_folder, pred_start, mod_start, lasso_val, base_dir, version, log_path, tr_log]
            threads[n] = mp.Process(target = ut.inhome_model, args=(multi_thread_parameters, file_name))
            threads[n].start()
        except Exception as e:
            cu.remove_dir(version, base_dir, 'output', 'input', 'pickle')
            logging.exception(str(e)+str(n))
            sys.exit('Error in training the skillset X minor market level model. Removing all generated model files for {version}')
            
    for m in range(1,cores):
        try:
            if threads[m]:
                threads[m].join()
        except Exception as e:
            cu.remove_dir(version, base_dir, 'output', 'input', 'pickle')
            logging.exception(str(e))
            sys.exit('Error in collating the skillset X minor market level results. Removing all generated model files for {version}')
            
    
    print("time_taken for process: ", round(time.time()-start_thread), "sec")
    print("Overall time taken:", str(round(time.time()-thread_st)))
    # -----------------------------------------------------------------------------
    results_f_df, train_all, test_all, feature_importance = cu.res_return(file_name)
    train_test_f = pd.concat([train_all, test_all], axis = 0)
    results_f_df = pd.merge(results_f_df, grp_mm_sparsity, on = model_base, how ="left")
    train_test_f = pd.merge(train_test_f, results_f_df[[model_base,"SPARSITY", "mape", "mod_type", "alpha", "model_col", "feat_slctn"]], how = "left")
    train_test_f["MAPE_LS"] = abs(train_test_f["LINE_ORDERS_ACTUAL"] - train_test_f["Prediction_Trf"])/train_test_f["LINE_ORDERS_ACTUAL"]
    feature_importance = pd.merge(feature_importance, results_f_df[[model_base,"SPARSITY", "mape", "mod_type", "alpha", "model_col", "feat_slctn"]], how = "left")
    results_f_df["mu_sigma_train"] = results_f_df["mean_train"] - results_f_df["std_train"]
    shutil.rmtree(file_name)
    res_fpg_all.append(results_f_df)
    feat_imp.append(feature_importance)
    tr_test.append(train_test_f)
    return res_fpg_all, feat_imp, tr_test, len(model_comb)


def service_hours_thread(fpg_all, df_all, cal_df, fisc_calender, pred_start, mnts_cols, tr_test, var_list, grp_mm_sparsity, cores, log_path, tr_log, base_dir, version):
    """
    Utiltiy to call Service hours prediction function for every FPG. Every FPG will be run concurrently. Service hours are predicted
    based on the predicted line orders and historical proportion between actual line orders and service hour metrics. The historical proportion
    is multiplied with the predicted line orders for the given combination to arrive at the predicted service hours.
    Input:
        fpg_all: List - fpg list
        df_all: Pandas dataframe - dataframe with DV
        cal_df: Pandas dataframe - Holiday calendar
        fisc_calender: Pandas dataframe - Fiscal calendar lookup
        pred_start: Prediction start date
        mnts_cols: List - SH columns
        tr_test: Pandas dataframe - FPG X MM model results
        var_list: List - Proportion list
        grp_mm_sparsity: Sparsity buckets
        cores: No of cores
    Output:
        df_res_sh: Pandas dataframe - FPG X MM SH predictions
    """
    # -----------------------------------------------------------------------------
    # Logging all the basic details of the model
    
    model_comb = fpg_all
    threads = [None] * (cores+1)
    start_thread = time.time()
    iter_per_core = math.ceil(len(model_comb)/(cores))
    start_thread = time.time()
    file_name = version+"_"+"IH_install_Service Hour"
    
    for n in range(1, cores+1):
        print(n)
    
        start_idx = (n-1)*iter_per_core
        end_idx = n*iter_per_core
        
        print("Indexes: ", str(start_idx), str(end_idx))
        multi_thread_parameters = [start_idx, end_idx, file_name, n, start_thread,  df_all, cal_df, fisc_calender, pred_start, mnts_cols, tr_test, var_list, model_comb, grp_mm_sparsity, log_path, tr_log, base_dir, version]
        threads[n] = mp.Process(target = ut.service_hours_predictions, args=(multi_thread_parameters, file_name))
        threads[n].start()

            
    for m in range(1,cores):
        try:
            if threads[m]:
                threads[m].join()
        except Exception as e:
            cu.remove_dir(version, base_dir, 'output', 'input', 'pickle')
            logging.exception(str(e))
            sys.exit('Error in calculating Service hours. Removing all generated model files for {version}')
            
            
    print("time_taken for process: ", round(time.time()-start_thread), "sec")
    print("Overall time taken:", str(round(time.time()-thread_st)))
    df_res, df_res_sh, df_res_all_dumm, dummy = cu.res_return(file_name)
    shutil.rmtree(file_name)
    return df_res_sh


res_fpg_all = []
feat_imp = []
tr_test = []
national_all = []
national_res_all = []
tz_all = []
tz_res_all = []
state_all = []
state_res_all = []
res_all_sp = []


if __name__ == "__main__":
    overall_st = time.time()
    version_check(version,data_dump)
    # Initiate logger object
    tr_log = cu.my_custom_logger(base_dir, os.path.join(log_path, version), "Training Steps")
    tr_log.info("Loading lookup files")
    # Load all the look up files which are necessary for model data prep and model building
    try:
        
        cal_df, fisc_calender, cov_buck, loc_mapping = dp.lookup_data_load(start, end, base_dir, lkp_dir, cal_param, cov_param, fisc_cal_params, loc_params, conn)
        check_dates(start, end, fisc_calender, num_test, tr_log)
    except:
        cu.remove_dir(version,base_dir, lkp_dir, model_dir, model_dump_folder)
        tr_log.exception("Error occured while loading lookup files") 
        sys.exit('Error Loading the lookup files')
        
    
    pred_start, pred_end, mod_start, fb_val, lasso_val, summ_end, summ_start, lst_mth, lst_qtr, lst_yr_start, start_rule = cu.get_date(fisc_calender, start, end, num_test) 
            
    
    
    if data_flag == 1:
        tr_log.info("Loading Line orders, Sales & Forecast data from flat files")
        # Load all the data required for modelling from flat files
        try:    
            df, sales_oth, forecast = dp.flat_data_load(start, end, cal_df, base_dir, lkp_dir, line_orders_df, sales_df, forecast_df)
        except:
            cu.remove_dir(version,base_dir, lkp_dir, model_dir, model_dump_folder)
            tr_log.exception("Error occured while loading loLine orders, Sales & Forecast data from flat files")    
            sys.exit('Error occured while loading loLine orders, Sales & Forecast data from flat files')
    else:
        tr_log.info("Loading Line orders, Sales & Forecast data from database")
        # Create volatile tables and load all the data required for modelling from DB
        
        try:                                       
            df, sales_oth, forecast = dp.data_load(start, end, cal_df, conn, hlpr_time_params, location_params, svc_sku_params, dt_params, sales_loc, svc_cls_lkp, base_dir, lkp_dir, version, line_orders_df, sales_df, forecast_df, data_dump)
        except:
            cu.remove_dir(version,base_dir, lkp_dir, model_dir, model_dump_folder)
            tr_log.exception("Error occured while loading loLine orders, Sales & Forecast data from database")            
            sys.exit('Unable to query data for training')
    
    df["FISC_WK_OF_MTH_ID"] = df["FISC_WK_OF_MTH_ID"].astype("int")
    # Split line orders data based on modellable combinations and combinations which requre rule based approach
    try:
        df_all, cov_sparsity, minimums = dp.modellable_combinations(base_dir, lkp_dir, sparsity_params, df, 1, cal_df, cov_buck, version, summ_start, summ_end, lst_mth, lst_qtr, lst_yr_start,data_dump)
    except:
        cu.remove_dir(version,base_dir, lkp_dir, model_dir, model_dump_folder)
        tr_log.exception("Error occured while identifying modellable combinations")    
        
    tr_log.info("Data extraction complete")
    model_base_var = "RSS"
    agg_col = "LINE_ORDERS"
    grp_cols = ["granularity", "FISC_WK_OF_MTH_ID"]


    df_gmb_all = df_all[df_all.SPARSITY == 'Prediction Model']
    df_sp_all = df_all[df_all.SPARSITY== 'Rule Based Prediction']
    
    fpg_gmb = df_gmb_all.groupby('RSS').LINE_ORDERS.sum().reset_index().sort_values('LINE_ORDERS', ascending = False).RSS.unique()
    fpg_sp = df_sp_all.groupby('RSS').LINE_ORDERS.sum().reset_index().sort_values('LINE_ORDERS', ascending = False).RSS.unique()
    fpg_all = df_all.RSS.unique()
    
    grp_mm_sparsity = df_all[['RSS_MM','SPARSITY']].drop_duplicates()
    
    if len(fpg_gmb) > 0:
        
        for fpg in fpg_gmb:
            try:
                
                tr_log.info("Model training started for "+fpg)
                # Subset the fpg
                df_grp = df_gmb_all[df_gmb_all[model_base_var] == fpg].copy()

                RSS_sales = sales_oth[sales_oth[model_base_var] == fpg]
                RSS_sales['SALES'].fillna('bfill', inplace = True)
                RSS_sales['SALES'].fillna('ffill', inplace = True)
                
                tr_log.info(fpg+" X National model building")
                # Execute the National model for the fpg
                    
                ntnl_rs_grp, national_all, national_res_all = national_model_train(df_grp, cal_df, model_base_var, agg_col, grp_cols, hol_cols, model_dump_folder, pred_start, fpg, fb_val, base_dir, version, national_all, national_res_all, RSS_sales, fcast_var = True)
                
                tr_log.info(fpg+" X Time Zone model building")
                # Execute the Time zone model for the fpg
                
                try:
                    model_base = 'TIME_ZONE'
                    file_name = "IH_Install_"+model_base+"_"+fpg+"_"+str(version)
                    file_name = file_name.replace('/', '-')
                    
                    time_zone_df, tz_all, tz_res_all, num_comb = state_timezone_model_train(df_grp, cal_df, model_base_var, agg_col, grp_cols, hol_cols, loc_mapping, cores, model_dump_folder, pred_start, fpg, fb_val, base_dir, version, tz_all, tz_res_all, log_path, RSS_sales,'TIME_ZONE', tr_log, file_name, fcast_var = True)
                    if time_zone_df[model_base].nunique() != num_comb:
                        raise  Exception("Results are not available for all the combinations")
                
                except:
                    if os.path.isdir(os.path.join(base_dir, file_name)):
                        shutil.rmtree(file_name)
                    tr_log.exception(f"Error occured while building models for {fpg} X TIME_ZONE combinations")  
                    cu.remove_dir(version, base_dir, lkp_dir, model_dir, model_dump_folder)
                    sys.exit("Error occured while scoring for RSS X Time Zone combinations")
                    
                tr_log.info(fpg+" X State model building")
                
                try:
                    model_base = 'STATE'
                    file_name = "IH_Install_"+model_base+"_"+fpg+"_"+str(version)
                    file_name = file_name.replace('/', '-')
                
                    # Execute the State model for the fpg
                    state_df, state_all, state_res_all, num_comb = state_timezone_model_train(df_grp, cal_df, model_base_var, agg_col, grp_cols, hol_cols, loc_mapping, cores, model_dump_folder, pred_start, fpg, fb_val, base_dir, version, state_all, state_res_all, log_path, RSS_sales,'STATE', tr_log, file_name, fcast_var = True)
                    # Sales data prep
                    if state_df[model_base].nunique() != num_comb:
                        raise  Exception("Results are not available for all the combinations")
                
                except:
                    if os.path.isdir(os.path.join(base_dir, file_name)):
                        shutil.rmtree(file_name)
                    tr_log.exception(f"Error occured while building models for {fpg} X State combinations")  
                    cu.remove_dir(version, base_dir, lkp_dir, model_dir, model_dump_folder)
                    sys.exit("Error occured while scoring for RSS X Time Zone combinations")
                
                
                sales_oth_part = sales_oth.copy()

                rss_sales = sales_oth[sales_oth[model_base_var]==fpg].copy()
                rss_sales = rss_sales.groupby(['FISC_WK_OF_MTH_ID', 'RSS']).agg({"SALES": "sum"}).reset_index()
                
                tr_log.info(fpg+" X Sales variable creation")
                                
                # Prepare MAPP variables for RSS MM's
                model_data_sales = fu.sales_data_prep(fpg, df, sales_oth_part, fisc_calender, cal_df, rss_sales, pred_start,'SALES')
                model_data_sales = model_data_sales[~model_data_sales.MM.str.contains("HD")]
                
                
                try:
                    model_base = 'RSS_MM'
                    file_name = "IH_Install_"+model_base+"_"+fpg+"_"+str(version)
                    file_name = file_name.replace('/', '-')
                           
                    res_fpg_all, feat_imp, tr_test, num_comb = fpg_mm_model_train(df_grp, model_cols, loc_mapping, cal_df, ntnl_rs_grp, time_zone_df, state_df, model_data_sales,  grp_mm_sparsity, fisc_calender, model_dump_folder, pred_start, mod_start, lasso_val, base_dir, cores, version, res_fpg_all, feat_imp, tr_test, log_path, model_base_var, fpg, agg_col, file_name, tr_log)
                    
                    if num_comb != tr_test[-1][model_base].nunique():
                        raise  Exception("Results are not available for all the combinations")
                
                except:
                    if os.path.isdir(os.path.join(base_dir, file_name)):
                        shutil.rmtree(file_name)
                    tr_log.exception(f"Error occured while building models for {fpg} X MM combinations")  
                    cu.remove_dir(version, base_dir, lkp_dir, model_dir, model_dump_folder)
                    sys.exit("Error occured while scoring for RSS X MM combinations")
                    
                    
                tr_log.info("Model training ended for "+fpg)

            except Exception as e:
                tr_log.exception(str(e))  
                tr_log.exception(f"Error occured while building models for {fpg}")  
          
            # -----------------------------------------------------------------------------
        cols_selec = ["FISC_WK_OF_MTH_ID", "LINE_ORDERS_ACTUAL", "Prediction_Trf", "SPARSITY", "MM", "RSS", "RSS_MM", "MAPE_LS"]
        # Collate all the model results and store it as a flat file
        res_fpg_all_df = pd.concat(res_fpg_all)

        feat_imp_df = pd.concat(feat_imp)
        tr_test_df = pd.concat(tr_test)
        national_all_df = pd.concat(national_all)
        national_res_all_df = pd.DataFrame(national_res_all)
        tz_all_df = pd.concat(tz_all)
        tz_res_all_df = pd.concat(tz_res_all)
        state_all_df = pd.concat(state_all)
        state_res_all_df = pd.concat(state_res_all)
        
        tr_log.info("Dumping results for Good, Medium & Bad FPG X MM combinations")

        if data_dump== 1:

            writer = pd.ExcelWriter(os.path.join(base_dir, model_dir, version, "line_orders_model_results_gmb.xlsx"))
            res_fpg_all_df.to_excel(writer, "Final Model Results", index=False)
            tr_test_df[cols_selec].to_excel(writer, "Train Test", index=False)
            feat_imp_df.to_excel(writer, "Feat Imp Results", index=False)
            writer.save()
            
            writer = pd.ExcelWriter(os.path.join(model_dir, version, "line_orders_higher_model_results_gmb.xlsx"))
            national_all_df.to_excel(writer, "Ntnl Train Test", index=False)
            national_res_all_df.to_excel(writer, "Ntnl Results", index=False)
            tz_all_df.to_excel(writer, "TZ Train Test", index=False)
            tz_res_all_df.to_excel(writer, "TZ Results", index=False)
            state_all_df.to_excel(writer, "State Train Test", index=False)
            state_res_all_df.to_excel(writer, "State Results", index=False)
            writer.save()
            
        
#        tr_test_df = pd.read_excel(os.path.join(base_dir, 'output', version, 'line_orders_model_results_gmb.xlsx'), sheet_name = 'Train Test')
        
    else:
        tr_test_df = pd.DataFrame(columns = ['FISC_WK_OF_MTH_ID', 'RSS_MM', 'Prediction_Trf', 'RSS', 'MM'])
        tr_log.info("No Good, Medium & Bad RSS X MM combinations")
    
    # -----------------------------------------------------------------------------
    tr_log.info("Prediction for Sparse RSS X MM combinations")
    # Every FPG will be run in loop for all the sparser combinations for rule based predictions

    if len(fpg_sp) > 0:
        avg_pred_prev = []
        avg_pred_curr = []
        for i, fpg in enumerate(fpg_sp):
            print(fpg)
            try:
                data_grp = df_sp_all[((df_sp_all[model_base_var] == fpg)&(df_sp_all["SPARSITY"].isin(["Rule Based Prediction"])))]
                avg_pred_prev = ut.sparse_predictions(data_grp, cal_df, fisc_calender, pred_start, fpg, avg_pred_prev)
                avg_pred_curr = ut.avg_predictions(data_grp, cal_df, fisc_calender, pred_start, fpg, avg_pred_curr)
            except:
                cu.remove_dir(version, base_dir, lkp_dir, model_dir, model_dump_folder)
                tr_log.exception(f"Error occured while scoring Sparse {fpg} combinations") 
                sys.exit(f"Error occured while scoring Sparse {fpg} combinations")


        tr_log.info("Calculating best rules for sparse prediction by RSS")
    
        avg_pred_prev_df = pd.concat(avg_pred_prev)
        avg_pred_curr_df = pd.concat(avg_pred_curr)
        avg_pred_curr_df["Variable_Type"] = "prev_5wk_avg_4wk_hld"
        avg_pred = avg_pred_prev_df.append(avg_pred_curr_df, ignore_index=True, sort=False)
                                                                                                                
        df_sp_all.FISC_WK_OF_MTH_ID = df_sp_all.FISC_WK_OF_MTH_ID.astype(int)                                    
                                                                                                                    
        avg_pred = pd.merge(avg_pred,df_sp_all[['FISC_WK_OF_MTH_ID', 'RSS_MM', 'LINE_ORDERS']], on=['FISC_WK_OF_MTH_ID', 'RSS_MM'], how="left")
                                                                                                                
        avg_pred['MAPE'] = abs(avg_pred['Prediction_Trf'] - avg_pred['LINE_ORDERS'])/avg_pred['LINE_ORDERS']
        
        avg_pred_20 = avg_pred[(avg_pred['FISC_WK_OF_MTH_ID'] >= summ_start) & ((avg_pred['FISC_WK_OF_MTH_ID'] <= summ_end))]
        
        var_pivot = pd.pivot_table(avg_pred_20, index = 'RSS', columns = 'Variable_Type', values = 'MAPE', aggfunc = 'mean')
        
        best_var = var_pivot.idxmin(axis = 1)
        
        best_var = best_var.reset_index()
        best_var.columns = ['RSS', 'Variable_Type']

        best_var['RSS_RULE'] = best_var['RSS']+'_'+best_var['Variable_Type']

        avg_pred['RSS_RULE'] = avg_pred['RSS']+'_'+avg_pred['Variable_Type'] 
        
        best_avg_pred = avg_pred[avg_pred['RSS_RULE'].isin(best_var['RSS_RULE'])].drop('RSS_RULE', axis = 1)
        
    else:
        res_all_sp_df = pd.DataFrame(columns = ['FISC_WK_OF_MTH_ID', 'RSS_MM', 'Prediction_Trf', 'RSS', 'MM'])
        tr_log.info("No Sparse RSS X MM combinations")  

    tr_log.info("Collating Good, Medium ,Bad & Sparse and Correcting the predictions of very sparse FPG X MM combinations")
    
    # Collate rule based and model predictions
    
    try:
        
        res_all_sp_df = best_avg_pred.copy()
        res_all_sp_df.rename({"LINE_ORDERS": "LINE_ORDERS_ACTUAL"}, axis=1, inplace=True)
        
        tr_test = pp.line_orders_collate(tr_test_df, res_all_sp_df, pred_start, df, start)
        tr_test['FISC_YR_NBR'] = tr_test['FISC_WK_OF_MTH_ID'].apply(lambda x : str(x)[0:4]).astype(int)
        
        tr_test.drop("SPARSITY", axis=1, inplace=True)

        zero_comb = grp_mm_sparsity[grp_mm_sparsity["SPARSITY"]=="Zero"]["RSS_MM"].values
        df_zero = df[df["RSS_MM"].isin(zero_comb)][['RSS', 'RSS_MM', 'FISC_WK_OF_MTH_ID', 'LINE_ORDERS', 'MM']]
        df_zero = df_zero.merge(cal_df[['FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_YR_NBR']], how="left")
        df_zero.rename({"LINE_ORDERS": "LINE_ORDERS_ACTUAL"}, axis=1, inplace=True)
        df_zero["Prediction_Trf"] = 0
        tr_test = tr_test.append(df_zero, ignore_index=True, sort=False)
        tr_test = tr_test.merge(grp_mm_sparsity[["RSS_MM", "SPARSITY"]], how="left")
        tr_test[tr_test.FISC_YR_NBR == 2020].LINE_ORDERS_ACTUAL.sum()
        
        if data_dump:
            tr_test.to_csv(os.path.join(base_dir,model_dir,version,'line_orders_results_all.tsv'), index = None)
        
#        tr_test = pd.read_csv(os.path.join(base_dir,model_dir,'IHI_FWF_Training_14092020_V1','line_orders_results_all.tsv'))
            
            
    except:
        tr_log.exception("Error occured while collating the predictions")  
    
    tr_log.info("Service hour calculation for all the FPG X MM combinations")
    
    try:
        var_list = ['pv_yr_wk', 'pv_yr', 'pv_yr_qtr', 'pv_yr_mth', 'pv_yr_mid_3wk', 'pv_yr_mid_5wk', 'pv_2wk', 'pv_4wk', 'pv_roll_5wk']
                
        data_all_df = service_hours_thread(fpg_all, df_all, cal_df, fisc_calender, pred_start, mnts_cols, tr_test, var_list, grp_mm_sparsity, cores, log_path,tr_log, base_dir, version)
    
    except:
        
        tr_log.exception("Error occured while performing service hours predictions")  
    

    # -----------------------------------------------------------------------------    

    tr_log.info("Model training completed for all the RSS X MM combinations and the results are dumped in the output folder")
    try:
        # Create results summary workbook from predictions and store it as a flat file
        df_grp_summary, stability, monthly_stability_check, mape_cov, mape_cov_grp, mape_grp, sh_mm_summary_df, sh_summary_grp, sh_prop_summary, df_overall_summay, best_method = pp.model_performance_summary(df, data_all_df.copy(), cov_buck, fisc_calender, mod_start, pred_start, summ_start, summ_end)
        writer = pd.ExcelWriter(os.path.join(base_dir, model_dir, version, "model_performance_summary.xlsx"))
        df_grp_summary.to_excel(writer, "RSS Distribution", index=False)
        stability.to_excel(writer, "By Covariance", index=False)
        monthly_stability_check.to_excel(writer, "By Covariance(monthly)", index=False)
        mape_cov.to_excel(writer, "Covariance X MAPE Bins", index=False)
        mape_cov_grp.to_excel(writer, "RSS X Covariance", index=False)
        mape_grp.to_excel(writer, "RSS X MAPE Bins", index=False)
        sh_prop_summary.to_excel(writer, "Proportion X MAPE Bins(MM SH)", index=False)
        sh_mm_summary_df.to_excel(writer, "By Proportion (MM SH)", index=False)
        sh_summary_grp.to_excel(writer, "RSS X MAPE Bins(GSD SH)", index=False)
        df_overall_summay.to_excel(writer, "By Covariance(RSS, MM)", index=False)
        writer.save()
    except:
        tr_log.exception("Error occured while generating report")  
        sys.exit("Error occured while generating report")  
        
    print("Overall time taken:", str(round(time.time() - overall_st)))
    # -----------------------------------------------------------------------------
    
    try:
        lookup_files = dp.lookup_data(base_dir, conn, lkp_dir, cov_param, svc_sku_params, svc_cls_lkp, hlpr_time_params, location_params, dt_params, loc_params, sales_loc)
        lookup_files.update({"Service hour proportion": best_method[0]})
        lookup_files.update({"Rule Based Prediction": best_var})
        
        with open(os.path.join(base_dir, model_dump_folder, version, 'lookupfiles.sav'), 'wb') as f:
            pickle.dump(lookup_files, f)   
        
        data_all_selec = data_all_df[data_all_df.Variable_Type == best_method[0]]
        
        if data_dump == 1:
            data_all_selec.to_csv(os.path.join(base_dir, model_dir, version, 'service_hours_results_all.tsv'), index = None)
        
    except:
        tr_log.exception("Error occured while dumping all the pickles and final output")  
        sys.exit("Error occured while dumping all the pickles and final output")
    
    
    tr_log.info("Report creation for all the RSS X MM combinations")
    # Create results report from predictions and store it as a flat file
    try:
        data_result_test = pp.report_creation(data_all_selec, pred_start)
        data_result_test.to_csv(os.path.join(base_dir, model_dir, version,"line_order_prediction_results_test.csv"), index = None)
    except:
        tr_log.exception("Error occured while generating report")  
        sys.exit("Error occured while generating report")  
        
    conn.close()
    