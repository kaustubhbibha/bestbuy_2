"""In-Home Install Future Work Force - Service Line Orders Forecasting

This script allows post processing of results for In-Home Delivery Run The Business (IHD-RTB)

This file requires following scripts to execute
    * fwf_install_common_utils.py - functions containing common utilities
    * fwf_install_data_prep_utils.py - functions containing data prep utils
"""


import pandas as pd
import numpy as np
import fwf_install_common_utils as cu
import fwf_install_data_prep_utils as dp


def line_orders_collate(recore_df_all_res, res_all_sp_df, pred_start, df, start):
    
    """
    Collate all the predictions from model and rule based approach to be further used for service hour predictions
    Input:
        recore_df_all_res: Pandas dataframe - Model predictions
        res_all_sp_df: Pandas dataframe - Rule based predictions
        minimums: Pandas dataframe - Combinations which needs to predicted minimums
        pred_start: Prediction start date
        df: Pandas dataframe - Line orders actuals
        start: Start Fiscal week
    Output:
        tr_test: Pandas dataframe - Line orders predictions collated both from model & rule based predictions
    """
    df_gmb = pd.merge(df, recore_df_all_res[["FISC_WK_OF_MTH_ID", "RSS_MM"]].drop_duplicates())
    df_sp = pd.merge(df, res_all_sp_df[["FISC_WK_OF_MTH_ID", "RSS_MM"]].drop_duplicates())
    df_gmb_sp = pd.concat([df_gmb, df_sp], axis = 0)
    tr_test = pd.concat([recore_df_all_res, res_all_sp_df], axis = 0)
    
    tr_test_cnt = df_gmb_sp[(df_gmb_sp.FISC_WK_OF_MTH_ID < pred_start) & (df_gmb_sp.FISC_WK_OF_MTH_ID >= start)].groupby("RSS_MM").agg({"LINE_ORDERS":["count", "mean"]}).reset_index()
    tr_test_cnt.columns = ["RSS_MM", "TRAIN_COUNT", "TRAIN_AVG"]
    tr_test = pd.merge(tr_test, tr_test_cnt, how = "left")
    
    return tr_test


def report_creation(data_all_df, pred_start):    
    """
    Generate report based on the predictions for all the FC X Minor Market combinations
    Input:
        data_all_df: Pandas dataframe - dataframe with DV
        pred_start: Prediction start date
    Output:
        data_result_test: Pandas dataframe - Final predictions
    """
    data_all_df["HD MM/GSD"] = np.where(data_all_df.MM.str.contains("HD"), "HD MM", "GSD")
    

    select_cols = ["IDV_LEAD_MNTS_Pred", "IDV_HELPER_MNTS_Pred","IDV_HELPER_OVR_Pred",
                "IDV_LD_UNCLAIMED_Pred", "IDV_HP_UNCLAIMED_Pred", "IDV_HPN_UNCLAIMED_Pred",
                "IDV_LD_DRIVE_Pred", "IDV_HP_DRIVE_Pred", "IDV_HPN_DRIVE_Pred"]

    for col in select_cols:
        data_all_df[col.replace('MNTS','HRS')] = round(data_all_df[col]/60,1)
    
    
    data_all_df["Lead Hrs/Appnt"] = data_all_df["IDV_LEAD_HRS_Pred"]/data_all_df["IDV_APPT_ID_Pred"]
    data_all_df["Helper Hrs/Appnt"] = data_all_df["IDV_HELPER_HRS_Pred"]/data_all_df["IDV_APPT_ID_Pred"]
    
    res_cols = ["MM", "HD MM/GSD", "FISC_WK_OF_MTH_ID", "RSS",
                "Prediction_Trf", "Lead Hrs/Appnt",
                "IDV_LEAD_HRS_Pred", "IDV_HELPER_HRS_Pred", "Helper Hrs/Appnt", 
                "IDV_HELPER_OVR_Pred",
                "IDV_LD_UNCLAIMED_Pred", "IDV_HP_UNCLAIMED_Pred", "IDV_HPN_UNCLAIMED_Pred",
                "IDV_LD_DRIVE_Pred", "IDV_HP_DRIVE_Pred", "IDV_HPN_DRIVE_Pred"]
    data_result = data_all_df[res_cols]
    data_result = data_result[data_result.FISC_WK_OF_MTH_ID >= pred_start]
    data_result.columns = ["MM", "HD MM/GSD", "FISCAL_WK_ID", "RSS",
                "Order Lines", "Lead Hrs/Appnt",
                "Lead Hrs", "Helper Hrs", "Helper Hrs/Appnt",
                "Helper Hrs Float",
                "Unclaimed Lead", "Unclaimed Helper", "Unclaimed Helper Hrs Float",
                "Lead Drive Time", "Helper Drive Time", "Helper Hrs Float Drive Time"]
    data_result['FISCAL_WK_ID'] = data_result['FISCAL_WK_ID'].astype(int)
    return data_result


def model_performance_summary(df, data_all_df, cov_buck, fisc_calender, mod_start, pred_start, summ_start, summ_end):    
    """                       
    Utiltiy to generate model performance summary. This module generates results summary for Line orders
    and Service hours at various levels like RSS, MAPE Bins, Covariance category for HD & GSD.
    Input:
        df: Pandas dataframe - Model input
        data_all_df: Pandas dataframe - Model results
        metro_map: Pandas dataframe - Metro mapping
        cov_buck: Pandas dataframe - Covariance buckets
        grp_mm_sparsity: Sparsity buckets
        fisc_calender: Pandas dataframe - Fiscal calendar
        fiscal_year: Year for analysis
        mod_start: model start date
        pred_start: Prediction start date
        rep_fiscal_year: year for which report is generated
    Output:
        df_grp_summary: Pandas dataframe - FPG Distribution
        stability: Pandas dataframe - MAPE by covariance
        monthly_stability_check: Pandas dataframe - Monthly summary by Covariance
        mape_cov: Pandas dataframe - Covariance X MAPE Bins summary
        mape_cov_grp: Pandas dataframe - FPG X Covariance summary
        mape_grp: Pandas dataframe - FPG X Mape bins summary
        sh_summary_grp: Pandas dataframe - GSD MM Service hour summary
        slo_summary_df: Pandas dataframe - HD MM Line order summary
    """
    var = "pv_yr"   
    agg_col = "LINE_ORDERS"
    select_cols = [ "LINE_ORDERS_ACT", "Prediction_Trf", "LEAD_MNTS_ACT", "LD_UNCLAIMED_ACT", "LD_DRIVE_ACT", 
               "IDV_LEAD_MNTS_Pred", "IDV_LD_UNCLAIMED_Pred", "IDV_LD_DRIVE_Pred",
                "IDV_LEAD_MNTS_ACT", "IDV_LD_UNCLAIMED_ACT", "IDV_LD_DRIVE_ACT",
               "HELPER_MNTS_ACT", "HP_UNCLAIMED_ACT", "HP_DRIVE_ACT",
               "IDV_HELPER_MNTS_Pred", "IDV_HP_UNCLAIMED_Pred", "IDV_HP_DRIVE_Pred",
               "IDV_HELPER_MNTS_ACT", "IDV_HP_UNCLAIMED_ACT", "IDV_HP_DRIVE_ACT",
               "HELPER_OVR_ACT", "HPN_UNCLAIMED_ACT", "HPN_DRIVE_ACT",
               "IDV_HELPER_OVR_Pred", "IDV_HPN_UNCLAIMED_Pred", "IDV_HPN_DRIVE_Pred",
               "IDV_HELPER_OVR_ACT", "IDV_HPN_UNCLAIMED_ACT", "IDV_HPN_DRIVE_ACT"
               ]

    for i in select_cols:
        if str(i) not in data_all_df.columns:
            data_all_df[str(i)] = np.nan

    cov_grp_mm = dp.cov_check("RSS_MM", df, cov_buck, agg_col, fisc_calender, summ_start, summ_end)
    cov_mm = dp.cov_check("MM", df, cov_buck, agg_col, fisc_calender, summ_start, summ_end)
    
    sample = data_all_df[data_all_df.Variable_Type == var].copy()
    sample = pd.merge(sample, cov_grp_mm[["RSS_MM", "CAT"]])
    sample.CAT = sample.CAT.astype(int)
    sample["MAPE"] = abs(sample["LINE_ORDERS_ACT"]-sample["Prediction_Trf"])/sample["LINE_ORDERS_ACT"]
    bins= [-1, 0.1, 0.15, 0.20, 0.30, 0.50, 1.00, np.inf]
    labels = [1, 2, 3, 4, 5, 6, 7]
    sample["MAPE_BIN"] = pd.cut(sample['MAPE'], bins=bins, labels=labels)
    wk_list = fisc_calender[fisc_calender.FISC_WK_OF_MTH_ID >= pred_start].FISC_WK_OF_MTH_ID.unique()
    ly_start = fisc_calender[fisc_calender.FISC_WK_OF_MTH_ID == wk_list[4]].LY_FISC_WK_OF_MTH_ID.values
    ly_end = fisc_calender[fisc_calender.FISC_WK_OF_MTH_ID == wk_list[16]].LY_FISC_WK_OF_MTH_ID.values
    train = sample[((sample.FISC_WK_OF_MTH_ID >= mod_start) & (sample.FISC_WK_OF_MTH_ID < pred_start))]
    test_in_train = sample[((sample.FISC_WK_OF_MTH_ID >= ly_start[0]) & (sample.FISC_WK_OF_MTH_ID <= ly_end[0]))]
    # The 4 th index of the list has the prediction start date after excluding the 4 week hold out period
    # for analysis purpose we are considereing only 13 weeks of test period
    test = sample[((sample.FISC_WK_OF_MTH_ID >= wk_list[4]) & (sample.FISC_WK_OF_MTH_ID <= wk_list[16]))]
    fy20 = sample[(sample.FISC_WK_OF_MTH_ID >= summ_start) & (sample.FISC_WK_OF_MTH_ID <= summ_end)]
    
    ####################################################################################
    
    df_grp_summary = fy20.groupby(["RSS"]).sum()["LINE_ORDERS_ACT"].reset_index()
    df_grp_summary["LINE_ORDERS_PERC"] = df_grp_summary["LINE_ORDERS_ACT"]/df_grp_summary["LINE_ORDERS_ACT"].sum()
    ####################################################################################
    
    df_sample_list = []
    df_sample_list = cu.mape_stability(train, "Train CAT", "CAT", "MAPE", "LINE_ORDERS_ACT", df_sample_list)
    df_sample_list = cu.mape_stability(test_in_train, "Test_in_train CAT", "CAT", "MAPE", "LINE_ORDERS_ACT", df_sample_list)
    df_sample_list = cu.mape_stability(test, "Test CAT", "CAT", "MAPE", "LINE_ORDERS_ACT", df_sample_list)
    df_sample_list = cu.mape_stability(fy20, "FY20 CAT", "CAT", "MAPE", "LINE_ORDERS_ACT", df_sample_list)
    stability = pd.concat(df_sample_list)
    stability.reset_index(inplace = True)
    ####################################################################################
    
    df_sample_res = cu.mape_stability_2d(fy20, "FISC_MTH_NBR", "CAT")
    grp_sum = fy20.groupby("FISC_MTH_NBR").sum()["LINE_ORDERS_ACT"].reset_index()
    monthly_stability_check = df_sample_res.merge(grp_sum)
    ####################################################################################
    
    mape_cov = cu.mape_stability_2d(fy20, "CAT", "MAPE_BIN")
    ####################################################################################
    
    mape_grp = cu.mape_stability_2d(fy20, "RSS", "MAPE_BIN")
    mape_grp["TYPE"] = "MAPE Buckets"
    ####################################################################################
    
    mape_cov_grp = cu.mape_stability_2d(fy20, "RSS", "CAT")
    mape_cov_grp["TYPE"] = "Covariance Category"
    ####################################################################################
 
    fy20 = cu.sh_mape(fy20)
    
    sh_summary = []
    for grp in fy20["RSS"].unique():
        sample_df = fy20[fy20["RSS"] == grp]

        sh_summary = cu.mape_stability(sample_df, grp, "MAPE_LD_BIN", "MAPE_LD", "TOTAL_LEAD_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, grp, "MAPE_HP_BIN", "MAPE_HP", "TOTAL_HELPER_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, grp, "MAPE_HPN_BIN", "MAPE_HPN", "TOTAL_HELPERN_ACT", sh_summary)
    sh_summary_grp = pd.concat(sh_summary)
    sh_summary_grp.reset_index(inplace = True)

    ####################################################################################

    sh_summary = []
    var_list = ["pv_yr", "pv_roll_5wk"]
    for var in var_list:
        sample_df = data_all_df[data_all_df.Variable_Type == var].groupby(["FISC_WK_OF_MTH_ID", "FISC_YR_NBR", "MM"]).sum()[select_cols].reset_index()
        sample_df = pd.merge(sample_df, cov_mm[["MM", "CAT"]])
        sample_df = sample_df[sample_df.CAT>0]
        sample_df = sample_df[(sample.FISC_WK_OF_MTH_ID >= summ_start) & (sample.FISC_WK_OF_MTH_ID <= summ_end)]
        sample_df = cu.sh_mape(sample_df)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_LDA_BIN", "MAPE_LD_ACT", "TOTAL_LEAD_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_HPA_BIN", "MAPE_HP_ACT", "TOTAL_HELPER_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_HPNA_BIN", "MAPE_HPN_ACT", "TOTAL_HELPERN_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_LD_BIN", "MAPE_LD", "TOTAL_LEAD_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_HP_BIN", "MAPE_HP", "TOTAL_HELPER_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_HPN_BIN", "MAPE_HPN", "TOTAL_HELPERN_ACT", sh_summary)
    sh_mm_summary_df = pd.concat(sh_summary)
    sh_mm_summary_df.reset_index(inplace = True)

    ####################################################################################

    sh_summary = []
    for var in data_all_df.Variable_Type.unique():
        sample_df = data_all_df[data_all_df.Variable_Type == var].groupby(["FISC_WK_OF_MTH_ID", "FISC_YR_NBR", "MM"]).sum()[select_cols].reset_index()
        sample_df[select_cols] = sample_df[select_cols].replace({0:np.nan})
        sample_df = pd.merge(sample_df, cov_mm[["MM", "CAT"]])
        sample_df = sample_df[sample_df.CAT>0]
        sample_df = sample_df[(sample.FISC_WK_OF_MTH_ID >= summ_start) & (sample.FISC_WK_OF_MTH_ID <= summ_end)]
        sample_df = cu.sh_mape(sample_df)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_LDAP_BIN", "MAPE_LD_ACT_PRED", "TOTAL_LEAD_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_HPAP_BIN", "MAPE_HP_ACT_PRED", "TOTAL_HELPER_ACT", sh_summary)
        sh_summary = cu.mape_stability(sample_df, var, "MAPE_HPNAP_BIN", "MAPE_HPN_ACT_PRED", "TOTAL_HELPERN_ACT", sh_summary)
    sh_prop_summary = pd.concat(sh_summary)
    sh_prop_summary.reset_index(inplace = True)
    
    best_mthd = sh_prop_summary.groupby(['IDENTIFIER']).mean()['WEIGHTED MAPE'].reset_index().sort_values(by = ["WEIGHTED MAPE"], ascending = True)
    var_list = best_mthd["IDENTIFIER"][:2].to_list()
    ####################################################################################

    sample_df = fy20.copy()
    sample_df = sample_df[sample_df.MM.str.contains("HD")]
    sample_df = sample_df.groupby(["FISC_WK_OF_MTH_ID", "MM"]).sum()[["LINE_ORDERS_ACT", "Prediction_Trf"]].reset_index()
    df_grp_all = sample_df.copy()
    df_grp_all[["LINE_ORDERS_ACT"]] = df_grp_all[[
                           "LINE_ORDERS_ACT"]].replace({0:np.nan})
    df_grp_all["MAPE_TA"] = abs(df_grp_all["LINE_ORDERS_ACT"]-df_grp_all["Prediction_Trf"])/df_grp_all["LINE_ORDERS_ACT"]
    bins= [-1, 0.1, 0.15, 0.20, 0.30, 0.50, 1.00, np.inf]
    labels = [1, 2, 3, 4, 5, 6, 7]
    df_grp_all["MAPE_TA_BIN"] = pd.cut(df_grp_all['MAPE_TA'], bins=bins, labels=labels)

    var = "pv_yr"   
    sample = data_all_df[data_all_df.Variable_Type == var].copy()
    
    sample = sample.merge(cov_grp_mm[["RSS_MM", "CAT"]])
    sample = sample[sample.CAT > 0]
    sample = sample[(sample.FISC_WK_OF_MTH_ID >= summ_start) & (sample.FISC_WK_OF_MTH_ID <= summ_end)]
    
    
    sample = data_all_df[data_all_df.Variable_Type == var].copy()
    sample = pd.merge(sample, cov_mm[["MM", "CAT"]])
    sample = sample[sample.CAT > 0]
    sample = sample[(sample.FISC_WK_OF_MTH_ID >= summ_start) & (sample.FISC_WK_OF_MTH_ID <= summ_end)]
    fy20_mm = sample.groupby(["FISC_WK_OF_MTH_ID", "MM", "CAT"]).sum()[["LINE_ORDERS_ACT", "Prediction_Trf"]].reset_index()
    fy20_mm[["LINE_ORDERS_ACT" ]] = fy20_mm[["LINE_ORDERS_ACT"]].replace({0:np.nan})
    fy20_mm["MAPE"] = abs(fy20_mm["LINE_ORDERS_ACT"]-fy20_mm["Prediction_Trf"])/fy20_mm["LINE_ORDERS_ACT"]
    bins= [-1, 0.1, 0.15, 0.20, 0.30, 0.50, 1.00, np.inf]
    labels = [1, 2, 3, 4, 5, 6, 7]
    fy20_mm["MAPE_BIN"] = pd.cut(fy20_mm['MAPE'], bins=bins, labels=labels)
    
    df_sample_list = []
    df_sample_list = cu.mape_stability(fy20, "FY20 RSS_MM", "CAT", "MAPE", "LINE_ORDERS_ACT", df_sample_list)
    df_sample_list = cu.mape_stability(fy20_mm, "FY20 MM", "CAT", "MAPE", "LINE_ORDERS_ACT", df_sample_list)
    df_overall_summay = pd.concat(df_sample_list)
    df_overall_summay.reset_index(inplace = True)
    return df_grp_summary, stability, monthly_stability_check, mape_cov, mape_cov_grp, mape_grp, sh_mm_summary_df, sh_summary_grp, sh_prop_summary, df_overall_summay, var_list

