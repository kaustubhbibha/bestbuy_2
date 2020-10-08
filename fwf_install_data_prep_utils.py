"""
In-Home Install Future Work Force - Service Line Orders Forecasting

This script conatins the functions to prepare data for Service Line orders model building/scoring,
Sales & Forecast for In-Home Future Work Force (IHI_FWF)

This file requires following scripts to execute
    * fwf_install_data_extraction.py - functions containing data extraction utils
"""

import os
import pandas as pd
import numpy as np
import fwf_install_data_extraction_utils as de


def cal_data_prep(cal_df_all, usholidays):
    """
    Module to subset BBY fiscal calendar to required fields,
    roll up to fiscal week and append binary flags for weeks
    with US holidays
    Input:
        cal_df_all: Pandas dataframe - Fiscal calendar data
        usholidays: Pandas dataframe - US Federal holidays
    Output:
        holiday: Pandas dataframe - Holiday lookup
    """
    holiday_list = {"Washington's Birthday":1, "Memorial Day":2, "Independence Day":3,
                    "Labor Day":4, "Columbus Day":5, "Veterans Day":6, "Thanksgiving Day":7, 
                    "Christmas Day":8, "New Year's Day":9, "Birthday of Martin Luther King, Jr.":10}
    
    cal_df_all["CALNDR_DT"] = pd.to_datetime(cal_df_all["CALNDR_DT"])
    cal_df = cal_df_all.copy()
    # load holidays list file
    usholidays["CALNDR_DT"] = pd.to_datetime(usholidays["CALNDR_DT"])
    # Roll-up at fiscal week level with flag if fiscal week contains holiday
    usholidays = pd.merge(cal_df_all[["CALNDR_DT", "FISC_WK_OF_MTH_ID"]], usholidays, on="CALNDR_DT", how="left")
    usholidays["Holiday_Type"] = usholidays["Holiday_Type"].map(holiday_list)
    usholidays["Holiday_Type"].fillna(0, inplace=True)
    usholidays["Holiday"] = usholidays["Holiday_Type"].apply(lambda x: 1 if x > 0 else 0)
    usholidays = usholidays[["FISC_WK_OF_MTH_ID", "Holiday_Type", "Holiday"]].groupby(["FISC_WK_OF_MTH_ID"]).sum().reset_index()
    # list of columns required for modelling exercise
    sel_cols = ["FISC_YR_NBR", "FISC_WK_OF_MTH_ID", "FISC_WK_OF_YR_NBR", "FISC_EOW_DT", "FISC_MTH_NBR", "FISC_QTR_NBR"]
    # subset the fiscal calendar file for duration of model analysis

    cal_df = cal_df[sel_cols].drop_duplicates()
    cal_df = pd.merge(cal_df, usholidays, on="FISC_WK_OF_MTH_ID", how="left")
    cal_df["FISC_EOW_DT"] = pd.to_datetime(cal_df["FISC_EOW_DT"])
    cal_df.sort_values(by=["FISC_EOW_DT"], inplace=True)
    holiday = cal_df.copy()
    for i in range(1, 11):
        hol_sub = holiday[holiday["Holiday_Type"]==i][["FISC_WK_OF_MTH_ID", "Holiday"]]
        hol_sub.columns = ["FISC_WK_OF_MTH_ID", "ADSTOCK_HOLIDAY_"+str(i)]
        holiday = pd.merge(holiday, hol_sub, how = "left" )
        holiday["ADSTOCK_HOLIDAY_"+str(i)] = holiday["ADSTOCK_HOLIDAY_"+str(i)].fillna(0)
        holiday["ADSTOCK_HOLIDAY_"+str(i)] = holiday["ADSTOCK_HOLIDAY_"+str(i)].shift(-1)+holiday["ADSTOCK_HOLIDAY_"+str(i)]+holiday["ADSTOCK_HOLIDAY_"+str(i)].shift(1)
        holiday["ADSTOCK_HOLIDAY_"+str(i)] = holiday["ADSTOCK_HOLIDAY_"+str(i)].fillna(0)
    # The Holidays corresponds to the Id 2, 3, 4, 7(refer holiday lis dict above) shows a peak very year, so creating a flag column
    # for those holidays
    hol_sub = holiday[holiday["Holiday_Type"].isin([2,3,4,7])][["FISC_WK_OF_MTH_ID", "Holiday"]]
    hol_sub.columns = ["FISC_WK_OF_MTH_ID", "HOLIDAY_SPIKE"]
    holiday = pd.merge(holiday, hol_sub, how = "left" )
    holiday["HOLIDAY_SPIKE"] = holiday["HOLIDAY_SPIKE"].fillna(0)
    # The Holidays corresponds to the Id 2, 3, 4, 7(refer holiday lis dict above) shows a peak very year, so creating a seperate/individual
    # flag column for those holidays and Id 5 shows a dip
    for i in [2,3,4,5,7]:
        hol_sub = holiday[holiday["Holiday_Type"]==i][["FISC_WK_OF_MTH_ID", "Holiday"]]
        hol_sub.columns = ["FISC_WK_OF_MTH_ID", "HOLIDAY_SPIKE_"+str(i)]
        holiday = pd.merge(holiday, hol_sub, how = "left")
        holiday["HOLIDAY_SPIKE_"+str(i)] = holiday["HOLIDAY_SPIKE_"+str(i)].fillna(0)
    
    holiday.rename(columns={'HOLIDAY_SPIKE_2':'HOLIDAY_SPIKE_N1',
                          'HOLIDAY_SPIKE_3':'HOLIDAY_SPIKE_N2',
                          'HOLIDAY_SPIKE_4':'HOLIDAY_SPIKE_N3',
                          'HOLIDAY_SPIKE_5':'HOLIDAY_VALLEY_N1',
                          'HOLIDAY_SPIKE_7': 'HOLIDAY_SPIKE_3'}, 
                 inplace=True)

    hol_sub = holiday[holiday["Holiday_Type"]==7][["FISC_WK_OF_MTH_ID", "Holiday"]]
    hol_sub.columns = ["FISC_WK_OF_MTH_ID", "HOLIDAY_SPIKE_AD1_7"]
    holiday = pd.merge(holiday, hol_sub, how = "left" )
    holiday["HOLIDAY_SPIKE_AD1_7"] = holiday["HOLIDAY_SPIKE_AD1_7"].fillna(0)
    holiday["HOLIDAY_SPIKE_AD2_7"] = holiday["HOLIDAY_SPIKE_AD1_7"]
    # Adding a flag column for id 7 which is a peak week with adstock effect. So peak week and peak week-1 will have 1's
    holiday["HOLIDAY_SPIKE_AD1_7"] = holiday["HOLIDAY_SPIKE_AD1_7"].shift(-1)+holiday["HOLIDAY_SPIKE_AD1_7"]
    holiday["HOLIDAY_SPIKE_AD1_7"] = holiday["HOLIDAY_SPIKE_AD1_7"].fillna(0)
    # Adding a flag column for id 7 which is a peak week with adstock effect. So peak week and peak week+1 will have 1's
    holiday["HOLIDAY_SPIKE_AD2_7"] = holiday["HOLIDAY_SPIKE_AD2_7"]+holiday["HOLIDAY_SPIKE_AD2_7"].shift(1)
    holiday["HOLIDAY_SPIKE_AD2_7"] = holiday["HOLIDAY_SPIKE_AD2_7"].fillna(0)
    holiday.rename(columns={'HOLIDAY_SPIKE_AD1_7':'HOLIDAY_SPIKE_N4',
                          'HOLIDAY_SPIKE_AD2_7':'HOLIDAY_SPIKE_N5'}, 
                 inplace=True)
    
    return holiday


def cov_check(cov_var, line_orders, new_cat, agg_col, fisc_calender_all, summ_start, summ_end):
    """
    Covariance buckets creation logic. Create covariance buckets as 1 to 7 based on Avergae line orders 
    per week, standard deviation & number of weeks present.
    Input:
        cov_var: Group X MM columns name
        line_orders: Line orders data
        new_cat: Pandas dataframe - Covaraince buckets
        spar: Pandas dataframe - Sparsity buckets
        agg_col: Line orders columns
        fisc_calender_all: Pandas dataframe - Fiscal calendar
        fiscal_year: Year for analysis
    Output:
        sample_20_grp: Pandas dataframe - Combinations with covariance buckets
     """

    sample_20_all = line_orders[(line_orders.FISC_WK_OF_MTH_ID >= summ_start) &
                                       (line_orders.FISC_WK_OF_MTH_ID <= summ_end)].copy()
    sam_pvt = pd.pivot_table(sample_20_all, index=cov_var, columns="FISC_WK_OF_MTH_ID", values="LINE_ORDERS", aggfunc="sum")
    sam_pvt = sam_pvt.fillna(0)
    sample_20_grp = pd.DataFrame(index=sam_pvt.index.values)
    sample_20_grp["LO_M"] = sam_pvt.mean(axis=1)
    sample_20_grp["LO_STD"] = sam_pvt.std(axis=1)
    sample_20_grp["LO_SUM"] = sam_pvt.sum(axis=1)
    sample_20_grp["LO_CNT"] = sam_pvt.count(axis=1)
    sample_20_grp.reset_index(inplace=True)
    sample_20_grp.rename({"index": cov_var}, axis=1, inplace=True)

    sample_20_grp["LO_COV"] = sample_20_grp["LO_STD"] / sample_20_grp["LO_M"]
    bins= [0, 7, 21, 42, 200, np.inf]
    
    labels = [1, 2, 3, 4, 5]
    sample_20_grp["AVG_BIN"] = pd.cut(sample_20_grp['LO_M'], bins=bins, labels=labels)
    sample_20_grp["AVG_BIN"] = sample_20_grp["AVG_BIN"].astype(int)
    bins= [0, 0.3, 0.4, 0.6, np.inf]
    labels = [1, 2, 3, 4]
    sample_20_grp["COV_BIN"] = pd.cut(sample_20_grp['LO_COV'], bins=bins, labels=labels)
    sample_20_grp["COV_BIN"] = sample_20_grp["COV_BIN"].astype(int)
    sample_20_grp["AVG_BIN"]  = np.where(sample_20_grp["AVG_BIN"] < 0, 0, sample_20_grp["AVG_BIN"] )
    sample_20_grp["COV_BIN"]  = np.where(sample_20_grp["COV_BIN"] < 0, 0, sample_20_grp["COV_BIN"] )
    sample_20_grp["COMB"] = sample_20_grp["AVG_BIN"].astype(int).astype(str)+sample_20_grp["COV_BIN"].astype(int).astype(str)
    sample_20_grp["COMB"] = sample_20_grp["COMB"].astype(int)
    sample_20_grp = pd.merge(sample_20_grp, new_cat[["COMB", "CAT"]], how = "left")
    
    return sample_20_grp


def comb_minimums(fisc_calender_all, df, summ_start, summ_end, lst_mth, lst_qtr, lst_yr_start):
    """
    Logic to categorize combinations which are tagged as minimums or very sparse. These are combinations
    which doesn't have enough data to model or produce rule based predictions 
    Input:
        fisc_calender_all: Pandas dataframe - Calendar data
        df: Line orders data
        fiscal_year: Year for analysis
    Output:
        sample_grp: Pandas dataframe - Combinations which needs to predicted minimums
     """
    fisc_calender = fisc_calender_all[['FISC_YR_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_MTH_NBR', 'FISC_QTR_NBR']].drop_duplicates()
    fisc_calender = fisc_calender.sort_values("FISC_WK_OF_MTH_ID")
    fisc_calender = fisc_calender.dropna()
    df_new = pd.merge(df, fisc_calender)
    sample = df_new[(df_new.FISC_WK_OF_MTH_ID >= summ_start) & (df_new.FISC_WK_OF_MTH_ID <= summ_end)]
    sample_yr = sample.groupby("RSS_MM").count()["FISC_WK_OF_MTH_ID"].reset_index()
    sample_qtr = sample[(sample.FISC_WK_OF_MTH_ID >= lst_qtr) & (sample.FISC_WK_OF_MTH_ID <= summ_end)].groupby("RSS_MM").count()["FISC_WK_OF_MTH_ID"].reset_index()
    sample_mth = sample[(sample.FISC_WK_OF_MTH_ID >= lst_mth) & (sample.FISC_WK_OF_MTH_ID <= summ_end)].groupby("RSS_MM").count()["FISC_WK_OF_MTH_ID"].reset_index()
    sample_all = pd.merge(sample_yr[["RSS_MM"]], sample_qtr[["RSS_MM"]])
    sample_all = pd.merge(sample_all[["RSS_MM"]], sample_mth[["RSS_MM"]])
    sample_all["COMMENTS_CURR"] = "Others"
    sample_19 = df_new[(df_new.FISC_WK_OF_MTH_ID >= lst_yr_start) & (df_new.FISC_WK_OF_MTH_ID <= summ_start)]
    sample_yr = sample_19.groupby("RSS_MM").count()["FISC_WK_OF_MTH_ID"].reset_index()
    # Check if atleast half the year is present in the previous year of the analysis
    sample_yr = sample_yr[sample_yr.FISC_WK_OF_MTH_ID > 26]
    sample_yr["COMMENTS_PV"] = "Others"
    sample_grp = sample[["RSS_MM"]].drop_duplicates()
    sample_grp = pd.merge(sample_grp, sample_all, how = "left")
    sample_grp["COMMENTS_CURR"] = sample_grp["COMMENTS_CURR"].fillna("Missing")
    sample_grp = pd.merge(sample_grp, sample_yr, how = "left")
    sample_grp["COMMENTS"]= sample_grp["COMMENTS_PV"].fillna(sample_grp["COMMENTS_CURR"])
    sample_grp["COMMENTS"] = sample_grp["COMMENTS"].fillna("Missing")
    sample_grp["COMMENTS"] = np.where(((sample_grp["COMMENTS_CURR"] == "Others") & (sample_grp["COMMENTS_PV"].isna())),"Average", sample_grp["COMMENTS_CURR"])
    
    return sample_grp


def get_sparsity_bucket(data, merge_df,summ_end):
    """
    Logic to categorize combinations into Prediction model or Rule based prediction or zero prediction based of historical volumne 
    of data.
    Input:
        data: Pandas dataframe - Line orders dataframe.
        merge_df: Pandas dataframe - Dataframe containing commments about historical volume for all combinations.
        summ_end: Last date of summary generation period.
    Output:
        RSS_mm_grp: Pandas dataframe - Combinations with there stats like prediction approach, mean line orders.
    """
    
    pvt_df = pd.pivot_table(data, index=["RSS_MM", "MM"], columns="FISC_WK_OF_MTH_ID", values="LINE_ORDERS", aggfunc="sum").reset_index()
    wks = data["FISC_WK_OF_MTH_ID"].unique()
    pvt_df.loc[:, wks] = pvt_df.loc[:, wks].fillna(0)
    
    data = data[data.FISC_WK_OF_MTH_ID <= summ_end]
    
    melt_df = pd.melt(pvt_df, id_vars=["RSS_MM", "MM"], value_vars=wks, value_name="LINE_ORDERS", var_name="FISC_WK_OF_MTH_ID")
    melt_df = melt_df.merge(merge_df[["RSS_MM", "CAT", "COMMENTS"]], how="left")
    
    
    wk_profile = data.groupby(["RSS_MM", "FISC_YR_NBR"]).agg({"FISC_WK_OF_MTH_ID": "count"}).reset_index()
    wk_pvt = pd.pivot_table(wk_profile, index="RSS_MM", columns="FISC_YR_NBR", values="FISC_WK_OF_MTH_ID", aggfunc="sum").reset_index()
    wk_pvt = wk_pvt.merge(merge_df[["RSS_MM", "CAT", "COMMENTS"]], how="left")
    wk_pvt["CAT"] = wk_pvt["CAT"].fillna(0)
    
    lst_two_yrs = wk_profile.FISC_YR_NBR.sort_values(ascending = False).unique()[:2]
    
    wk_pvt["COMMENTS"] = wk_pvt["COMMENTS"].fillna("Current yr Missing")

    RSS_mm_grp = melt_df[["RSS_MM"]].drop_duplicates()
    RSS_mm_grp = RSS_mm_grp.merge(merge_df, how="left")
    RSS_mm_grp = RSS_mm_grp.merge(wk_pvt[["RSS_MM", lst_two_yrs[1], lst_two_yrs[0]]], how="left")
    
    RSS_mm_grp["Method"] = np.where(RSS_mm_grp["CAT"].isna(), "Current yr Missing", "Others")
    RSS_mm_grp["Flag_1"] = np.where(RSS_mm_grp["CAT"].isin(range(1,6)), "1_5", "Others")
    RSS_mm_grp["Flag_2"] = np.where(RSS_mm_grp["LO_M"]>7, "Mean", "Others")
    RSS_mm_grp["Flag_3"] = np.where(((RSS_mm_grp[lst_two_yrs[1]]>=26) & (RSS_mm_grp[lst_two_yrs[0]]>=26)), "Wk", "Others")
    RSS_mm_grp["Sparsity_Method"] = "Zero"
    RSS_mm_grp["Sparsity_Method"] = np.where(((RSS_mm_grp["Flag_1"]=="1_5")
                                              & (RSS_mm_grp["Flag_2"]=="Mean")
                                              & (RSS_mm_grp["Flag_3"]=="Wk")
                                              & (RSS_mm_grp["COMMENTS"]=="Others")), "Prediction Model", RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"] = np.where(((RSS_mm_grp["Flag_1"]=="1_5")
                                              & ((RSS_mm_grp["Flag_2"]=="Others")
                                              | (RSS_mm_grp["Flag_3"]=="Others"))), "Rule Based Prediction", RSS_mm_grp["Sparsity_Method"])
    
    RSS_mm_grp["Sparsity_Method"] = np.where((RSS_mm_grp["Flag_1"]=="Others")
                                             & (RSS_mm_grp["COMMENTS"]=="Others")
                                             & (RSS_mm_grp["Flag_3"]=="Wk"), "Rule Based Prediction", RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"] = np.where((RSS_mm_grp["Flag_1"]=="Others")
                                             & (RSS_mm_grp["COMMENTS"]=="Others")
                                             & (RSS_mm_grp["Flag_3"]=="Others"), "Rule Based Prediction", RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"] = np.where((RSS_mm_grp["Flag_1"]=="Others")
                                             & (RSS_mm_grp["COMMENTS"]=="Average")
                                             & (RSS_mm_grp["Flag_3"]=="Wk"), "Rule Based Prediction", RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"] = np.where((RSS_mm_grp["Flag_1"]=="Others")
                                             & (RSS_mm_grp["COMMENTS"]=="Average")
                                             & (RSS_mm_grp["Flag_3"]=="Others"), "Rule Based Prediction", RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"] = np.where((RSS_mm_grp["COMMENTS"]=="Missing"), "Zero", RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"] = np.where(RSS_mm_grp[lst_two_yrs[0]]<=13, "Zero", RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"] = np.where((RSS_mm_grp[lst_two_yrs[0]]>40)&(RSS_mm_grp[lst_two_yrs[1]]>26)&(RSS_mm_grp['LO_M']> 5) , 'Prediction Model', RSS_mm_grp["Sparsity_Method"])
    RSS_mm_grp["Sparsity_Method"].value_counts()
    
    RSS_map = dict(zip(data["RSS_MM"], data["RSS"]))
    mm_map = dict(zip(data["RSS_MM"], data["MM"]))
    RSS_mm_grp["RSS"] = RSS_mm_grp["RSS_MM"].map(RSS_map)
    RSS_mm_grp["MM"] = RSS_mm_grp["RSS_MM"].map(mm_map)
    RSS_mm_grp.rename({"LO_M": "Mean",
                       "LO_STD": "Std. Dev",
                       "LO_SUM": "Total Line Orders",
                       "LO_COV": "Co-variance",
                       lst_two_yrs[1]: "No. of Weeks-prv year",
                       lst_two_yrs[0]: "No. of Weeks-curr year",
                       "Method": f"FY{lst_two_yrs[0]} Data Avail.",
                       "Flag_1": "Avg/Covar Bucket",
                       "Flag_2": "Avg. Line Orders/Week",
                       "Flag_3": "Min. 26 wk Data in prv yr & curr yr",
                       "Sparsity_Method": "Sparsity"
                       }, axis=1, inplace=True)
    
    RSS_mm_grp = RSS_mm_grp[['RSS_MM', 'RSS', 'MM', 'Mean', 'Std. Dev', 'Total Line Orders',
                             'Co-variance', 'AVG_BIN', 'COV_BIN', 'COMB', 'CAT', 'COMMENTS', 'No. of Weeks-curr year',
                             "No. of Weeks-prv year", f'FY{lst_two_yrs[0]} Data Avail.', 'Avg/Covar Bucket',
                             'Avg. Line Orders/Week', 'Min. 26 wk Data in prv yr & curr yr', 'Sparsity']]
    return RSS_mm_grp


def modellable_combinations(base_dir, lkp_dir, sparsity_params, df, lkp_refresh, fisc_calender_all, cov_buck, version, summ_start, summ_end, lst_mth, lst_qtr, lst_yr_start, data_flag):
    
    """
    Loads Sparsity, Covariance & Minimum categories to seperate line orders data based on modellable and non-modellable combinations.
    Input:
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
        sparsity_params: Dict - Sparsity parameters  
        minimums_params: Dict - Parameters for combinations which needs to predicted minimums
        df: Pandas dataframe - Line orders data
        lkp_refresh: Flag to denote lookup refresh
        fisc_calender_all: Pandas dataframe - Calendar data
        cov_buck: Covariance bucket file name
        version: Model trained date
        summ_start: Sparsity start
        summ_end: Sparsity end
        
    Output:
        df_all: Pandas dataframe - Line orders data for all sparsity combinations
        minimums: Pandas dataframe - Combinations which needs to predicted minimums
        RSS_mm_sparsity: Pandas dataframe - Sparsity lookup
     """
    agg_col = "LINE_ORDERS"
    if lkp_refresh == 0:
        RSS_mm_grp = pd.read_excel(os.path.join(base_dir, lkp_dir, sparsity_params['filename']), sheet_name = sparsity_params['sheet_name'])
        RSS_mm_grp = RSS_mm_grp[RSS_mm_grp.CAT > 0]
        
        RSS_mm_grp = RSS_mm_grp[["RSS_MM", "RSS", "Sparsity"]]
        
        RSS_mm_grp.columns = ["RSS_MM", "RSS", "SPARSITY"]
        df_all = pd.merge(df.copy(), RSS_mm_grp, how = "inner")
        
        return df_all, RSS_mm_grp
    else:                        
        cov_sparsity = cov_check("RSS_MM", df.copy(), cov_buck, agg_col, fisc_calender_all, summ_start, summ_end)
        cov_sparsity = cov_sparsity[cov_sparsity.CAT >0]
        cov_sparsity_1 = cov_sparsity.copy()
        cov_sparsity = cov_sparsity[["RSS_MM", "CAT"]]
        minimums = comb_minimums(fisc_calender_all, df, summ_start, summ_end, lst_mth, lst_qtr, lst_yr_start)
        merge_df = cov_sparsity_1.merge(minimums, how="outer")
        RSS_mm_map = dict(zip(df["RSS_MM"], df["MM"]))
        merge_df["MM"] = merge_df["RSS_MM"].map(RSS_mm_map)
        RSS_mm_grp = get_sparsity_bucket(df, merge_df, summ_end)
        
        RSS_mm_grp = RSS_mm_grp[~RSS_mm_grp.CAT.isna()]
        
        if data_flag:
            if not os.path.isdir(os.path.join(base_dir, lkp_dir, version)):
                os.mkdir(os.path.join(base_dir, lkp_dir, version))
            RSS_mm_grp.to_excel(os.path.join(base_dir, lkp_dir, version, sparsity_params['filename']), sheet_name = sparsity_params['sheet_name'], index = False)

        RSS_mm_grp.to_excel(os.path.join(base_dir, lkp_dir, sparsity_params['filename']), sheet_name = sparsity_params['sheet_name'], index = False)
        
    RSS_mm_grp = RSS_mm_grp[["RSS_MM", "RSS", "Sparsity"]]
    RSS_mm_grp.columns = ["RSS_MM", "RSS", "SPARSITY"]
    df_all = pd.merge(df.copy(), RSS_mm_grp, how = "inner")
    return df_all, RSS_mm_grp, minimums

                     
def lookup_data_load(start, pred_end, base_dir, lkp_dir, cal_param, cov_param, fisc_cal_params, loc_params, conn):
    """
    Loads all the necessary lookup required for model building, evaluation & scoring
    Input:
       start: Data extraction start Fiscal week
        end: Data extraction end Fiscal week
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
        cal_param: Calendar parameters
        cov_param: Covariance buckets
        conn: Terradata connection
    Output:
        cal_df: Pandas dataframe - Holiday lookup
        fisc_calender: Pandas dataframe - Fiscal calendar lookup
        cov_buck: Pandas dataframe - Covariance bucket 
        loc_state_grp: Pandas dataframe - Location lookup
    """
    fisc_calender_all = de.fiscal_clndr_info(conn)
    fisc_calender = fisc_calender_all[['FISC_YR_NBR', 'FISC_MTH_NBR', 'FISC_WK_OF_MTH_ID', 'FISC_WK_OF_YR_NBR', 'FISC_QTR_NBR', 'LY_FISC_WK_OF_MTH_ID']].drop_duplicates()
    fisc_calender = fisc_calender.sort_values("FISC_WK_OF_MTH_ID")
    fisc_calender = fisc_calender.fillna(method="ffill")
    fisc_calender = fisc_calender.dropna()
    fisc_calender = fisc_calender.reset_index(drop = True)
    fisc_calender.to_csv(os.path.join(base_dir, lkp_dir, fisc_cal_params['filename']), index = False)
    fisc_calender = fisc_calender[(fisc_calender.FISC_WK_OF_MTH_ID >= start) & (fisc_calender.FISC_WK_OF_MTH_ID <= pred_end)]
    usholidays = pd.read_csv(os.path.join(base_dir, lkp_dir, cal_param))
    cal_df_all = cal_data_prep(fisc_calender_all, usholidays)
    cal_df = cal_df_all[cal_df_all.FISC_WK_OF_MTH_ID >= start]
    cal_df = cal_df[cal_df.FISC_WK_OF_MTH_ID <= pred_end]
    cal_df = pd.merge(cal_df, fisc_calender, how = "left")
    cov_buck = pd.read_csv(os.path.join(base_dir,lkp_dir,cov_param))
    loc_state_grp = de.state_tz_info(conn)
    
    fisc_calender.to_csv(os.path.join(base_dir, lkp_dir, fisc_cal_params['filename']), index = False)
    loc_state_grp.to_csv(os.path.join(base_dir,  lkp_dir, loc_params['filename']), index = False)
    
    cal_df_dummy = pd.get_dummies(cal_df['Holiday_Type'].astype(int), prefix = 'Holiday')
    cal_df = pd.concat([cal_df, cal_df_dummy], axis = 1)

    return cal_df, fisc_calender, cov_buck, loc_state_grp
    

def data_load(start, end, cal_df, conn, hlpr_time_params, location_params, svc_sku_params, dt_params, sales_loc, svc_cls_lkp, base_dir, lkp_dir, version, line_orders_df, sales_df, forecast_df, data_flag):
    """
    Loads all the necessary data for model building & evaluation from DB
    Input:
        start: Data extraction start Fiscal week
        end: Data extraction end Fiscal week
        for_end: Data extraction end Fiscal week for forecast
        cal_df: Pandas dataframe - Holiday lookup
        conn: Terradata connection
        svc_cls_db_lkp: Dict - Class Subclass to FPG mapping parameters
        hlpr_time_params: Dict - helper override time standards parameters
        location_params: Dict - Zipcode to Minor Market mapping filename
        svc_sku_params: Dict - SKU to RSS, RSS mapping parameters
        dt_params: Dict - drive time standards filename
        loc_sls: Dict - Location to Minor Market mapping filename
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
        version: Model training date
        line_orders_df: Line orders file name
        sales_df: Sales file name
        forecast_df: Forecast file name
    Output:
        df: Pandas dataframe - Line orders data
        sales: Pandas dataframe - Sales data
        forecast: Pandas dataframe - Forecast at Week X FPG
    """
                           
    df = de.get_lineorders(start, end, conn, hlpr_time_params, location_params, svc_sku_params, dt_params, base_dir, lkp_dir)

    df.to_csv(os.path.join(base_dir, lkp_dir, line_orders_df), index = None)
    agg_col = "LINE_ORDERS"
    df[agg_col].replace({0: np.nan}, inplace=True)

    sales = de.get_sales(start, end, df, conn, svc_cls_lkp, sales_loc, base_dir, lkp_dir)
    
    sales.to_csv(os.path.join(base_dir, lkp_dir, sales_df), index = None)
    
    forecast_cls_scls = de.get_forecasts(start, end, conn, svc_cls_lkp, sales_loc, base_dir, lkp_dir)

    forecast = forecast_cls_scls.groupby(["FISC_WK_OF_MTH_ID", "RSS"]).sum()["FORECAST_WP"].reset_index()
    forecast.to_csv(os.path.join(base_dir, lkp_dir, forecast_df), index = False)
    
    if data_flag:
        df.to_csv(os.path.join(base_dir, lkp_dir, version, line_orders_df), index = None)
        sales.to_csv(os.path.join(base_dir, lkp_dir, version, sales_df), index = None)
        forecast.to_csv(os.path.join(base_dir, lkp_dir, version, forecast_df), index = None)
        
    df = df.merge(cal_df[["FISC_EOW_DT", "FISC_YR_NBR", "FISC_WK_OF_MTH_ID", "FISC_WK_OF_YR_NBR"]], on = "FISC_WK_OF_MTH_ID", how="left")
    agg_col = "LINE_ORDERS"
    df[agg_col].replace({0: np.nan}, inplace=True)
    sales = sales.merge(cal_df[["FISC_WK_OF_MTH_ID","FISC_EOW_DT", "FISC_YR_NBR", "FISC_WK_OF_YR_NBR"]], on = "FISC_WK_OF_MTH_ID", how="left")
    
    
    return df, sales, forecast
    # -----------------------------------------------------------------------------


def flat_data_load(start, end, cal_df, base_dir, lkp_dir, line_orders_df, sales_df, forecast_df):
    """
    Loads all the necessary data for model building & evaluation from flat files
    Input:
        start: Data extraction start Fiscal week
        end: Data extraction end Fiscal week
        cal_df: Pandas dataframe - Holiday lookup
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
        line_orders_df: Line Orders frozen data name
        sales_df: Sales frozen data name
        forecast_class_subclass_df: Forecast Class Subclass frozen data name
    Output:
        df: Pandas dataframe - Line orders data
        sales: Pandas dataframe - Sales data
        forecast: Pandas dataframe - Forecast at Week X FPG
    """
    
    df = pd.read_csv(os.path.join(base_dir, lkp_dir, line_orders_df))
    df = df.merge(cal_df[["FISC_EOW_DT", "FISC_YR_NBR", "FISC_WK_OF_MTH_ID", "FISC_WK_OF_YR_NBR"]], on = "FISC_WK_OF_MTH_ID", how="left")
    agg_col = "LINE_ORDERS"
    df[agg_col].replace({0: np.nan}, inplace=True)
    sales = pd.read_csv(os.path.join(base_dir, lkp_dir, sales_df))
    sales = sales.merge(cal_df[["FISC_EOW_DT", "FISC_WK_OF_MTH_ID", "FISC_YR_NBR", "FISC_WK_OF_YR_NBR"]], on = "FISC_WK_OF_MTH_ID", how="left")

    forecast = pd.read_csv(os.path.join(base_dir, lkp_dir, forecast_df))
    
    
    return df, sales,forecast
    # -----------------------------------------------------------------------------

def lookup_data(base_dir, conn, lkp_dir, cov_param, sku_line_orders_params, cls_sales_params, hlpr_time_params, location_params, dt_params, state_tz_params, sales_loc_params):
    """
    Creates a dictionary with lookup files used
    Input:
        base_dir: Model framework parent directory
        conn: Teradata connection
        lkp_dir: Model lookup folder directory
        cov_param: Covariance buckets
        sku_line_orders_params: Dict - SKU to FC mapping parameters
        cls_line_orders_params: Dict - Class Subclass to FC mapping parameters
        cls_sales_params: Dict - Class Subclass to FC mapping parameters
        hlpr_time_params: Dict - helper override time standards parameters
        location_params: Dict - Location to Minor Market mapping parameters
        dt_params: Dict - drive time standards parameters
        state_tz_params: MM to State TZ name
        sales_loc_params: Sales location file name
        cov_sparsity_params: Sparsity params
    Output:
        lookup_files: Dict - Lookup file dictionary
    """

    cov_buck = pd.read_csv(os.path.join(base_dir, lkp_dir, cov_param))
    loc_state_grp = pd.read_csv(os.path.join(base_dir, lkp_dir, state_tz_params['filename']))
    sku_mapp = pd.read_excel(os.path.join(base_dir, lkp_dir, sku_line_orders_params['filename']), sheet_name=sku_line_orders_params['sheet_name'])
    hlpr_time = pd.read_excel(os.path.join(base_dir, lkp_dir, hlpr_time_params['filename']), sheet_name = hlpr_time_params['sheet_name'])[hlpr_time_params['columns']]
    dt = pd.read_csv(os.path.join(base_dir, lkp_dir, dt_params['filename']))
    service_cls_db = pd.read_excel(os.path.join(base_dir, lkp_dir, cls_sales_params['filename']), sheet_name=cls_sales_params['sheet_name'])
    location = pd.read_excel(os.path.join(base_dir, lkp_dir, location_params['filename']), sheet_name = location_params['sheet_name'])
    sales_loc_map = pd.read_csv(os.path.join(base_dir, lkp_dir, sales_loc_params['filename']))
    lookup_files = {
                    cov_param: cov_buck,
                    state_tz_params['filename']: loc_state_grp,
                    sku_line_orders_params['filename']+"-"+sku_line_orders_params['sheet_name']: sku_mapp,
                    hlpr_time_params['filename']: hlpr_time,
                    dt_params['filename']: dt,
                    cls_sales_params['filename']+ '-' + cls_sales_params['sheet_name']: service_cls_db,
                    location_params['filename']: location,
                    sales_loc_params['filename']: sales_loc_map,
                    }
    
    return lookup_files