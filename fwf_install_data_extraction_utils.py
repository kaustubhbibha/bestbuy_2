"""
In-Home Install Future Work Force - Service Line Orders Forecasting

This script conatins the functions to extract data for Service Line orders,
Sales & Forecast for In-Home Install Future Work Force (IHI-FWF)

"""

import pandas as pd
import os


def create_lookup_orderlines(conn, base_dir, lkp_dir, hlpr_time_params,location_params, svc_sku_params, dt_params):
    """
    Creates volatile tables for lookups used in creating service line orders table
    Input:
        conn: Terradata connection
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
        hlpr_time_params: Dict - helper override time standards parameters
        location_params: Dict - Location to Minor Market mapping parameters
        svc_sku_params: Dict - SKU to  RSS mapping parameters
        dt_params: Dict - drive time standards parameters
    Output:
        Creates volatile tables created in db
    """
    # Helper time volatile table creation
    
    
    hlpr_time = pd.read_excel(os.path.join(base_dir, lkp_dir, hlpr_time_params['filename']), sheet_name=hlpr_time_params['sheet_name'])
    hlpr_time = hlpr_time[hlpr_time_params['columns']].drop_duplicates()
    
    q = '''CREATE VOLATILE TABLE HLPR_TIME (
        SKU_ID INTEGER
        , HLPR_SVC_OVERRIDE INTEGER
    )
    PRIMARY INDEX(SKU_ID) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    for idx, row in hlpr_time.iterrows():
        q = '''INSERT INTO HLPR_TIME (%d, %d)'''%(row.parent_sku_id,row.tm_dur_qty)
        print (q)
        conn.execute(q)
    
    q = '''
    SELECT * FROM prodbbysvcadhocdb.svc_fms_zip_mnr
    '''
    
    location = pd.read_sql_query(q, conn)
    location = location[location.Dept != "HD"]
    location = location[~location.Mnr_Mkt.isna()]
    location = location[['Zip_Cde', 'Dept', 'Mnr_Mkt']]
    location.Zip_Cde = location.Zip_Cde.astype(int)

    q = '''CREATE VOLATILE TABLE LOC_N (
        ZIPCODE INTEGER
        ,DEPT  VARCHAR(300)
        , MNR_MKT_NM VARCHAR(300)
    )
    PRIMARY INDEX(ZIPCODE) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    for idx, row in location.iterrows():
        q = '''INSERT INTO LOC_N (%d, '%s', '%s')'''%(row.Zip_Cde, row.Dept, row.Mnr_Mkt)
        print (q)
        conn.execute(q)
    
    # Location volatile table creation
    location.to_excel(os.path.join(base_dir,lkp_dir,location_params['filename']), sheet_name = location_params['sheet_name'], index = False)

    
    # SKU_ID to Skillset volatile table creation
    service_sku = pd.read_excel(os.path.join(base_dir, lkp_dir, svc_sku_params['filename']) , sheet_name = svc_sku_params['sheet_name'])
    service_sku = service_sku[svc_sku_params['columns']].copy()
    q = '''CREATE VOLATILE TABLE service_sku (
        SKU_ID INTEGER      
        , RSS VARCHAR(200)
        
    )
    PRIMARY INDEX(SKU_ID) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    for idx, row in service_sku.iterrows():
        q = '''INSERT INTO service_sku (%d,'%s')'''%(row["SKU_ID"],row["RSS"])
        print (q)
        conn.execute(q)
    
    # Drive time volatile table creation
    
    q = """
        SELECT MM, SUM(TOTAL_TIME) TT, SUM(TOTAL_STOPS) TS, COALESCE(SUM(TOTAL_TIME)/SUM(TOTAL_STOPS), 30) DRIVE_TIME FROM
        (SELECT DISTINCT LOC_ID, MNR_MKT_NM MM FROM PRODBBYSVCADHOCDB.DM_LOC_CRNT
        WHERE MNR_MKT_NM IS NOT NULL) A
        LEFT JOIN
        (SELECT work_crew_loc_id AS LOC_ID,
          SUM(plan_drive_dur_mm_qty) AS TOTAL_TIME,
          SUM(plan_cust_stops_cnt_qty) AS TOTAL_STOPS
           FROM  PRODBBYSVCADHOCDB.svc_logistics_route_mart
           WHERE  to_number(plan_drive_dur_mm_qty) IS NOT NULL
           AND to_number(plan_cust_stops_cnt_qty) IS NOT NULL
           AND schd_sys_nm = 'Click' 
           GROUP BY 1
           ) B
        ON A.LOC_ID = B.LOC_ID
        GROUP BY MM
    """
    
    dt = pd.read_sql_query(q,conn)
    
    q = '''CREATE VOLATILE TABLE drive_ts (
        MM VARCHAR(200)
        , DRIVE_TIME_NEW FLOAT
    )
    PRIMARY INDEX(MM) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    for idx, row in dt.iterrows():
        q = '''INSERT INTO drive_ts ('%s',%f)'''%(row["MM"],row["DRIVE_TIME"])
        print (q)
        conn.execute(q)
        
    dt.to_csv(os.path.join(base_dir, lkp_dir, dt_params['filename']), index = False)
    
    


def create_lookup_sales(conn, base_dir, lkp_dir, svc_cls_lkp, sales_loc, forecast = False):
    """
    Create volatile tables for lookups used in creating sales & forecast table
    Input:
        conn: Terradata connection
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
        svc_cls_lkp: Dict - Class subclass to FPG mapping
        forecast: Flag to pull forecast data
    Output:
        Volatile tables created in db
    """

    # Class Subclass to Skillset volatile table creation
    service_cls_db = pd.read_excel(os.path.join(base_dir, lkp_dir, svc_cls_lkp['filename']), sheet_name=svc_cls_lkp['sheet_name'])
    service_cls_db = service_cls_db[svc_cls_lkp['columns']].drop_duplicates()

    q = '''CREATE VOLATILE TABLE MAPP_PH_SVC_CLS (
        CLASS_ID INTEGER
        , SCLS_ID INTEGER
        , RSS VARCHAR(300)
    )
    PRIMARY INDEX(CLASS_ID, SCLS_ID) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    for idx, row in service_cls_db.iterrows():
        q = '''INSERT INTO MAPP_PH_SVC_CLS (%d, %d, '%s')'''%(row.CLASS_ID, row.SCLS_ID, row.RSS)
        print (q)
        conn.execute(q)
        
    if forecast:
        return 0

    # Location volatile table creation
    loc = """
        SELECT 
        DISTINCT 
        loc_id
        , mnr_mkt_nm AS Install_Repair_MM
        , hd_mnr_mkt_nm AS Delivery_MM 
        FROM prodbbysvcadhocdb.dm_loc_crnt 
        WHERE COALESCE(mnr_mkt_nm,hd_mnr_mkt_nm) IS NOT NULL
        ORDER BY 1"""
    location_all_comb = pd.read_sql(loc, conn)
    location_all_comb.rename({"Install_Repair_MM": "MNR_MKT_NM"}, axis=1, inplace=True)
    location_all_comb["MM_CNT"] = 1
    
    q = '''CREATE VOLATILE TABLE LOC_ALL (
        LOC_ID INTEGER,
        MM_CNT INTEGER,
        MNR_MKT_NM VARCHAR(200)
    )
    PRIMARY INDEX(LOC_ID) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    for idx, row in location_all_comb.iterrows():
        q = '''INSERT INTO LOC_ALL (%d, %d, '%s')'''%(row.LOC_ID, row.MM_CNT, row.MNR_MKT_NM)
        print (q)
        conn.execute(q)
    
    location_all_comb.to_csv(os.path.join(base_dir, lkp_dir, sales_loc['filename']), index = False)
    

def get_lineorders(start, end, conn, hlpr_time_params, location_params, svc_sku_params, dt_params, base_dir, lkp_dir):
    """
    Creates service line orders volatile table
    Input:
        start: Start Fiscal week
        end: End Fiscal week
        conn: Terradata connection
        sku_mapp_params: Dict - SKU to FPG mapping parameters
        svc_cls_db_lkp: Dict - Class Subclass to FPG mapping parameters
        hlpr_time_params: Dict - helper override time standards parameters
        location_params: Dict - Location to Minor Market mapping parameters
        svc_sku_params: Dict - SKU to CSS, RSS mapping parameters
        dt_params: Dict - drive time standards parameters
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
    Output:
        del_orders: Pandas dataframe - Line orders data
        conn: Session connection
    """
                             
    create_lookup_orderlines(conn, base_dir, lkp_dir, hlpr_time_params, location_params, svc_sku_params, dt_params)
    
    q = '''CREATE VOLATILE TABLE VOL_FCT_IH_FLFLMNT1 AS (
    SELECT DISTINCT ord_crt_dt AS REPORTING_DATE
      , FISC_WK_OF_MTH_ID
      , FLMT.SKU_ID
      , ORD_LN_ID
      , APPT_ID
      , COALESCE(LD_SVC_MNTS,0) LD_SVC_MNTS
      , COALESCE(HLPR_SVC_MNTS,0) HLPR_SVC_MNTS
      , HLPR_SVC_OVERRIDE
      , CASE WHEN HLPR_SVC_OVERRIDE IS NOT NULL 
             THEN HLPR_SVC_OVERRIDE ELSE HLPR_SVC_MNTS END AS HLPR_SVC_MNTS_NEW
      , (COALESCE(LD_SVC_MNTS,0) + COALESCE(HLPR_SVC_MNTS,0)) AS TOTAL_SVC_MNTS
      , CAST(TOTAL_SVC_MNTS AS float) / CAST(60 AS float) AS TOTAL_SVC_HR
      , FLMT.rprtng_loc_id  AS ORD_CUST_LOC_KEY
      , LOC.ZIPCODE
      , rprtng_loc_id RPRT_LOC
      , LOC.MNR_MKT_NM AS MM
      , LOC.DEPT
      , CLS.CLASS_ID
      , CLS.SCLS_ID
      , DRIVE_TIME_NEW
      , CASE WHEN CLS.CLASS_ID in (134, 265, 503) THEN 'Home Theater'
              WHEN CLS.CLASS_ID in (190, 404, 596) THEN 'PC'
              WHEN CLS.CLASS_ID in (154) THEN 'Repair'
              WHEN CLS.CLASS_ID in (133) THEN 'Appliance'
              WHEN CLS.CLASS_ID in (109, 299, 628, 631) THEN 'Delivery'
              WHEN CLS.CLASS_ID in (678) THEN 'Connected Home'
              WHEN CLS.CLASS_ID in (654, 655) THEN 'Health'
              WHEN CLS.CLASS_ID in (718) THEN 'Fitness'
              ELSE 'UNMAPPED'
              END AS WORK_TYPE
      , APPT_CST_ALCN_GRP
      , CASE 
        WHEN TRIM(APPT_CST_ALCN_GRP) IN 
        (
         'RPR-GD',  'RPR-ED',  'RPR-VAC',  'RPR-IMW',  'RPR-FRZ',
         'RPR-REF',  'RPR-WAS',  'RPR-TC',  'RPR-VCT',  'RPR-NC',
         'RPR-TV1',  'RPR-AC', 'RPR-TV3',  'RPR-DW',  'Fit',
         'RPR-PC',  'RPR-UPD',  'RPR-TV2',  'RPR-GR',  'RPR-ER'
         ) THEN 'Repair'
        WHEN TRIM(APPT_CST_ALCN_GRP) IN 
         (
         'ABHD', 'HTBHD'
        ) THEN 'Delivery'
        WHEN TRIM(APPT_CST_ALCN_GRP) IN 
        ('ADI', 'CnctHm', 'Health',  'HTDI', 'MDCSI', 'PCIR', 'PCTS', 'PC247', 'Fit', 'FITDI'
         ) THEN 'Install'
        WHEN TRIM(APPT_CST_ALCN_GRP) IN 
        ('Other') THEN 'Other' END AS APPT_TYPE
        , CASE WHEN WORK_TYPE in ('Delivery') THEN NULL 
               WHEN RSS = 'nan' THEN NULL
               WHEN APPT_TYPE = 'Install'
            AND WORK_TYPE = 'Repair' THEN NULL
               WHEN APPT_TYPE = 'Repair'
            AND WORK_TYPE LIKE ANY ('%Appliance%', '%Home Theater%', '%PC%', '%Connected Home%', '%Health%', '%Fitness%') THEN NULL ELSE RSS END AS RSS
   
    FROM PRODBBYSVCADHOCDB.FCT_IH_FLFLMNT FLMT
    JOIN PRODBBYSVCADHOCVWS.DM_FLFLMNT_FLG F2
    ON F2.FLFLMNT_FLG_KEY = FLMT.FLFLMNT_FLG_KEY
    AND F2.LN_CMPLT_FLG = 1
    AND FLFLMNT_STAT_KEY NOT IN (SELECT FLFLMNT_STAT_KEY FROM PRODBBYSVCADHOCVWS.DM_FLFLMNT_STAT WHERE "LINE TYPE" = 'DETACHMENT')
    JOIN PRODBBYSVCADHOCVWS.DM_WRK_TYP T
    ON T.WRK_TYP_KEY = FLMT.WRK_TYP_KEY
    INNER JOIN 
        LOC_N AS LOC
    ON 
        FLMT.SVC_ADDR_ZIP_CDE = LOC.ZIPCODE
    LEFT JOIN PRODBBYVWS.TBEND_BU_FISC_DT DT
    ON FLMT.ord_crt_dt = DT.CALNDR_DT
    LEFT JOIN
        (SELECT  DISTINCT SKU_ID, CLASS_ID, SCLS_ID FROM PRODBBYRPTVWS.ALL_CHNL_MERCH_SKU_CURR) CLS
    ON
        FLMT.SKU_ID = CLS.SKU_ID
    LEFT JOIN service_sku SVC
    ON FLMT.SKU_ID = SVC.SKU_ID
    LEFT JOIN
    HLPR_TIME H
    ON H.SKU_ID = FLMT.SKU_ID
    LEFT JOIN
    drive_ts DRIVE
    ON LOC.MNR_MKT_NM = DRIVE.MM 
    WHERE FISC_WK_OF_MTH_ID >= start
    AND FISC_WK_OF_MTH_ID <= end
    AND ln_cmplt_flg = 1
    )
    WITH DATA PRIMARY INDEX(ord_ln_id) ON COMMIT PRESERVE ROWS'''.replace('start',  ' '+str(start) + ' ').replace('end', ' '+str(end) + ' ')
    conn.execute(q)
    

    q = '''CREATE VOLATILE TABLE VOL_FCT_IH_FLFLMNT_APPT AS (
    SELECT DISTINCT APPT_ID,
        COUNT(RSS) RSS_APPT,
        SUM(CASE WHEN RSS IS NOT NULL AND LD_SVC_MNTS > 0 THEN 1 ELSE 0 END) RSS_LD_APPT,
        SUM(CASE WHEN RSS IS NOT NULL AND HLPR_SVC_MNTS > 0 THEN 1 ELSE 0 END) RSS_HP_APPT,
        SUM(CASE WHEN RSS IS NOT NULL AND HLPR_SVC_MNTS_NEW > 0 THEN 1 ELSE 0 END) RSS_HPN_APPT,
        SUM(CASE WHEN RSS IS NULL THEN LD_SVC_MNTS ELSE 0 END) AS RSS_LD_MNTS,
        SUM(CASE WHEN RSS IS NULL THEN HLPR_SVC_MNTS ELSE 0 END) AS RSS_HP_MNTS,
        SUM(CASE WHEN RSS IS NULL THEN HLPR_SVC_MNTS_NEW ELSE 0 END) AS RSS_HPN_MNTS,
        SUM(CASE WHEN WORK_TYPE LIKE ANY ('%Appliance%', '%Home Theater%', '%PC%', '%Connected Home%', '%Health%', '%Fitness%') THEN TOTAL_SVC_MNTS ELSE 0 END) AS INS_MNTS, 
        SUM(CASE WHEN WORK_TYPE LIKE ANY ('%Delivery%') THEN TOTAL_SVC_MNTS ELSE 0 END) AS DEL_MNTS
    FROM VOL_FCT_IH_FLFLMNT1
    GROUP BY APPT_ID
    WHERE FISC_WK_OF_MTH_ID >= start
    AND FISC_WK_OF_MTH_ID <= end
    )
    WITH DATA PRIMARY INDEX(APPT_ID) ON COMMIT PRESERVE ROWS'''.replace('start', ' '+ str(start) + ' ').replace('end', ' ' + str(end) + ' ')
    conn.execute(q)


    q = '''CREATE VOLATILE TABLE VOL_FCT_IH_FLFLMNT_ALL AS (
    SELECT A.*, 
        B.RSS_LD_APPT,
        B.RSS_HP_APPT,
        B.RSS_HPN_APPT, INS_MNTS, DEL_MNTS,
        CAST(DRIVE_TIME_NEW AS float) / NULLIF(CAST(CASE WHEN A.APPT_TYPE NOT IN ('Delivery') AND RSS IS NOT NULL THEN RSS_APPT ELSE NULL END AS float),0) AS RSS_LD_STOPS,
        CASE WHEN HLPR_SVC_MNTS > 0 
            THEN CAST(DRIVE_TIME_NEW AS float) / NULLIF(CAST(CASE WHEN A.APPT_TYPE NOT IN ('Delivery') AND RSS IS NOT NULL THEN RSS_HP_APPT ELSE NULL END AS float),0)
             ELSE 0 END AS RSS_HP_STOPS,
        CASE WHEN HLPR_SVC_MNTS_NEW > 0 
            THEN CAST(DRIVE_TIME_NEW AS float) / NULLIF(CAST(CASE WHEN A.APPT_TYPE NOT IN ('Delivery') AND RSS IS NOT NULL THEN RSS_HPN_APPT ELSE NULL END AS float),0)
             ELSE 0 END AS RSS_HPN_STOPS,
        CAST(RSS_LD_MNTS AS float) / NULLIF(CAST(CASE WHEN A.APPT_TYPE NOT IN ('Delivery') AND 
            RSS IS NOT NULL AND LD_SVC_MNTS > 0 
            THEN RSS_LD_APPT ELSE NULL END AS float),0) AS RSS_LD_UNCLAIMED,
        CAST(RSS_HP_MNTS AS float) / NULLIF(CAST(CASE WHEN A.APPT_TYPE NOT IN ('Delivery') AND 
            RSS IS NOT NULL AND HLPR_SVC_MNTS > 0 
            THEN RSS_HP_APPT ELSE NULL END AS float),0) AS RSS_HP_UNCLAIMED,
        CAST(RSS_HPN_MNTS AS float) / NULLIF(CAST(CASE WHEN A.APPT_TYPE NOT IN ('Delivery') AND
            RSS IS NOT NULL AND HLPR_SVC_MNTS_NEW > 0 
            THEN RSS_HPN_APPT ELSE NULL END AS float),0) AS RSS_HPN_UNCLAIMED
    FROM VOL_FCT_IH_FLFLMNT1 A
    LEFT JOIN
    VOL_FCT_IH_FLFLMNT_APPT B
    ON A.APPT_ID = B.APPT_ID
    )
    WITH DATA PRIMARY INDEX(ORD_LN_ID) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    

    q = '''
    SELECT FISC_WK_OF_MTH_ID
        , RSS
        , MM
        , RSS||'_'||MM as RSS_MM
        , COUNT(DISTINCT ORD_LN_ID) LINE_ORDERS
        , COUNT(DISTINCT APPT_ID) APPT_ID
        , SUM(LD_SVC_MNTS) LEAD_MNTS
        , SUM(HLPR_SVC_MNTS) HELPER_MNTS
        , SUM(HLPR_SVC_MNTS_NEW) HELPER_OVR
        , SUM(RSS_LD_UNCLAIMED) LD_UNCLAIMED
        , SUM(RSS_HP_UNCLAIMED) HP_UNCLAIMED
        , SUM(RSS_HPN_UNCLAIMED) HPN_UNCLAIMED
        , SUM(RSS_LD_STOPS) LD_DRIVE
        , SUM(RSS_HP_STOPS) HP_DRIVE
        , SUM(RSS_HPN_STOPS) HPN_DRIVE
        , SUM(CASE WHEN LD_SVC_MNTS = 0 
                    THEN 1 
                    ELSE 0 END) LEAD_ZERO
        , SUM(CASE WHEN HLPR_SVC_MNTS = 0 
                    THEN 1 
                    ELSE 0 END) HELPER_ZERO
        , SUM(CASE WHEN HLPR_SVC_MNTS_NEW = 0 
                    THEN 1 
                    ELSE 0 END) HELPERN_ZERO
        , SUM(CASE WHEN LD_SVC_MNTS > 0 
                    THEN 1 
                    ELSE 0 END) LEAD_ALL
        , SUM(CASE WHEN HLPR_SVC_MNTS > 0 
                    THEN 1 
                    ELSE 0 END) HELPER_ALL
        , SUM(CASE WHEN HLPR_SVC_MNTS_NEW > 0 
                    THEN 1 
                    ELSE 0 END) HELPERN_ALL
           FROM VOL_FCT_IH_FLFLMNT_ALL
    GROUP BY FISC_WK_OF_MTH_ID
        , RSS
        , MM
        , RSS||'_'||MM
    WHERE APPT_TYPE = 'Install'
        AND RSS IS NOT NULL   
    '''

    inst_orders = pd.read_sql_query(q, conn)
    conn.execute("""DROP TABLE HLPR_TIME""")
    conn.execute("""DROP TABLE LOC_N""")
    conn.execute("""DROP TABLE service_sku""")
    conn.execute("""DROP TABLE drive_ts""")
    conn.execute("""DROP TABLE VOL_FCT_IH_FLFLMNT1""")
    conn.execute("""DROP TABLE VOL_FCT_IH_FLFLMNT_APPT""")
    conn.execute("""DROP TABLE VOL_FCT_IH_FLFLMNT_ALL""")
   
    return inst_orders


def get_sales(start, end, orderlines_data, conn, svc_cls_lkp, sales_loc, base_dir, lkp_dir):
    
    """
    Create sales volatile table
    Input:
        start: Start Fiscal week
        end: End Fiscal week
        orderlines_data: Order line data
        conn: Terradata connection
        svc_cls_lkp: Dict - Class subclass to FPG mapping
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
    Output:
         sales: Pandas dataframe- Sales data for Minor MArkets other than HD
         sales_hd: PAndas dataframe - Sales data for HD Minor Mararkets
    """
    
    create_lookup_sales(conn, base_dir, lkp_dir, svc_cls_lkp, sales_loc, forecast = False)

    df_mm = orderlines_data.copy()
    df_mm = df_mm[df_mm.FISC_WK_OF_MTH_ID >= start]
    df_mm = df_mm[df_mm.FISC_WK_OF_MTH_ID <= end]
    df_mm = df_mm.groupby("MM").sum()["LINE_ORDERS"].reset_index()
    df_mm["MM_PROP"] = df_mm["LINE_ORDERS"]/df_mm["LINE_ORDERS"].sum()
    df_mm["JOIN_KEY"] = 1
    df_mm.columns = ["MNR_MKT_NM", "LINE_ORDERS", "MM_PROP", "JOIN_KEY"]
    
    q = '''CREATE VOLATILE TABLE MM_ALL (
        JOIN_KEY INTEGER
        , MNR_MKT_NM VARCHAR(300)
        , MM_PROP FLOAT
    )
    PRIMARY INDEX(MNR_MKT_NM) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    for idx, row in df_mm.iterrows():
        q = '''INSERT INTO MM_ALL (%d, '%s', %f)'''%(row.JOIN_KEY,row.MNR_MKT_NM, row.MM_PROP)
        print (q)
        conn.execute(q)
    
    q = '''CREATE VOLATILE TABLE SALES_NEW_ALL AS (
    SELECT SLS_BSNS_DT, 
    CASE WHEN LOC_ID IS NULL THEN 99999999 ELSE LOC_NEW END AS LOC_NEW, 
    RSS_ACT, 
    SUM(SLS_LN_QTY) SALES 
    FROM
    (
        SELECT 
            SLS_BSNS_DT, A.SKU_ID, 
            CASE WHEN LOC.LOC_ID IS NULL THEN ALL_CHNL_RPT_LOC_ID 
                 ELSE LOC.LOC_ID END AS LOC_NEW,
            ALL_CHNL_RPT_LOC_ID, FISC_WK_OF_MTH_ID, FISC_MTH_NBR, FISC_YR_ID, SLS_LN_QTY
            FROM PRODBBYVWS.TBEND_SA_ITEM A
            INNER JOIN PRODBBYMEADHOCVWS.TBEND_BU_FISC_DT B
            ON SLS_BSNS_DT = CALNDR_DT
            LEFT JOIN 
            (SELECT DISTINCT LOC_ID FROM LOC_ALL) AS LOC
                    ON A.LOC_ID = LOC.LOC_ID
        WHERE
        FISC_WK_OF_MTH_ID <= end
        AND
        FISC_WK_OF_MTH_ID >= start
    ) A
    JOIN 
    (
            SELECT  DISTINCT 
                    SKU_ID, SKU_DESC, CLASS_ID, SCLS_ID, CLASS_NM, SCLS_NM 
            FROM PRODBBYRPTVWS.ALL_CHNL_MERCH_SKU_CURR
    ) CLS
    ON A.SKU_ID = CLS.SKU_ID
    LEFT JOIN
    (SELECT DISTINCT LOC_ID FROM LOC_ALL) AS LOC
    ON LOC.LOC_ID = A.LOC_NEW
    INNER JOIN
    (
    SELECT DISTINCT CLASS_ID, SCLS_ID,
    RSS AS RSS_ACT FROM MAPP_PH_SVC_CLS
    ) M
    ON CLS.CLASS_ID = M.CLASS_ID
    AND CLS.SCLS_ID = M.SCLS_ID
    GROUP BY 
    SLS_BSNS_DT,
     CASE WHEN LOC_ID IS NULL THEN 99999999 ELSE LOC_NEW END
    , RSS_ACT
    )
    WITH DATA PRIMARY INDEX(SLS_BSNS_DT, LOC_NEW, RSS_ACT) ON COMMIT PRESERVE ROWS'''.replace('start', str(start)).replace('end', str(end))
    conn.execute(q)


    q = '''CREATE VOLATILE TABLE SALES_ALL_LOC_MM AS (
    SELECT SLS_BSNS_DT
    , RSS_ACT
    , LOC.MNR_MKT_NM, 
    SUM(SALES/CAST((CASE WHEN LOC.MM_CNT IS NULL THEN 1 ELSE LOC.MM_CNT END) AS FLOAT)) SALES_ACT FROM 
    SALES_NEW_ALL A
    INNER JOIN 
    (
    SELECT DISTINCT LOC_ID, MNR_MKT_NM, MM_CNT FROM LOC_ALL
    WHERE MNR_MKT_NM NOT LIKE '%HD%' AND
    MNR_MKT_NM NOT LIKE '%UNMAPPED%'
    ) LOC
    ON A.LOC_NEW = LOC.LOC_ID
    GROUP BY SLS_BSNS_DT
    , RSS_ACT
    , LOC.MNR_MKT_NM
    
    UNION ALL
    
    SELECT SLS_BSNS_DT
    , RSS_ACT
    , MNR_MKT_NM, SUM(SALES*CAST (MALL.MM_PROP AS FLOAT)) SALES_ACT
           FROM 
    
    (SELECT A.*,
    1 AS JOIN_KEY FROM SALES_NEW_ALL A
    LEFT JOIN
    (
    SELECT DISTINCT LOC_ID FROM LOC_ALL
    WHERE MNR_MKT_NM NOT LIKE '%HD%' AND
    MNR_MKT_NM NOT LIKE '%UNMAPPED%'
    ) LOC
    ON A.LOC_NEW = LOC.LOC_ID
    WHERE LOC_ID IS NULL
    ) A
    
    LEFT JOIN MM_ALL MALL
    ON A.JOIN_KEY = MALL.JOIN_KEY
    GROUP BY SLS_BSNS_DT
    , RSS_ACT
    , MNR_MKT_NM
    )
    WITH DATA PRIMARY INDEX(SLS_BSNS_DT, MNR_MKT_NM, RSS_ACT) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    
    
    q = '''CREATE VOLATILE TABLE SALES_ALL_MM AS (
    SELECT FISC_WK_OF_MTH_ID
    , RSS_ACT||'_'||MNR_MKT_NM RSS_MM, RSS_ACT RSS
    , MNR_MKT_NM MM, SUM(COALESCE(SALES_ACT,0)) SALES
    FROM SALES_ALL_LOC_MM A
    INNER JOIN PRODBBYMEADHOCVWS.TBEND_BU_FISC_DT B
            ON SLS_BSNS_DT = CALNDR_DT
    GROUP BY FISC_WK_OF_MTH_ID
    , RSS_ACT||'_'||MNR_MKT_NM, RSS_ACT
    , MNR_MKT_NM
    )
    WITH DATA PRIMARY INDEX(FISC_WK_OF_MTH_ID, RSS, MM) ON COMMIT PRESERVE ROWS'''
    conn.execute(q)
    
    q = '''
    SELECT * FROM SALES_ALL_MM
    '''
    
    sales = pd.read_sql_query(q, conn)
    
    conn.execute("""DROP TABLE MM_ALL""")
    conn.execute("""DROP TABLE LOC_ALL""")
    conn.execute("""DROP TABLE MAPP_PH_SVC_CLS""")
    conn.execute("""DROP TABLE SALES_NEW_ALL""")
    conn.execute("""DROP TABLE SALES_ALL_LOC_MM""")
    conn.execute("""DROP TABLE SALES_ALL_MM""")
    
    return sales


def get_forecasts(start, end, conn, svc_cls_db_lkp, sales_loc, base_dir, lkp_dir):    
    """
    Create forecast volatile table
    Input:
        start: Start Fiscal week
        end: End Fiscal week
        conn: Terradata connection
        svc_cls_db_lkp: Dict - Class subclass to FPG mapping
        base_dir: Model framework parent directory
        lkp_dir: Model lookup folder directory
    Output:
        forecast: Pandas dataframe - Forecast data
    """
        
    create_lookup_sales(conn, base_dir, lkp_dir, svc_cls_db_lkp, sales_loc, forecast = True)
    
    q = '''CREATE VOLATILE TABLE VOL_FORECAST_NEW_ALL_WP AS (
    SELECT FISC_WK_OF_MTH_ID, CLASS_ID, SCLS_ID
        , RSS_ACT RSS
        , SUM( UNIT_QTY) FORECAST_WP FROM
    	(SELECT FISC_WK_OF_MTH_ID
         ,FISC_YR_ID
        , CATLOCPLN_SCLS_ID
        , CATLOCPLN_ROLE_CDE
        , CATLOCPLN_VER_CDE
        , CATLOCPLN_VER_CNT
        , CATLOCPLN_4_DGT_CLASS_NBR AS CLASS_ID
        , CATLOCPLN_4_DGT_SCLS_NBR AS SCLS_ID
    	,COALESCE(BRKM_UNIT_QTY,0)+COALESCE(ECM_UNIT_QTY,0) AS UNIT_QTY
    	,REC_CRT_TS
        , RSS_ACT
    FROM PRODBBYVWS.TBEN_FI_CATPL_SC_FW A
    INNER JOIN
    (
    SELECT DISTINCT CLASS_ID, SCLS_ID, RSS AS RSS_ACT FROM MAPP_PH_SVC_CLS
    ) M
    ON A.CATLOCPLN_4_DGT_CLASS_NBR = M.CLASS_ID
    AND A.CATLOCPLN_4_DGT_SCLS_NBR = M.SCLS_ID
    ) AB
    GROUP BY FISC_WK_OF_MTH_ID, CLASS_ID, SCLS_ID
    , RSS_ACT
    WHERE 
        FISC_WK_OF_MTH_ID >= start
        AND FISC_WK_OF_MTH_ID <= end
        AND CATLOCPLN_VER_CDE='WP'
    )
    WITH DATA PRIMARY INDEX(FISC_WK_OF_MTH_ID, CLASS_ID, SCLS_ID, RSS) ON COMMIT PRESERVE ROWS'''.replace('start', str(start)).replace('end', str(end))
    conn.execute(q)

    q = '''
    SELECT * FROM VOL_FORECAST_NEW_ALL_WP
    '''
    forecast = pd.read_sql_query(q, conn)
    
    conn.execute("""DROP TABLE MAPP_PH_SVC_CLS""")
    conn.execute("""DROP TABLE VOL_FORECAST_NEW_ALL_WP""")
    
    return forecast


def fiscal_clndr_info(conn):
    """
    Extracts fiscal calendar data
    """
    q = '''
    SELECT DISTINCT CALNDR_DT, FISC_EOW_DT, FISC_YR_NBR, FISC_MTH_NBR, 
    FISC_WK_OF_MTH_ID, FISC_WK_OF_YR_NBR, FISC_QTR_NBR, LY_FISC_WK_OF_MTH_ID 
    FROM PRODBBYVWS.TBEND_BU_FISC_DT
    '''
    fisc_calender_all = pd.read_sql_query(q, conn)
    return fisc_calender_all


def state_tz_info(conn):
    """
    Extracts Minor Market to State, Time zone mapping
    """
    q = '''
    SELECT DISTINCT MNR_MKT_NM MM, TM_ZONE_ID TIME_ZONE FROM PRODBBYSVCADHOCDB.DM_LOC_CRNT
    WHERE MNR_MKT_NM IS NOT NULL
    '''
    loc_state_grp = pd.read_sql_query(q, conn)
    loc_state_grp = loc_state_grp.drop_duplicates("MM")
    loc_state_grp["STATE"] = loc_state_grp["MM"].str.split(" - ", expand = True)[0]
    return loc_state_grp