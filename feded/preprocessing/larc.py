import pandas as pd

SID_COLNAME = "STDNT_ID"
TERM_COLNAME = "TERM_CD"
SNAPSHT_COLNAME = "SNPSHT_RPT_DT"

def make_prev_term_gpa_column(df):
    return df