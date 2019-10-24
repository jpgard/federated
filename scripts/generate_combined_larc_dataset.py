"""
Read the individual LARC tables and join them into a single student-term-course dataset.

Example usage:
python scripts/generate_combined_larc_dataset.py \
    --prsn_identifying_info_fp data/larc-mock/PRSN_IDNTFYNG_INFO.csv \
    --stdnt_info_fp data/larc-mock/STDNT_INFO.csv \
    --stdnt_term_class_info_fp data/larc-mock/STDNT_TERM_CLASS_INFO.csv \
    --stdnt_term_info_fp data/larc-mock/STDNT_TERM_INFO.csv \
    --stdnt_term_trnsfr_info_fp data/larc-mock/STDNT_TERM_TRNSFR_INFO.csv \
    --out_fp data/larc-mock/larc-mock.csv
"""

import pandas as pd
import argparse

def main(prsn_identifying_info_fp,
         stdnt_info_fp,
         stdnt_term_class_info_fp,
         stdnt_term_info_fp,
         stdnt_term_trnsfr_info_fp,
         out_fp):
    # common arguments to use when reading the csv files
    read_csv_args = {
        "na_values": ('', ' '),
        "keep_default_na": True,
        "dtype": "object",
    }
    # for this table, we rename column PRSN_ID before setting the index
    prsn_identifying_info = pd.read_csv(prsn_identifying_info_fp,
                                        **read_csv_args).rename(columns={"PRSN_ID":
                                                                         "STDNT_ID"})\
                                        .astype({"STDNT_ID": "int64"})\
                                        .set_index(["SNPSHT_RPT_DT", "STDNT_ID"])
    stdnt_info = pd.read_csv(stdnt_info_fp,
                             index_col=("SNPSHT_RPT_DT", "STDNT_ID"),
                             **read_csv_args)
    stdnt_term_class_info = pd.read_csv(stdnt_term_class_info_fp,
                                        index_col=("SNPSHT_RPT_DT", "TERM_CD",
                                                   "STDNT_ID", "CLASS_NBR"),
                                        **read_csv_args)
    # drop the duplicated column
    stdnt_term_info = pd.read_csv(stdnt_term_info_fp,
                                  index_col=("SNPSHT_RPT_DT", "TERM_CD", "STDNT_ID"),
                                  **read_csv_args).drop(columns="TERM_SHORT_DES")
    # stdnt_term_trnsfr_info = pd.read_csv(stdnt_term_trnsfr_info_fp, **read_csv_args)

    # we want a student-term-class level dataset; join the other data onto this df
    larc = stdnt_term_class_info\
        .join(stdnt_term_info, how="left", on=("SNPSHT_RPT_DT", "STDNT_ID", "TERM_CD"))\
        .join(stdnt_info, how="left", on=("SNPSHT_RPT_DT", "STDNT_ID"))\
        .join(prsn_identifying_info, how="left", on=("SNPSHT_RPT_DT", "STDNT_ID"))
    # ensure no data is lost or duplicated in joins
    assert len(larc) == len(stdnt_term_class_info)
    larc.reset_index().to_csv(out_fp, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prsn_identifying_info_fp", help="person identifying info csv")
    parser.add_argument("--stdnt_info_fp", help="student info csv")
    parser.add_argument("--stdnt_term_class_info_fp", help="student term class info csv")
    parser.add_argument("--stdnt_term_info_fp", help="student term info csv")
    parser.add_argument("--stdnt_term_trnsfr_info_fp", help="student term transfer csv",
                        required=False) # currently unused
    parser.add_argument("--out_fp", help="path to write csv to")
    args = parser.parse_args()
    main(**vars(args))