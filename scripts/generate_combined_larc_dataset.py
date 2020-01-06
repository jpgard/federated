"""
Read the individual LARC tables and join them into a single student-term-course dataset.

Example usage:

# full LARC dataset (excluding transfer table)
python scripts/generate_combined_larc_dataset.py \
    --stdnt_info_fp data/LARC/LARC_20190924_STDNT_INFO.csv.bz2 \
    --stdnt_term_class_info_fp data/LARC/LARC_20190924_STDNT_TERM_CLASS_INFO.csv.bz2 \
    --stdnt_term_info_fp data/LARC/LARC_20190924_STDNT_TERM_INFO.csv.bz2 \
    --outdir data/larc-split/ \
    --generate_ttv
"""

import argparse
import os
import os.path as osp
import bz2

import pandas as pd

TERM_CDS = {
    # the term codes to use for the validation dataset; should be the second-to-last full
    # academic year in the dataset
    "val": [
        "2160",  # FA 2017
        "2170"  # WN 2018
    ],
    # the term codes to use for the test dataset; should be the last full academic year
    # in the data
    "test": [
        "2210",  # Fall 2018
        "2220"  # WN 2019
    ]
}


def read_csv_from_bz2(fp, index_cols, **read_csv_args):
    print("[INFO] reading {}".format(fp))
    with bz2.open(fp) as file:
        df = pd.read_csv(file, index_col=index_cols, **read_csv_args)
    return df


def check_single_snapshot(df: pd.DataFrame):
    """Verify that there is only one database snapshot represented in the df.

    If this check fails, SNPSHT_RPT_DT should be added as an index for proper joining.
    """
    if "SNPSHT_RPT_DT" in df.columns:
        assert df["SNPSHT_RPT_DT"].nunique() == 1


def main(stdnt_info_fp,
         stdnt_term_class_info_fp,
         stdnt_term_info_fp,
         outdir,
         prsn_identifying_info_fp=None,
         stdnt_term_trnsfr_info_fp=None,
         generate_ttv=False
         ):
    """
    Generate a combined dataset by merging the CSV files along the correct indices.
    :param stdnt_info_fp:
    :param stdnt_term_class_info_fp:
    :param stdnt_term_info_fp:
    :param outdir:
    :param prsn_identifying_info_fp:
    :param stdnt_term_trnsfr_info_fp:
    :param generate_ttv: boolean indicator for whether to generate train-test-val split.
    :return:
    """

    # common arguments to use when reading the csv files
    read_csv_args = {
        "na_values": ('', ' '),
        "keep_default_na": True,
        "dtype": "object",
    }

    stdnt_info = read_csv_from_bz2(
        stdnt_info_fp, index_cols=("STDNT_ID",), **read_csv_args)
    stdnt_term_class_info = read_csv_from_bz2(
        stdnt_term_class_info_fp,
        index_cols=("TERM_CD", "STDNT_ID", "CLASS_NBR"),
        **read_csv_args)
    # drop the duplicated and redundant column when reading stdnt_term_info
    stdnt_term_info = read_csv_from_bz2(
        stdnt_term_info_fp,
        index_cols=("TERM_CD", "STDNT_ID"),
        **read_csv_args).drop(columns="TERM_SHORT_DES")

    for df in (stdnt_info, stdnt_term_class_info, stdnt_term_info):
        check_single_snapshot(df)
        df.drop(columns="SNPSHT_RPT_DT", inplace=True)

    # we want a student-term-class level dataset; join the other data onto this df
    larc = stdnt_term_class_info \
        .join(stdnt_term_info, how="left", on=("STDNT_ID", "TERM_CD")) \
        .join(stdnt_info, how="left", on=("STDNT_ID",))
    if prsn_identifying_info_fp:
        print("[INFO] reading {}".format(prsn_identifying_info_fp))
        # for this table, we rename column PRSN_ID before setting the index
        prsn_identifying_info = pd.read_csv(
            prsn_identifying_info_fp, index_col=["STDNT_ID", ],
            **read_csv_args).rename(
            columns={"PRSN_ID": "STDNT_ID"}) \
            .astype({"STDNT_ID": "int64"})

        larc = larc.join(prsn_identifying_info, how="left", on=("STDNT_ID",))
    if stdnt_term_trnsfr_info_fp:
        print("[ERROR] transfer data join not implemented")
        raise NotImplementedError
    larc.reset_index(inplace=True)
    import ipdb;
    ipdb.set_trace()
    # ensure no data is lost or duplicated in joins
    assert len(larc) == len(stdnt_term_class_info)
    if not generate_ttv:  # write the entire dataset
        out_fp = osp.join(outdir, "larc.csv")
        larc.to_csv(out_fp, index=False)
    else:  # generate train/test/val split by year

        # cutoff_term_cd is the first val/test term; no training data should be taken
        # from any temporal period after this term, even if it is not one of the
        # specified val/test terms.

        cutoff_term_cd = min([int(t) for key in TERM_CDS.values() for t in key])
        for mode in ("train", "test", "val"):  # create the directories
            mode_out_dir = osp.join(outdir, mode)
            mode_out_fp = osp.join(mode_out_dir, mode + ".csv")
            if not osp.exists(mode_out_dir):
                os.mkdir(mode_out_dir)
            if mode == "train":
                # filter the data using cutoff_term_cd
                larc[larc["TERM_CD"].astype(int) < cutoff_term_cd] \
                    .to_csv(mode_out_fp, index=False)
            else:  # mode is val or test; write that data
                larc[larc["TERM_CD"].isin(TERM_CDS[mode])] \
                    .to_csv(mode_out_fp, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prsn_identifying_info_fp", help="person identifying info "
                                                           "csv", required=False)
    parser.add_argument("--stdnt_info_fp", help="student info csv")
    parser.add_argument("--stdnt_term_class_info_fp", help="student term class info csv")
    parser.add_argument("--stdnt_term_info_fp", help="student term info csv")
    parser.add_argument("--stdnt_term_trnsfr_info_fp", help="student term transfer csv",
                        required=False)  # currently unused
    parser.add_argument("--outdir", help="directory to write dataset to")
    parser.add_argument("--generate_ttv", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
