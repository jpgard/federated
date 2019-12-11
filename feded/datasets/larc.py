import itertools
import re
from typing import List

import numpy as np
import tensorflow as tf

from feded.training import TrainingConfig
from feded.datasets import TabularDataset
from feded.preprocessing import read_csv, filter_df_by_values, \
    make_binary_indicator_column, generate_categorical_feature_dict, \
    minmax_scale_numeric_columns, generate_missing_value_indicator

# the prediction target

DEFAULT_LARC_TARGET_COLNAME = "CRSE_GRD_OFFCL_CD"

# the column to use for generating clients; this should be a column which universities
# might use to partition their data
DEFAULT_LARC_CLIENT_COLNAME = "SBJCT_CD"

DEFAULT_LARC_CATEGORICAL_FEATURES = [
    "ADMSSN_VTRN_IND",  # Admission Veteran Indicator (SENSITIVE ATTRIBUTE)
    "CLASS_GRDD_IND",  # Class Graded Indicator TODO(jpgard): filter for 1 ("yes") only
    "CLASS_HONORS_IND",  # Class Honors Indicator
    # "CRER_LVL_CD",  # Career Level Code
    "CRSE_CMPNT_CD",  # Course Component Code
    "CRSE_GRD_OFFCL_CD",  # Course Grade Official Code (The official grade that appears
    # on a student's transcript)
    "EST_GROSS_FAM_INC_CD",  # Estimated Gross Family Income Code (SENSITIVE ATTRIBUTE)
    "HS_CALC_IND",  # High School Calculus Indicator (self-reported)
    "HS_CHEM_LAB_IND",  # High School Chemistry Laboratory Indicator (self-reported)
    "HS_STATE_CD", # High School State Code (note: only exists for US/CA students)
    "GRD_BASIS_ENRL_CD",
    # "PRMRY_CRER_CD",  # Primary Career Code
    "PRNT_MAX_ED_LVL_CD",  # Parent Maximum Education Level Code (SENSITIVE ATTRIBUTE)
    # "RES_CD",  # Residency Code (SENSITIVE ATTRIBUTE)
    "SBJCT_CD",  # Subject Code TODO(jpgard): might move to embedding columns later
    "SNGL_PRNT_IND",  # Single Parent Indicator (SENSITIVE ATTRIBUTE)
    # "SPPLMNT_STUDY_IND",  # Supplemental Study Indicator
    "STDNT_ASIAN_IND",  # Student Asian Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_BLACK_IND",  # Student Black Indicator (SENSITIVE ATTRIBUTE)
    ## "STDNT_CTZN_STAT_CD",  # Student Citizenship Status Code (SENSITIVE ATTRIBUTE) (
    # (INVALID 'N' value)
    "STDNT_DMSTC_UNDREP_MNRTY_CD",
    # Student Domestic Underrepresented Minority (URM) Code (SENSITIVE ATTRIBUTE)
    "STDNT_ETHNC_GRP_CD",  # Student Ethnic Group Code (SENSITIVE ATTRIBUTE)
    "STDNT_GNDR_CD",  # Student Gender Code (SENSITIVE ATTRIBUTE)
    "STDNT_HSPNC_IND",  # Student Hispanic Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_HSPNC_LATINO_IND",  # Student Hispanic Latino Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_HWIAN_IND",  # Student Hawaiian Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_INTL_IND",  # Student International Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_MULTI_ETHNC_IND",  # Student Multi Ethnic Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_NTV_AMRCN_IND",  # Student Native American Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_NTV_ENG_SPKR_IND",  # Student Native English Speaker Indicator (SENSITIVE
    # ATTRIBUTE)
    "STDNT_WHITE_IND",  # Student White Indicator (SENSITIVE ATTRIBUTE)
]

# TODO(jpgard): several of these should be categorical or ordinal features,
#  NOT numeric, as they are really just numeric identifiers for categorical features.

DEFAULT_LARC_NUMERIC_FEATURES = [  # numeric and binary features
    # "ACAD_MAJOR_CNT",  # Academic Major Count
    # "ACAD_MINOR_CNT",  # Academic Minor Count
    "CLASS_ENRL_TOTAL_NBR",  # Class Enrollment Total Number
    # TODO(jpgard): take the leading digit of catlg_nbr as a feature; note that some
    #  courses have invalid (non-numeric) catlg_nbr, such as '300HNSP.U'
    # "CATLG_NBR",
    "CMBN_CLASS_ENRL_TOTAL_NBR",  # Combined Class Enrollment Total Number
    # "EXCL_CLASS_CUM_GPA",
    "HS_GPA",  # High School Grade Point Average
    # "PREV_TERM_CUM_GPA",  # Previous Term Cumulative Grade Point Average
    "STDNT_BIRTH_YR",  # Student Birth Year TODO(jpgard): take (birth year - term year)
    # to obtain an additional feature for approximate age
    "TERM_CD",  # Term Code
]

DEFAULT_LARC_EMBEDDING_FEATURES = [
    # "FIRST_US_PRMNNT_RES_P STL_5_CD", # First Student United States Permanent
    # Residence Postal Five Code (SENSITIVE ATTRIBUTE)
    # "HS_CEEB_CD", # High School College Entrance Examination Board Code
    # "HS_PSTL_CD", # High School Postal Code
]

LARC_VALID_GRADES = [''.join(x) for x in itertools.product(['A', 'B', 'C', 'D', 'E', 'F'],
                                                           ['+', '', '-'])]

# A+, A, A-, B+, ...
# The "passing" grades; these are the positive labels
LARC_PASSING_GRADES = [g for g in LARC_VALID_GRADES if re.match("[ABC].*", g)]


class LarcDataset(TabularDataset):
    """A class to represent the LARC dataset."""

    def __init__(self, client_id_col: str = DEFAULT_LARC_CLIENT_COLNAME,
                 target_column: str = DEFAULT_LARC_TARGET_COLNAME,
                 categorical_columns: List[str] = DEFAULT_LARC_CATEGORICAL_FEATURES,
                 embedding_columns: List[str] = DEFAULT_LARC_EMBEDDING_FEATURES,
                 numeric_columns: List[str] = DEFAULT_LARC_NUMERIC_FEATURES,
                 ):
        super(LarcDataset, self).__init__(client_id_col=client_id_col,
                                          target_column=target_column,
                                          categorical_columns=categorical_columns,
                                          embedding_columns=embedding_columns,
                                          numeric_columns=numeric_columns
                                          )

    def read_data(self, fp):
        # make a dictionary mapping feature types to dtypes
        dtypes = dict()
        dtypes[self.client_id_col] = object
        for cc in self.categorical_columns + self.embedding_columns:
            dtypes[cc] = object
        for nc in self.numeric_columns:
            dtypes[nc] = np.float64
        # read the data
        colnames_to_keep = [self.client_id_col] + self.feature_column_names
        df = read_csv(fp, usecols=colnames_to_keep, na_values=('', ' '),
                      keep_default_na=True, dtype=dtypes)
        self.df = self._preprocessing_fn(df)
        return

    def _preprocessing_fn(self, df):
        """
        The function applied to preprocessing the raw DataFrame from reading CSV.

        Handles procedures like preprocessing the label column, filtering for desired
        values, handling missing values, normalizing inputs to range [0,1],
        etc. Optionally, also prints some  information about how much data is
        discarded, etc for easy monitoring at execution time. Note that
        Ultimately, dataset CANNOT contain any missing values and nan
        cannot be in the vocab for any FeatureColumns (otherwise this leads to
        TypeError when converting to Tensor).
        """
        print("[INFO] raw dataset rows: {}".format(df.shape[0]))
        df = filter_df_by_values(df, self.target_column, LARC_VALID_GRADES)
        df = make_binary_indicator_column(df, self.target_column,
                                          positive_vals=LARC_PASSING_GRADES, replace=True)
        print("[INFO] dataset rows after target filtering: {}".format(df.shape[0]))
        print("[INFO]: label value counts:")
        print(df[self.target_column].value_counts())

        df = generate_missing_value_indicator(df, self.categorical_columns)
        print("[INFO] null counts after creating indicator for categorical columns:")
        print(df.isnull().sum(axis=0))
        df.dropna(inplace=True)
        print("[INFO] dataset rows after dropping NAs: {}".format(df.shape[0]))
        df = minmax_scale_numeric_columns(df, self.numeric_columns)
        print("final preprocessed dataset description:")
        print(df.describe(include='all').T.sort_values(by='unique'))
        return df

    def create_tf_dataset_for_client(self, client_id, training_config: TrainingConfig):
        # filter the dataset by client_id, keeping only the feature and target colnames
        df = self.df[self.df[self.client_id_col] == client_id][self.feature_column_names]
        if len(df):
            dataset = tf.data.Dataset.from_tensor_slices(df.to_dict('list'))
            dataset = dataset.shuffle(training_config.shuffle_buffer).batch(1).repeat(
                training_config.epochs)
            return dataset
        else:
            print("[WARNING] no data for specified client_id {}".format(client_id))
            return None

    def create_tf_dataset(self,  training_config: TrainingConfig):
        """"Create a single tf.Dataset to represent the entire (non-federated) dataset."""
        dataset = tf.data.Dataset.from_tensor_slices(self.df.to_dict('list'))
        dataset = dataset.shuffle(training_config.shuffle_buffer).batch(1).repeat(
            training_config.epochs)
        return dataset

    def make_feature_layer(self):
        """
        Utility function to assemble a feature layer; this builds a single dense feature
        from a list of feature columns.
        :return: a tf.keras.layers.DenseFeatures layer.
        """
        categorical_feature_values = generate_categorical_feature_dict(
            self.df, self.categorical_columns)
        # generate a list of feature columns
        feature_columns = list()
        for nc in self.numeric_columns:
            feature_columns.append(tf.feature_column.numeric_column(nc))
        for cc in self.categorical_columns:
            # create one-hot encoded columns for each categorical column
            fco = tf.feature_column.categorical_column_with_vocabulary_list(
                cc, categorical_feature_values[cc])
            fco_ohe = tf.feature_column.indicator_column(fco)
            feature_columns.append(fco_ohe)
        # TODO(jpgard): embedding columns here
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        return feature_layer
