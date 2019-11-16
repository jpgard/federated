"""
Functions to load and transform datasets.
"""
import collections
import functools
import itertools
import re

import pandas as pd
import tensorflow as tf
import numpy as np

from abc import ABC, abstractmethod
from feded.preprocessing import generate_categorical_feature_dict, filter_df_by_values, \
    make_binary_indicator_column, read_csv
from feded.config import TrainingConfig
from typing import Optional, Tuple, List

# the prediction target
DEFAULT_LARC_TARGET_COLNAME = "CRSE_GRD_OFFCL_CD"

# the column to use for generating clients; this should be a column which universities
# might use to partition their data (e.g. sharing across schools or departments might
# be restricted, so this would be a good choice).
DEFAULT_LARC_CLIENT_COLNAME = "SBJCT_CD"

# explicitly specify the categorical features and the values they can take
# TODO(jpgard): create a function which reads these directly from the data for a list
#  of specified categorical feature names; or even better, implement these as part of
#  an object representing the dataset.
DEFAULT_LARC_CATEGORICAL_FEATURES = [
    # "CRER_LVL_CD",  # Career Level Code
    "CRSE_CMPNT_CD",  # Course Component Code
    "CRSE_GRD_OFFCL_CD",  # Course Grade Official Code (The official grade that appears
    # on a student's transcript)
    "EST_GROSS_FAM_INC_CD",  # Estimated Gross Family Income Code (SENSITIVE ATTRIBUTE)
    # "HS_STATE_CD", # High School State Code (note: only exists for US/CA students)
    "GRD_BASIS_ENRL_CD",
    # "PRMRY_CRER_CD",  # Primary Career Code
    "PRNT_MAX_ED_LVL_CD",  # Parent Maximum Education Level Code (SENSITIVE ATTRIBUTE)
    # "RES_CD",  # Residency Code (SENSITIVE ATTRIBUTE)
    "SBJCT_CD",  # Subject Code TODO(jpgard): might move to embedding columns later
    "STDNT_DMSTC_UNDREP_MNRTY_CD",
    # Student Domestic Underrepresented Minority (URM) Code (SENSITIVE ATTRIBUTE)
    "STDNT_ETHNC_GRP_CD",  # Student Ethnic Group Code (SENSITIVE ATTRIBUTE)
    "STDNT_GNDR_CD",  # Student Gender Code (SENSITIVE ATTRIBUTE)
]

# explicitly specify the numeric features
# TODO(jpgard): create a function which reads these directly from the data for a list
#  of specified feature names; or even better, implement these as part of an object
#  representing the dataset.
# TODO(jpgard): several of these should be categorical or ordinal features,
#  NOT numeric, as they are really just numeric identifiers for categorical features.
DEFAULT_LARC_NUMERIC_FEATURES = [  # numeric and binary features
    # "ACAD_MAJOR_CNT",  # Academic Major Count
    # "ACAD_MINOR_CNT",  # Academic Minor Count
    "ADMSSN_VTRN_IND",  # Admission Veteran Indicator (SENSITIVE ATTRIBUTE)
    "CLASS_ENRL_TOTAL_NBR",  # Class Enrollment Total Number
    "CLASS_GRDD_IND",  # Class Graded Indicator TODO(jpgard): filter for 1 ("yes") only
    "CLASS_HONORS_IND",  # Class Honors Indicator
    # TODO(jpgard): take the leading digit of catlg_nbr as a feature; note that some
    #  courses have invalid (non-numeric) catlg_nbr, such as '300HNSP.U'
    # "CATLG_NBR",
    "CMBN_CLASS_ENRL_TOTAL_NBR",  # Combined Class Enrollment Total Number
    # "EXCL_CLASS_CUM_GPA",
    "HS_CALC_IND",  # High School Calculus Indicator (self-reported)
    "HS_CHEM_LAB_IND",  # High School Chemistry Laboratory Indicator (self-reported)
    "HS_GPA",  # High School Grade Point Average
    # "PREV_TERM_CUM_GPA",  # Previous Term Cumulative Grade Point Average
    "SNGL_PRNT_IND",  # Single Parent Indicator (SENSITIVE ATTRIBUTE)
    # "SPPLMNT_STUDY_IND",  # Supplemental Study Indicator
    "STDNT_ASIAN_IND",  # Student Asian Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_BIRTH_YR",  # Student Birth Year TODO(jpgard): take (birth year - term year)
    # to obtain an additional feature for approximate age
    "STDNT_BLACK_IND",  # Student Black Indicator (SENSITIVE ATTRIBUTE)
    ## "STDNT_CTZN_STAT_CD",  # Student Citizenship Status Code (SENSITIVE ATTRIBUTE) (
    # (INVALID 'N' value)
    "STDNT_HSPNC_IND",  # Student Hispanic Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_HSPNC_LATINO_IND",  # Student Hispanic Latino Indicator (SENSITIVE ATTRIBITE)
    "STDNT_HWIAN_IND",  # Student Hawaiian Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_INTL_IND",  # Student International Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_MULTI_ETHNC_IND",  # Student Multi Ethnic Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_NTV_AMRCN_IND",  # Student Native American Indicator (SENSITIVE ATTRIBUTE)
    "STDNT_NTV_ENG_SPKR_IND",  # Student Native English Speaker Indicator (SENSITIVE
    # ATTRIBUTE)
    "STDNT_WHITE_IND",  # Student White Indicator (SENSITIVE ATTRIBUTE)
    "TERM_CD",  # Term Code
]

DEFAULT_LARC_EMBEDDING_FEATURES = [
    # "FIRST_US_PRMNNT_RES_P STL_5_CD", # First Student United States Permanent
    # Residence Postal Five Code (SENSITIVE ATTRIBUTE)
    # "HS_CEEB_CD", # High School College Entrance Examination Board Code
    # "HS_PSTL_CD", # High School Postal Code
]
# A+, A, A-, B+, ...
LARC_VALID_GRADES = [''.join(x) for x in itertools.product(['A', 'B', 'C', 'D', 'E', 'F'],
                                                           ['+', '', '-'])]
# The "passing" grades; these are the positive labels
LARC_PASSING_GRADES = [g for g in LARC_VALID_GRADES if re.match("[ABC].*", g)]


def preprocess(dataset, feature_layer, training_config: TrainingConfig,
               target_feature=DEFAULT_LARC_TARGET_COLNAME):
    """Preprocess data with a single-element label (label of length one)."""
    num_epochs = training_config.epochs
    shuffle_buffer = training_config.shuffle_buffer
    batch_size = training_config.batch_size

    def element_fn(element):
        # element_fn extracts feature and label vectors from each element;
        # 'x' and 'y' names are required by keras.
        feature_vector = feature_layer(element)

        return collections.OrderedDict([
            ('x', tf.reshape(feature_vector, [feature_vector.shape[1]])),
            ('y', tf.reshape(element[target_feature], [1])),
        ])

    return dataset.repeat(num_epochs).map(element_fn).shuffle(shuffle_buffer).batch(
        batch_size)


class TabularDataset(ABC):
    """A class to represent tabular datasets."""

    def __init__(self, client_id_col: str,
                 target_column: str,
                 categorical_columns: Optional[List[str]] = None,
                 embedding_columns: Optional[List[str]] = None,
                 numeric_columns: Optional[List[str]] = None,
                 ):
        self.client_id_col = client_id_col
        self.categorical_columns = categorical_columns
        self.df = None
        self.embedding_columns = embedding_columns
        self.numeric_columns = numeric_columns
        self.target_column = target_column

    @property
    def feature_column_names(self):
        """The name of all feature columns; excluding the client_id column."""
        return self.categorical_columns + self.embedding_columns + self.numeric_columns

    @abstractmethod
    def read_data(self, fp):
        raise  # this line should never be reached

    @abstractmethod
    def create_tf_dataset_for_client(self, client_id, training_config: TrainingConfig):
        """"Creates a tf.Dataset for the given client id."""
        raise  # this line should never be reached

    @abstractmethod
    def make_feature_layer(self):
        """Create a keras.DenseFeatures layer."""
        raise


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
        # df = pd.read_csv(fp, usecols=colnames_to_keep, na_values=('', ' '),
        #                  keep_default_na=True, dtype=dtypes)
        df = read_csv(fp, usecols=colnames_to_keep, na_values=('', ' '),
                      keep_default_na=True, dtype=dtypes)
        print("[INFO] raw dataset rows: {}".format(df.shape[0]))
        df = filter_df_by_values(df, self.target_column, LARC_VALID_GRADES)
        df = make_binary_indicator_column(df, self.target_column,
                                          positive_vals=LARC_PASSING_GRADES, replace=True)
        print("[INFO] dataset rows after target filtering: {}".format(df.shape[0]))
        print("[INFO] null counts:")
        print(df.isnull().sum(axis=0))

        # TODO(jpgard): handle NA values instead of dropping incomplete cases; for
        #  some features (e.g. categorical features) we can generate an indicator
        #  for missingness; for missing numeric features these will likely need to
        #  be dropped. Ultimately, dataset CANNOT contain any missing values and nan
        #  cannot be in the vocab for any FeatureColumns (otherwise this leads to
        #  TypeError when converting to Tensor).
        df.dropna(inplace=True)
        print("[INFO] dataset rows after dropping NAs: {}".format(df.shape[0]))
        self.df = df
        return

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
            # TODO(jpgard): normalize the numeric columns
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
