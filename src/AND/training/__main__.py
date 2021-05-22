import AND.model.cleaning as cleaning
import AND.utils.logger as logger
from AND.utils.helpers import *
from AND.utils.file import *
from AND.training.data_preparation import *
from AND.model.feature_engineering import *

if __name__ == "__main__":
    # ----- load config file and data sets
    print("# ---- Start training")
    logger.logging.info("Start training model")

    logger.logging.info("Loading config file")
    config = read_configurations()
    file_util = FileUtil(config)

    logger.logging.info("Loading data sets")
    data,gt,persons  = file_util.read_data()

    logger.logging.info("Combine ground truth and contributions data")
    df = combine_data_sets(data,gt)

    # ----- clean data set
    logger.logging.info("Cleaning the text data columns")
    df = cleaning.cleaning_procedure(df)

    # ----- create train test split
    logger.logging.info("Creating the train test split")
    df_train, df_test = create_train_test(df, list(df["personId"].unique()))
    print("# ---- Data loaded, data cleaned and created train test split")

    # ----- create contribution pairs
    logger.logging.info("Creating contribution pairs")
    print("# ---- Start creating contribution pairs for traning data")
    df_train = create_contribution_pairs(df_train, n = 6)
    print("# ---- Start creating contribution pairs for test data")
    df_test = create_contribution_pairs(df_test, n = 6)

    # ----- compute features
    logger.logging.info("Ccomputing features")
    print("# ---- Start computing features")
    df_train = compute_features(df_train)
    df_test = compute_features(df_test)

    print(df_test)
    # ----- train random forest

    # ----- evaluate random forest

    # ----- create graph

    # ----- evaluate disconnected subgraphs
    print("# ---- Training completed")