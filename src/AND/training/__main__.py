import AND.model.cleaning as cleaning
import AND.utils.logger as logger
from AND.utils.helpers import *
from AND.utils.file import *
from AND.model.rf_classifier import *
from AND.training.data_preparation import *
from AND.model.feature_engineering import *

if __name__ == "__main__":
    logger.logging.info("Start training model")
    print("################################")
    print("# ---- Start training")
    print("################################")

    # ----- load config file and data sets
    logger.logging.info("Loading config file")
    config = read_configurations()
    file_util = FileUtil(config)

    logger.logging.info("Loading data sets")
    data, gt, persons = file_util.read_data()

    logger.logging.info("Combine ground truth and contributions data")
    df = combine_data_sets(data, gt)

    # ----- clean data set
    logger.logging.info("Cleaning the text data columns")
    df = cleaning.cleaning_procedure(df)

    # ----- create train test split
    logger.logging.info("Creating the train test split")
    df_train, df_test = create_train_test(df, list(df["personId"].unique()))

    # ----- create contribution pairs
    logger.logging.info("Creating contribution pairs for train data set")
    print("# ---- Start creating contribution pairs for traning data")
    df_train = create_contribution_pairs(df_train, n=6)
    logger.logging.info("Creating contribution pairs for test data set")
    print("# ---- Start creating contribution pairs for test data")
    df_test = create_contribution_pairs(df_test, n=6)

    # ----- compute features
    logger.logging.info("Ccomputing features for training data set")
    print("# ---- Start computing features for train data set")
    df_train = compute_features(df_train)
    logger.logging.info("Ccomputing features for test data set")
    print("# ---- Start computing features for test data set")
    df_test = compute_features(df_test)

    # ----- evaluate random forest with cross validation
    clfObject = rfClassifier(config["rfClassifier"])
    scores = clfObject.run_cross_validation(df_train)
    print(scores)

    # ----- train random forest on full training data set

    # ----- create graph from test data set

    # ----- evaluate disconnected subgraphs

    print("################################")
    print("# ---- Training completed")
    print("################################")
