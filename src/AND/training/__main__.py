import AND.model.cleaning as cleaning
import AND.utils.logger as logger
from AND.utils.helpers import *
from AND.utils.file import *
from AND.model.rf_classifier import *
from AND.training.data_preparation import *
from AND.model.feature_engineering import *
from AND.model.graph import *

def run_traning():
    logger.logging.info("##----------------------------------------")
    logger.logging.info("Start training model")
    logger.logging.info("##----------------------------------------")

    # ----- load config file and data sets
    logger.logging.info("## Loading config file")
    config = read_configurations()
    file_util = FileUtil(config)

    logger.logging.info("## Loading data sets")
    data, gt, persons = file_util.read_data()
    data = data.sample(500)

    logger.logging.info("## Combine ground truth and contributions data")
    df = combine_data_sets(data, gt)

    # ----- clean data set
    logger.logging.info("## Cleaning the text data columns")
    df = cleaning.cleaning_procedure(df)

    # ----- create train test split
    logger.logging.info("## Creating the train test split")
    df_train, df_test = create_train_test(df, list(df["personId"].unique()), ratio=config["train_test_split"])

    # ----- create contribution pairs
    logger.logging.info("## Creating contribution pairs for train data set")
    df_train = create_contribution_pairs(df_train, n=config["left_out_negative_sample_rate"])
    logger.logging.info("## Creating contribution pairs for test data set")
    df_test = create_contribution_pairs(df_test, n=config["left_out_negative_sample_rate"])

    # ----- compute features
    logger.logging.info("## Computing features for training data set")
    df_train = compute_features(df_train)
    logger.logging.info("## Computing features for test data set")
    df_test = compute_features(df_test)

    # ----- evaluate random forest with cross validation
    logger.logging.info("## Running cross validation")
    clfObject = rfClassifier(config["rfClassifier"])
    cv_scores = clfObject.run_cross_validation(df_train)
    file_util.report_cv_scores(cv_scores)
    logger.logging.info("Cross validation f1-sores are: {}".format(' '.join(map(str, cv_scores["test_f1"]))))

    # ----- train random forest on full training data set
    logger.logging.info("## Fitting random forest model")
    feature_importances = clfObject.fit_classifier(df_train)
    file_util.report_feature_importances(feature_importances)

    logger.logging.info("##----------------------------------------")
    logger.logging.info("Start evaluation on test data set")
    logger.logging.info("##----------------------------------------")

    # ----- create graph from test data set
    logger.logging.info("## Creating graph of the ontributions")
    df_test = clfObject.predict(df_test)
    contribution_graph = create_graph(df_test)

    # ----- create author profiles -> find the disconnected subgraphs
    logger.logging.info("## Creating the author profiles")
    profiles = get_disconnected_subgraphs(contribution_graph)
    file_util.report_profiles(profiles,df_test)

    # evaluate performance
    # file_util.report_test_results(test_results)

if __name__ == "__main__":
    run_traning()