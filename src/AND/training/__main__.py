import AND.model.cleaning as cleaning
import AND.utils.logger as logger
from AND.utils.helpers import *
from AND.utils.file import *
from AND.training.data_preparation import *


if __name__ == "__main__":
    print("# ---- Start training")
    logger.logging.info("Start training model")

    logger.logging.info("Loading config file")
    config = read_configurations()
    file_util = FileUtil(config)

    logger.logging.info("Loading data sets")
    data,gt,persons  = file_util.read_data()

    logger.logging.info("Combine ground truth and contributions data")
    df = combine_data_sets(data,gt)

    logger.logging.info("Cleaning the text data columns")
    df = cleaning.cleaning_procedure(df)

    logger.logging.info("Creating the train test split")
    df_train, df_test = create_train_test(df, list(df["personId"].unique()))
    print("# ---- Data loaded, data cleaned and created train test split")

    # ----- create contribution pairs

    # ----- create features

    # ----- train random forest

    # ----- evaluate random forest

    # ----- create graph

    # ----- evaluate disconnected subgraphs
    print("# ---- Training completed")