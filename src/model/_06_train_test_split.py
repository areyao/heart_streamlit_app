from src.utils._run_scripts import *
from src.model._run_scripts import *

conf = get_conf()
split_ratio = conf['train_val_test_split']


def split_data(dataframe, custom_split = None):
    """
    split_data : Split the dataset into training, validation set and test set.
    Use a stratified sampling method for equal split.

    :param dataframe : dataset to split
    :param custom_split (list ; optional) : modify the train-validation-test split ratio

    :return train_set, validation_set, test_set (PySpark dataframes) - percentage split based on the provided values
    """

    # split dataframes between 0s and 1s
    no_heartAttack = dataframe.filter(dataframe["HeartDisease"] == 0)
    with_heartAttack = dataframe.filter(dataframe["HeartDisease"] == 1)

    # split dataframes into training and testing
    train0, validation0, test0 = no_heartAttack.randomSplit(split_ratio, seed=1234)
    train1, validation1, test1 = with_heartAttack.randomSplit(split_ratio, seed=1234)

    if custom_split:
        train0, validation0, test0 = no_heartAttack.randomSplit(custom_split, seed=1234)
        train1, validation1, test1 = with_heartAttack.randomSplit(custom_split, seed=1234)

    # stack datasets back together
    train_set = train0.union(train1)
    validation_set = validation0.union(validation1)
    test_set = test0.union(test1)

    return train_set, validation_set, test_set
