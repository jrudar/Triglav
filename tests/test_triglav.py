from triglav import Triglav

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd

from pathlib import Path

if __name__ == "__main__":

    dirpath = Path(__file__).parent

    expected_output = dirpath / 'data/expected_output.csv'

    def test_triglav():

        #Create the dataset
        X, y = make_classification(n_samples = 200,
                                   n_features = 20,
                                   n_informative = 5,
                                   n_redundant = 2,
                                   n_repeated = 0,
                                   n_classes = 2,
                                   shuffle = False,
                                   random_state = 0)

        #Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size = 0.2, 
                                                            random_state = 0, 
                                                            stratify = y)

        #Set up Triglav
        model = Triglav(n_jobs = 4,
                        verbose = 3,
                        metric = "euclidean",
                        linkage = "ward", 
                        criterion="maxclust",
                        thresh = 9,
                        transformer=StandardScaler())

        #Identify predictive features
        model.fit(X_train, y_train)

        features_selected = model.selected_
        features_best = model.selected_best_

        df = pd.DataFrame(data = [features_best, features_selected], index = ["Selected Best", "Selected"], columns = [i for i in range(0, 20)])
    
        test_df = pd.read_csv(expected_output)

        pd.assert_frame_equal(df, test_df, check_dtype = False)

