#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Model module."""

import logging
import os
import pickle
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split

from .FeatureEngine import Features

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

seed = 7
np.random.seed(seed)


class RFC:
    """RandomForest classifier for crystals."""

    def __init__(self, crystal_sys: str = "all"):
        """Initialize RandomForest classifier.

        Args:
            crystal_sys: crystal systems to be used.

                "all" for all the crystal systems.
                Other seven options are:
                "monoclinic", "triclinic", "orthorhombic", "trigonal",
                "hexagonal", "cubic", "tetragonal"
        """

        self.crystal_sys = crystal_sys

    def load_data(self, file_name: str) -> pd.DataFrame:
        """Load dataset.

        Args:
            file_name: path of the dataset.

        Returns:
            Dataframe with the loaded dataset.
        """
        feature_eng = Features(formula_file=file_name)
        features = feature_eng.get_features()
        df = pd.DataFrame(features)
        # df = df.drop([9])
        df = df.dropna()

        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp.fit(df.values.tolist())
        data_list = imp.transform(df.values.tolist())
        df = pd.DataFrame(data_list)

        if self.crystal_sys == "monoclinic":
            # Selecting only the monoclinic materials
            df = df.loc[df[1] == 1]
        elif self.crystal_sys == "triclinic":
            # Selecting only the triclinic materials
            df = df.loc[df[2] == 1]
        elif self.crystal_sys == "orthorhombic":
            # Selecting only the orthorhombic materials
            df = df.loc[df[3] == 1]
        elif self.crystal_sys == "trigonal":
            # Selecting only the trigonal materials
            df = df.loc[df[4] == 1]
        elif self.crystal_sys == "hexagonal":
            # Selecting only the hexagonal materials
            df = df.loc[df[5] == 1]
        elif self.crystal_sys == "cubic":
            # Selecting only the cubic materials
            df = df.loc[df[6] == 1]
        elif self.crystal_sys == "tetragonal":
            # Selecting only the tetragonal materials
            df = df.loc[df[7] == 1]
        elif self.crystal_sys == "all":
            # Selecting  the all materials
            pass

        return df

    def split_data(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[Any, Any, Any, Any]:
        """Load dataset.

        Args:
            df: dataset's dataframe.
            test_size: size of the test set.

        Returns:
            Training and testing sets.
        """
        data_X, data_y = df.iloc[:, 1:].values, df.iloc[:, 0].values

        train_x, test_x, train_y, test_y = train_test_split(
            data_X, data_y, test_size=test_size, random_state=42
        )

        return train_x, test_x, train_y, test_y

    def normalize_data(self, train_x:Any, test_x:Any, train_y:Any, test_y:Any) -> Tuple[Any, Any, Any, Any]:
        """Normalize dataset.

        Args:
            train_x: training set's input.
            test_x: testing set's input.
            train_y: training set's groundtruth.
            test_y: testing set's groundtruth.

        Returns:
            Training and testing sets.
        """
        xx = abs(train_x)
        maxm = xx.max(axis=0)
        maxm[maxm == 0.0] = 1

        train_x /= maxm
        test_x /= maxm

        return train_x, test_x, train_y, test_y, maxm

    def train(self, x: Any, y: Any) -> RandomForestClassifier:
        """Train a RandomForest model.

        Args:
            x: training set's input.
            y: training set's groundtruth.

        Returns:
            Trained model.
        """

        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features="auto",
            max_depth=70,
            bootstrap=False,
        )

        scores = cross_val_score(clf, x, y, cv=10, scoring="accuracy")
        model = clf.fit(x, y)

        logger.info("Mean Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))

        return model

    def save(self, path: str, model: RandomForestClassifier, maxm) -> None:
        """Save model.

        Args:
            path: path to store the model.
            model: a trained model.
            maxm: normalized parameters of the trained model.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        # save the model to disk
        pickle.dump(model, open(os.path.join(path, "model.sav"), "wb"))
        # save the normalizing parameters
        df_maxm = pd.DataFrame(maxm)
        df_maxm.to_csv(os.path.join(path, "maxm.csv"), index=False, header=None)

    def load_model(self, path: str) -> Tuple[Any, Any]:
        """Save model.

        Args:
            path: path where the file is located.

        Returns:
            The pretrained model and its normalized parameters.
        """
        # load the model from disk
        loaded_model = pickle.load(open(os.path.join(path, "model.sav"), "rb"))

        # load normalizing parameters
        df_maxm_load = pd.read_csv(os.path.join(path, "maxm.csv"), header=None)
        maxm = np.array([x[0] for x in df_maxm_load.values.tolist()])

        return loaded_model, maxm

    def predict(
        self, model: RandomForestClassifier, maxm: Any, pred_x: Any
    ) -> List[str]:
        """Predict.

        Args:
            model: a trained model.
            maxm: the normalized parameters of the model.
            pred_x: input.

        Returns:
            Predictions
        """

        pred_x /= maxm
        y_rbf_pred = model.predict(pred_x)
        y_rbf_pred = list(y_rbf_pred)

        y_pred_label = ["metal" if x == 0 else "non-metal" for x in y_rbf_pred]

        return y_pred_label
