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
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split

from .feature_engine import Features

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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

        self.model = RandomForestClassifier(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features="auto",
            max_depth=70,
            bootstrap=False,
        )

        self.maxm: np.ndarray

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def normalize_data(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        self.maxm = xx.max(axis=0)
        self.maxm[self.maxm == 0.0] = 1

        train_x /= self.maxm
        test_x /= self.maxm

        return train_x, test_x, train_y, test_y

    def train(self, x: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train a RandomForest model.

        Args:
            x: training set's input.
            y: training set's groundtruth.

        Returns:
            Trained model.
        """

        if self.maxm is None:
            raise ValueError("Dataset should be normalized before the training.")

        scores = cross_val_score(self.model, x, y, cv=10, scoring="accuracy")
        model = self.model.fit(x, y)

        logger.info("Mean Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))

        return model

    def save(self, path: str) -> None:
        """Save model.

        Args:
            path: path to store the model.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        # save the model to disk
        pickle.dump(self.model, open(os.path.join(path, "model.sav"), "wb"))
        # save the normalizing parameters
        df_maxm = pd.DataFrame(self.maxm)
        df_maxm.to_csv(os.path.join(path, "maxm.csv"), index=False, header=None)

    def load_model(self, path: str) -> None:
        """Save model.

        Args:
            path: path where the file is located.
        """
        # load the model from disk
        self.model = pickle.load(open(os.path.join(path, "model.sav"), "rb"))

        # load normalizing parameters
        df_maxm_load = pd.read_csv(os.path.join(path, "maxm.csv"), header=None)
        self.maxm = np.array([x[0] for x in df_maxm_load.values.tolist()])

    def predict(self, pred_x: np.ndarray) -> List[str]:
        """Predict.

        Args:
            pred_x: input.

        Returns:
            Predictions
        """

        if self.maxm is None:
            raise ValueError("Model is not initialized.")

        pred_x /= self.maxm
        y_rbf_pred = self.model.predict(pred_x)
        y_rbf_pred = list(y_rbf_pred)

        y_pred_label = ["metal" if x == 0 else "non-metal" for x in y_rbf_pred]

        return y_pred_label
