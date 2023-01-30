# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""Feature engine for crystals."""

import os
import re
from typing import Dict, List

import pandas as pd
import pkg_resources  # type: ignore
from pymatgen.core.composition import Composition


class Features:
    """Feature generator for crystals."""

    def __init__(self, formula_file: str):
        """Initialize Feature engine.

        Args:
            formula_file: file of formulas
        """
        self.formula_file = formula_file
        self.atomic_data_file = pkg_resources.resource_filename(  # type: ignore
            "gt4sd", os.path.join("frameworks", "crystals_rfc", "atomic_data.csv")
        )

    def make_features(
        self,
        atomic_descriptors,
        formula_list,
        targets,
        encoded_sym,
        add_avg: bool = True,
        add_aad: bool = True,
        add_md: bool = False,
        add_cv: bool = False,
    ) -> List[List[float]]:
        """Initialize Feature engine.

        Args:
            atomic_descriptors: atomic descriptors.
            formula_list: file of formulas.
            targets: targets.
            encoded_sym: encoded_sym.
            add_avg: include Weighted Average.
            add_aad: include Average Absolute Deviation.
            add_md: include maximum difference.
            add_cv: include element ratio vector.

        Returns:
            List of all features.
        """

        all_feature_list = []
        for indx0, formula in enumerate(formula_list):

            feature_list = []
            atom_symbols = list(atomic_descriptors[0].keys())

            comp = Composition(formula)
            formula = comp.formula
            s = re.findall(r"([A-Z][a-z]?)([0-9]?\.?[0-9]*)", formula)
            comp_vector = [0 for _ in range(0, len(atom_symbols))]

            feature_list.append(targets[indx0])

            for d in encoded_sym[indx0]:
                feature_list.append(d)

            # Calculating the total number of atoms in the chemical formula
            total = 0.0
            for elem, num in s:
                if num == "":
                    num = 1
                total += int(num)

            # Calculating Weighted Average
            avg = 0.0
            for des in atomic_descriptors:
                des_list = []
                for elem, num in s:
                    if num == "":
                        num = 1
                    num = int(num)
                    avg += des[elem] * num
                    des_list.append((des[elem], num))
                avg = avg / total
                if add_avg:
                    feature_list.append(avg)

                # Calculating Average Absolute Deviation
                if add_aad:
                    avg_ad = 0.0
                    for y, num in des_list:
                        ad = abs(y - avg) * num
                        avg_ad += ad
                    avg_ad = avg_ad / total

                    feature_list.append(avg_ad)

                # Calculating maximum difference
                if add_md:
                    dif_list = []
                    for y1, num1 in des_list:
                        for y2, num2 in des_list:
                            dif = abs(y1 - y2)
                            dif_list.append(dif)
                    max_dif = max(dif_list)

                    feature_list.append(max_dif)

            # Creting Element Ratio Vector
            for elem, num in s:
                if num == "":
                    num = 1
                num = int(num)
                index = atom_symbols.index(elem)
                comp_vector[index] = int(num) / total  # type: ignore

            # Uncomment if the element ratio vector is required
            if add_cv:
                for ratio in comp_vector:
                    feature_list.append(ratio)

            all_feature_list.append(feature_list)

        return all_feature_list

    def get_formula_list(self) -> List[str]:
        """Get formula list.

        Returns:
            List of formulas.
        """
        df_mat = pd.read_csv(self.formula_file, header=None)
        formula_list = [x[0] for x in df_mat.values.tolist()]

        return formula_list

    def get_encoded_sym(self) -> List[List[int]]:
        """Get encoded systems.

        Returns:
            List of encoded systems.
        """
        df_mat = pd.read_csv(self.formula_file, header=None)

        if len(df_mat.columns) == 3:
            sym_list = [x[2] for x in df_mat.values.tolist()]
        elif len(df_mat.columns) == 2:
            sym_list = [x[1] for x in df_mat.values.tolist()]
        else:
            raise ValueError(
                "The provided csv file should contain two or three columns."
            )

        encoded_sym = []
        for sym in sym_list:
            if sym == "monoclinic":
                digits = [1, 0, 0, 0, 0, 0, 0]
            elif sym == "triclinic":
                digits = [0, 1, 0, 0, 0, 0, 0]
            elif sym == "orthorhombic":
                digits = [0, 0, 1, 0, 0, 0, 0]
            elif sym == "trigonal":
                digits = [0, 0, 0, 1, 0, 0, 0]
            elif sym == "hexagonal":
                digits = [0, 0, 0, 0, 1, 0, 0]
            elif sym == "cubic":
                digits = [0, 0, 0, 0, 0, 1, 0]
            elif sym == "tetragonal":
                digits = [0, 0, 0, 0, 0, 0, 1]

            encoded_sym.append(digits)
        return encoded_sym

    def get_targets(self) -> List[float]:
        """Get targets.

        Returns:
            List of targets.
        """
        df_mat = pd.read_csv(self.formula_file, header=None)

        if len(df_mat.columns) == 3:
            targets = [x[1] for x in df_mat.values.tolist()]
        elif len(df_mat.columns) == 2:
            targets = [0 for _ in df_mat.values.tolist()]
        else:
            raise ValueError(
                "The provided csv file should contain two or three columns."
            )

        return targets

    def get_atomic_descriptors(self) -> List[Dict[str, float]]:
        """Get atomic descriptors.

        Returns:
               List of atomic descriptors.
        """
        df_des = pd.read_csv(self.atomic_data_file, header=None)

        atomic_descriptors = []

        elements = [x[0] for x in df_des.values.tolist()]
        for i in range(1, len(df_des.columns)):
            tmp = [x[i] for x in df_des.values.tolist()]
            des_dict = dict(zip(elements, tmp))

            atomic_descriptors.append(des_dict)

        return atomic_descriptors

    def get_features(self) -> List[List[float]]:
        """Get features.

        Returns:
            List of all features.
        """
        formula_list = self.get_formula_list()
        encoded_sym = self.get_encoded_sym()
        targets = self.get_targets()
        atomic_descriptors = self.get_atomic_descriptors()

        features = self.make_features(
            atomic_descriptors=atomic_descriptors,
            encoded_sym=encoded_sym,
            targets=targets,
            formula_list=formula_list,
        )

        return features
