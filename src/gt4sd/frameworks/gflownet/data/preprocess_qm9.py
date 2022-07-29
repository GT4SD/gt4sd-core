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
import os

import pandas as pd

data_path = "/"

# https://www.kaggle.com/code/rmonge/predicting-molecule-properties-based-on-its-smiles/notebook


def read_xyz(path):
    """Reads the xyz files in the directory on 'path'.

    Args:
        path: the path to the folder to be read.

    Returns:
        atoms: list with the characters representing the atoms of a molecule.
        coordinates: list with the cartesian coordinates of each atom.
        smile: list with the SMILE representation of a molecule.
        prop: list with the scalar properties.
    """
    atoms = []
    coordinates = []

    with open(path, "r") as file:
        lines = file.readlines()
        n_atoms = int(lines[0])  # the number of atoms
        smile = lines[n_atoms + 3].split()[0]  # smiles string
        prop = lines[1].split()[2:]  # scalar properties

        # to retrieve each atmos and its cartesian coordenates
        for atom in lines[2 : n_atoms + 2]:
            line = atom.split()
            # which atom
            atoms.append(line[0])

            # its coordinate
            # Some properties have '*^' indicading exponentiation
            try:
                coordinates.append((float(line[1]), float(line[2]), float(line[3])))
            except ValueError:
                coordinates.append(
                    (
                        float(line[1].replace("*^", "e")),
                        float(line[2].replace("*^", "e")),
                        float(line[3].replace("*^", "e")),
                    )
                )

    return atoms, coordinates, smile, prop


# Reads the xyz files and return the properties, smiles and coordinates
data = []
smiles = []
properties = []
i = 0
for file in os.listdir(data_path):
    try:
        path = os.path.join(data_path, file)
        atoms, coordinates, smile, prop = read_xyz(path)
        data.append((atoms, coordinates))  # A tuple with the atoms and its coordinates
        smiles.append(smile)  # The SMILES representation
        properties.append(prop)  # The molecules properties
    except ValueError:
        print(path)
    i += 1

# rename relevant properties to match GFN schema
labels = [
    "rA",
    "rB",
    "rC",
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]
df = pd.DataFrame(properties, columns=labels)
df["SMILES"] = smiles

print(df)
df.to_hdf("qm9.h5", key="df", mode="w")


# convert from smiles
