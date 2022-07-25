import os
import pandas as pd

data_path = "/"
# property_path = "/u/giorgio/GFN/gflownet/src/data/raw/gdb9.sdf.csv"

# https://www.kaggle.com/code/rmonge/predicting-molecule-properties-based-on-its-smiles/notebook


def read_xyz(path):
    """
    Reads the xyz files in the directory on 'path'
    Input
    path: the path to the folder to be read

    Output
    atoms: list with the characters representing the atoms of a molecule
    coordinates: list with the cartesian coordinates of each atom
    smile: list with the SMILE representation of a molecule
    prop: list with the scalar properties
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
    except:
        print(path)
    i += 1

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
