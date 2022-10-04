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
import ipaddress
import json
from typing import Any, Callable, Dict, List, Tuple, Type, Union

from rdkit import Chem
from modlamp.descriptors import GlobalDescriptor
from tdc import Oracle
from tdc.chem_utils.oracle.oracle import fp2fpfunc
from tdc.metadata import download_oracle_names

from ..domains.materials import MacroMolecule, SmallMolecule
from .core import ApiTokenParameters, PropertyValue
from .scores import SCORING_FACTORY


def to_mol(mol: SmallMolecule) -> Chem.Mol:
    """Safely convert a string or a rdkit.Chem.Mol to a rdkit.Chem.Mol.

    Args:
        mol: a string or a rdkit.Chem.Mol object.

    Raises:
        TypeError: if wrong type is given.

    Returns:
        a rdkit.Chem.Mol object.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    elif isinstance(mol, Chem.Mol):
        pass
    else:
        raise TypeError(
            f"Please provide SMILES string or rdkit.Chem.Mol object not {type(mol)}"
        )
    return mol


def to_smiles(mol: SmallMolecule) -> str:
    """Safely returns a SMILES string from a SMILES or a rdkit.Chem.Mol object.

    Args:
        SmallMolecule: either a SMILES or a rdkit.Chem.Mol object.

    Returns:
        a SMILES string.
    """
    if isinstance(mol, str):
        try:
            mol = Chem.MolFromSmiles(mol)
        except Exception:
            raise ValueError(
                f"Could not convert SMILES string to rdkit.Chem.Mol: {mol}"
            )
    elif isinstance(mol, Chem.Mol):
        pass
    else:
        raise TypeError(
            f"Pass a SMILES string or rdkit.Chem.Mol object not {type(mol)}"
        )

    return Chem.MolToSmiles(mol, canonical=True)


def get_similarity_fn(
    target_mol: SmallMolecule, fp_key: str = "FCFP4"
) -> Callable[[SmallMolecule], PropertyValue]:
    """Get a similarity function for a target molecule.

    Args:
        target_mol: a target molecule as SMILES or rdkit.Chem.Mol object.
        fp_key: The type of fingerprint to use. One of `ECFP4`, `ECFP6`, `FCFP4` and `AP`.

    Returns:
        a similarity function that can be called with a `SmallMolecule`.
    """
    if fp_key not in fp2fpfunc.keys():
        raise ValueError(f"Choose fp_key from {fp2fpfunc.keys()}.")
    target_smiles = to_smiles(target_mol)
    return Oracle(name="similarity_meta", target_smiles=target_smiles, fp=fp_key)


def get_activity_fn(target: str) -> Callable[[SmallMolecule], PropertyValue]:
    """Get a function to measure activity/affinity against a protein target.

    Args:
        target: name of the target protein.

    Returns:
        an affinity function that can be called with a `SmallMolecule`.
    """
    if target not in download_oracle_names:
        raise ValueError(
            f"Supported targets are: {download_oracle_names}, not {target}"
        )
    return Oracle(name=target)


# for proteins
def get_sequence(protein: MacroMolecule) -> str:
    """Safely returns an amino acid sequence of a macromolecule

    Args:
        protein: either an AA sequence or a rdkit.Chem.Mol object that can be converted to FASTA.

    Raises:
        TypeError: if the input was none of the above types.
        ValueError: if the sequence was empty or could not be parsed into FASTA.

    Returns:
        an AA sequence.
    """
    if isinstance(protein, str):
        seq = protein.upper().strip()
        return seq
    elif isinstance(protein, Chem.Mol):
        seq = Chem.MolToFASTA(protein).split()
    else:
        raise TypeError(f"Pass a string or rdkit.Chem.Mol object not {type(protein)}")
    if seq == []:
        raise ValueError(
            f"Sequence was empty or rdkit.Chem.Mol could not be converted: {protein}"
        )
    return seq[-1]


def get_descriptor(protein: MacroMolecule) -> GlobalDescriptor:
    """Convert a macromolecule to a modlamp GlobalDescriptor object.

    Args:
        protein: either an AA sequence or a rdkit.Chem.Mol object that can be converted to FASTA.

    Returns:
        GlobalDescriptor object.
    """
    seq = get_sequence(protein)
    return GlobalDescriptor(seq)


def get_target_parameters(
    target: Union[str, Dict[str, Any]]
) -> Tuple[List[Type[Any]], List[float]]:
    """Generates a tuple of scorers and weight list

    Args:
        target: scoring functions and parameters related to it

    Return:
        A tuple containing scoring functions and weight list
    """
    score_list = []
    weights = []
    target_dictionary: Dict[str, Any] = {}
    if isinstance(target, str):
        target_dictionary = json.loads(target)
    elif isinstance(target, dict):
        target_dictionary = target
    else:
        raise ValueError(
            f"{target} of type {type(target)} is not supported: provide 'str' or 'Dict[str, Any]'"
        )
    for scoring_function_name, parameters in target_dictionary.items():
        weight = 1.0
        if "weight" in parameters:
            weight = parameters.pop("weight")
        score_list.append(SCORING_FACTORY[scoring_function_name](**parameters))
        weights.append(weight)
    return (score_list, weights)


def validate_ip(ip: str, message: str = "") -> None:
    """
    Validates whether the parameter configuration contains a correct IP
    address.

    Args:
        ip: The IP address to validate.
        message: Additional error message to be displayed.
    """

    try:
        ipaddress.ip_address(ip)
    except ValueError:
        raise ValueError(f"{ip} is not a IPv4 or IPv6 address\n {message}")


def validate_api_token(parameters: ApiTokenParameters, message: str = "") -> None:
    """
    Validates whether the parameter configuration contains something
    that _could_ be a valid API key.

    Args:
        parameters: ApiTokenParameters.
        message: Additional error message to be displayed.
    """

    if not hasattr(parameters, "api_token"):
        raise AttributeError(f"API key missing in {parameters}")

    if not isinstance(parameters.api_token, str):
        raise TypeError(
            f"API key has to be a string not {parameters.api_token}\n {message}"
        )


def docking_import_check() -> None:
    """
    Verifies that __some__ of the required packages for docking are installed.

    Raises:
        ModuleNotFoundError: If a necessary module was not found.
    """
    try:
        import openbabel
        import pdbfixer
        import pyscreener

        openbabel, pdbfixer, pyscreener
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "You dont seem to have a valid installation for docking. You at "
            "least need `pdbfixer`, `openbabel` and `pyscreener` installed."
            "See here for details: https://tdcommons.ai/functions/oracles/#docking-scores"
        )
