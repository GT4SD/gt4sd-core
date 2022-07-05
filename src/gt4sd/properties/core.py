from ..domains.materials import Property, Protein, SmallMolecule, MacroMolecule
from pydantic import BaseModel
from typing import Any, Union, Callable

class PropertyPredictorConfiguration(BaseModel):
    """Abstract class for property prediction in molecules and proteins.
    """
    pass

class PropertyPredictor(BaseModel):
    """Property predictor.
    """
    def __init__(self, parameters: PropertyPredictorConfiguration = None) -> None:
        """
        Args:
            parameters
        """
        self.parameters = parameters

    @staticmethod
    def from_json(json_file: str) -> PropertyPredictor:
        """Instantiate from json configuration.

        # pydantic from dict

        Args:
            json_file (str): configuration file

        Returns:
            PropertyPredictor
        """
        pass

    def to_json(self) -> str:
        """Convert instance PropertyPrediction in json configuration.

        Returns:
            str: json file
        """
        pass

    def __call__(self, sample : Any) -> Property:
        """generic call method for all properties.

        pp = PropertyPredictor(params) 
        property_value = pp(sample)

        Args:
            sample: 

        Returns:
            Property:
        """
        pass 

class CallablePropertyPredictor(PropertyPredictor):

    def __init__(self, parameters: PropertyPredictorConfiguration, callable_fn: Callable) -> None:
        self.callable_fn = callable_fn
        super().__init__(parameters=parameters)
    
    def __call__(self, sample : Union[SmallMolecule, MacroMolecule, Protein]) -> Property:
        """generic call method for all properties.

        pp = PropertyPredictor(params) 
        property_value = pp(sample)

        Args:
            sample: input string (SMILES)

        Returns:
            Property: callable function to compute a generic property
        """
        return self.callable_fn(sample)

class SmallMoleculeProperty(PropertyPredictor):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, molecule: SmallMolecule) -> Property:
        pass

class ProteinProperty(PropertyPredictor):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, protein: MacroMolecule) -> Property:
        pass