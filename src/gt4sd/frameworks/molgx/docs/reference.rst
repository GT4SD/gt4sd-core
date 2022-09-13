.. _reference:

********************
Python API Reference
********************

.. _molecule:

molgx.Molecule module
=====================

.. py:module:: molgx.Molecule

* :py:class:`MolType`	       
	       
* :py:class:`Molecule`

  - :py:class:`SimpleMolecule`

    + :py:class:`GeneratedMolecule`

* :py:class:`Property`

* :py:class:`PropertySet`

----

.. autoclass:: MolType
    :members:

----

.. autoclass:: Molecule
    :members:

----

.. autoclass:: SimpleMolecule
    :members:
	       
----

.. autoclass:: GeneratedMolecule
    :members:	       

----

.. autoclass:: Property
    :members:

----

.. autoclass:: PropertySet
    :members:

----

|

.. _databox:

molgx.DataBox module
============================

.. py:module:: molgx.DataBox

* :py:class:`MolData`

----

.. autoclass:: MolData
    :members:

----		  

|
			
.. _feature_extraction:

molgx.FeatureExtraction module
======================================

.. py:module:: molgx.FeatureExtraction

* :py:class:`Feature`

  - :py:class:`IntFeature`

* :py:class:`FeatureSet`

  - :py:class:`MergedFeatureSet`

* :py:class:`FeatureExtractor`

  - :py:class:`StructureCounting`

    + :py:class:`HeavyAtomExtractor`

    + :py:class:`RingExtractor`

    + :py:class:`AromaticRingExtractor`

    + :py:class:`StructureExtractor`

      + :py:class:`FingerPrintStructureExtractor`

  - :py:class:`FeatureOperator`

    + :py:class:`FeatureSumOperator`

* :py:func:`print_feature_extractor()`

----

.. autoclass:: Feature
    :members:

----

.. autoclass:: IntFeature
    :members:

----

.. autoclass:: FeatureSet
    :members:

----

.. autoclass:: MergedFeatureSet
    :members:	       

----

.. autoclass:: FeatureExtractor
    :members:

----

.. autoclass:: StructureCounting
    :members:

----

.. autoclass:: HeavyAtomExtractor
    :members:	       

----

.. autoclass:: RingExtractor
    :members:	       

----

.. autoclass:: AromaticRingExtractor
    :members:	       

----

.. autoclass:: StructureExtractor
    :members:	       

----

.. autoclass:: FingerPrintStructureExtractor       
    :members:

----

.. autoclass:: FeatureOperator
   :members:

----

.. autoclass:: FeatureSumOperator
   :members:	       

----

.. autofunction:: print_feature_extractor()

|
       
.. _prediction:

molgx.Prediction module
===============================

.. py:module:: molgx.Prediction

* :py:class:`RegressionModel`

  - :py:class:`SklearnRegressionModel`

    + :py:class:`SklearnLinearRegressionModel`

      + :py:class:`LinearRegressionModel`

      + :py:class:`RidgeRegressionModel`

      + :py:class:`LassoRegressionModel`

      + :py:class:`ElasticNetRegressionModel`

    + :py:class:`RandomForestRegressionModel`

* :py:class:`ModelSnapShot`

* :py:func:`print_regression_model()`


----

.. autoclass:: RegressionModel
    :members:

----

.. autoclass:: SklearnRegressionModel
    :members:

----

.. autoclass:: SklearnLinearRegressionModel
    :members:

----

.. autoclass:: LinearRegressionModel
    :members:

----

.. autoclass:: RidgeRegressionModel
    :members:

----

.. autoclass:: LassoRegressionModel
    :members:

----

.. autoclass:: ElasticNetRegressionModel
    :members:

----

.. autoclass:: RandomForestRegressionModel
    :members:

----

.. autoclass:: ModelSnapShot
    :members:

----

.. autofunction:: print_regression_model()

----

|

.. _feature_estimation:

molgx.FeatureEstimation module
======================================

.. py:module:: molgx.FeatureEstimation

* :py:class:`DesignParam`

* :py:class:`ComponentFixCondition`

* :py:class:`FeatureEstimator`
  
* :py:class:`FeatureEvaluator`

* :py:class:`StructuralConstraint`  

* :py:class:`FeasibleFingerPrintVector`  

* :py:class:`ParticleSwarmOptimization`

* :py:class:`FeatureEstimationResult`


----
	       
	       
.. autoclass:: DesignParam
    :members:	       

----

.. autoclass:: ComponentFixCondition
    :members:	       

----

.. autoclass:: FeatureEstimator
    :members:	       

----

.. autoclass:: FeatureEvaluator
    :members:	       

----

.. autoclass:: StructuralConstraint
    :members:	       

----

.. autoclass:: FeasibleFingerPrintVector
    :members:	       

----

.. autoclass:: ParticleSwarmOptimization
    :members:

----       

.. autoclass:: FeatureEstimationResult
    :members:

----

|

.. _generation:

molgx.Generation module
===============================

.. py:module:: molgx.Generation

* :py:class:`MoleculeGenerator`

----

.. autoclass:: MoleculeGenerator
    :members:

----

|

.. _utility:

molgx.Utility module
====================

.. py:module:: molgx.Utility

* :py:func:`fetch_QM9()`

* :py:func:`draw_molecules()`

* :py:func:`draw_rdkit_mols()`  

* :py:func:`draw_all_molecule()`

* :py:func:`draw_all_smiles()`

* :py:func:`draw_molecule_with_atom_index()`


----
  
.. autofunction:: fetch_QM9

----

.. autofunction:: draw_molecules

----

.. autofunction:: draw_rdkit_mols

----

.. autofunction:: draw_all_molecule

----

.. autofunction:: draw_all_smiles

----

.. autofunction:: draw_molecule_with_atom_index

