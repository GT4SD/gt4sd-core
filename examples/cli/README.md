# GT4SD CLI
The command-line interface (CLI) provides access to four commands:
 - `gt4sd-inference`: To generate data with pretrained models.
 - `gt4sd-trainer`: To train (or finetune) generative models on custom data.
 - `gt4sd-saving`: To save a trained model so that it can be used via `gt4sd-inference`.
 - `gt4sd-upload`: Uploading a model to be used by others via `gt4sd-inference`.


## `gt4sd-inference`

To get started, you might want to have a look at all algorithms that you can use for inference.

```py
from gt4sd.algorithms.registry import ApplicationsRegistry
algorithms = ApplicationsRegistry.list_available()
for a in algorithms:
    print(a)
```

This will generate something like:

```txt
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'GuacaMolGenerator', 'algorithm_application': 'SMILESGAGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'GuacaMolGenerator', 'algorithm_application': 'GraphGAGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'GuacaMolGenerator', 'algorithm_application': 'GraphMCTSGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'GuacaMolGenerator', 'algorithm_application': 'SMILESLSTMHCGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'GuacaMolGenerator', 'algorithm_application': 'SMILESLSTMPPOGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'MosesGenerator', 'algorithm_application': 'AaeGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'MosesGenerator', 'algorithm_application': 'VaeGenerator', 'algorithm_version': 'fast-example-v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'MosesGenerator', 'algorithm_application': 'VaeGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'MosesGenerator', 'algorithm_application': 'OrganGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'nlp', 'algorithm_name': 'KeywordBERTGenerationAlgorithm', 'algorithm_application': 'KeyBERTGenerator', 'algorithm_version': 'distilbert-base-nli-mean-tokens'}
{'algorithm_type': 'conditional_generation', 'domain': 'nlp', 'algorithm_name': 'KeywordBERTGenerationAlgorithm', 'algorithm_application': 'KeyBERTGenerator', 'algorithm_version': 'circa_bert_v2'}
{'algorithm_type': 'conditional_generation', 'domain': 'nlp', 'algorithm_name': 'KeywordBERTGenerationAlgorithm', 'algorithm_application': 'KeyBERTGenerator', 'algorithm_version': 'circa_bert_v2_cls'}
{'algorithm_type': 'conditional_generation', 'domain': 'nlp', 'algorithm_name': 'KeywordBERTGenerationAlgorithm', 'algorithm_application': 'KeyBERTGenerator', 'algorithm_version': 'circa_bert_cls_v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'nlp', 'algorithm_name': 'KeywordBERTGenerationAlgorithm', 'algorithm_application': 'KeyBERTGenerator', 'algorithm_version': 'circa_scibert_cls'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'PaccMannRL', 'algorithm_application': 'PaccMannRLProteinBasedGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'PaccMannRL', 'algorithm_application': 'PaccMannRLOmicBasedGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'RegressionTransformer', 'algorithm_application': 'RegressionTransformerMolecules', 'algorithm_version': 'logp_and_synthesizability'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'RegressionTransformer', 'algorithm_application': 'RegressionTransformerMolecules', 'algorithm_version': 'pfas'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'RegressionTransformer', 'algorithm_application': 'RegressionTransformerMolecules', 'algorithm_version': 'solubility'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'RegressionTransformer', 'algorithm_application': 'RegressionTransformerMolecules', 'algorithm_version': 'qed'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'RegressionTransformer', 'algorithm_application': 'RegressionTransformerProteins', 'algorithm_version': 'stability'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'Reinvent', 'algorithm_application': 'ReinventGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'conditional_generation', 'domain': 'materials', 'algorithm_name': 'Template', 'algorithm_application': 'TemplateGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'controlled_sampling', 'domain': 'materials', 'algorithm_name': 'AdvancedManufacturing', 'algorithm_application': 'CatalystGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'controlled_sampling', 'domain': 'materials', 'algorithm_name': 'PaccMannGP', 'algorithm_application': 'PaccMannGPGenerator', 'algorithm_version': 'v1'}
{'algorithm_type': 'controlled_sampling', 'domain': 'materials', 'algorithm_name': 'PaccMannGP', 'algorithm_application': 'PaccMannGPGenerator', 'algorithm_version': 'v10'}
{'algorithm_type': 'controlled_sampling', 'domain': 'materials', 'algorithm_name': 'PaccMannGP', 'algorithm_application': 'PaccMannGPGenerator', 'algorithm_version': 'v11'}
{'algorithm_type': 'controlled_sampling', 'domain': 'materials', 'algorithm_name': 'PaccMannGP', 'algorithm_application': 'PaccMannGPGenerator', 'algorithm_version': 'v12'}
{'algorithm_type': 'controlled_sampling', 'domain': 'materials', 'algorithm_name': 'PaccMannGP', 'algorithm_application': 'PaccMannGPGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'HuggingFaceGenerationAlgorithm', 'algorithm_application': 'HuggingFaceXLMGenerator', 'algorithm_version': 'xlm-mlm-en-2048'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'HuggingFaceGenerationAlgorithm', 'algorithm_application': 'HuggingFaceCTRLGenerator', 'algorithm_version': 'ctrl'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'HuggingFaceGenerationAlgorithm', 'algorithm_application': 'HuggingFaceGPT2Generator', 'algorithm_version': 'gpt2'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'HuggingFaceGenerationAlgorithm', 'algorithm_application': 'HuggingFaceGPT2Generator', 'algorithm_version': 'circa-gpt2'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'HuggingFaceGenerationAlgorithm', 'algorithm_application': 'HuggingFaceOpenAIGPTGenerator', 'algorithm_version': 'openai-gpt'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'HuggingFaceGenerationAlgorithm', 'algorithm_application': 'HuggingFaceXLNetGenerator', 'algorithm_version': 'xlnet-large-cased'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'HuggingFaceGenerationAlgorithm', 'algorithm_application': 'HuggingFaceTransfoXLGenerator', 'algorithm_version': 'transfo-xl-wt103'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'MoLeR', 'algorithm_application': 'MoLeRDefaultGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'PGT', 'algorithm_application': 'PGTGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'PGT', 'algorithm_application': 'PGTEditor', 'algorithm_version': 'v0'}
{'algorithm_type': 'generation', 'domain': 'nlp', 'algorithm_name': 'PGT', 'algorithm_application': 'PGTCoherenceChecker', 'algorithm_version': 'v0'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'PolymerBlocks', 'algorithm_application': 'PolymerBlocksGenerator', 'algorithm_version': 'v0'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'TorchDrugGenerator', 'algorithm_application': 'TorchDrugGCPN', 'algorithm_version': 'qed_v0'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'TorchDrugGenerator', 'algorithm_application': 'TorchDrugGCPN', 'algorithm_version': 'zinc250k_v0'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'TorchDrugGenerator', 'algorithm_application': 'TorchDrugGCPN', 'algorithm_version': 'plogp_v0'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'TorchDrugGenerator', 'algorithm_application': 'TorchDrugGraphAF', 'algorithm_version': 'qed_v0'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'TorchDrugGenerator', 'algorithm_application': 'TorchDrugGraphAF', 'algorithm_version': 'zinc250k_v0'}
{'algorithm_type': 'generation', 'domain': 'materials', 'algorithm_name': 'TorchDrugGenerator', 'algorithm_application': 'TorchDrugGraphAF', 'algorithm_version': 'plogp_v0'}
{'algorithm_type': 'prediction', 'domain': 'materials', 'algorithm_name': 'PaccMann', 'algorithm_application': 'AffinityPredictor', 'algorithm_version': 'v0'}
{'algorithm_type': 'prediction', 'domain': 'nlp', 'algorithm_name': 'TopicsZeroShot', 'algorithm_application': 'TopicsPredictor', 'algorithm_version': 'dbpedia'}
```
