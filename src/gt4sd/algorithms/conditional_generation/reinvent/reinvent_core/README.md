<!--
MIT License

Copyright (c) 2022 GT4SD team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->
# MolecularAI Reinvent Code Explanation

The code for getting unique sample sequences, randomizing scaffolds, and generation of the dataset as well as the dataloader was taken from the original implementation of [Molecular Reinvent](https://github.com/MolecularAI/Reinvent) and can be found in the class [ReinventBase](/gt4sd/algorithms/conditional_generation/reinvent/reinvent_core/core.py) which is in the subdirectory **reinvent_core**.

We have a created a new function *get_dataloader* which is a modified version of the function *[run](https://github.com/MolecularAI/Reinvent/blob/982b26dd6cfeb8aa84b6d7e4a8c2a7edde2bad36/running_modes/lib_invent/rl_actions/sample_model.py#:~:text=def%20run(self%2C%20scaffold_list%3A%20List%5Bstr%5D)%20-%3E%20List%5BSampledSequencesDTO%5D%3A)* that returns an instance of the dataloader instead of the sampled sequences and it can be found in the [ReinventBase](/gt4sd/algorithms/conditional_generation/reinvent/reinvent_core/core.py) class.

Moreover, we have not included [BaseAction](https://github.com/MolecularAI/Reinvent/blob/982b26dd6cfeb8aa84b6d7e4a8c2a7edde2bad36/running_modes/lib_invent/rl_actions/sample_model.py#:~:text=class%20BaseAction(abc.ABC)%3A) as a parent class for the [ReinventBase](/gt4sd/algorithms/conditional_generation/reinvent/reinvent_core/core.py) where we have added all the functions of [Molecular Reinvent](https://github.com/MolecularAI/Reinvent). 