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

from .graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext
from .mol_building_env import MolBuildingEnvContext


# convert to a factory/registry
def build_env_context(
    environment_name: str = "graph_building_env",
    context_name: str = "graph_building_env_context",
):

    """Builds an environment and context environment.

    Args:
        environment_name: The name of the environment to build.
        context_name: The name of the context environment to build.

    Returns:
        tuple with selected environment and context environment.
    """

    env, context = None, None
    if environment_name == "graph_building_env":
        env = GraphBuildingEnv

    if context_name == "graph_building_env_context":
        context = GraphBuildingEnvContext
    elif context_name == "mol_building_env_context":
        context = MolBuildingEnvContext

    return env, context
