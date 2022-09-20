# -*- coding:utf-8 -*-
"""
ChemGraphGenPath.py

Package for IBM Molecule Generation Experience

MIT License

Copyright (c) 2022 International Business Machines Corporation

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
"""


from rdkit.Chem.rdchem import BondType

from .ChemGraph import *

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GenerationPath(object):
    """Class for representing generation path of a molecule.
    A generation path is a sequence of operations for adding atom/ring resource.
    """

    class Operation(object):
        """Class for an adding operation of a generation path

        Attributes:
            expand_index (int): vertex index of expanding ring resource
            connecting_index (int): vertex index of connecting position
            resource (GraphGenResourceManager.Resource): resource of adding vertex
            resource_connecting_index (int): vertex index of connecting position within a ring
        """

        def __init__(self, e_index, c_index, bond_type, resource, r_index):
            """Constructor of GenerationPath.Element

            Args:
                e_index (int): vertex index of expanding ring resource
                c_index (int): vertex index of connecting position
                bond_type (BondType): type of connecting bond
                resource (GraphGenResourceManager.Resource): resource of adding vertex
                r_index (int): vertex index of connecting position within a ring
            """
            self.expand_index = e_index
            self.connecting_index = c_index
            self.bond_type = bond_type
            self.resource = resource
            self.resource_connecting_index = r_index

        def to_string(self):
            """Get a string representation of an operation of a generation path

            Returns:
                str: string representation
            """
            op_str = '/'
            if self.expand_index is not None:
                op_str += 'e{0}:'.format(self.expand_index)
            if self.connecting_index is not None:
                op_str += '{0}{1}:'.format(self.connecting_index,
                                           ChemGraph.get_bond_char(self.bond_type))
            op_str += format(self.resource.get_symbol())
            if self.resource_connecting_index is not None:
                op_str += ':{0}'.format(self.resource_connecting_index)
            return op_str

    @staticmethod
    def op_sequence_to_str(op_sequence):
        """Get a string representation of an operation sequence

        Args:
            op_sequence (list): a list of operation sequence

        Returns:
            str: a string representation
        """
        gen_str = ''
        for operation in op_sequence:
            gen_str += operation.to_string()
        return gen_str
