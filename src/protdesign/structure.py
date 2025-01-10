"""
Biomolecular structure-related functionality (PDB structures etc.)
"""
from typing import Dict, List

class StructureChain:
    """
    Single chain/entity in biomolecular structure
    """
    pass


class Structure:
    """
    Biomolecular structure comprised from one or multiple chains
    """
    pass

# allow to receive single chain, or map from identifier to single chain or list of chains
StructureChainMap = StructureChain | Dict[str, StructureChain | List[StructureChain]]