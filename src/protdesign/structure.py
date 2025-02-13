"""
Biomolecular structure-related functionality (PDB structures etc.)

Thin wrapper around biotite structures for more convenient, unified access to PDBx and PDB formats; also
decouples internal codebase from biotite API through abstractions that we know work well from EVcouplings
package development.
"""
from typing import Literal, TextIO, BinaryIO, Self

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
import pandas as pd

# allow to receive single chain, or map from identifier to single chain or list of chains
StructureFormat = Literal["bcif", "cif", "pdb"]
_INVALID_FORMAT_MSG = "Invalid PDB file type, options are: 'bcif', 'cif', 'pdb'"


class Model:
    def __init__(self, atom_array: struc.AtomArray):
        self.atom_array = atom_array

        # dataframe representation
        self._df = None

    def df(
        self,
        sse: bool = True,
        sasa: bool = False,
    ):
        """
        Return dataframe representation of model

        Note: do not mutate returned dataframe without creating a copy

        Parameters
        ----------
        sse
            If true, compute column with secondary structure elements
            (will be retained on subsequent calls even if sse = False)
        sasa
            If true, add column solvent accessibility
            (will be retained on subsequent calls even if sasa = False)

        Returns
        -------
        Dataframe representation of model
        """
        # built dataframe if not already existring
        if self._df is None:
            cols = self.atom_array.get_annotation_categories()
            _df_raw = {
                col: self.atom_array.get_annotation(col) for col in cols
            }

            # unpack 3D coordinates into separate columns
            for i, col in enumerate(["x", "y", "z"]):
                _df_raw[col] = self.atom_array.coord[:, i]

            # replace masked values in custom fields not handled by biotite
            self._df = pd.DataFrame(
                _df_raw
            ).replace(
                {"?": pd.NA, ".": pd.NA}
            )

            # the following custom fields do not have mask values replaced by biotite, do this here
            for col in [
                "label_entity_id", "label_seq_id", "auth_seq_id"
            ]:
                if col in self._df.columns:
                    self._df.loc[:, col] = self._df.loc[:, col].astype("Int64")

        # annotate secondary structure, use 3-state DSSP nomenclature (even if different algorithm
        # used by biotite)
        if sse and "sse" not in self._df.columns:
            sse = pd.Series(
                struc.annotate_sse(self.atom_array)
            ).replace({
                "a": "H",
                "b": "E",
                "c": "C",
                "": pd.NA,
            })

            # TODO: have to use get_residues()
            # print(len(sse))
            # print(len(self._df))
            # print(len(struc.annotate_sse(self.atom_array)))
            # assert len(sse) == len(self._df)
            # self._df.loc[:, "sse"] = sse

            # TODO: separate atom and residue df?

        if sasa and "sasa" not in self._df.columns:
            print("compute sasa")
            # TODO: annotate_sse(struc)

        return self._df

    def chains(self):
        # return all available chains
        # TODO: implement
        raise NotImplementedError()

    def get_chain(
        self
    ):
        # TODO: get_chains(struc)
        # TODO: document
        # TODO: check chain is valid
        pass

    # def map_indices(self):
    #     # TODO: implement; how to handle multiple chains or only do single chain?
    #     # TODO: remove anything not mapped?
    #
    #     # TODO: raise ValueError if insertion codes are present
    #     raise NotImplementedError("Structure mapping not yet implemented")

    @classmethod
    def concat(
        cls,
        models: list[Self]
    ):
        """
        Create new model by concatenating given models

        Note: Caller is responsible for making sure there are no duplicated residues or chains

        Parameters
        ----------
        models
            Concatenate these models into new model

        Returns
        -------
        Concatenated model
        """
        return cls(
            struc.concatenate([
                model.atom_array for model in models
            ])
        )

    def to_file(
        self,
        file: TextIO | BinaryIO | str,
        format: StructureFormat="cif"  # noqa
    ) -> None:
        """
        Save model coordinates to a file

        Parameters
        ----------
        file
            File-like object or path to file
        format
            PDB format to write file as
        """
        if format == "cif":
            out_file = pdbx.CIFFile()
            pdbx.set_structure(out_file, self.atom_array)
        elif format == "bcif":
            out_file = pdbx.BinaryCIFFile()
            pdbx.set_structure(out_file, self.atom_array)
        elif format == "pdb":
            out_file = pdb.PDBFile()
            pdb.set_structure(out_file, self.atom_array)
        else:
            raise ValueError(_INVALID_FORMAT_MSG)

        out_file.write(file)


class Structure:
    """
    Biomolecular 3D structure
    """
    # extra fields that can be added for any structure type, we retrieve these by default
    _extra_fields = [
        "atom_id", "b_factor", "occupancy", "charge"
    ]

    # extra fields only available through CIF/PDBx formats, we also retrieve these by default even
    # if redundant to default fields extracted by biotite so we have all information available whenever needed
    _extra_fields_pdbx = _extra_fields + [
        "label_entity_id", "label_asym_id", "auth_asym_id", "label_seq_id", "auth_seq_id", "pdbx_PDB_ins_code"
    ]

    def __init__(
        self,
        file: TextIO | BinaryIO | str,
        format: StructureFormat,  # noqa
    ):
        """
        Load existing PDB structure

        Parameters
        ----------
        file
            Path or file handle to read structure from
        format
            Indicates whether provided structure is mmCIF ('cif'), binaryCIF ('bcif'),
            or legacy PDB format ('pdb')
        """

        if format == "bcif":
            self.data = pdbx.BinaryCIFFile.read(file)
            self.pdbx = True
        elif format == "cif":
            self.data = pdbx.CIFFile.read(file)
            self.pdbx = True
        elif format == "pdb":
            self.data = pdb.PDBFile.read(file)
            self.pdbx = False
        else:
            raise ValueError(_INVALID_FORMAT_MSG)

    @classmethod
    def from_id(cls, pdb_id: str):
        """
        Load structure by fetching from RCSB PDB

        Parameters
        ----------
        pdb_id
            4-letter PDB identifier code to fetch

        Returns
        -------
        Loaded structure
        """
        # fetch as bCIF by default for quicker fetching/loading
        pdb_data = rcsb.fetch(pdb_id, format="bcif")
        return cls(pdb_data, format="bcif")

    def assemblies(self) -> dict[str, str | None]:
        """
        List biological assemblies for structure

        Returns
        -------
        Mapping from assembly identifier to description (description
        will be None for old PDB format)
        """
        if self.pdbx:
            return pdbx.list_assemblies(self.data)
        else:
            # for pdb, we only get list of assembly identifiers, so turn into dictionary
            return {
                id_: None for id_ in pdb.list_assemblies(self.data)
            }

    def model_count(self) -> int:
        """
        Return number of models contained in structure file

        Returns
        -------
        Number of models
        """
        if self.pdbx:
            return pdbx.get_model_count(self.data)
        else:
            return pdb.get_model_count(self.data)

    def get_model(
        self,
        model: int = 1,
        altloc: Literal["first", "occupancy", "all"] = "occupancy",
        use_author_fields: bool = True,
        include_bonds: bool = False,
    ) -> Model:
        """
        Extract one model from asymmetric unit

        Parameters
        ----------
        model
            Number of model to extract (numbering starts from 1,
            check model_count() for total number of models)
        altloc
            Multiple location (altloc) per atom resolution strategy (see biotite documentation for details)
        use_author_fields
            If True, use author chain and residue numbering (possibly containing insertion codes),
            otherwise use label_seq_id and label_asym_id
        include_bonds
            If True, include bond list (see biotite documentation for details)

        Returns
        -------
        Extracted model
        """

        if not use_author_fields and not self.pdbx:
            raise ValueError(
                "Legacy PDB format only supports use_author_fields = True"
            )

        if self.pdbx:
            coords = pdbx.get_structure(
                self.data,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields_pdbx,
                use_author_fields=use_author_fields,
                include_bonds=include_bonds
            )
        else:
            coords = pdb.get_structure(
                self.data,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields,
                include_bonds=include_bonds
            )

        return Model(coords)

    def get_assembly_model(
        self,
        assembly_id: str | None = None,
        model: int = 1,
        altloc: Literal["first", "occupancy", "all"] = "occupancy",
        use_author_fields: bool = True,
        include_bonds: bool = False,
    ) -> Model:
        """
        Extract one model from biological assembly

        Parameters
        ----------
        assembly_id
            Assembly to extract, check for available assemblies with assemblies() method
        model
            Number of model to extract (numbering starts from 1,
            check model_count() for total number of models)
        altloc
            Multiple location (altloc) per atom resolution strategy (see biotite documentation for details)
        use_author_fields
            If True, use author chain and residue numbering (possibly containing insertion codes),
            otherwise use label_seq_id and label_asym_id
        include_bonds
            If True, include bond list (see biotite documentation for details)

        Returns
        -------
        Extracted model from biological assembly
        """
        if not use_author_fields and not self.pdbx:
            raise ValueError(
                "Legacy PDB format only supports use_author_fields = True"
            )

        if self.pdbx:
            coords = pdbx.get_assembly(
                self.data,
                assembly_id=assembly_id,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields_pdbx,
                use_author_fields=use_author_fields,
                include_bonds=include_bonds
            )
        else:
            coords = pdb.get_assembly(
                self.data,
                assembly_id=assembly_id,
                model=model,
                altloc=altloc,
                extra_fields=self._extra_fields,
                include_bonds=include_bonds
            )

        return Model(coords)

    def sequences(self) -> dict[str, str]:
        """
        Extract biopolymer chain sequences

        Returns
        -------
        Map from chain identifier to sequences
        """
        if self.pdbx:
            return {
                id_: str(seq) for id_, seq in pdbx.get_sequence(self.data).items()
            }
        else:
            raise NotImplementedError(
                "Sequences not available for legacy PDB format (not implemented in biotite)"
            )
