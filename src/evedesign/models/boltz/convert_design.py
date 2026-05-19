"""
Translation layer between evedesign data structures
and BoltzGen inputs/outputs.

This module converts evedesign System/SystemInstance
objects into BoltzGen design specification YAMLs and
parses BoltzGen's generated structures back into
SystemInstance objects.

NOTE: This module is the only place that knows about
BoltzGen file conventions. boltzgen.py calls these
functions without importing boltzgen directly.
"""

from evedesign.system import (
    Entity,
)


# ─── Entity classification ────────────────────────────


def _is_design_entity(entity: Entity) -> bool:
    """
    Determine whether an entity should be designed
    by BoltzGen.

    Returns True when:
    - rep is None (no sequence specified), OR
    - min_length / max_length are specified (length
      range for design)
    """
    if entity.rep is None:
        return True
    if (
        getattr(entity, "min_length", None) is not None
        or getattr(entity, "max_length", None) is not None
    ):
        return True
    return False
