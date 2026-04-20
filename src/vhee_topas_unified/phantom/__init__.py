"""Phantom generation and materialization utilities."""

from .common import (
    add_tube_segment,
    capped_cylinder_along_y_mask,
    centered_axis_mm,
    combine_polylines,
    cylinder_along_y_mask,
    ellipsoid_mask,
    polyline_tube_mask,
)
from .detailed_headneck import (
    DENSITY_G_CM3,
    build_detailed_headneck_phantom,
    build_detailed_plan_phantom,
)
from .materials import (
    MATERIAL_SPECS,
    MaterialSpec,
    build_density_from_tags,
    build_material_tag_grid,
    render_materials_include,
    write_image_cube,
)
from .simple_headneck import (
    AIR_TAG,
    BONE_TAG,
    SOFT_TAG,
    build_simple_headneck_phantom,
)

__all__ = [
    "AIR_TAG",
    "BONE_TAG",
    "DENSITY_G_CM3",
    "MATERIAL_SPECS",
    "SOFT_TAG",
    "MaterialSpec",
    "add_tube_segment",
    "build_density_from_tags",
    "build_detailed_headneck_phantom",
    "build_detailed_plan_phantom",
    "build_material_tag_grid",
    "build_simple_headneck_phantom",
    "capped_cylinder_along_y_mask",
    "centered_axis_mm",
    "combine_polylines",
    "cylinder_along_y_mask",
    "ellipsoid_mask",
    "polyline_tube_mask",
    "render_materials_include",
    "write_image_cube",
]
