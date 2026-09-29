"""
ifc_parser.py
Extracts spaces, doors, columns and relationships from an IFC model.

"""

import os
import math
import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.placement
from shapely.geometry import Point, Polygon


# ── Settings ─────────────────────────────────────────────────────────────
def _make_settings():
    """
    Build ifcopenshell geometry settings with world coordinates enabled.

    Returns:
        ifcopenshell.geom.settings
    Called by: extract_all.
    """
    ifc_open_shell_settings = ifcopenshell.geom.settings()
    ifc_open_shell_settings.set(ifc_open_shell_settings.USE_WORLD_COORDS, True)
    return ifc_open_shell_settings


def load_ifc(ifc_path: str):
    """
    Open an IFC file.

    Args:
        ifc_path: path to the .ifc file.
    Returns:
        ifcopenshell model.
    Called by: main.py (run).
    """
    if not os.path.exists(ifc_path):
        raise FileNotFoundError(f"IFC file not found: {ifc_path}")
    return ifcopenshell.open(ifc_path)


# ── Unit ────────────────────────────────────────────────────────────────
def get_length_unit_scale(model) -> float:
    """
    Determine the scale factor to convert IFC numerical values to metres.

    Args:
        model: ifcopenshell model.
    Returns:
        float: scale factor (1.0 = already metres, 0.001 = millimetres, ...).
    Called by: extract_all, check_space_coverage.
    """
    try:
        unit_assignments = model.by_type("IfcUnitAssignment")
        for unit_assignment in unit_assignments:
            for unit in unit_assignment.Units:
                if not hasattr(unit, "UnitType"):
                    continue
                if unit.UnitType != "LENGTHUNIT":
                    continue
                prefix = getattr(unit, "Prefix", None)
                name   = getattr(unit, "Name",   None)
                if name == "METRE":
                    if prefix == "MILLI":  return 0.001
                    if prefix == "CENTI":  return 0.01
                    return 1.0
                if name == "FOOT":  return 0.3048
                if name == "INCH":  return 0.0254
    except Exception:
     return 0.001


def _apply_scale_if_needed(pts: list, scale: float) -> list:
    """
    Apply `scale` to a point list, but only when the coordinates still
    look like millimetres (max coordinate > 100 m).

    Args:
        pts: list of (x, y) points.
        scale: unit scale factor.
    Returns:
        list[tuple]: scaled points (or the input unchanged).
    Called by: get_space_footprint.
    """
    if not pts or scale == 1.0:
        return pts
    if max(abs(p[0]) for p in pts) > 100:
        return [(p[0] * scale, p[1] * scale) for p in pts]
    return pts


# ── IFC property helpers ──────────────────────────────────────────────────
def get_property(element, pset_name: str, property_name: str):
    """
    Read one property value from an IFC element's property sets.

    Args:
        element: an IFC entity.
        pset_name: property set name to match, or "any" to match all.
        property_name: property name to read.
    Returns:
        The property's NominalValue, or None if not found.
    Called by: _is_flagged_obstacle, get_ramps, is_fire_exit_door, extract_all.
    """
    try:
        for relation in element.IsDefinedBy:
            if relation.is_a("IfcRelDefinesByProperties"):
                pset = relation.RelatingPropertyDefinition
                if pset.is_a("IfcPropertySet"):
                    if pset_name == "any" or pset.Name == pset_name:
                        for property in pset.HasProperties:
                            if property.Name == property_name:
                                if hasattr(property, "NominalValue") and property.NominalValue:
                                    return property.NominalValue.wrappedValue
    except Exception:
        pass
    return None

# ── Geometry helpers (curves) ──────────────────────────────────────────────

def _apply_transform(matrix, x: float, y: float) -> tuple:
    """
    Transform a local (x, y) point by a 4x4 placement matrix into world
    coordinates.

    Args:
        matrix: 4x4 placement matrix.
        x, y: local coordinates.
    Returns:
        tuple: (world_x, world_y).
    Called by: _read_indexed_poly_curve, _read_trimmed_curve, _profile_to_points.
    """
    wx = matrix[0][0]*x + matrix[0][1]*y + matrix[0][3]
    wy = matrix[1][0]*x + matrix[1][1]*y + matrix[1][3]
    return (wx, wy)


def _interpolate_arc(p1, pmid, p2, n=8) -> list:
    """
    Interpolate an arc through three points (start, mid, end) into a
    polyline of n+1 points.

    Args:
        p1, pmid, p2: (x, y) points defining the arc.
        n: number of segments.
    Returns:
        list[tuple]: interpolated (x, y) points.
    Called by: _read_indexed_poly_curve.
    """
    ax, ay = p1; bx, by = pmid; cx, cy = p2
    D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
    if abs(D) < 1e-10:
        return [p1, p2]

    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    r  = np.sqrt((ax-ux)**2 + (ay-uy)**2)
    a0 = np.arctan2(ay-uy, ax-ux)
    am = np.arctan2(by-uy, bx-ux)
    a1 = np.arctan2(cy-uy, cx-ux)
    ccw_0m = (am-a0) % (2*np.pi)
    ccw_01 = (a1-a0) % (2*np.pi)
    if ccw_0m < ccw_01:
        angles = np.linspace(a0, a0+ccw_01, n+1)
    else:
        angles = np.linspace(a0, a0-(2*np.pi-ccw_01), n+1)
    return [(ux+r*np.cos(a), uy+r*np.sin(a)) for a in angles]


def _read_indexed_poly_curve(curve, matrix) -> list:
    """
    Read an IfcIndexedPolyCurve into a list of world-coordinate points,
    interpolating any arc segments.

    Args:
        curve: an IfcIndexedPolyCurve.
        matrix: world placement matrix.
    Returns:
        list[tuple]: (x, y) points, or [] if not an IfcIndexedPolyCurve.
    Called by: _profile_to_points.
    """
    if not curve.is_a("IfcIndexedPolyCurve"):
        return []
    coord_list = curve.Points.CoordList
    def _pt(idx):
        x, y = coord_list[idx-1]
        return _apply_transform(matrix, float(x), float(y))
    segments = curve.Segments
    if not segments:
        return [_pt(i+1) for i in range(len(coord_list))]
    pts = []
    for seg in segments:
        indices = list(seg.wrappedValue)
        if seg.is_a("IfcLineIndex"):
            for idx in indices[:-1]:
                pts.append(_pt(idx))
        elif seg.is_a("IfcArcIndex") and len(indices) >= 3:
            arc_pts = _interpolate_arc(_pt(indices[0]), _pt(indices[1]), _pt(indices[2]))
            pts.extend(arc_pts[:-1])
    return pts


def _read_trimmed_curve(curve, matrix, n=8) -> list:
    """
    Read an IfcTrimmedCurve (circular arc or line segment) into world
    coordinate points.

    Args:
        curve: an IfcTrimmedCurve.
        matrix: world placement matrix.
        n: number of segments when the basis curve is a circle.
    Returns:
        list[tuple]: (x, y) points, or [] on any unsupported/malformed input.
    Called by: _profile_to_points.
    """
    try:
        basis = curve.BasisCurve
        trim1 = curve.Trim1
        trim2 = curve.Trim2
        same  = curve.SenseAgreement
        if basis.is_a("IfcCircle"):
            centre = basis.Position
            loc    = centre.Location.Coordinates if hasattr(centre, "Location") else None
            cx_w   = _apply_transform(matrix, float(loc[0]), float(loc[1])) if loc else _apply_transform(matrix, 0, 0)
            cx_w, cy_w = cx_w[0], cx_w[1]
            scale  = np.sqrt(matrix[0][0]**2 + matrix[1][0]**2)
            r      = float(basis.Radius) * scale
            def _ang(ts):
                for t in ts:
                    try:
                        if hasattr(t, "wrappedValue"):
                            return np.radians(float(t.wrappedValue))
                        if hasattr(t, "Coordinates"):
                            px, py = float(t.Coordinates[0]), float(t.Coordinates[1])
                            wx, wy = _apply_transform(matrix, px, py)
                            return np.arctan2(wy-cy_w, wx-cx_w)
                    except Exception:
                        continue
                return 0.0
            a0 = _ang(trim1); a1 = _ang(trim2)
            if not same: a0, a1 = a1, a0
            if a1 < a0: a1 += 2*np.pi
            return [(cx_w+r*np.cos(a), cy_w+r*np.sin(a))
                    for a in np.linspace(a0, a1, n+1)]
        elif basis.is_a("IfcLine"):
            pts = []
            for ts in [trim1, trim2]:
                for t in ts:
                    try:
                        if hasattr(t, "Coordinates"):
                            pts.append(_apply_transform(matrix, float(t.Coordinates[0]), float(t.Coordinates[1])))
                            break
                    except Exception:
                        continue
            return pts if len(pts) == 2 else []
    except Exception:
        pass
    return []


def _profile_to_points(profile, matrix) -> list:
    """
    Convert an IFC profile definition (arbitrary closed profile, composite
    curve, rectangle or circle) into world-coordinate 2D points.

    Args:
        profile: an IFC profile definition entity.
        matrix: world placement matrix.
    Returns:
        list[tuple]: (x, y) points, or [] for an unsupported profile type.
    Called by: get_space_footprint.
    """
    ptype = profile.is_a()
    if ptype in ("IfcArbitraryClosedProfileDef", "IfcArbitraryProfileDefWithVoids"):
        curve = profile.OuterCurve
        if curve.is_a("IfcIndexedPolyCurve"):
            return _read_indexed_poly_curve(curve, matrix)
        if curve.is_a("IfcPolyline"):
            return [_apply_transform(matrix, float(p.Coordinates[0]), float(p.Coordinates[1]))
                    for p in curve.Points]
        if curve.is_a("IfcCompositeCurve"):
            pts = []
            for seg in curve.Segments:
                sub  = seg.ParentCurve
                same = getattr(seg, "SameSense", True)
                if sub.is_a("IfcIndexedPolyCurve"):
                    sp = _read_indexed_poly_curve(sub, matrix)
                elif sub.is_a("IfcPolyline"):
                    sp = [_apply_transform(matrix, float(p.Coordinates[0]), float(p.Coordinates[1]))
                          for p in sub.Points]
                elif sub.is_a("IfcTrimmedCurve"):
                    sp = _read_trimmed_curve(sub, matrix)
                else:
                    continue
                pts.extend(sp if same else sp[::-1])
            return pts
        if curve.is_a("IfcTrimmedCurve"):
            return _read_trimmed_curve(curve, matrix)
    elif ptype == "IfcRectangleProfileDef":
        w, h = profile.XDim/2.0, profile.YDim/2.0
        return [_apply_transform(matrix, x, y) for x, y in [(-w,-h),(w,-h),(w,h),(-w,h)]]
    elif ptype == "IfcCircleProfileDef":
        r = profile.Radius
        return [_apply_transform(matrix, r*np.cos(a), r*np.sin(a))
                for a in np.linspace(0, 2*np.pi, 16, endpoint=False)]
    return []


# ── Space footprint ────────────────────────────────────────────────────────
def get_space_footprint(space, settings, scale: float) -> list:
    """
    Extract the 2D footprint polygon of an IfcSpace, in world metres.
    Tries the space's own extruded-solid profile first (accurate, handles
    curves); falls back to projecting the lowest mesh face if that fails.

    Args:
        space: an IfcSpace.
        settings: ifcopenshell.geom settings (see _make_settings).
        scale: unit scale factor.
    Returns:
        list[tuple]: (x, y) footprint points, or [] if none could be extracted.
    Called by: extract_all, check_space_coverage.
    """
    try:
        rep = space.Representation
        if rep:
            for sub_rep in rep.Representations:
                for item in sub_rep.Items:
                    solid = item
                    while solid.is_a("IfcBooleanClippingResult"):
                        solid = solid.FirstOperand
                    if not solid.is_a("IfcExtrudedAreaSolid"):
                        continue
                    profile  = solid.SweptArea
                    spacematrix   = ifcopenshell.util.placement.get_local_placement(space.ObjectPlacement)
                    sp_pos   = solid.Position
                    if sp_pos:
                        worldmatrix = spacematrix @ ifcopenshell.util.placement.get_axis2placement(sp_pos)
                    else:
                        worldmatrix = spacematrix
                    pts = _profile_to_points(profile, worldmatrix)
                    pts = _apply_scale_if_needed(pts, scale)
                    if len(pts) >= 3:
                        return pts
    except Exception:
        pass
    try:
        shape  = ifcopenshell.geom.create_shape(settings, space)
        verts  = shape.geometry.verts
        pts_3d = [(verts[i], verts[i+1], verts[i+2]) for i in range(0, len(verts), 3)]
        if not pts_3d:
            return []
        z_vals  = sorted(set(round(p[2], 3) for p in pts_3d))
        floor_z = z_vals[0]
        floor_pts = list(set((p[0], p[1]) for p in pts_3d if abs(p[2]-floor_z) < 0.001))
        return floor_pts
    except Exception:
        return []


def get_space_centroid(footprint: list) -> tuple:
    """
    Average (centroid) of a footprint's points.

    Args:
        footprint: list of (x, y) points.
    Returns:
        tuple: (x, y), or (0.0, 0.0) if footprint is empty.
    Called by: extract_all.
    """
    if not footprint:
        return (0.0, 0.0)
    return (sum(p[0] for p in footprint)/len(footprint),
            sum(p[1] for p in footprint)/len(footprint))


def get_element_location_geometry(element, settings) -> tuple:
    """
    Bounding-box centre of an element's mesh (a rough centre point).

    Deliberately NOT a raw average of every vertex: that average is
    skewed toward whichever side of the mesh happens to be more densely
    triangulated (e.g. door hardware/frame detail concentrated on one
    side), even though the mesh's true extent is symmetric. Confirmed on
    5 real doors — the vertex average sat consistently 36-37 cm away from
    this bounding-box centre.

    Args:
        element: an IFC entity.
        settings: ifcopenshell.geom settings.
    Returns:
        tuple: (x, y, z), or (0.0, 0.0, 0.0) on failure.
    Called by: extract_all (door locations).
    """
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = shape.geometry.verts
        if verts:
            xs = verts[0::3]; ys = verts[1::3]; zs = verts[2::3]
            return ((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2)
    except Exception:
        pass
    return (0.0, 0.0, 0.0)


def get_element_location_placement(element, scale: float) -> tuple:
    """
    An element's placement origin (not a geometric centroid).

    Args:
        element: an IFC entity.
        scale: unit scale factor.
    Returns:
        tuple: (x, y, z) in metres, or (0.0, 0.0, 0.0) on failure.
    Called by: extract_all (space location fallback when no footprint).
    """
    try:
        m = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
        return (m[0][3]*scale, m[1][3]*scale, m[2][3]*scale)
    except Exception:
        return (0.0, 0.0, 0.0)


def get_storey_elevation(storey, scale: float) -> float:
    """
    An IfcBuildingStorey's elevation, in metres.

    Args:
        storey: an IfcBuildingStorey.
        scale: unit scale factor.
    Returns:
        float: elevation in metres, or 0.0 if unavailable.
    Called by: get_columns, get_obstacles, get_mep_obstacles, get_ramps,
        extract_all, check_space_coverage.
    """
    try:
        if hasattr(storey, "Elevation") and storey.Elevation is not None:
            return float(storey.Elevation) * scale
    except Exception:
        pass
    return 0.0


# ── Columns ─────────────────────────────────────────────────────────────
def get_columns(model, settings, scale: float) -> list:
    """
    Extract IfcColumn footprint polygons as obstacle zones for pathfinding.

    The plan footprint is the union of every projected mesh triangle at
    the column's lowest z (handles concave cross-sections, e.g. L-shaped
    "Angle" columns, which a convex hull alone would fill in incorrectly).
    Falls back to the convex hull of the projected points if that
    projection does not yield a valid polygon, so no column is dropped.

    Args:
        model: ifcopenshell model.
        settings: ifcopenshell.geom settings.
        scale: unit scale factor.
    Returns:
        list[dict]: {id, name, footprint, storey_elevation_m, z_min_m,
            z_max_m} — z_min_m/z_max_m are the column's real vertical
            extent, for matching against every floor it physically spans
            (see RT_SCIE_checker._columns_on_floor / visualizer's
            _draw_ramp_context), not just its base storey.
    Called by: extract_all.
    """
    columns = []
    storey_elevation_of = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        for element in rel.RelatedElements:
            if element.is_a("IfcColumn"):
                structure = rel.RelatingStructure
                if structure.is_a("IfcBuildingStorey"):
                    storey_elevation_of[element.GlobalId] = \
                        get_storey_elevation(structure, scale)

    for column in model.by_type("IfcColumn"):
        try:
            shape = ifcopenshell.geom.create_shape(settings, column)
            verts = shape.geometry.verts
            faces = shape.geometry.faces
            points_3d = [(verts[i], verts[i+1], verts[i+2])
                         for i in range(0, len(verts), 3)]
            if not points_3d:
                continue
            z_values  = sorted(set(round(p[2], 3) for p in points_3d))
            floor_z   = z_values[0]
            from shapely.geometry import Polygon as _ShpPoly, MultiPoint
            from shapely.ops import unary_union

            # Fallback footprint, used only if the accurate projection below
            # fails: the convex hull of every projected point.
            hull_fallback = MultiPoint([(p[0], p[1]) for p in points_3d]).convex_hull

            footprint_poly = None
            try:
                triangles = []
                for f in range(0, len(faces or []), 3):
                    a, b, c = faces[f], faces[f+1], faces[f+2]
                    tri = _ShpPoly([(verts[3*a], verts[3*a+1]),
                                    (verts[3*b], verts[3*b+1]),
                                    (verts[3*c], verts[3*c+1])])
                    if tri.is_valid and tri.area > 1e-9:
                        triangles.append(tri)
                if triangles:
                    merged = unary_union(triangles)
                    if merged.geom_type == "MultiPolygon":
                        merged = max(merged.geoms, key=lambda g: g.area)
                    if merged.geom_type == "Polygon" and not merged.is_empty:
                        footprint_poly = merged
            except Exception:
                footprint_poly = None

            if footprint_poly is None:
                if hull_fallback.geom_type != "Polygon" or hull_fallback.is_empty:
                    continue
                footprint_poly = hull_fallback

            elevation = storey_elevation_of.get(column.GlobalId, floor_z)
            columns.append({
                "id":                column.GlobalId,
                "name":              column.Name or "Column",
                "footprint":         list(footprint_poly.exterior.coords),
                "storey_elevation_m": elevation,
                # Real vertical extent, in world coords — a column is often
                # IFC-contained in only its base storey even though it
                # physically runs through several floors above that (e.g. a
                # full-height column from Ground to Roof). Floor-filtering
                # code should check whether a floor's elevation falls in
                # this span, not just match storey_elevation_m.
                "z_min_m":           z_values[0],
                "z_max_m":           z_values[-1],
            })
        except Exception:
            continue

    return columns


# The architect marks fixed furniture that blocks evacuation with this
# property (any Pset) set to Yes/True/1. Only tagged furniture is treated
# as an obstacle; movable furniture is left out to avoid false NO-PATH
# results.
OBSTACLE_FLAG_PROPERTY = "Egress_Obstacle"


def _is_flagged_obstacle(element) -> bool:
    """
    Check whether an element carries Egress_Obstacle = Yes/True/1.

    Args:
        element: an IFC entity.
    Returns:
        bool
    Called by: get_obstacles.
    """
    val = get_property(element, "any", OBSTACLE_FLAG_PROPERTY)
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("yes", "true", "1")
    return val is True or val == 1


def get_obstacles(model, settings, scale: float) -> list:
    """
    Extract footprint polygons of fixed furniture tagged as evacuation
    obstacles (Egress_Obstacle = Yes). Mirrors get_columns: convex hull of
    the lowest face is the plan footprint. Targets IfcFurnishingElement,
    which in IFC4 also covers IfcFurniture and IfcSystemFurnitureElement.
    Untagged furniture is ignored.

    Args:
        model: ifcopenshell model.
        settings: ifcopenshell.geom settings.
        scale: unit scale factor.
    Returns:
        list[dict]: {id, name, footprint, storey_elevation_m}
    Called by: extract_all.
    """
    obstacles = []
    storey_elevation_of = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        for element in rel.RelatedElements:
            if element.is_a("IfcFurnishingElement"):
                structure = rel.RelatingStructure
                if structure.is_a("IfcBuildingStorey"):
                    storey_elevation_of[element.GlobalId] = \
                        get_storey_elevation(structure, scale)

    for furniture in model.by_type("IfcFurnishingElement"):
        if not _is_flagged_obstacle(furniture):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, furniture)
            verts = shape.geometry.verts
            points_3d = [(verts[i], verts[i+1], verts[i+2])
                         for i in range(0, len(verts), 3)]
            if not points_3d:
                continue
            z_values  = sorted(set(round(p[2], 3) for p in points_3d))
            floor_z   = z_values[0]
            # Furniture obstructs at its widest point (a table's lowest
            # face is just its leg tips), so the plan footprint uses the
            # full vertical projection of every vertex, not only the
            # lowest-face points used for columns.
            proj_points = list(set((p[0], p[1]) for p in points_3d))
            if len(proj_points) < 3:
                continue
            from shapely.geometry import MultiPoint
            hull = MultiPoint(proj_points).convex_hull
            if hull.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            elevation = storey_elevation_of.get(furniture.GlobalId, floor_z)
            obstacles.append({
                "id":                 furniture.GlobalId,
                "name":               furniture.Name or "Furniture",
                "footprint":          list(hull.exterior.coords),
                "storey_elevation_m": elevation,
            })
        except Exception:
            continue

    return obstacles


def get_mep_obstacles(model, settings, scale: float) -> list:
    """
    Extract footprint polygons of MEP / plumbing fixtures tagged as
    evacuation obstacles. Mirrors get_obstacles: the same
    Egress_Obstacle = Yes/True/1 tag (any Pset) is required, so an
    untagged toilet, sink or piece of equipment is left out exactly like
    untagged furniture, avoiding false NO-PATH results from fixtures that
    don't actually block egress. IfcDistributionElement is the IFC4
    superclass covering IfcFlowTerminal, IfcFlowSegment,
    IfcFlowController and their subtypes.

    Uses the same projected-triangle-union footprint method as
    get_columns (handles concave shapes like a toilet bowl), falling back
    to the convex hull of all vertices if that projection fails.

    Args:
        model: ifcopenshell model.
        settings: ifcopenshell.geom settings.
        scale: unit scale factor.
    Returns:
        list[dict]: {id, name, footprint, storey_elevation_m}
    Called by: extract_all.
    """
    obstacles = []
    storey_elevation_of = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        for element in rel.RelatedElements:
            if element.is_a("IfcDistributionElement"):
                structure = rel.RelatingStructure
                if structure.is_a("IfcBuildingStorey"):
                    storey_elevation_of[element.GlobalId] = \
                        get_storey_elevation(structure, scale)

    from shapely.geometry import Polygon as _ShpPoly, MultiPoint
    from shapely.ops import unary_union

    for fixture in model.by_type("IfcDistributionElement"):
        if not _is_flagged_obstacle(fixture):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, fixture)
            verts = shape.geometry.verts
            faces = shape.geometry.faces
            points_3d = [(verts[i], verts[i+1], verts[i+2])
                         for i in range(0, len(verts), 3)]
            if not points_3d:
                continue
            z_values = sorted(set(round(p[2], 3) for p in points_3d))
            floor_z  = z_values[0]

            hull_fallback = MultiPoint([(p[0], p[1])
                                        for p in points_3d]).convex_hull

            footprint_poly = None
            try:
                triangles = []
                for f in range(0, len(faces or []), 3):
                    a, b, c = faces[f], faces[f+1], faces[f+2]
                    tri = _ShpPoly([(verts[3*a], verts[3*a+1]),
                                    (verts[3*b], verts[3*b+1]),
                                    (verts[3*c], verts[3*c+1])])
                    if tri.is_valid and tri.area > 1e-9:
                        triangles.append(tri)
                if triangles:
                    merged = unary_union(triangles)
                    if merged.geom_type == "MultiPolygon":
                        merged = max(merged.geoms, key=lambda g: g.area)
                    if merged.geom_type == "Polygon" and not merged.is_empty:
                        footprint_poly = merged
            except Exception:
                footprint_poly = None

            if footprint_poly is None:
                if hull_fallback.geom_type != "Polygon" or hull_fallback.is_empty:
                    continue
                footprint_poly = hull_fallback

            elevation = storey_elevation_of.get(fixture.GlobalId, floor_z)
            obstacles.append({
                "id":                 fixture.GlobalId,
                "name":               fixture.Name or "MEP element",
                "footprint":          list(footprint_poly.exterior.coords),
                "storey_elevation_m": elevation,
            })
        except Exception:
            continue

    return obstacles


def get_ramps(model, settings, scale: float) -> tuple:
    """
    Extract IfcRamp elements for the DL 163/2006 accessibility ramp check,
    one entry per flight (lanço) — the unit Anexo 2.5.1 regulates — plus a
    separate, visualization-only entry per landing.

    Scope: only ramps carrying Pset_RampCommon.HandicapAccessible = True
    are measured. A ramp with the flag False, or missing, is skipped here
    before any measurement is made.

    Flights and landings are told apart by decomposing IfcRelAggregates
    children, considering only children of type IfcRampFlight or IfcSlab —
    a ramp's decomposition can carry either or both for the same physical
    run (confirmed on a real export where one flight was typed
    IfcRampFlight and the next, otherwise identical in kind, was typed
    IfcSlab), and can also include IfcRailing (guardrails), never a
    walking surface and always excluded. A ramp with no IfcRampFlight/
    IfcSlab children is treated as a single flight, using its own geometry
    directly. A child is a LANDING if its own top-surface rise is under
    2 cm, otherwise a FLIGHT.

    Each flight is measured independently:
        rise_m       : vertical height climbed by the top walking surface.
        projection_m : horizontal run measured along the FALL LINE (the
                       direction of steepest ascent on the top surface),
                       not the flight's bounding box — a turning flight
                       (e.g. L-shaped) can run along a different axis than
                       the ramp's overall footprint, and the bounding
                       box's max(extent_x, extent_y) would mix the
                       flight's own run with the landing's perpendicular
                       offset.
        slope_pct    : 100 * rise_m / projection_m.
        width_m      : the shorter plan dimension of the flight.
        corners      : a 4-point 3D quad approximating the flight's walking
                       surface (real world coordinates), for drawing — see
                       visualizer.generate_violation_snapshot.
        entity_id    : GlobalId of the real IFC entity this flight came
                       from (the decomposed child, or the ramp itself in
                       the fallback case) — the "id" field above is a
                       synthetic composite (ramp GlobalId + flight index)
                       that does not exist as a model element, so a BCF
                       viewpoint must select by entity_id, not by "id".

    Landing size is not checked (Anexo 2.5.5-2.5.6 rest-platform sizing is
    out of scope) and a landing is never a compliance unit under Anexo
    2.5.1, so landings are kept in a SEPARATE return list — geometry only
    (id, ramp_id, entity_id, name, corners, storey info), for drawing a
    landing alongside its ramp's flights without it ever reaching the
    slope check.
    Curved-flight radius (Anexo 2.5.3) is out of scope; the fall-line
    projection assumes a single straight sloped plane per flight.

    Args:
        model: ifcopenshell model.
        settings: ifcopenshell.geom settings.
        scale: unit scale factor.
    Returns:
        tuple: (flights, landings)
            flights: list[dict], one entry per flight (geometry only — no
                compliance decision is made here).
            landings: list[dict], one entry per landing (visualization
                only — never checked against any rule).
    Called by: extract_all.
    """
    ramps = []
    landings_out = []

    storey_map = {}       # ramp GlobalId -> elevation
    storey_name_map = {}  # ramp GlobalId -> storey name
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        for el in rel.RelatedElements:
            if el.is_a("IfcRamp"):
                st = rel.RelatingStructure
                if st.is_a("IfcBuildingStorey"):
                    storey_map[el.GlobalId] = get_storey_elevation(st, scale)
                    storey_name_map[el.GlobalId] = st.Name or None

    def _segment_points(entity):
        """World-coordinate (x,y,z) points of one entity's mesh, or []."""
        try:
            shape = ifcopenshell.geom.create_shape(settings, entity)
            verts = shape.geometry.verts
            return [(verts[i], verts[i+1], verts[i+2])
                    for i in range(0, len(verts), 3)]
        except Exception:
            return []

    def _top_surface(pts_3d):
        """Keep only the highest z at each 5 cm plan cell (top walking surface)."""
        top_z_by_cell = {}
        for (px, py, pz) in pts_3d:
            cell = (round(px / 0.05), round(py / 0.05))
            if cell not in top_z_by_cell or pz > top_z_by_cell[cell][2]:
                top_z_by_cell[cell] = (px, py, pz)
        return list(top_z_by_cell.values())

    def _z_bands(top_pts, gap_tol=0.02):
        """
        Group top-surface points into bands by z: sorted ascending, a new
        band starts wherever the step to the next point exceeds `gap_tol`
        (2cm, matching the landing-vs-flight rise threshold elsewhere).

        A single decomposed child can carry a landing directly fused to a
        short rise in one mesh (observed on a real model: one IfcSlab
        whose top surface was five points flat at one z — a genuine
        landing-sized plateau — plus two points at a second z only ~11cm
        higher). Grouping by z-gap recovers that structure so the landing
        can be reported as its own entry and the flight measured between
        the two ends, instead of one misshapen "flight" spanning both.

        Returns:
            list[list[tuple]]: bands, ascending by z.
        """
        pts_sorted = sorted(top_pts, key=lambda p: p[2])
        bands = [[pts_sorted[0]]]
        for p in pts_sorted[1:]:
            if p[2] - bands[-1][-1][2] > gap_tol:
                bands.append([p])
            else:
                bands[-1].append(p)
        return bands

    def _band_plan_area(band):
        """Plan (x,y) bounding-box area of a band's points, 0 if fewer than 3."""
        if len(band) < 3:
            return 0.0
        xs = [p[0] for p in band]; ys = [p[1] for p in band]
        return (max(xs)-min(xs)) * (max(ys)-min(ys))

    def _band_center(band):
        """Plan (x,y) bounding-box CENTRE of a band, at its own (near-constant)
        z — robust against an uneven number of sample points per band,
        unlike a raw coordinate average, which a lopsided or duplicated
        interior sample point can pull off-centre."""
        xs = [p[0] for p in band]; ys = [p[1] for p in band]
        z = sum(p[2] for p in band) / len(band)
        return ((min(xs)+max(xs))/2.0, (min(ys)+max(ys))/2.0, z)

    def _nearest_edge(band, reference_xy):
        """
        The half of `band`'s points closest (in plan) to `reference_xy`.

        Used when a flight's low end is really the near edge of a wider
        embedded landing (see _z_bands): the landing's own full extent —
        both its near and far edge — is not where the flight begins, only
        the near edge is. Keeps at least 2 points.
        """
        if len(band) <= 2:
            return band
        ranked = sorted(band, key=lambda p: (p[0]-reference_xy[0])**2
                                            + (p[1]-reference_xy[1])**2)
        keep_n = max(2, -(-len(band) // 2))  # ceil(len/2), at least 2
        return ranked[:keep_n]

    def _measure_flight(lo_band, hi_band):
        """
        rise/projection/width/slope for one flight, plus a 4-point 3D quad
        (corners) approximating its walking surface for drawing: a
        rectangle from the low end to the high end, offset by half the
        width on each side, perpendicular to the fall line. Not a
        substitute for the true mesh (real ramps can be curved or have a
        non-rectangular outline) — good enough to show a flight's real
        position, slope and extent in an illustration.

        Three failure modes, all confirmed on a real model, are guarded
        against when turning the low/high bands (see _z_bands) into a
        single lo_point/hi_point pair:
          - min()/max() on the whole point set: a flight's low (and high)
            edge is usually two points tied at the same z — the two
            corners of that edge — and picking one arbitrary point from
            each tied pair, rather than the edge's centre, fabricates a
            diagonal fall line for a perfectly straight, axis-aligned
            flight. Fixed by using each band's own bounding-box centre
            instead of a single min()/max() point.
          - a raw coordinate average of a band with an uneven point count
            (e.g. an extra duplicate/interior sample at one end) pulls
            that end's average off the true rectangle centre. Fixed by
            using the bounding-box centre, immune to point count.
          - the low and high ends genuinely differing in width (e.g. an
            embedded landing wider than the flight it feeds, split out by
            the caller): comparing each end's own bounding-box CENTRE is
            still wrong here, because the two ends' centres are offset by
            half of whatever the width difference is, fabricating a small
            tilt even though the flight itself runs straight. The real
            signal is that one side of the two ends (call it left or
            right, whichever axis) lines up exactly while the other
            tapers — confirmed on a real model where the low end's near
            edge was 1.35m wide against a 1.2m high end, yet their
            minimum-x sides matched to the millimetre. Fixed by anchoring
            the flight's centreline to whichever side (min or max, on
            whichever axis) matches best between the two ends, rather
            than trusting each end's own independently-computed centre.

        Args:
            lo_band, hi_band: point lists from _z_bands (already the two
                ends this flight runs between — a multi-band segment may
                also contain an embedded landing, handled by the caller,
                which also narrows a landing-derived lo_band to just its
                near edge via _nearest_edge before calling this).
        """
        lo_center = _band_center(lo_band)
        hi_center = _band_center(hi_band)
        rise_m = hi_center[2] - lo_center[2]

        # Width from the high end alone: in every case observed the low
        # end is the one that can also be a landing's own (wider) edge,
        # so it is not a reliable width reference; the high end is always
        # the flight's own simple, unambiguous edge.
        hi_xs = [p[0] for p in hi_band]; hi_ys = [p[1] for p in hi_band]
        width_m = max(max(hi_xs)-min(hi_xs), max(hi_ys)-min(hi_ys))
        if width_m < 1e-6:
            lo_xs = [p[0] for p in lo_band]; lo_ys = [p[1] for p in lo_band]
            width_m = max(max(lo_xs)-min(lo_xs), max(lo_ys)-min(lo_ys))

        # Anchor the centreline to whichever side matches best between the
        # two ends (see docstring) instead of each end's own centre.
        lo_xs = [p[0] for p in lo_band]; hi_xs = [p[0] for p in hi_band]
        lo_ys = [p[1] for p in lo_band]; hi_ys = [p[1] for p in hi_band]
        candidates = [
            ("x", "min", min(lo_xs), min(hi_xs)),
            ("x", "max", max(lo_xs), max(hi_xs)),
            ("y", "min", min(lo_ys), min(hi_ys)),
            ("y", "max", max(lo_ys), max(hi_ys)),
        ]
        axis, extreme, lo_v, hi_v = min(candidates, key=lambda c: abs(c[2]-c[3]))
        anchor = (lo_v + hi_v) / 2.0
        offset = width_m/2.0 if extreme == "min" else -width_m/2.0
        if axis == "x":
            lo_point = (anchor + offset, lo_center[1], lo_center[2])
            hi_point = (anchor + offset, hi_center[1], hi_center[2])
        else:
            lo_point = (lo_center[0], anchor + offset, lo_center[2])
            hi_point = (hi_center[0], anchor + offset, hi_center[2])

        dx = hi_point[0] - lo_point[0]
        dy = hi_point[1] - lo_point[1]
        line_len = math.hypot(dx, dy)

        all_pts = lo_band + hi_band
        xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]

        if line_len > 1e-6:
            ux, uy = dx / line_len, dy / line_len
            projections = [(p[0]-lo_point[0])*ux + (p[1]-lo_point[1])*uy
                          for p in all_pts]
            projection_m = max(projections) - min(projections)

            half_w = width_m / 2.0
            perp_x, perp_y = -uy, ux
            corners = [
                (lo_point[0] + perp_x*half_w, lo_point[1] + perp_y*half_w, lo_point[2]),
                (lo_point[0] - perp_x*half_w, lo_point[1] - perp_y*half_w, lo_point[2]),
                (hi_point[0] - perp_x*half_w, hi_point[1] - perp_y*half_w, hi_point[2]),
                (hi_point[0] + perp_x*half_w, hi_point[1] + perp_y*half_w, hi_point[2]),
            ]
        else:
            # Degenerate (near-vertical or single-point) case: fall back to
            # the raw bounding box rather than divide by zero.
            projection_m = max(max(xs)-min(xs), max(ys)-min(ys))
            cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
            half_w = width_m / 2.0
            corners = [(cx-half_w, cy-half_w, lo_point[2]),
                      (cx+half_w, cy-half_w, lo_point[2]),
                      (cx+half_w, cy+half_w, hi_point[2]),
                      (cx-half_w, cy+half_w, hi_point[2])]

        slope_pct = 100.0 * rise_m / projection_m if projection_m > 0 else 0.0
        return rise_m, projection_m, width_m, slope_pct, corners

    def _measure_landing(top_pts):
        """
        A flat 4-point 3D quad for one landing, at its own (near-constant)
        z. Unlike a flight, a landing has no meaningful fall-line
        direction (its top surface is flat, so the lowest/highest-z points
        used for a flight's fall line are arbitrary here) — its footprint
        is simply the raw plan bounding box of its top-surface points.
        """
        xs = [p[0] for p in top_pts]; ys = [p[1] for p in top_pts]
        z = sum(p[2] for p in top_pts) / len(top_pts)
        corners = [(min(xs), min(ys), z), (max(xs), min(ys), z),
                  (max(xs), max(ys), z), (min(xs), max(ys), z)]
        return corners

    for ramp in model.by_type("IfcRamp"):
        try:
            # ── Scope gate: only explicitly tagged accessible ramps ────────
            handicap_flag = get_property(ramp, "any", "HandicapAccessible")
            is_accessible = (handicap_flag is True or handicap_flag == "True"
                             or handicap_flag == 1)
            if not is_accessible:
                continue

            # ── Gather segments: decomposing children, or the ramp itself ──
            # Each segment carries the GlobalId of the REAL IFC entity it
            # came from (the child, or the ramp itself in the fallback
            # case) — needed downstream so a BCF viewpoint can select the
            # actual model element, not a synthetic composite id.
            segments = []  # list of (entity_name, points, entity_global_id)
            for rel in getattr(ramp, "IsDecomposedBy", []) or []:
                if not rel.is_a("IfcRelAggregates"):
                    continue
                for child in rel.RelatedObjects:
                    # IfcRampFlight or IfcSlab children are the walking-
                    # surface candidates (either can carry a real flight or
                    # landing in this kind of export). IfcRailing
                    # (guardrails) and anything else is excluded — its
                    # sparse post/baluster mesh can otherwise measure as a
                    # multi-metre-wide "flight" with a fabricated slope
                    # violation.
                    if not (child.is_a("IfcRampFlight") or child.is_a("IfcSlab")):
                        continue
                    pts = _segment_points(child)
                    if pts:
                        segments.append((child.Name or child.is_a(), pts, child.GlobalId))

            if not segments:
                pts = _segment_points(ramp)
                if pts:
                    segments.append((ramp.Name or "Ramp", pts, ramp.GlobalId))

            if not segments:
                continue

            # ── Classify each segment as a landing or a flight ─────────────
            # A segment whose overall rise is under 2cm is a landing outright.
            # A climbing segment can still carry an embedded landing plateau
            # fused directly to a short rise in the same mesh (see _z_bands'
            # docstring) — that plateau is split out as its own landing entry
            # before the flight is measured between the two ends.
            LANDING_AREA_MIN = 0.3  # m2 — a real plateau, not a flight's own edge
            flights = []
            landings = []
            for seg_name, pts, entity_id in segments:
                top_pts = _top_surface(pts)
                if len(top_pts) < 2:
                    continue
                zs = [p[2] for p in top_pts]
                seg_rise = max(zs) - min(zs)
                if seg_rise < 0.02:
                    landings.append((seg_name, top_pts, entity_id))
                    continue

                bands = _z_bands(top_pts)
                lo_band, hi_band = bands[0], bands[-1]
                if len(bands) >= 2 and _band_plan_area(lo_band) >= LANDING_AREA_MIN:
                    landings.append((f"{seg_name} (landing)", lo_band, entity_id))
                    # The flight's own low end is the landing's edge
                    # nearest the rise, not the landing's full extent —
                    # narrow to that near edge before measuring.
                    lo_band = _nearest_edge(lo_band, _band_center(hi_band))
                flights.append((seg_name, lo_band, hi_band, entity_id))

            if not flights:
                continue

            elev = storey_map.get(ramp.GlobalId, min(p[2] for _, pts, _ in segments for p in pts))
            storey_name = storey_name_map.get(ramp.GlobalId)

            for index, (seg_name, lo_band, hi_band, entity_id) in enumerate(flights, start=1):
                rise_m, projection_m, width_m, slope_pct, corners = _measure_flight(lo_band, hi_band)
                all_pts = lo_band + hi_band
                xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
                ramps.append({
                    "id":                 f"{ramp.GlobalId}::flight{index}",
                    "ramp_id":            ramp.GlobalId,
                    "entity_id":          entity_id,
                    "name":               (f"{ramp.Name or 'Ramp'} "
                                           f"[flight {index}/{len(flights)}]"),
                    "flight_index":       index,
                    "flight_count":       len(flights),
                    "has_landing":        bool(landings),
                    "rise_m":             round(rise_m, 3),
                    "projection_m":       round(projection_m, 3),
                    "width_m":            round(width_m, 3),
                    "slope_pct":          round(slope_pct, 2),
                    "corners":            corners,
                    "handicap_flag":      handicap_flag,
                    "storey_elevation_m": elev,
                    "storey_name":        storey_name,
                    "location":           (sum(xs)/len(xs), sum(ys)/len(ys),
                                           min(p[2] for p in all_pts)),
                })

            for index, (seg_name, top_pts, entity_id) in enumerate(landings, start=1):
                corners = _measure_landing(top_pts)
                landings_out.append({
                    "id":                 f"{ramp.GlobalId}::landing{index}",
                    "ramp_id":            ramp.GlobalId,
                    "entity_id":          entity_id,
                    "name":               (f"{ramp.Name or 'Ramp'} "
                                           f"[landing {index}/{len(landings)}]"),
                    "landing_index":      index,
                    "corners":            corners,
                    "storey_elevation_m": elev,
                    "storey_name":        storey_name,
                })
        except Exception:
            # A single malformed ramp must not stop extraction
            continue

    return ramps, landings_out


# ── Door helpers ────────────────────────────────────────────────────────
def is_fire_exit_door(door, model) -> bool:
    """
    Check whether a door is tagged as a fire exit (FireExit / IsFireExit).

    Args:
        door: an IfcDoor.
        model: ifcopenshell model (unused; kept for interface symmetry).
    Returns:
        bool
    Called by: extract_all.
    """
    for prop_name in ["FireExit", "IsFireExit"]:
        val = get_property(door, "any", prop_name)
        if val is True or val == "True" or val == 1:
            return True
    return False


def get_virtual_boundary_lengths(model, scale: float) -> dict:
    """
    Collect the VIRTUAL space boundaries of every space (a room separator
    exported as IfcRelSpaceBoundary with PhysicalOrVirtualBoundary =
    VIRTUAL — the two spaces are open to each other).

    The export used here does not write IfcVirtualElement, so the two
    spaces sharing one boundary cannot be paired through a common element.
    Instead, the LENGTH of each virtual boundary is recorded per space
    (from its connection geometry, in the file's own raw length unit);
    the routing engine matches those lengths against the shared edge the
    two rooms really have, which is measured in metres — so this raw
    length must be scaled to metres too, or the two will never match on
    any project not already declared in metres. Confirmed on a real
    project declared in millimetres: an un-scaled virtual-boundary length
    of "4000.0" was compared against a real shared edge of 1.436 m and
    never matched within tolerance, so a room that WAS physically open to
    its neighbour (touching, no wall between them) was still reported as
    disconnected and unreachable.

    Args:
        model: ifcopenshell model.
        scale: unit scale factor (see get_length_unit_scale).
    Returns:
        dict: {space_global_id: [length_m, ...]}
    Called by: extract_all.
    """
    lengths = {}
    for rel in model.by_type("IfcRelSpaceBoundary"):
        kind = str(getattr(rel, "PhysicalOrVirtualBoundary", "")).upper()
        if kind != "VIRTUAL":
            continue
        space = rel.RelatingSpace
        if not space:
            continue
        try:
            points = rel.ConnectionGeometry \
                        .SurfaceOnRelatingElement \
                        .OuterBoundary.Points
            along = [p.Coordinates[0] * scale for p in points]
        except Exception:
            continue
        if not along:
            continue
        lengths.setdefault(space.GlobalId, []).append(max(along) - min(along))

    return lengths


def extract_all(model, rules: dict) -> dict:
    """
    Extract all IFC data needed by the pipeline. Unit scale is detected
    once here and applied everywhere.

    Args:
        model: ifcopenshell model.
        rules: rules dict (currently unused; kept for interface symmetry).
    Returns:
        dict: ifc_data with keys spaces, non_occupiable_spaces, doors,
            storeys, columns, obstacles, ramps, ramp_landings,
            space_boundaries, virtual_boundaries, scale.
    Called by: main.py (run).
    """
    settings = _make_settings()

    # ── Detect unit scale ONCE ────────────────────────────────────
    scale = get_length_unit_scale(model)

    # ── Map openings → doors ──────────────────────────────────────
    opening_to_door = {}
    for rel in model.by_type("IfcRelFillsElement"):
        try:
            opening = rel.RelatingOpeningElement
            element = rel.RelatedBuildingElement
            if opening and element and element.is_a("IfcDoor"):
                opening_to_door[opening.GlobalId] = element.GlobalId
        except Exception:
            continue

    # ── Storeys ───────────────────────────────────────────────────
    storey_elevations = []
    storeys = []
    for storey in model.by_type("IfcBuildingStorey"):
        elevation = get_storey_elevation(storey, scale)
        storey_elevations.append(elevation)
        storeys.append({
            "id":           storey.GlobalId,
            "name":         (storey.Name or "Unknown").replace(" ", "_"),
            "elevation_m":  elevation,
            "is_basement":  elevation < 0,
            "is_high_floor": elevation > 28.0,
            "ifc_ref":      storey,
        })
    storey_elevations = sorted(set(round(e, 1) for e in storey_elevations))

    def snap_to_storey(z):
        if not storey_elevations:
            return 0.0
        candidates = [e for e in storey_elevations if abs(e-z) <= 1.5]
        return min(candidates, key=lambda e: abs(e-z)) \
            if candidates else storey_elevations[0]

    # ── Spaces ────────────────────────────────────────────────────
    spaces = []
    non_occupiable_spaces = []
    for space in model.by_type("IfcSpace"):
        footprint = get_space_footprint(space, settings, scale)
        centroid_x, centroid_y = get_space_centroid(footprint)

        # Storey lookup happens before location is built below, so the
        # space's Z coordinate is its real storey elevation, needed for a
        # correct BCF viewpoint camera height on floors above ground.
        storey_id   = None
        storey_name = None
        storey_elev = 0.0
        try:
            for rel in model.by_type("IfcRelAggregates"):
                if space in rel.RelatedObjects \
                        and rel.RelatingObject.is_a("IfcBuildingStorey"):
                    storey_id   = rel.RelatingObject.GlobalId
                    storey_name = rel.RelatingObject.Name or None
                    storey_elev = get_storey_elevation(rel.RelatingObject, scale)
                    break
        except Exception:
            pass

        location = (centroid_x, centroid_y, storey_elev) if footprint \
            else get_element_location_placement(space, scale)

        name      = space.LongName or space.Name or "Space"
        long_name = space.LongName \
            if hasattr(space, "LongName") and space.LongName else ""

        # Skip spaces explicitly marked non_occupiable = Yes/True/1; all
        # other spaces are treated as occupiable by default. Kept out of
        # routing/checks, but recorded separately (non_occupiable_spaces
        # below) so main.py can still warn about and draw them.
        non_occ = get_property(space, "any", "non_occupiable")
        if non_occ is True or str(non_occ).lower() in ("yes", "true", "1"):
            non_occupiable_spaces.append({
                "id":                space.GlobalId,
                "name":              name,
                "long_name":         long_name,
                "footprint":         footprint,
                "storey_id":         storey_id,
                "storey_name":       storey_name,
                "storey_elevation_m": storey_elev,
            })
            continue

        spaces.append({
            "id":                space.GlobalId,
            "name":              name,
            "long_name":         long_name,
            "location":          location,
            "footprint":         footprint,
            "storey_id":         storey_id,
            "storey_name":       storey_name,
            "storey_elevation_m": storey_elev,
            "ifc_ref":           space,
        })

    # ── Doors ─────────────────────────────────────────────────────
    # Storey lookup via IfcRelContainedInSpatialStructure — the same
    # relation already used for columns, furniture and MEP elements.
    # (Snapping a door's raw world-Z to the nearest storey elevation is
    # kept only as a fallback below: on a model using large georeferenced
    # site coordinates, a door's world Z can carry a site-level offset the
    # storey's own Elevation attribute lacks, making that snap match the
    # wrong storey.)
    door_storey_elev = {}
    door_storey_name  = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        for element in rel.RelatedElements:
            if element.is_a("IfcDoor"):
                structure = rel.RelatingStructure
                if structure.is_a("IfcBuildingStorey"):
                    door_storey_elev[element.GlobalId] = \
                        get_storey_elevation(structure, scale)
                    door_storey_name[element.GlobalId] = structure.Name or None

    doors = []
    for door in model.by_type("IfcDoor"):
        location = get_element_location_geometry(door, settings)
        # Width and height in metres
        width  = getattr(door, "OverallWidth",  None)
        height = getattr(door, "OverallHeight", None)
        width_m  = float(width)  * scale if width  else 0.915  # default 915mm
        height_m = float(height) * scale if height else 2.100
        # Orientation angle from placement matrix (degrees, in XY plane)
        try:
            import math
            matrix = ifcopenshell.util.placement.get_local_placement(
                door.ObjectPlacement)
            angle = math.degrees(math.atan2(matrix[1][0], matrix[0][0]))
        except Exception:
            angle = 0.0

        if door.GlobalId in door_storey_elev:
            door_elev = door_storey_elev[door.GlobalId]
        else:
            # Fallback for the rare door with no direct spatial-structure
            # containment relation.
            door_elev = snap_to_storey(
                location[2] if len(location) > 2 else 0.0)

        doors.append({
            "id":                    door.GlobalId,
            "name":                  door.Name or "Door",
            "location":              location,
            "storey_elevation_m":    door_elev,
            "storey_name":           door_storey_name.get(door.GlobalId),
            "width_m":               width_m,
            "height_m":              height_m,
            "angle_deg":             angle,
            "is_fire_exit":          is_fire_exit_door(door, model),
            "ifc_ref":               door,
        })

    # ── Columns ───────────────────────────────────────────────────
    columns = get_columns(model, settings, scale)

    # ── Fixed-furniture obstacles (architect-tagged) ──────────────
    obstacles = get_obstacles(model, settings, scale)

    # ── MEP / plumbing fixture obstacles (architect-tagged, same as
    # furniture) ────────────────────────────────────────────────────
    obstacles += get_mep_obstacles(model, settings, scale)

    # Ramps for the DL 163/2006 accessibility check (geometry only; the
    # compliance decision happens in rule_checker.check_ramps). ramp_landings
    # is visualization-only and never checked against any rule.
    ramps, ramp_landings = get_ramps(model, settings, scale)

    # ── Space boundaries ──────────────────────────────────────────
    space_by_id = {s["id"]: s for s in spaces}
    door_by_id  = {d["id"]: d for d in doors}
    space_boundaries = []

    for rel in model.by_type("IfcRelSpaceBoundary"):
        try:
            space   = rel.RelatingSpace
            element = rel.RelatedBuildingElement
            if not space or not element:
                continue
            space_record = space_by_id.get(space.GlobalId)
            if not space_record:
                continue

            space_elevation = space_record.get("storey_elevation_m", 0.0)

            if element.is_a("IfcDoor"):
                door_record = door_by_id.get(element.GlobalId)
                if not door_record:
                    continue
                # Elevation check — door must be on the same floor as space
                door_elevation = door_record.get("storey_elevation_m", 0.0)
                if abs(door_elevation - space_elevation) > 1.0:
                    continue
                space_boundaries.append((space.GlobalId, element.GlobalId))

            elif element.is_a("IfcOpeningElement"):
                door_id = opening_to_door.get(element.GlobalId)
                if not door_id:
                    continue
                door_record = door_by_id.get(door_id)
                if not door_record:
                    continue
                door_elevation = door_record.get("storey_elevation_m", 0.0)
                if abs(door_elevation - space_elevation) > 1.0:
                    continue
                space_boundaries.append((space.GlobalId, door_id))

        except Exception:
            continue

    space_boundaries = list(set(space_boundaries))

    return {
        "spaces":           spaces,
        "non_occupiable_spaces": non_occupiable_spaces,
        "doors":            doors,
        "storeys":          storeys,
        "columns":          columns,
        "obstacles":        obstacles,
        "ramps":            ramps,
        "ramp_landings":    ramp_landings,
        "space_boundaries": space_boundaries,
        "virtual_boundaries": get_virtual_boundary_lengths(model, scale),
        "scale":            scale,
    }


# ── Space coverage check (data-quality warning) ───────────────────────────
def check_space_coverage(model, min_residual_area_m2: float = 2.0) -> list:
    """
    Data-quality check run before the main parse: for every storey,
    compare the storey's floor footprint (from IfcSlab) against the union
    of all IfcSpace footprints on that storey. Any leftover area larger
    than `min_residual_area_m2` is reported as a possible missing room.
    Warning-only — never blocks the pipeline, since a gap can also be an
    intentionally space-less area (a void, an unenclosed balcony, a slab
    edge not meant to be occupiable).

    Args:
        model: ifcopenshell model.
        min_residual_area_m2: minimum uncovered area to report, m2.
    Returns:
        list[dict]: {id, storey_name, storey_elevation_m,
            residual_area_m2, residual_centroid, footprint} — footprint is
            the gap polygon's own outline (list of (x, y) points), for
            drawing it gray on the floor plan (see main.py, visualizer).
    Called by: main.py (run).
    """
    import ifcopenshell.geom
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

    scale = get_length_unit_scale(model)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # Build the space -> storey map once, via IfcRelAggregates (this model
    # aggregates spaces into their storey rather than "containing" them,
    # so IfcRelContainedInSpatialStructure alone finds zero spaces here).
    space_storey_id = {}
    for rel in model.by_type("IfcRelAggregates"):
        if rel.RelatingObject.is_a("IfcBuildingStorey"):
            for obj in rel.RelatedObjects:
                if obj.is_a("IfcSpace"):
                    space_storey_id[obj.GlobalId] = rel.RelatingObject.GlobalId

    warnings = []

    for storey in model.by_type("IfcBuildingStorey"):
        elevation = get_storey_elevation(storey, scale)

        # Collect every IfcSlab in this storey and project its footprint
        # onto the XY plane.
        slab_polys = []
        for rel in getattr(storey, "ContainsElements", []):
            for element in rel.RelatedElements:
                if not element.is_a("IfcSlab"):
                    continue
                try:
                    shape = ifcopenshell.geom.create_shape(settings, element)
                    verts = shape.geometry.verts
                    faces = shape.geometry.faces
                    triangles = []
                    for f in range(0, len(faces), 3):
                        a, b, c = faces[f], faces[f+1], faces[f+2]
                        # verts from create_shape are already metre-
                        # normalised by the geometry kernel (same as
                        # get_columns/get_obstacles); multiplying by
                        # `scale` again here would double-convert.
                        tri = Polygon([(verts[3*a], verts[3*a+1]),
                                       (verts[3*b], verts[3*b+1]),
                                       (verts[3*c], verts[3*c+1])])
                        if tri.is_valid and tri.area > 1e-6:
                            triangles.append(tri)
                    if triangles:
                        slab_polys.append(unary_union(triangles))
                except Exception:
                    continue

        if not slab_polys:
            # No slab on this storey — nothing to compare against.
            continue

        floor_area = unary_union(slab_polys)

        # Only IfcSpace objects actually aggregated into THIS storey may
        # cover its residual area (avoids a federated model's stacked
        # storeys wrongly "covering" each other's gaps).
        space_polys = []
        for space in model.by_type("IfcSpace"):
            if space_storey_id.get(space.GlobalId) != storey.GlobalId:
                continue
            footprint = get_space_footprint(space, settings, scale)
            if footprint and len(footprint) >= 3:
                poly = Polygon(footprint)
                if poly.is_valid and poly.area > 0.01:
                    space_polys.append(poly)

        covered = unary_union(space_polys) if space_polys else None
        residual = floor_area.difference(covered) if covered else floor_area

        # A residual can be a Polygon or MultiPolygon; check each piece
        # separately so several small gaps do not cancel out.
        pieces = list(residual.geoms) if isinstance(residual, MultiPolygon) \
            else [residual]

        for index, piece in enumerate(pieces):
            if piece.area < min_residual_area_m2:
                continue
            if piece.geom_type != "Polygon" or piece.is_empty:
                # A degenerate slice from difference() (a line, a point) —
                # not something that can be drawn or is a real room gap.
                continue
            warnings.append({
                "id":                 f"{storey.GlobalId}-gap-{index}",
                "storey_name":        storey.Name or "Unknown",
                "storey_elevation_m": round(elevation, 2),
                "residual_area_m2":   round(piece.area, 1),
                "residual_centroid":  (round(piece.centroid.x, 1),
                                       round(piece.centroid.y, 1)),
                "footprint":          list(piece.exterior.coords),
            })

    return warnings
