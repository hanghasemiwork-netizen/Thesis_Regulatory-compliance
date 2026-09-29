"""
RT_SCIE_checker.py
Evacuation router based on a visibility graph.

The walkable area of a floor is:
  - the footprint of every IfcSpace on the floor (kept separate per room), plus
  - a small oriented rectangle over every door opening — the only place a
    path may cross from one room to the next.

Rooms are only joined at their doors, so a path built on this area can
never cross a wall or leave the building.
"""

import math
from itertools import product

import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union, nearest_points
from shapely.prepared import prep


# Standard clearance the whole route keeps from every wall, column and
# obstacle, applied consistently including at the start and exit points.
# RT-SCIE Article 57 does not itself specify a numeric clearance value.
# This project instead adopts the value given for the equivalent measurement
# in a different, but directly analogous, regulation: Malaysia's Uniform
# Building By-Laws 1984, By-law 165(1), which defines the travel distance
# to an exit as measured "starting 0.300 metres from the most remote point
# of occupancy, curving around any corners or obstructions with 0.300-metre
# clearance therefrom" (quoted in Ismail et al., 2023) — the same dual role
# (start-point offset and ongoing obstacle clearance) EGRESS_CLEARANCE_M
# plays here. RT-SCIE does not provide its own equivalent numeric clause,
# so this is a deliberate cross-regulation borrowing, not a value drawn
# from RT-SCIE itself, and is documented as such rather than presented as
# an RT-SCIE requirement.
EGRESS_CLEARANCE_M = 0.30

# Coordinate grid used to snap geometries before a union (see _safe_union).
# Two rooms exported with footprints that share a wall exactly (distance
# 0.0) can make GEOS's union noding produce a degenerate MultiPolygon with
# a tiny sliver bleeding into a hole that sits on that shared edge — e.g. a
# piece of furniture pushed flush into a corner where two walls meet. That
# sliver is far smaller than the clearance erosion (EGRESS_CLEARANCE_M), but it corrupts
# the polygon's local topology enough that the eroded boundary near the
# hole collapses to near-zero clearance instead of wrapping around it.
# Snapping every vertex to a coarse-but-still-submillimetre grid before
# the union removes the floating-point mismatch that triggers this.
UNION_PRECISION_GRID = 1e-4

# How deep a door rectangle reaches into each side, in metres. Must be
# large enough to bridge the wall gap between two rooms, small enough not
# to leak into a room it does not belong to, and strictly greater than
# EGRESS_CLEARANCE_M plus half the real wall gap between the two rooms:
# after a room is eroded inward by the clearance (see
# _erode_with_clearance), the re-opened door rectangle must reach far
# enough past that eroded boundary to still overlap it, or the door fails
# to reconnect the two eroded rooms and silently splits the walkable area
# at that doorway — the room on the far side then vanishes from routing
# entirely (build_walkable_for_chain/build_walkable_area keep only the
# larger of the resulting fragments, on the assumption a smaller one is a
# rounding sliver), so every space in it gets its start point snapped to
# the nearest surviving point outside the room instead of its own
# interior. The door's own recorded location sits at the wall gap's own
# centre, not flush against either room, so half_depth must clear
# EGRESS_CLEARANCE_M plus half that gap, not just the clearance alone. A
# +0.05 margin was enough only for a near-zero-thickness modelled wall;
# confirmed on a real model (TEST MODEL 2.ifc) where the space-footprint
# gap between rooms — i.e. the modelled wall thickness — commonly ran
# 0.10-0.20 m (up to 0.20 m), silently breaking the bridge at 50 of 53
# inter-room doors. Sized here for gaps up to 0.30 m (a generous
# real-world wall) with the same +0.05 safety margin beyond that.
DOOR_HALF_DEPTH = EGRESS_CLEARANCE_M + 0.05 + 0.15


def build_door_rectangle(door):
    """
    Build a small oriented rectangle covering one door opening: as wide as
    the door leaf (plus a small margin), reaching DOOR_HALF_DEPTH into the
    room on each side, aligned with the door's own direction.

    Args:
        door: one door dict (needs location, width_m, angle_deg).
    Returns:
        shapely.Polygon, or None if the door has no usable location or the
        resulting rectangle is degenerate.
    Called by: _rooms_really_at_door, _door_candidates_between,
        build_walkable_for_chain, build_walkable_area.
    """
    location = door.get("location")
    if not location:
        return None

    # Half the door width, plus a small margin so the opening is not too tight
    half_width = (door.get("width_m", 0.9) or 0.9) / 2.0 + 0.05
    half_depth = DOOR_HALF_DEPTH

    # The door's rotation. angle_deg is the direction along the door width.
    angle = math.radians(door.get("angle_deg", 0.0) or 0.0)

    # Unit vector ALONG the door width, and the perpendicular (through-wall)
    along_x, along_y = math.cos(angle), math.sin(angle)
    thru_x, thru_y = -math.sin(angle), math.cos(angle)

    cx, cy = location[0], location[1]

    # Four corners of the oriented rectangle
    corners = [
        (cx + along_x*half_width + thru_x*half_depth,
         cy + along_y*half_width + thru_y*half_depth),
        (cx - along_x*half_width + thru_x*half_depth,
         cy - along_y*half_width + thru_y*half_depth),
        (cx - along_x*half_width - thru_x*half_depth,
         cy - along_y*half_width - thru_y*half_depth),
        (cx + along_x*half_width - thru_x*half_depth,
         cy + along_y*half_width - thru_y*half_depth),
    ]
    rectangle = Polygon(corners)
    if rectangle.is_valid and rectangle.area > 0.001:
        return rectangle
    return None


def same_floor_boundaries(ifc_data, floor_elevation):
    """
    Return the (space_id, door_id) links on the given floor: the recorded
    IfcRelSpaceBoundary pairs, plus a geometric rescue for any door with no
    recorded boundary at all.

    Filters the recorded pairs to both space and door sitting on this
    floor, which also drops cross-floor links caused by a ground-floor
    door wrongly linked to a space one floor above (a known export issue
    when two spaces share the same X,Y footprint).

    A door can be a genuine fire exit or internal door and still carry
    ZERO recorded IfcRelSpaceBoundary entries — observed on a real model
    where Revit's boundary computation missed a door sitting right at a
    re-entrant corner of an L-shaped room, even though the door's own
    crossing rectangle demonstrably overlaps that room by a real, if thin,
    sliver. Left unpatched, such a door is invisible to build_space_graph:
    exit_space_ids never includes its room, and every room on the floor
    comes back unreachable even though the exit is real and correctly
    placed. For every door with no recorded boundary on this floor, this
    rescues it by testing its crossing rectangle against every room's
    footprint here and adding a pair for each one it overlaps by more than
    0.01 m2 — the same overlap threshold _rooms_really_at_door already
    uses to validate a recorded boundary, applied here to recover a
    missing one instead.

    Args:
        ifc_data: parsed data dict (uses spaces, doors, space_boundaries).
        floor_elevation: elevation of the floor to filter to.
    Returns:
        list[tuple]: (space_id, door_id) pairs.
    Called by: _prepare_routing_context, build_space_graph,
        build_walkable_for_chain, build_walkable_area.
    """
    space_elev = {s["id"]: s.get("storey_elevation_m", 0.0)
                  for s in ifc_data["spaces"]}
    door_elev = {d["id"]: d.get("storey_elevation_m", 0.0)
                 for d in ifc_data["doors"]}

    kept = []
    for space_id, door_id in ifc_data["space_boundaries"]:
        s_elev = space_elev.get(space_id)
        d_elev = door_elev.get(door_id)
        if s_elev is None or d_elev is None:
            continue
        if abs(s_elev - floor_elevation) <= 1.0 and \
           abs(d_elev - floor_elevation) <= 1.0:
            kept.append((space_id, door_id))

    # ── Geometric rescue for doors with no recorded boundary on this floor ──
    OVERLAP_MIN = 0.01   # m2, matches _rooms_really_at_door
    doors_with_boundary = set(door_id for _, door_id in kept)
    rooms_on_floor = [s for s in ifc_data["spaces"]
                      if abs(s.get("storey_elevation_m", 0.0) - floor_elevation) <= 1.0
                      and s.get("footprint")]
    for door in ifc_data["doors"]:
        if door["id"] in doors_with_boundary:
            continue
        if abs(door.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        rectangle = build_door_rectangle(door)
        if rectangle is None:
            continue
        for space in rooms_on_floor:
            room_polygon = _as_single_polygon(Polygon(space["footprint"]))
            if rectangle.intersection(room_polygon).area > OVERLAP_MIN:
                kept.append((space["id"], door["id"]))

    return kept


def _farthest_corner_from(space_poly, reference_point):
    """
    The footprint corner farthest from a single reference point (typically
    the exit door of the room), nudged ~5 cm inside so it is strictly
    within the room.

    Args:
        space_poly: shapely.Polygon of the room.
        reference_point: (x, y) to measure distance from.
    Returns:
        tuple: (x, y) point.
    Called by: route_all_spaces.
    """
    corners = list(space_poly.exterior.coords)[:-1]
    if not corners:
        return (space_poly.centroid.x, space_poly.centroid.y)
    corner = max(corners, key=lambda c: math.hypot(c[0]-reference_point[0],
                                                   c[1]-reference_point[1]))
    cx, cy = space_poly.centroid.x, space_poly.centroid.y
    dx, dy = cx - corner[0], cy - corner[1]
    length = math.hypot(dx, dy)
    if length > 0.001:
        return (corner[0] + dx/length*0.05, corner[1] + dy/length*0.05)
    return corner


def _farthest_corner_from_nearest(space_poly, reference_points):
    """
    The footprint corner that maximizes the distance to its NEAREST point
    in `reference_points`, nudged ~5 cm inside the room.

    Used for a "single-room chain": a space that already has one or more
    fire-exit doors on its own boundary, so find_space_chain returns just
    that space with no other room to pass through. The Art. 57 worst-case
    point of such a room is a single, fixed point — the point most remote
    from whichever of its own exits is nearest to it — not a different
    corner re-picked separately for each candidate exit. Picking a new
    corner per exit compares each exit against a different starting point
    instead of the one true worst point against every exit, and can hide a
    genuinely short route to a nearby exit behind a corner that was only
    ever measured against a distant one.

    Args:
        space_poly: shapely.Polygon of the room.
        reference_points: list of (x, y) exit points on this floor.
    Returns:
        tuple: (x, y) point.
    Called by: route_all_spaces.
    """
    corners = list(space_poly.exterior.coords)[:-1]
    if not corners or not reference_points:
        return (space_poly.centroid.x, space_poly.centroid.y)

    def nearest_dist(corner):
        return min(math.hypot(corner[0]-p[0], corner[1]-p[1])
                   for p in reference_points)

    corner = max(corners, key=nearest_dist)
    cx, cy = space_poly.centroid.x, space_poly.centroid.y
    dx, dy = cx - corner[0], cy - corner[1]
    length = math.hypot(dx, dy)
    if length > 0.001:
        return (corner[0] + dx/length*0.05, corner[1] + dy/length*0.05)
    return corner


def _collect_corners(walkable):
    """
    Collect every corner point of a walkable area (outer and hole
    boundaries) — the points a taut string can bend around.

    Args:
        walkable: shapely Polygon or MultiPolygon.
    Returns:
        list[tuple]: (x, y) points.
    Called by: build_corner_graph.
    """
    corners = []
    geoms = walkable.geoms if walkable.geom_type == "MultiPolygon" else [walkable]
    for poly in geoms:
        corners.extend(list(poly.exterior.coords)[:-1])
        for hole in poly.interiors:
            corners.extend(list(hole.coords)[:-1])
    return corners


def _can_see(a, b, walkable_prepared):
    """
    True if the straight segment from a to b stays inside the walkable
    area (no wall in between).

    Args:
        a, b: (x, y) points.
        walkable_prepared: prepared, buffered walkable geometry (see
            _prepare_walkable) — passed in rather than built here, since
            preparing it once per walkable area is what keeps the
            visibility graph fast.
    Returns:
        bool
    Called by: build_corner_graph, route_from_point.
    """
    line = LineString([a, b])
    return walkable_prepared.contains(line)


def _prepare_walkable(walkable):
    """
    Buffer the walkable area by a tiny amount (robust when a sight-line
    runs along an edge) and wrap it as a prepared geometry for fast,
    repeated contains() tests.

    Args:
        walkable: shapely Polygon or MultiPolygon.
    Returns:
        shapely.prepared.PreparedGeometry
    Called by: build_corner_graph, route_from_point.
    """
    return prep(walkable.buffer(0.001))


def shortest_taut_path(nodes, neighbours, start_index=0, exit_index=1):
    """
    Dijkstra shortest path from start to exit over the visibility graph.

    Args:
        nodes: list of (x, y) points, indexed by node id.
        neighbours: dict node_index -> list of (other_index, distance).
        start_index, exit_index: node ids for start and exit.
    Returns:
        tuple: (list_of_points, total_distance), or (None, None) if unreachable.
    Called by: route_from_point.
    """
    import heapq
    best_cost = {start_index: 0.0}
    came_from = {start_index: None}
    heap = [(0.0, start_index)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == exit_index:
            path = []
            current = exit_index
            while current is not None:
                path.append(nodes[current])
                current = came_from[current]
            path.reverse()
            return path, cost
        if cost > best_cost.get(node, float("inf")):
            continue
        for (neighbour, distance) in neighbours[node]:
            new_cost = cost + distance
            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                came_from[neighbour] = node
                heapq.heappush(heap, (new_cost, neighbour))
    return None, None


def _safe_union(geoms):
    """
    Union a list of shapely polygons, snapping their coordinates to
    UNION_PRECISION_GRID first to avoid a GEOS union degeneracy where
    polygons touching along a shared edge produce a corrupted result near
    a hole that sits on that edge (see UNION_PRECISION_GRID).

    Args:
        geoms: list of shapely Polygon/MultiPolygon.
    Returns:
        shapely geometry: the union, or None if `geoms` is empty.
    Called by: _erode_with_clearance, build_walkable_for_chain,
        build_walkable_area.
    """
    if not geoms:
        return None
    snapped = [shapely.set_precision(g, UNION_PRECISION_GRID) for g in geoms]
    return unary_union(snapped)


def _erode_with_clearance(walkable, door_rectangles, clearance=EGRESS_CLEARANCE_M):
    """
    Shrink the walkable area inward by `clearance` from every wall, column
    and obstacle, then re-open each door at its full original width (a
    typical door opening, ~0.9 m, is narrower than two clearance margins,
    so a flat erosion would otherwise seal most doors shut).

    Building the taut-string path directly on this eroded area, rather
    than computing a path first and nudging it away from obstacles
    afterwards, keeps the required clearance along the full length of
    every route segment, not just its waypoints — a straight segment
    between two individually safe points can still pass close to an
    obstacle partway along its length, which routing on the pre-eroded
    area rules out.

    Args:
        walkable: shapely Polygon/MultiPolygon of the chain's walkable area.
        door_rectangles: door rectangles to re-open after erosion.
        clearance: erosion distance, metres.
    Returns:
        shapely geometry: the eroded-with-doors-reopened polygon, or the
            original `walkable` if erosion leaves nothing.
    Called by: route_all_spaces.
    """
    inner = walkable.buffer(-clearance)
    if inner.is_empty:
        return walkable
    if door_rectangles:
        inner = _safe_union([inner] + list(door_rectangles))
    return inner


def build_corner_graph(walkable):
    """
    Build the visibility graph over the corners of a walkable area, once.
    Each space's own start and exit points are then connected to this
    shared graph separately (see route_from_point), which is much faster
    than rebuilding the whole graph per space.

    Args:
        walkable: shapely Polygon/MultiPolygon.
    Returns:
        tuple: (corners: list[(x,y)], corner_edges: dict corner_index ->
            list of (other_index, distance)).
    Called by: route_all_spaces.
    """
    corners = _collect_corners(walkable)
    walkable_prepared = _prepare_walkable(walkable)   # buffer + prep once
    corner_edges = {i: [] for i in range(len(corners))}
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if _can_see(corners[i], corners[j], walkable_prepared):
                distance = math.hypot(corners[i][0]-corners[j][0],
                                      corners[i][1]-corners[j][1])
                corner_edges[i].append((j, distance))
                corner_edges[j].append((i, distance))
    return corners, corner_edges


def route_from_point(start_point, exit_point, corners, corner_edges, walkable):
    """
    Find the shortest taut path from start_point to exit_point, reusing a
    pre-built corner graph and connecting only the two extra nodes (start,
    exit) to it.

    Args:
        start_point, exit_point: (x, y) points.
        corners, corner_edges: from build_corner_graph.
        walkable: shapely geometry the path must stay inside.
    Returns:
        tuple: (points, distance) or (None, None) if unreachable.
    Called by: _route_through_gates.
    """
    # Node 0 = start, node 1 = exit, nodes 2.. = the corners
    # (corner index c in corner_edges becomes node c + 2 here).
    nodes = [start_point, exit_point] + corners
    neighbours = {i: [] for i in range(len(nodes))}
    walkable_prepared = _prepare_walkable(walkable)   # buffer + prep once

    # copy the corner-to-corner edges, shifting indices by 2
    for corner_i, edges in corner_edges.items():
        for (corner_j, distance) in edges:
            neighbours[corner_i + 2].append((corner_j + 2, distance))

    # connect start and exit to every corner they can see
    for node_index, point in [(0, start_point), (1, exit_point)]:
        for corner_index, corner in enumerate(corners):
            if _can_see(point, corner, walkable_prepared):
                distance = math.hypot(point[0]-corner[0], point[1]-corner[1])
                neighbours[node_index].append((corner_index + 2, distance))
                neighbours[corner_index + 2].append((node_index, distance))

    # connect start directly to exit, if they can see each other
    if _can_see(start_point, exit_point, walkable_prepared):
        distance = math.hypot(start_point[0]-exit_point[0],
                              start_point[1]-exit_point[1])
        neighbours[0].append((1, distance))
        neighbours[1].append((0, distance))

    return shortest_taut_path(nodes, neighbours, 0, 1)


def _door_candidates_between(space_a, space_b, boundaries, door_by_id,
                             space_by_id=None):
    """
    Every door point that genuinely connects space_a and space_b — ALL of
    them, not just one. Two rooms sharing more than one real door is not
    rare (an open-plan pair with two separate doorways, say), and which
    one gives the shorter real route depends on the rest of the path,
    which this function has no way to know — so it hands every valid
    candidate back to the caller (_chain_door_combinations) to actually
    try and compare, rather than guessing here.

    Validates each candidate door the same way throughout this module (a
    door's rectangle must truly overlap both room footprints to count as
    "real"; otherwise every door on the raw shared list is returned as a
    fallback) but returns the full list instead of collapsing it to one
    point.

    Args:
        space_a, space_b: space_ids.
        boundaries: same-floor (space_id, door_id) pairs.
        door_by_id: door lookup by id.
        space_by_id: space lookup by id (enables overlap validation).
    Returns:
        list[tuple]: (x, y) door points (possibly empty).
    Called by: _chain_door_combinations.
    """
    doors_a = set(d for s, d in boundaries if s == space_a)
    doors_b = set(d for s, d in boundaries if s == space_b)
    shared = doors_a & doors_b

    real_points = []
    fallback_points = []
    for door_id in shared:
        door = door_by_id.get(door_id)
        if not door or not door.get("location"):
            continue
        point = (door["location"][0], door["location"][1])
        fallback_points.append(point)

        if space_by_id is not None:
            rectangle = build_door_rectangle(door)
            room_a = space_by_id.get(space_a)
            room_b = space_by_id.get(space_b)
            if rectangle is not None and room_a and room_b \
                    and room_a.get("footprint") and room_b.get("footprint"):
                poly_a = Polygon(room_a["footprint"])
                poly_b = Polygon(room_b["footprint"])
                overlap_a = rectangle.intersection(poly_a).area
                overlap_b = rectangle.intersection(poly_b).area
                if overlap_a > 0.01 and overlap_b > 0.01:
                    real_points.append(point)

    return real_points if real_points else fallback_points


def _chain_door_combinations(chain, boundaries, door_by_id, space_by_id=None):
    """
    Every combination of door choices across the chain's hops — one gate
    point per hop per combination — so the caller can route through each
    combination and keep whichever produces the shortest real path.

    Picking a single "best" door per hop up front by overlap area alone
    says nothing about which door actually shortens the walked route: on a
    real model, two rooms connected by both a nearby door and a farther
    one, the farther one having the larger overlap footprint, produced a
    visibly longer straight-line detour even though the nearer door was
    an equally valid, shorter-routing choice. Trying every combination
    and measuring the resulting path removes that guess.

    A hop with no valid door at all is treated as an open gap (no gate
    forced there) rather than blocking the whole chain, matching the
    previous lenient behaviour.

    Args:
        chain: list of space_ids (start .. exit space).
        boundaries: same-floor (space_id, door_id) pairs.
        door_by_id: door lookup by id.
        space_by_id: space lookup by id (for door-overlap validation).
    Returns:
        list[list[tuple]]: each inner list is one full ordered sequence
            of door gate points to try (always at least one, possibly
            empty, entry).
    Called by: route_all_spaces.
    """
    per_hop = []
    for i in range(len(chain) - 1):
        candidates = _door_candidates_between(chain[i], chain[i+1],
                                              boundaries, door_by_id,
                                              space_by_id)
        per_hop.append(candidates if candidates else [None])

    if not per_hop:
        return [[]]
    return [[pt for pt in combo if pt is not None]
            for combo in product(*per_hop)]


def _unreachable_result(space):
    """
    Build the result dict for a space with no path to any exit.

    Args:
        space: one space dict.
    Returns:
        dict: {space_id, name, start, path, distance, reachable, has_two_exits}
    Called by: route_all_spaces, route_building.
    """
    return {
        "space_id":  space.get("id"),
        "name":      space.get("name", "?"),
        "start":     None,
        "path":      None,
        "distance":  None,
        "reachable": False,
        "has_two_exits": False,
    }


def _prepare_routing_context(ifc_data, floor_elevation):
    """
    Gather everything the per-space routing needs, once per floor.

    Args:
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        dict: {neighbours, boundaries, door_by_id, exit_space_ids,
            space_centroid, space_farthest_point, space_by_id,
            open_passage_points}
    Called by: route_all_spaces.
    """
    neighbours, open_passage_points = build_space_graph(ifc_data, floor_elevation)
    boundaries = same_floor_boundaries(ifc_data, floor_elevation)
    door_by_id = {door["id"]: door for door in ifc_data["doors"]}

    exit_space_ids = set()
    for space_id, door_id in boundaries:
        door = door_by_id.get(door_id)
        if door and door.get("is_fire_exit"):
            exit_space_ids.add(space_id)

    space_centroid = {}
    space_farthest_point = {}
    for space in ifc_data["spaces"]:
        if space.get("footprint"):
            polygon = Polygon(space["footprint"])
            if polygon.is_valid and polygon.area > 0:
                centroid = (polygon.centroid.x, polygon.centroid.y)
                space_centroid[space["id"]] = centroid
                # The room's own vertex farthest from its own centroid, used
                # instead of the centroid itself as the semantic-graph edge
                # reference point (find_space_chain). Chain-independent by
                # construction (no door involved), so it carries none of the
                # wrong-door risk _worst_case_start had before its own fix;
                # it is a more conservative (larger, worst-case-leaning)
                # distance estimate than centroid-to-centroid for an
                # elongated or L-shaped room, at the same negligible cost.
                space_farthest_point[space["id"]] = _farthest_corner_from(
                    polygon, centroid)

    return {
        "neighbours": neighbours,
        "boundaries": boundaries,
        "door_by_id": door_by_id,
        "exit_space_ids": exit_space_ids,
        "space_centroid": space_centroid,
        "space_farthest_point": space_farthest_point,
        "space_by_id": {s["id"]: s for s in ifc_data["spaces"]},
        "open_passage_points": open_passage_points,
    }


def _snap_point_into_area(point, area):
    """
    Pull a point to the nearest position inside `area`, for a fire-exit
    door that sits right on the building edge, just outside the walkable
    area.

    Args:
        point: (x, y).
        area: shapely geometry.
    Returns:
        tuple: (x, y), inside `area`.
    Called by: route_all_spaces.
    """
    if area.contains(Point(point)):
        return point
    _, inside = nearest_points(Point(point), area)
    return (inside.x, inside.y)


def _route_through_gates(gates, corners, corner_edges, walkable):
    """
    Route a taut-string path through each gate point in order
    (start -> chain door 1 -> ... -> exit), leg by leg, so the path
    cannot short-cut across a thin wall.

    Args:
        gates: ordered list of (x, y) points to pass through.
        corners, corner_edges: from build_corner_graph.
        walkable: shapely geometry the path must stay inside.
    Returns:
        list[tuple]: the joined path, or None if any leg has no route.
    Called by: route_all_spaces.
    """
    path = [gates[0]]
    for i in range(len(gates) - 1):
        leg, _ = route_from_point(gates[i], gates[i + 1],
                                  corners, corner_edges, walkable)
        if leg is None:
            return None
        path.extend(leg[1:])       # skip the first point (already in path)
    return path


def route_all_spaces(ifc_data, floor_elevation, exit_points):
    """
    Route every space on a floor to the nearest fire exit, using the
    model's semantic door connectivity, then a taut-string path.

    For each space and each candidate exit:
      1. Find the shortest chain of rooms to an exit space, using the
         semantic space graph, so the route only ever passes through
         doors that genuinely connect two rooms.
      2. Build a walkable area restricted to just those chain rooms and
         their connecting doors.
      3. Compute the taut-string path on that restricted area. The
         worst-case start is the corner farthest from the door the route
         leaves the room by; when the room has more than one door onto
         the next room, that door is the one closest to the fire exit
         (shortest routed distance from the door on to the nearest exit).
         For a single-room chain it is the corner farthest from the
         room's nearest own exit. The path is then pushed
         EGRESS_CLEARANCE_M off the walls.

    Args:
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
        exit_points: (x, y) tuple, or list of them — every fire exit on
            this floor. The route to each is computed and the shortest is
            kept, so the reported distance is the distance to the nearest
            exit (Art. 57).
    Returns:
        tuple: (full_area_for_drawing, list of per-space route dicts).
    Called by: route_building.
    """
    if exit_points and not isinstance(exit_points, list):
        exit_points = [exit_points]
    exit_points = exit_points or []

    # A floor served by two or more fire exits allows the 30 m limit;
    # a single-exit floor is a dead-end situation limited to 15 m.
    floor_has_two_exits = len(exit_points) >= 2

    # Full area is kept only for drawing the floor plan behind the paths.
    full_area = build_walkable_area(ifc_data, floor_elevation)

    context = _prepare_routing_context(ifc_data, floor_elevation)
    neighbours     = context["neighbours"]
    boundaries     = context["boundaries"]
    door_by_id     = context["door_by_id"]
    exit_space_ids = context["exit_space_ids"]
    space_farthest_point = context["space_farthest_point"]
    space_by_id    = context["space_by_id"]
    open_passage_points = context["open_passage_points"]

    results = []
    for space in ifc_data["spaces"]:
        if abs(space.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        footprint = space.get("footprint")
        if not footprint or len(footprint) < 3:
            continue
        poly = Polygon(footprint)
        poly = _as_single_polygon(poly)
        if poly.area < 0.05:
            continue

        # 1. Semantic chain of rooms to an exit
        chain = find_space_chain(space["id"], exit_space_ids,
                                 neighbours, space_farthest_point)
        if not chain:
            results.append(_unreachable_result(space))
            continue

        # 2. Walkable area restricted to the chain rooms + their doors
        chain_walkable, chain_door_rects = build_walkable_for_chain(
            chain, ifc_data, floor_elevation)
        if chain_walkable is None or chain_walkable.is_empty:
            results.append(_unreachable_result(space))
            continue

        # Route on the version of this area already eroded EGRESS_CLEARANCE_M from
        # every wall, column and obstacle, doors reopened to full width.
        routing_area = _erode_with_clearance(chain_walkable, chain_door_rects,
                                             clearance=EGRESS_CLEARANCE_M)

        # Simplify to a 2 cm tolerance before building the corner graph.
        # The visibility graph checks every pair of corners against each
        # other (cost grows with the square of the corner count), and an
        # eroded circular column can leave >100 corners; 2 cm is well
        # inside the EGRESS_CLEARANCE_M clearance and negligible against the 15/30 m
        # Article 57 limits.
        routing_area = routing_area.simplify(0.02, preserve_topology=True)

        # Erosion can split a genuinely thin connecting passage into a
        # MultiPolygon: the main room area, plus a sliver a few
        # centimetres across left over from buffer/simplify rounding —
        # not a real second room. Confirmed on a real model: reducing
        # EGRESS_CLEARANCE_M from 0.40 m to 0.30 m turned one such passage
        # from "fully eroded away" into "barely surviving as a 0.013 m²
        # fragment", and a room's own start point happened to snap into
        # that fragment, stranding it from every door and exit gate even
        # though the real, substantial walkable area was still fully
        # connected. Same philosophy as _as_single_polygon: keep only the
        # largest piece, treating the rest as a data-quality artefact of
        # this erosion pass rather than a real, separate walkable area.
        if routing_area.geom_type == "MultiPolygon":
            routing_area = max(routing_area.geoms, key=lambda g: g.area)

        corners, corner_edges = build_corner_graph(routing_area)

        # 3. Try every fire exit on this floor and keep the shortest route.
        best_path = None
        best_dist = None
        best_start = None

        # Single-room chain (len(chain) == 1): the space itself already
        # carries one or more fire-exit doors on its own boundary. The
        # worst-case start point is found once, using all of this floor's
        # exits jointly, and reused for every candidate exit below — see
        # _farthest_corner_from_nearest for why it must NOT be recomputed
        # per exit_point.
        single_room_start = None
        # For a single-room chain, only THIS room's own recorded fire-exit
        # door(s) are real candidate exits — trying every exit on the
        # floor (as the multi-room case does further below via the chain
        # graph) has no chain of rooms behind it here to make those other
        # exits reachable, so snapping a distant, unrelated exit onto this
        # one room's small boundary produces a fake "shortest path" that
        # is really just the nearest wall point to that distant exit, not
        # a real route to it. Confirmed on a real model: a room whose own
        # exit door was correctly found still kept routing to a snapped
        # point on its far wall, because that snap distance for an
        # unrelated, far-away floor exit happened to come out shorter.
        own_exit_points = exit_points
        if len(chain) == 1:
            own_exit_points = [
                (door_by_id[door_id]["location"][0], door_by_id[door_id]["location"][1])
                for (space_id, door_id) in boundaries
                if space_id == chain[0]
                and door_by_id.get(door_id, {}).get("is_fire_exit")
                and door_by_id[door_id].get("location")
            ] or exit_points   # fallback: keep old behaviour if none recorded
            single_room_start = _farthest_corner_from_nearest(poly, own_exit_points)

        # The room's own first hop (chain[0] -> chain[1]) is handled
        # separately from the rest of the chain, because it also fixes the
        # worst-case start point: Article 57 measures from the corner
        # farthest from wherever the route actually leaves the room, so
        # that corner must be found relative to the SAME door (or
        # open-passage midpoint) the route then uses for this hop.
        #
        # When the room has more than one door onto the next room, the
        # occupant is taken to leave through whichever of them is closest
        # to the fire exit, and the start corner and the whole path are
        # measured against that one door (see the collapse step below).
        # This is a deliberate, defined choice — not the earlier bug where
        # the start corner came from one single-door heuristic while the
        # path separately tried every door, so a two-door room could have
        # its farthest corner measured against a door the route never used.
        if len(chain) >= 2 and single_room_start is None:
            hop0_candidates = _door_candidates_between(
                chain[0], chain[1], boundaries, door_by_id, space_by_id)
            if not hop0_candidates:
                open_pt = (open_passage_points or {}).get((chain[0], chain[1]))
                hop0_candidates = [open_pt] if open_pt else [None]
        else:
            hop0_candidates = [None]

        # Every combination of door choices across the REMAINING hops
        # (chain[1:]); reused as-is regardless of which hop-0 candidate is
        # being tried, since later hops do not affect the start point.
        rest_combos = (_chain_door_combinations(chain[1:], boundaries,
                                                door_by_id, space_by_id)
                       if len(chain) > 2 else [[]])

        # Collapse several hop-0 doors to the single one nearest the fire
        # exit: the occupant leaves by the door closest to the exit, then
        # every step below (worst-case corner, full route) runs against
        # that one door. "Nearest" = shortest routed distance from the
        # door on through the rest of the chain to the nearest exit;
        # straight-line distance to the nearest exit is the fallback when
        # no such route can be built.
        real_hop0 = [p for p in hop0_candidates if p is not None]
        if len(real_hop0) > 1:
            def _hop0_distance_to_exit(door_pt):
                gate0 = _snap_point_into_area(door_pt, routing_area)
                best_routed = None
                for exit_point in own_exit_points:
                    exit_in_area = _snap_point_into_area(exit_point, routing_area)
                    for rest_points in rest_combos:
                        rest_snapped = [_snap_point_into_area(p, routing_area)
                                        for p in rest_points]
                        tail = _route_through_gates(
                            [gate0] + rest_snapped + [exit_in_area],
                            corners, corner_edges, routing_area)
                        if not tail:
                            continue
                        d = sum(math.hypot(tail[k + 1][0] - tail[k][0],
                                           tail[k + 1][1] - tail[k][1])
                                for k in range(len(tail) - 1))
                        if best_routed is None or d < best_routed:
                            best_routed = d
                if best_routed is not None:
                    return (0, best_routed)
                straight = min(math.hypot(door_pt[0] - ep[0], door_pt[1] - ep[1])
                               for ep in own_exit_points)
                return (1, straight)
            hop0_candidates = [min(real_hop0, key=_hop0_distance_to_exit)]

        for exit_point in own_exit_points:
            exit_in_area = _snap_point_into_area(exit_point, routing_area)

            for hop0_pt in hop0_candidates:
                if single_room_start is not None:
                    start_point = single_room_start
                elif hop0_pt is not None:
                    start_point = _farthest_corner_from(poly, hop0_pt)
                else:
                    # No door and no open passage recorded for this hop —
                    # same last-resort fallback the old code used.
                    start_point = _farthest_corner_from(poly, exit_point)

                # The EGRESS_CLEARANCE_M clearance applies at the start and exit points
                # too, not only along the interior of the path: a raw room
                # corner or door threshold is not itself a valid standing
                # position, so both are snapped to the nearest point
                # inside routing_area.
                start_point = _snap_point_into_area(start_point, routing_area)

                # 4. Gates the path must pass through, in order: start ->
                #    hop-0 door (the same one the start point was measured
                #    against) -> every remaining chain door -> exit. When a
                #    later hop has more than one real connecting door,
                #    every combination is tried here and the shortest
                #    resulting route kept — see _chain_door_combinations.
                #    Every gate is snapped into routing_area first, not
                #    only the start and exit points: a real door survives
                #    erosion because its rectangle is reopened to full
                #    width afterwards, but an open-passage midpoint (no
                #    discrete door, e.g. a Space Separator) gets no such
                #    reopening and sits exactly EGRESS_CLEARANCE_M outside
                #    the eroded area on the original, un-eroded boundary —
                #    an unsnapped gate there makes the route fail entirely,
                #    wrongly reporting the room unreachable even though the
                #    chain and the underlying geometry were both fine.
                hop0_gate = ([_snap_point_into_area(hop0_pt, routing_area)]
                            if hop0_pt is not None else [])
                for rest_points in rest_combos:
                    rest_snapped = [_snap_point_into_area(p, routing_area)
                                    for p in rest_points]
                    door_points = hop0_gate + rest_snapped
                    gates = [start_point] + door_points + [exit_in_area]
                    path = _route_through_gates(gates, corners, corner_edges,
                                                routing_area)
                    if not path:
                        continue

                    dist = sum(math.hypot(path[k+1][0]-path[k][0],
                                          path[k+1][1]-path[k][1])
                               for k in range(len(path)-1))

                    # Keep the route to the nearest exit, over every
                    # hop-0/rest-of-chain combination and every candidate
                    # exit.
                    if best_dist is None or dist < best_dist:
                        best_path, best_dist, best_start = path, dist, start_point

        results.append({
            "space_id":      space.get("id"),
            "name":          space.get("name", "?"),
            "start":         best_start,
            "path":          best_path,
            "distance":      best_dist,
            "reachable":     best_path is not None,
            "has_two_exits": floor_has_two_exits,
        })
    return full_area, results


def find_floor_exits(ifc_data, floor_elevation):
    """
    Find the fire-exit door points on one floor (doors already marked
    is_fire_exit in the model, whose own storey matches this floor).

    Args:
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        list[tuple]: (x, y) exit points (usually one per floor).
    Called by: route_building.
    """
    exits = []
    for door in ifc_data["doors"]:
        if not door.get("is_fire_exit"):
            continue
        if abs(door.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        location = door.get("location")
        if location:
            exits.append((location[0], location[1]))
    return exits


def list_floor_elevations(ifc_data):
    """
    Return the sorted list of distinct storey elevations that have spaces.

    Args:
        ifc_data: parsed data dict.
    Returns:
        list[float]
    Called by: route_building.
    """
    elevations = set()
    for space in ifc_data["spaces"]:
        elevations.add(round(space.get("storey_elevation_m", 0.0), 2))
    return sorted(elevations)


def route_building(ifc_data):
    """
    Route every occupiable space of every storey to a fire exit on its own
    storey. Each floor is processed independently — there is no travel
    between floors.

    Args:
        ifc_data: parsed data dict.
    Returns:
        dict: floor_elevation -> {area, exit, exits, results}
            (see route_all_spaces for "results").
    Called by: build_path_results.
    """
    building = {}
    for elevation in list_floor_elevations(ifc_data):
        exits = find_floor_exits(ifc_data, elevation)
        if not exits:
            # No fire exit on this floor: report all its spaces as unreachable.
            area = build_walkable_area(ifc_data, elevation)
            results = []
            for space in ifc_data["spaces"]:
                if abs(space.get("storey_elevation_m", 0.0) - elevation) > 1.0:
                    continue
                results.append(_unreachable_result(space))
            building[elevation] = {"area": area, "exit": None,
                                   "exits": [],
                                   "results": results}
            continue

        # Route to ALL fire exits on this floor; route_all_spaces keeps the
        # shortest route per space (nearest-exit rule of Art. 57) and marks
        # the floor as two-exit when more than one exit is present.
        area, results = route_all_spaces(ifc_data, elevation, exits)
        building[elevation] = {"area": area,
                               "exit": exits[0],   # kept for compatibility
                               "exits": exits,     # all exits on this floor
                               "results": results}
    return building


def build_path_results(ifc_data):
    """
    Run the whole-building routing and translate it into the path_results
    list that rule_checker, bcf_reporter and the visualizer read.

    Args:
        ifc_data: parsed data dict. Spaces tagged non_occupiable=Yes are
            excluded upstream by ifc_parser and never appear here.
    Returns:
        tuple: (path_results, building)
            path_results: list[dict] {space_id, space_name, reachable,
                travel_distance_m, has_two_exits, path, start,
                storey_elevation_m, storey_name}.
            building: from route_building.
    Called by: main.py (run).
    """
    building = route_building(ifc_data)

    # Index spaces by their unique GlobalId (room names are not unique).
    space_by_id = {space["id"]: space for space in ifc_data["spaces"]}

    path_results = []
    for elevation, floor in building.items():
        for result in floor["results"]:
            space_id  = result.get("space_id")
            space     = space_by_id.get(space_id, {})
            reachable = result["reachable"]

            # Single travel distance: farthest walkable point -> fire exit.
            distance = round(result["distance"], 2) if reachable else None

            path_results.append({
                "space_id":          space_id if space_id else result["name"],
                "space_name":        space.get("name", result["name"]),
                "reachable":         reachable,
                "travel_distance_m": distance,
                "has_two_exits":     result.get("has_two_exits", False),
                "path":              result["path"],
                "start":             result["start"],
                "storey_elevation_m": round(elevation, 2),
                "storey_name":       space.get("storey_name"),
            })
    return path_results, building


def build_space_graph(ifc_data, floor_elevation):
    """
    Build the semantic room-to-room graph from IfcRelSpaceBoundary data: a
    door that connects two spaces becomes an edge between those two rooms.
    Uses the model's own connectivity data, not geometric guessing, so a
    path can only pass through doors that genuinely link two rooms.

    Args:
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        tuple: (neighbours: dict space_id -> list of (other_space_id,
            door_id), open_passage_points: from _add_open_passages).
    Called by: _prepare_routing_context.
    """
    boundaries = same_floor_boundaries(ifc_data, floor_elevation)
    door_by_id = {d["id"]: d for d in ifc_data["doors"]}
    space_by_id = {s["id"]: s for s in ifc_data["spaces"]}

    # group boundary records by door -> which spaces each door touches
    spaces_of_door = {}
    for space_id, door_id in boundaries:
        spaces_of_door.setdefault(door_id, [])
        if space_id not in spaces_of_door[door_id]:
            spaces_of_door[door_id].append(space_id)

    # A door only really connects two rooms if its rectangle actually
    # reaches both of them. Validating the semantic link against the
    # geometry removes phantom connections (an exterior door listed for an
    # inner room, or a door where three rooms nearly meet).
    valid_rooms_of_door = {}
    for door_id, space_ids in spaces_of_door.items():
        valid_rooms_of_door[door_id] = _rooms_really_at_door(
            space_ids, door_id, door_by_id, space_by_id)

    # Count how many valid links each room would keep, so a room is never
    # left with no link at all (a doubtful link beats an isolated room).
    valid_link_count = {}
    for door_id, rooms in valid_rooms_of_door.items():
        if len(rooms) >= 2:
            for room in rooms:
                valid_link_count[room] = valid_link_count.get(room, 0) + 1

    neighbours = {}
    for door_id, space_ids in spaces_of_door.items():
        rooms = valid_rooms_of_door[door_id]
        if len(rooms) < 2:
            # keep the raw link only if a room would otherwise be isolated
            if any(valid_link_count.get(room, 0) == 0 for room in space_ids):
                rooms = space_ids
            else:
                continue

        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                room_a, room_b = rooms[i], rooms[j]
                neighbours.setdefault(room_a, []).append((room_b, door_id))
                neighbours.setdefault(room_b, []).append((room_a, door_id))

    open_passage_points = _add_open_passages(neighbours, ifc_data, floor_elevation)
    return neighbours, open_passage_points


# A room separator only counts as a passage if it is at least this wide,
# otherwise a few centimetres of geometric noise would become a short-cut.
MIN_OPEN_PASSAGE_M = 0.90


def _add_open_passages(neighbours, ifc_data, floor_elevation):
    """
    Connect spaces separated only by a room separator (a virtual boundary
    — e.g. an open kitchen and the living room next to it) exactly like
    two rooms sharing a door.

    Recognised by two matching facts:
      1. geometry — the two footprints touch along an edge of at least
         MIN_OPEN_PASSAGE_M, and
      2. semantics — at least one of the two spaces declares a VIRTUAL
         space boundary whose length matches that shared edge.
    Requiring (2) at all — on either side — prevents a false passage in a
    model whose space footprints are drawn to wall centre-lines, where
    every neighbour would touch even though the boundary is physical.
    Requiring it on BOTH sides, tried first, turned out too strict for a
    real model where a newly added space had its own boundary classified
    PHYSICAL against a neighbour that correctly declared the same edge
    VIRTUAL: the two rooms visibly touched, one side properly declared the
    opening, yet the pair was rejected because the other side didn't — an
    asymmetric export, not a genuine wall. Either side confirming is
    enough: it still requires a real architect-placed separator to exist
    somewhere on the boundary, which is what rules out the wall-centreline
    false positive (2) exists to guard against.

    No gate point is forced at this opening: build_walkable_for_chain
    unions the two touching footprints into one polygon, so the
    taut-string path is free to cross anywhere along the shared edge.

    Args:
        neighbours: the room graph being built (mutated in place).
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        dict: {(id_a, id_b): (mx, my), (id_b, id_a): (mx, my)} — the
            midpoint of each shared open edge, used only as a reference
            direction by route_all_spaces when a chain's first hop has no
            discrete door, not as a routing constraint.
    Called by: build_space_graph.
    """
    virtual_lengths = ifc_data.get("virtual_boundaries") or {}
    if not virtual_lengths:
        return {}

    open_passage_points = {}
    rooms = []
    for space in ifc_data["spaces"]:
        if abs(space.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        if not space.get("footprint"):
            continue
        polygon = Polygon(space["footprint"])
        polygon = _as_single_polygon(polygon)
        if polygon.area > 0.05:
            rooms.append((space["id"], polygon))

    LENGTH_TOLERANCE = 0.6      # m, between the shared edge and the boundary
    for i in range(len(rooms)):
        id_a, poly_a = rooms[i]
        for j in range(i + 1, len(rooms)):
            id_b, poly_b = rooms[j]
            if poly_a.distance(poly_b) > 0.001:
                continue                      # a wall stands between them

            shared = poly_a.buffer(0.001).intersection(poly_b.buffer(0.001))
            if shared.is_empty:
                continue
            shared_length = shared.length / 2.0
            if shared_length < MIN_OPEN_PASSAGE_M:
                continue                      # too narrow to walk through

            declared_a = any(abs(length - shared_length) < LENGTH_TOLERANCE
                             for length in virtual_lengths.get(id_a, []))
            declared_b = any(abs(length - shared_length) < LENGTH_TOLERANCE
                             for length in virtual_lengths.get(id_b, []))
            if not (declared_a or declared_b):
                continue                      # not declared as a separator

            neighbours.setdefault(id_a, []).append((id_b, None))
            neighbours.setdefault(id_b, []).append((id_a, None))

            midpoint = shared.centroid
            open_passage_points[(id_a, id_b)] = (midpoint.x, midpoint.y)
            open_passage_points[(id_b, id_a)] = (midpoint.x, midpoint.y)

    return open_passage_points


def _rooms_really_at_door(space_ids, door_id, door_by_id, space_by_id):
    """
    Return only the rooms whose footprint actually overlaps the door
    rectangle, dropping a boundary recorded in the IFC that has no
    geometric meaning.

    Args:
        space_ids: candidate space_ids the door is linked to.
        door_id: the door's id.
        door_by_id, space_by_id: lookups by id.
    Returns:
        list[str]: space_ids the door genuinely reaches.
    Called by: build_space_graph.
    """
    door = door_by_id.get(door_id)
    if not door:
        return space_ids
    rectangle = build_door_rectangle(door)
    if rectangle is None:
        return space_ids

    OVERLAP_MIN = 0.01   # m2 — a real doorway overlaps clearly more than this
    really = []
    for space_id in space_ids:
        space = space_by_id.get(space_id)
        if not space or not space.get("footprint"):
            continue
        room_polygon = Polygon(space["footprint"])
        room_polygon = _as_single_polygon(room_polygon)
        if rectangle.intersection(room_polygon).area > OVERLAP_MIN:
            really.append(space_id)
    return really


def find_space_chain(start_space_id, exit_space_ids, neighbours,
                     space_reference_point):
    """
    Find the shortest chain of rooms from the start space to any exit
    space, using Dijkstra over the semantic space graph. Edge cost =
    distance between each room's own reference point (its vertex farthest
    from its own centroid, not the centroid itself — see
    _prepare_routing_context) — used only to choose among alternative
    chains; the final travel distance comes from the taut-string path,
    not from here. Using the farthest-vertex reference rather than the
    centroid is a more conservative (larger) distance estimate for an
    elongated or L-shaped room, at the same cost, and does not depend on
    which door the chain eventually uses (unlike the room's own
    worst-case start point; see _farthest_corner_from's other caller).

    Args:
        start_space_id: space_id to route from.
        exit_space_ids: set of space_ids that contain a fire-exit door.
        neighbours: room graph (see build_space_graph).
        space_reference_point: space_id -> (x, y); see
            _prepare_routing_context's space_farthest_point.
    Returns:
        list[str] or None: space_ids (start .. exit), or None if unreachable.
    Called by: route_all_spaces.
    """
    import heapq
    if start_space_id in exit_space_ids:
        return [start_space_id]

    best_cost = {start_space_id: 0.0}
    came_from = {start_space_id: None}
    heap = [(0.0, start_space_id)]
    while heap:
        cost, room = heapq.heappop(heap)
        if room in exit_space_ids:
            chain = []
            current = room
            while current is not None:
                chain.append(current)
                current = came_from[current]
            chain.reverse()
            return chain
        if cost > best_cost.get(room, float("inf")):
            continue
        for (neighbour, door_id) in neighbours.get(room, []):
            point_a = space_reference_point.get(room)
            point_b = space_reference_point.get(neighbour)
            if point_a and point_b:
                step_distance = math.hypot(point_a[0]-point_b[0],
                                           point_a[1]-point_b[1])
            else:
                step_distance = 1.0
            new_cost = cost + step_distance
            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                came_from[neighbour] = room
                heapq.heappush(heap, (new_cost, neighbour))
    return None


def _columns_on_floor(ifc_data, floor_elevation):
    """
    Return the footprint polygons of the IfcColumns on this floor — solid
    obstacles people walk around, not through.

    Matches on the column's own base (z_min_m), not a floor it merely
    passes through on the way up: a column rising from Ground Floor to
    Roof belongs to Ground Floor only, even though it physically
    continues through First Floor. The base may sit up to
    COLUMN_BASE_ABOVE_FLOOR_M above the floor's own elevation (a plinth,
    pedestal, or base plate raised slightly above finished floor level)
    but never below it — a column's base can't be attributed to a floor
    underneath where it actually starts.

    Args:
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        list[shapely.Polygon]
    Called by: build_walkable_for_chain, build_walkable_area.
    """
    COLUMN_BASE_ABOVE_FLOOR_M = 0.5
    column_polygons = []
    for column in ifc_data.get("columns", []):
        z_min = column.get("z_min_m")
        base = z_min if z_min is not None else column.get("storey_elevation_m", 0.0)
        if not (floor_elevation - 0.02 <= base
                <= floor_elevation + COLUMN_BASE_ABOVE_FLOOR_M):
            continue
        footprint = column.get("footprint")
        if not footprint or len(footprint) < 3:
            continue
        polygon = Polygon(footprint)
        polygon = _as_single_polygon(polygon)
        if polygon.area > 0.001:
            column_polygons.append(polygon)
    return column_polygons


def _obstacles_on_floor(ifc_data, floor_elevation):
    """
    Return the footprint polygons of the fixed obstacles on this floor —
    IfcFurnishingElements and IfcDistributionElements (MEP/plumbing
    fixtures) tagged Egress_Obstacle=Yes — walked around exactly like
    columns.

    Args:
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        list[shapely.Polygon]
    Called by: build_walkable_for_chain, build_walkable_area.
    """
    obstacle_polygons = []
    for obstacle in ifc_data.get("obstacles", []):
        if abs(obstacle.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        footprint = obstacle.get("footprint")
        if not footprint or len(footprint) < 3:
            continue
        polygon = Polygon(footprint)
        polygon = _as_single_polygon(polygon)
        if polygon.area > 0.001:
            obstacle_polygons.append(polygon)
    return obstacle_polygons


def _as_single_polygon(polygon):
    """
    Repair an invalid polygon with buffer(0) and guarantee a single
    Polygon result, never a MultiPolygon. A genuinely self-intersecting
    input can legitimately split into several pieces under buffer(0); only
    the largest piece by area is kept, treating the smaller pieces as a
    data-quality artefact rather than a real second room.

    Args:
        polygon: shapely.Polygon (possibly invalid).
    Returns:
        shapely.Polygon: valid, single polygon.
    Called by: route_all_spaces, _add_open_passages, _rooms_really_at_door,
        _columns_on_floor, _obstacles_on_floor, build_walkable_for_chain,
        build_walkable_area.
    """
    if polygon.is_valid:
        return polygon
    fixed = polygon.buffer(0)
    if fixed.geom_type == "MultiPolygon":
        return max(fixed.geoms, key=lambda g: g.area)
    return fixed


def _subtract_polygons(room_polygon, obstacle_polygons):
    """
    Cut a set of obstacle polygons (columns, furniture, etc.) out of a
    room's polygon, so routing treats them as holes to walk around. Skips
    an obstacle if removing it would break the room apart or eat most of
    its area — safer to ignore an obstacle than make a real space
    unreachable.

    Also guards against a subtler version of the same failure, one step
    downstream: several obstacles can each individually pass every check
    above, yet once every one of them is a hole, the routing clearance
    erosion that follows (EGRESS_CLEARANCE_M, in _erode_with_clearance)
    pinches off a thin connecting web between them and fragments the room
    anyway — confirmed on a real model where 11 small furniture obstacles
    (each ~0.36 m2) fragmented one room's routing area into 7 disconnected
    pieces under 0.4m clearance, though the same room's plain outer
    boundary eroded cleanly with none of them cut out. If cutting the
    obstacles out this way would disconnect a room that erodes fine
    without them, this ignores all of them for this room rather than sever
    its own connectivity — the same principle as the checks above, just
    applied past the direct cut to the clearance step that follows it.

    Args:
        room_polygon: shapely.Polygon of the room.
        obstacle_polygons: list of shapely.Polygon obstacles.
    Returns:
        shapely.Polygon: room polygon with obstacles cut out.
    Called by: build_walkable_for_chain, build_walkable_area.
    """
    result = room_polygon
    for obstacle in obstacle_polygons:
        if not result.intersects(obstacle):
            continue
        cut = result.difference(obstacle)
        if cut.is_empty:
            continue                      # would erase the room
        if cut.geom_type != "Polygon":
            continue                      # would split the room in pieces
        if cut.area < 0.5 * result.area:
            continue                      # would eat most of the room
        result = cut

    if result is not room_polygon:
        original_eroded = room_polygon.buffer(-EGRESS_CLEARANCE_M)
        was_connected = (original_eroded.geom_type == "Polygon"
                         and not original_eroded.is_empty)
        if was_connected:
            result_eroded = result.buffer(-EGRESS_CLEARANCE_M)
            still_connected = (result_eroded.geom_type == "Polygon"
                              and not result_eroded.is_empty)
            if not still_connected:
                return room_polygon

    return result


def build_walkable_for_chain(chain, ifc_data, floor_elevation):
    """
    Build a walkable area containing only the rooms in `chain` plus the
    doors connecting consecutive rooms in it, forcing the taut string
    through the correct semantic doors and never through an unrelated one.

    Args:
        chain: list of space_ids (start .. exit space).
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        tuple: (walkable_polygon, door_rectangles)
            walkable_polygon: union of the chain's room footprints (columns
                and obstacles cut out as holes) and connecting door
                rectangles, or None if empty.
            door_rectangles: the door rectangles on their own, needed
                separately to re-open doorways after clearance erosion
                (see _erode_with_clearance).
    Called by: route_all_spaces.
    """
    space_by_id = {s["id"]: s for s in ifc_data["spaces"]}
    boundaries = same_floor_boundaries(ifc_data, floor_elevation)
    chain_rooms = set(chain)

    # 1. Footprints of the chain's rooms, with columns/obstacles cut out
    column_polygons = _columns_on_floor(ifc_data, floor_elevation)
    obstacle_polygons = _obstacles_on_floor(ifc_data, floor_elevation)
    # Door/bridge rectangles are re-opened at their full raw width AFTER
    # the walkable area is eroded by EGRESS_CLEARANCE_M (see
    # _erode_with_clearance) — otherwise a doorway narrower than twice the
    # clearance would be sealed shut. But a rectangle re-opened at raw
    # width can itself run flush against an obstacle (e.g. an open-plan
    # bridge spanning most of a shared wall, with a piece of furniture
    # pushed into that same wall) and silently reintroduce a zero-clearance
    # gap right next to it. Cutting the CLEARANCE-BUFFERED obstacle/column
    # footprints out of every door/bridge rectangle below, instead of their
    # raw footprints, keeps the EGRESS_CLEARANCE_M clearance intact wherever a
    # rectangle is re-opened.
    obstacle_clearance_polygons = [o.buffer(EGRESS_CLEARANCE_M) for o in obstacle_polygons]
    column_clearance_polygons = [c.buffer(EGRESS_CLEARANCE_M) for c in column_polygons]
    parts = []
    for space_id in chain:
        space = space_by_id.get(space_id)
        if not space or not space.get("footprint"):
            continue
        room_polygon = Polygon(space["footprint"])
        room_polygon = _as_single_polygon(room_polygon)
        if room_polygon.area > 0.05:
            if column_polygons:
                room_polygon = _subtract_polygons(room_polygon, column_polygons)
            if obstacle_polygons:
                room_polygon = _subtract_polygons(room_polygon, obstacle_polygons)
            parts.append(room_polygon)

    # 2. Doors connecting two rooms that are BOTH in the chain
    rooms_of_door = {}
    for space_id, door_id in boundaries:
        rooms_of_door.setdefault(door_id, []).append(space_id)

    door_by_id = {d["id"]: d for d in ifc_data["doors"]}
    door_rectangles = []
    for door_id, touched_rooms in rooms_of_door.items():
        rooms_in_chain = [r for r in touched_rooms if r in chain_rooms]
        door = door_by_id.get(door_id)
        # An interior (pass-through) door needs both rooms it connects
        # present in the chain, so a path cannot shortcut into a room the
        # chain never intended to cross. A fire-exit door is different: it
        # is always the chain's own TERMINAL door for whichever room it
        # sits on, even when the model also happens to link that same
        # door to a second, unrelated room (e.g. a fire-exit door shared
        # between a room and the corridor it also opens onto) — so one
        # matching chain room is enough to let it through, regardless of
        # how many rooms the recorded boundary otherwise touches.
        # Left unpatched, a single-room chain (a room carrying its own
        # fire-exit door) never satisfies ">= 2" — the chain has only one
        # room — so that door is never added to its routing area, and the
        # exit point gets snapped to the nearest point on the room's
        # plain boundary instead of the real door: a straight line that
        # lands near, but not on, the door it is meant to reach.
        is_own_exit_door = bool(door and door.get("is_fire_exit") and rooms_in_chain)
        if len(rooms_in_chain) >= 2 or is_own_exit_door:
            if door:
                door_rectangle = build_door_rectangle(door)
                if door_rectangle is not None:
                    if column_clearance_polygons:
                        door_rectangle = _subtract_polygons(door_rectangle, column_clearance_polygons)
                    if obstacle_clearance_polygons:
                        door_rectangle = _subtract_polygons(door_rectangle, obstacle_clearance_polygons)
                    if not door_rectangle.is_empty:
                        door_rectangles.append(door_rectangle)
    parts.extend(door_rectangles)

    # 3. Virtual (open-plan) boundaries between consecutive chain rooms.
    # Two footprints that merely sit very close together (common in IFC
    # exports, off by a fraction of a millimetre) are not automatically
    # merged by a plain union — shapely only merges polygons that truly
    # touch or overlap. This bridges that gap the same way a physical door
    # is bridged: a polygon spanning the full shared edge, reaching
    # DOOR_HALF_DEPTH into each room so it survives clearance erosion.
    virtual_lengths = ifc_data.get("virtual_boundaries") or {}
    if virtual_lengths:
        chain_polys = {sid: Polygon(space_by_id[sid]["footprint"])
                       for sid in chain if space_by_id.get(sid)
                       and space_by_id[sid].get("footprint")}
        for i in range(len(chain) - 1):
            id_a, id_b = chain[i], chain[i + 1]
            poly_a, poly_b = chain_polys.get(id_a), chain_polys.get(id_b)
            if poly_a is None or poly_b is None:
                continue
            if poly_a.distance(poly_b) > 0.001:
                continue                          # a wall stands between them
            shared = poly_a.buffer(0.001).intersection(poly_b.buffer(0.001))
            if shared.is_empty:
                continue
            shared_length = shared.length / 2.0
            if shared_length < MIN_OPEN_PASSAGE_M:
                continue
            declared_a = any(abs(length - shared_length) < 0.6
                             for length in virtual_lengths.get(id_a, []))
            declared_b = any(abs(length - shared_length) < 0.6
                             for length in virtual_lengths.get(id_b, []))
            # Either side declaring the separator is enough — see
            # _add_open_passages for why requiring both sides is too
            # strict for a real, asymmetrically-classified export.
            if not (declared_a or declared_b):
                continue                          # not declared as a separator
            bridge = shared.buffer(DOOR_HALF_DEPTH, cap_style=2)
            if bridge.is_empty:
                continue
            # The bridge must keep the same clearance from columns and
            # obstacles as the room polygons in step 1, otherwise an
            # obstacle sitting near the shared edge gets silently filled
            # back in when this bridge is re-opened after erosion.
            if column_clearance_polygons:
                bridge = _subtract_polygons(bridge, column_clearance_polygons)
            if obstacle_clearance_polygons:
                bridge = _subtract_polygons(bridge, obstacle_clearance_polygons)
            if bridge.is_empty:
                continue
            parts.append(bridge)
            door_rectangles.append(bridge)

    walkable_polygon = _safe_union(parts)
    return walkable_polygon, door_rectangles


def build_walkable_area(ifc_data, floor_elevation):
    """
    Build the walkable area of one floor: every space's footprint (columns
    and obstacles cut out as holes) plus a rectangle over every door that
    connects spaces on this floor.

    Args:
        ifc_data: parsed data dict.
        floor_elevation: this floor's elevation.
    Returns:
        dict: {
            "spaces": list of (space_id, name, Polygon) — id kept so
                callers (the visualizer) can match a drawn room to a
                compliance result by unique id, not name.
            "doors": list of Polygon door rectangles.
            "columns": list of Polygon.
            "obstacles": list of Polygon.
            "walkable": one shapely geometry = union of rooms + doors.
        }
    Called by: route_all_spaces, route_building.
    """
    column_polygons = _columns_on_floor(ifc_data, floor_elevation)
    obstacle_polygons = _obstacles_on_floor(ifc_data, floor_elevation)
    room_polygons = []
    for space in ifc_data["spaces"]:
        if abs(space.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        footprint = space.get("footprint")
        if not footprint or len(footprint) < 3:
            continue
        polygon = Polygon(footprint)
        polygon = _as_single_polygon(polygon)
        if polygon.area > 0.05:
            if column_polygons:
                polygon = _subtract_polygons(polygon, column_polygons)
            if obstacle_polygons:
                polygon = _subtract_polygons(polygon, obstacle_polygons)
            room_polygons.append((space["id"], space.get("name", "?"),
                                  polygon))

    # Only doors that connect spaces on this floor (same_floor_boundaries),
    # which also drops any cross-floor data errors.
    boundaries = same_floor_boundaries(ifc_data, floor_elevation)
    door_ids_here = set(door_id for (_, door_id) in boundaries)
    door_by_id = {d["id"]: d for d in ifc_data["doors"]}

    door_rectangles = []
    for door_id in door_ids_here:
        door = door_by_id.get(door_id)
        if not door:
            continue
        rectangle = build_door_rectangle(door)
        if rectangle is not None:
            door_rectangles.append(rectangle)

    all_parts = [polygon for (_, _, polygon) in room_polygons] + door_rectangles
    walkable = _safe_union(all_parts)

    return {"spaces": room_polygons, "doors": door_rectangles,
            "columns": column_polygons,
            "obstacles": obstacle_polygons,
            "walkable": walkable}
