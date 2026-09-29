"""
visualizer.py
Presentation layer for the SCIE / accessibility pipeline.

Pure renderer: takes the data structures the routing engine already
produced (the walkable `area` dict and the routed `results` list) and
draws them. Never re-computes geometry, so what is drawn is exactly what
the engine measured.
"""


def _draw_ramp_context(ax, ifc_data, ramp_id, violating_ids, floor_elev):
    """
    Draw one ramp's floor-plan context on an existing Axes: every room and
    column on its floor, every landing (grey, labelled "L<n>"), and every
    flight (blue, labelled "F<n>", or red + labelled if its id is in
    `violating_ids`). Shared by generate_violation_snapshot (one BCF
    topic's own flight red) and draw_ramp_snapshot (every currently-known
    violating flight on the ramp red).

    Args:
        ax: a matplotlib Axes already created by the caller.
        ifc_data: parsed data dict (spaces, columns, ramps, ramp_landings).
        ramp_id: the parent IfcRamp GlobalId to draw flights/landings for.
        violating_ids: set of flight ids (see ifc_parser.get_ramps "id")
            to fill red instead of blue.
        floor_elev: the ramp's storey elevation, for room/column filtering.
    Returns:
        list[tuple]: (minx, miny, maxx, maxy) bounds of every flight drawn
            — the zoom window the caller should fit to. Empty if the ramp
            has no flights with corners.
    Called by: generate_violation_snapshot, draw_ramp_snapshot.
    """
    from shapely.geometry import Polygon

    # Rooms on the same floor, for context — a ramp commonly starts below
    # the first room-bearing storey, so this can legitimately be empty.
    for other in ifc_data.get("spaces", []):
        if abs(other.get("storey_elevation_m", 0.0) - floor_elev) > 1.0:
            continue
        if not other.get("footprint"):
            continue
        poly = Polygon(other["footprint"])
        if not poly.is_valid or poly.area <= 0:
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="#eaf4ff", alpha=0.5, ec="#3070b0",
                lw=0.8, zorder=1)

    # Columns on the same floor — extra spatial context on a floor with
    # few or no modelled rooms, so the ramp is never the only thing on an
    # otherwise featureless plot. Matched on the column's own base
    # (z_min_m), not a floor it merely passes through — see
    # RT_SCIE_checker._columns_on_floor.
    COLUMN_BASE_ABOVE_FLOOR_M = 0.5
    for column in ifc_data.get("columns", []):
        z_min = column.get("z_min_m")
        base = z_min if z_min is not None else column.get("storey_elevation_m", 0.0)
        if not (floor_elev - 0.02 <= base
                <= floor_elev + COLUMN_BASE_ABOVE_FLOOR_M):
            continue
        if not column.get("footprint"):
            continue
        poly = Polygon(column["footprint"])
        if not poly.is_valid or poly.area <= 0:
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="#8B4513", alpha=0.9, ec="#4a2508",
                lw=0.6, zorder=1)

    for landing in ifc_data.get("ramp_landings", []):
        if landing["ramp_id"] != ramp_id:
            continue
        corners = landing.get("corners")
        if not corners:
            continue
        xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
        ax.fill(xs, ys, color="#c4c9cf", alpha=0.85, ec="#5a5f66",
                lw=1.0, zorder=2)
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        ax.text(cx, cy, f"L{landing.get('landing_index', '')}",
                ha="center", va="center", fontsize=5.5, zorder=5)

    bounds = []
    for flight in ifc_data.get("ramps", []):
        if flight["ramp_id"] != ramp_id:
            continue
        corners = flight.get("corners")
        if not corners:
            continue
        xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
        if flight["id"] in violating_ids:
            ax.fill(xs, ys, color="#f2c0c0", alpha=0.85, ec="#c0392b",
                    lw=1.8, zorder=3)
        else:
            ax.fill(xs, ys, color="#aecbe8", alpha=0.85, ec="#2166ac",
                    lw=1.2, zorder=3)
        bounds.append((min(xs), min(ys), max(xs), max(ys)))
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        ax.text(cx, cy, f"F{flight.get('flight_index', '')}",
                ha="center", va="center", fontsize=6, zorder=5)

    return bounds


def _ramp_legend_handles():
    """
    The three-entry legend (compliant / violating flight, landing) shared
    by generate_violation_snapshot and draw_ramp_snapshot.

    Returns:
        list[matplotlib.patches.Patch]
    Called by: generate_violation_snapshot, draw_ramp_snapshot.
    """
    from matplotlib.patches import Patch
    return [Patch(facecolor="#aecbe8", ec="#2166ac", label="compliant flight"),
            Patch(facecolor="#f2c0c0", ec="#c0392b", label="violating flight"),
            Patch(facecolor="#c4c9cf", ec="#5a5f66", label="landing")]


def draw_ramp_snapshot(ifc_data, ramp_id, violating_ids, out_path, ramp_name="Ramp"):
    """
    Render one PNG for a whole ramp: a floor-plan crop of its own floor
    with every flight and landing drawn (see _draw_ramp_context) — every
    flight in `violating_ids` red, the rest blue, landings grey. Unlike
    generate_violation_snapshot (one BCF topic's own flight highlighted at
    a time), this shows every currently-known violation on the ramp at
    once, for a single overview image per ramp.

    Args:
        ifc_data: parsed data dict (spaces, columns, ramps, ramp_landings).
        ramp_id: the parent IfcRamp GlobalId to draw.
        violating_ids: set of flight ids (ramp["id"]) that failed a check.
        out_path: where to save the PNG.
        ramp_name: title label.
    Returns:
        str: out_path on success, or None if this ramp has no flights.
    Called by: main.py (run).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flights_here = [f for f in ifc_data.get("ramps", []) if f["ramp_id"] == ramp_id]
    if not flights_here:
        return None
    floor_elev = flights_here[0].get("storey_elevation_m", 0.0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bounds = _draw_ramp_context(ax, ifc_data, ramp_id, violating_ids, floor_elev)
    if not bounds:
        plt.close(fig)
        return None

    minx = min(b[0] for b in bounds); miny = min(b[1] for b in bounds)
    maxx = max(b[2] for b in bounds); maxy = max(b[3] for b in bounds)
    margin = max(2.0, 0.15 * max(maxx-minx, maxy-miny))
    ax.set_xlim(minx-margin, maxx+margin)
    ax.set_ylim(miny-margin, maxy+margin)

    ax.legend(handles=_ramp_legend_handles(), loc="upper right", fontsize=6)
    ax.set_title(ramp_name, fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6, length=2, colors="#888888")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def generate_violation_snapshot(violation, path_results, ifc_data, out_path):
    """
    Render a small PNG for one BCF issue's <Snapshot>: a floor-plan crop
    around the violation, on its own floor.

    For a room violation: every room on the floor (context), the violating
    room filled red, its own evacuation path drawn if one was computed.
    For a ramp violation: every room on the floor (context) plus every
    flight and landing of that same ramp — the violating flight filled
    red, its sibling flights blue, landings grey — so the snapshot shows
    where on the floor plan the ramp sits and which part of it fails,
    not just an isolated shape with no surroundings.

    Args:
        violation: one violation dict (has at least space_id, space_name;
            a ramp violation also has domain == "DL163").
        path_results: full list of per-space route results from
            RT_SCIE_checker.build_path_results (may be None for an
            accessibility-only run).
        ifc_data: parsed data dict (spaces, doors, columns, obstacles,
            ramps, ramp_landings).
        out_path: where to save the PNG.
    Returns:
        str: out_path on success, or None if there was nothing to draw
            (e.g. a ramp flight with no corners) — the caller treats None
            as "no snapshot for this topic" and skips it.
    Called by: bcf_reporter._write_bcf_file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon

    is_ramp = violation.get("domain") == "DL163"

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bounds = []  # (minx, miny, maxx, maxy) tuples to compute the zoom window

    if is_ramp:
        flight = next((r for r in ifc_data.get("ramps", [])
                      if r["id"] == violation["space_id"]), None)
        if flight is None or not flight.get("corners"):
            plt.close(fig)
            return None

        floor_elev = flight.get("storey_elevation_m", 0.0)
        ramp_id = flight["ramp_id"]

        # This BCF topic is about ONE flight, so only that flight is red
        # here — see draw_ramp_snapshot for the "every known violation on
        # this ramp at once" overview instead.
        bounds.extend(_draw_ramp_context(ax, ifc_data, ramp_id,
                                         {violation["space_id"]}, floor_elev))
        ax.legend(handles=_ramp_legend_handles(), loc="upper right", fontsize=6)

    else:
        space = next((s for s in ifc_data.get("spaces", [])
                     if s["id"] == violation["space_id"]), None)
        if space is None or not space.get("footprint"):
            plt.close(fig)
            return None

        floor_elev = space.get("storey_elevation_m", 0.0)

        # Draw every space on the same floor for context, so the violating
        # room's neighbours and doors are visible; only the violating
        # space itself is filled red.
        for other in ifc_data.get("spaces", []):
            if abs(other.get("storey_elevation_m", 0.0) - floor_elev) > 1.0:
                continue
            if not other.get("footprint"):
                continue
            poly = Polygon(other["footprint"])
            if not poly.is_valid or poly.area <= 0:
                continue
            xs, ys = poly.exterior.xy
            if other["id"] == violation["space_id"]:
                ax.fill(xs, ys, color="#f2c0c0", alpha=0.8, ec="#c0392b",
                        lw=1.8, zorder=3)
            else:
                ax.fill(xs, ys, color="#eaf4ff", alpha=0.5, ec="#3070b0",
                        lw=0.8, zorder=1)

        # The violating space's own footprint sets the zoom window.
        target_poly = Polygon(space["footprint"])
        ax.text(target_poly.centroid.x, target_poly.centroid.y,
                violation["space_name"], ha="center", va="center",
                fontsize=6, zorder=5)
        bounds.append(target_poly.bounds)

        # Draw this space's own evacuation path, if one was computed, so
        # the snapshot shows WHY it is a violation (a long or absent
        # route), not just its location.
        if path_results:
            match = next((r for r in path_results
                         if r.get("space_id") == violation["space_id"]), None)
            if match and match.get("reachable") and match.get("path"):
                px = [p[0] for p in match["path"]]
                py = [p[1] for p in match["path"]]
                ax.plot(px, py, "-", color="#c0392b", lw=2.0, zorder=8)
                minx, miny, maxx, maxy = (min(px), min(py), max(px), max(py))
                bounds.append((minx, miny, maxx, maxy))

    # Zoom to the violating element (plus its path, if any) with a fixed
    # margin — a BCF snapshot is a small thumbnail and should read clearly
    # at that scale, not show the whole floor.
    minx = min(b[0] for b in bounds); miny = min(b[1] for b in bounds)
    maxx = max(b[2] for b in bounds); maxy = max(b[3] for b in bounds)
    margin = max(2.0, 0.15 * max(maxx-minx, maxy-miny))
    ax.set_xlim(minx-margin, maxx+margin)
    ax.set_ylim(miny-margin, maxy+margin)

    ax.set_title(violation["space_name"], fontsize=9)
    ax.set_aspect("equal")
    # Ticks stay on (small font) specifically so the grid has something to
    # draw at: matplotlib's grid lines are drawn at tick positions, so an
    # empty tick list — the previous behaviour — silently produced NO grid
    # at all, leaving a blank canvas with nothing to judge scale or
    # position by whenever there was little else on the plot (e.g. a ramp
    # on a floor with no modelled rooms around it).
    ax.tick_params(labelsize=6, length=2, colors="#888888")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def draw_all_paths(area, results, exit_point, floor_elevation, out_path,
                   violating_ids=None, unreachable_ids=None,
                   missing_long_name_ids=None, non_occupiable_spaces=None,
                   coverage_gaps=None, exit_points=None):
    """
    Draw the walkable area plus every space's taut-string path.

    Args:
        area: walkable-area dict (see RT_SCIE_checker.build_walkable_area).
        results: per-space route dicts (RT_SCIE_checker.route_all_spaces).
        exit_point: the primary exit (kept for backward compatibility).
        floor_elevation: this floor's elevation, for the title.
        out_path: where to save the PNG.
        violating_ids: space ids that failed a compliance check -> filled
            red. Matched by unique space id, not name.
        unreachable_ids: space ids with no path to any exit -> filled
            purple. Purple takes priority over red: "no route at all" is a
            different, more serious signal than "route too long".
        missing_long_name_ids: space ids with no IfcSpace.LongName -> filled
            yellow, a naming/identification warning (IDS-07), not a
            compliance failure — so it only shows on a space that is
            otherwise compliant; purple/red take priority.
        non_occupiable_spaces: ifc_data["non_occupiable_spaces"] entries
            (id, name, footprint, storey_elevation_m) -> filled green
            (IDS-08). These spaces are excluded from routing upstream
            (ifc_parser.extract_all), so they never appear in
            area["spaces"] at all — drawn here as their own layer,
            filtered to this floor by storey_elevation_m.
        coverage_gaps: ifc_parser.check_space_coverage() entries (id,
            storey_name, storey_elevation_m, residual_area_m2,
            residual_centroid, footprint) -> filled gray. Floor area with
            no IfcSpace covering it at all — a possible missing room, not
            a compliance issue — so like non_occupiable_spaces it is drawn
            from its own footprint rather than area["spaces"], filtered to
            this floor by storey_elevation_m.
        exit_points: all fire exits on this floor. When given, every exit
            is drawn, since Art. 57 measures distance to the NEAREST exit
            and a floor with two or more exits gets the longer 30 m limit.
    Called by: main.py (run).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from shapely.geometry import Polygon

    # Normalize to sets for fast lookup; None means "nothing to mark"
    violating_ids         = set(violating_ids) if violating_ids else set()
    unreachable_ids       = set(unreachable_ids) if unreachable_ids else set()
    missing_long_name_ids = set(missing_long_name_ids) \
        if missing_long_name_ids else set()
    non_occupiable_spaces = non_occupiable_spaces or []
    coverage_gaps         = coverage_gaps or []

    fig, ax = plt.subplots(figsize=(16, 11))

    first_missing_name = True
    for space_id, name, poly in area["spaces"]:
        xs, ys = poly.exterior.xy
        # Priority: no-path (purple) > other violation (red) >
        # missing LongName (yellow) > compliant (blue)
        if space_id in unreachable_ids:
            ax.fill(xs, ys, color="#d9b3e6", alpha=0.7, ec="#8e44ad",
                    lw=1.6, zorder=2)
        elif space_id in violating_ids:
            ax.fill(xs, ys, color="#f2c0c0", alpha=0.7, ec="#c0392b",
                    lw=1.6, zorder=2)
        elif space_id in missing_long_name_ids:
            ax.fill(xs, ys, color="#f9e79b", alpha=0.7, ec="#b7950b",
                    lw=1.6, zorder=2,
                    label="missing Long Name" if first_missing_name else None)
            first_missing_name = False
        else:
            ax.fill(xs, ys, color="#93c5fd", alpha=0.7, ec="#1d4ed8",
                    lw=1.3, zorder=1)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, name, ha="center", va="center", fontsize=7, zorder=5)

    # Non-occupiable spaces (IDS-08): excluded from routing entirely, so
    # drawn from their own footprint rather than area["spaces"]. Green,
    # filtered to this floor with the same elevation tolerance used
    # elsewhere for matching obstacles to a storey (RT_SCIE_checker
    # _columns_on_floor / _obstacles_on_floor).
    first_non_occupiable = True
    for space in non_occupiable_spaces:
        if abs(space.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        footprint = space.get("footprint")
        if not footprint or len(footprint) < 3:
            continue
        try:
            poly = Polygon(footprint)
            if not poly.is_valid or poly.is_empty:
                continue
        except Exception:
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="#b8e6b8", alpha=0.7, ec="#1e8449", lw=1.6,
                zorder=2,
                label="non-occupiable" if first_non_occupiable else None)
        first_non_occupiable = False
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, space.get("name", "Space"), ha="center",
                va="center", fontsize=7, zorder=5)

    # Coverage gaps: floor area with no IfcSpace at all (a possible missing
    # room). Gray, drawn from its own footprint like non-occupiable spaces
    # above — this area was never an IfcSpace, so it cannot come from
    # area["spaces"] either. zorder=0 — strictly BELOW every real room, not
    # just equal to the compliant-room baseline (zorder=1): matplotlib only
    # respects draw order as a tiebreaker between equal zorders, and this
    # loop runs after the room loop above, so a merely-equal zorder here
    # let a large gap polygon paint over and hide real, routed rooms
    # underneath it (confirmed on a real model: a big open-plan gap
    # covered a Lobby space with its own footprint and computed route).
    first_gap = True
    for gap in coverage_gaps:
        if abs(gap.get("storey_elevation_m", 0.0) - floor_elevation) > 1.0:
            continue
        footprint = gap.get("footprint")
        if not footprint or len(footprint) < 3:
            continue
        try:
            poly = Polygon(footprint)
            if not poly.is_valid or poly.is_empty:
                continue
        except Exception:
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="#c2c2c2", alpha=0.7, ec="#6e6e6e", lw=1.6,
                zorder=0, label="no space assigned" if first_gap else None)
        first_gap = False
        # The gap's area (residual_area_m2) is intentionally NOT drawn on
        # the plan — the gray fill plus the "no space assigned" legend
        # entry already identify it, and the m2 figure overlapped the real
        # room label it sits under. The number is still in the text/JSON
        # report for anyone who needs it.

    for rect in area["doors"]:
        xs, ys = rect.exterior.xy
        ax.fill(xs, ys, color="#2ca02c", alpha=0.7, zorder=3)

    # Columns are obstacles cut out of the walkable area — drawn so it is
    # clear why a path bends around a particular spot.
    for index, column in enumerate(area.get("columns", [])):
        xs, ys = column.exterior.xy
        ax.fill(xs, ys, color="#8B4513", alpha=0.95, ec="#4a2508",
                lw=1.0, zorder=6,
                label="column" if index == 0 else None)

    # Fixed furniture tagged as an evacuation obstacle: cut out of the
    # walkable area just like columns.
    for index, obstacle in enumerate(area.get("obstacles", [])):
        xs, ys = obstacle.exterior.xy
        ax.fill(xs, ys, color="#e67e22", alpha=0.9, ec="#a04000",
                lw=1.0, zorder=6,
                label="furniture obstacle" if index == 0 else None)

    colors = cm.tab20(np.linspace(0, 1, max(len(results), 1)))
    for idx, r in enumerate(results):
        if not r["reachable"]:
            continue
        px = [p[0] for p in r["path"]]
        py = [p[1] for p in r["path"]]
        ax.plot(px, py, "-", color=colors[idx], lw=2.0, alpha=0.85, zorder=8)
        ax.plot(r["start"][0], r["start"][1], "o", color=colors[idx],
                markersize=7, zorder=9)

    # Draw every fire exit on this floor, since Art. 57 measures distance
    # to the NEAREST exit — showing all of them explains why a given route
    # was chosen.
    all_exits = exit_points if exit_points else (
        [exit_point] if exit_point else [])
    for index, ex in enumerate(all_exits):
        ax.plot(ex[0], ex[1], "*", color="green",
                markersize=20, zorder=10,
                label="fire exit" if index == 0 else None)

    ax.set_title(f"STEP 3 - all spaces routed to exit (floor {floor_elevation:.1f} m)",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)


def draw_walkable_area(area, floor_elevation, out_path):
    """
    Draw the rooms and door rectangles of one floor, for visual QA of the
    walkable-area geometry.

    Args:
        area: walkable-area dict (see RT_SCIE_checker.build_walkable_area).
        floor_elevation: this floor's elevation, for the title.
        out_path: where to save the PNG.
    Called by: not currently called in the pipeline (debug helper).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 11))

    # Draw each room footprint in light blue with its name
    for space_id, name, poly in area["spaces"]:
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, color="#cfe8ff", alpha=0.6, ec="#3070b0",
                lw=1.5, zorder=1)
        cx, cy = poly.centroid.x, poly.centroid.y
        ax.text(cx, cy, name, ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=5)

    # Draw each door rectangle in solid green
    for rect in area["doors"]:
        xs, ys = rect.exterior.xy
        ax.fill(xs, ys, color="#2ca02c", alpha=0.9, ec="#186a18",
                lw=1.0, zorder=4)

    ax.set_title(f"STEP 1 — walkable area (rooms + door rectangles) "
                 f"— floor {floor_elevation:.1f} m",
                 fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
