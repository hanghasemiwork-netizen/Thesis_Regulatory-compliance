"""
bcf_reporter.py
Generates BCF (BIM Collaboration Format) issues for each violation.
Outputs:
  - violations_report.json  (machine-readable, custom)
  - violations_report.txt   (human-readable)
  - violations_report.bcf   (BCF 2.1 ZIP container, openable in IFC viewers
                             such as BIMcollab ZOOM, Solibri Anywhere, and
                             in Revit / Archicad through their BCF add-ins)

All three are organized the same way, top to bottom:
  1. Evacuation Distance (RT-SCIE Art.57) — compliant (TXT only, see
     generate_bcf_issues) / non-compliant, split into Unreachable and
     Exceeds Allowed Distance.
  2. Ramp Accessibility (DL163/2006 Anexo 2.5.1) — violations.
  3. Warnings (data quality, never a compliance failure) — split by
     check: missing LongName (IDS-05), non-occupiable spaces (IDS-06),
     coverage gaps (COVERAGE-GAP), floors with no fire exit door
     (NO-FIRE-EXIT).
BCF has no folder/category concept beyond per-topic fields, so
categorization there means TopicType + an extra Label per topic
(_violation_category, _WARNING_TOPIC_TYPES) — grouping in the viewer's own
UI, not a literal directory structure.
"""

import json
import math
import os
import uuid
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


# Regulations this report can cover — shared by the JSON metadata and the
# TXT header so the two never drift out of sync with each other.
_REGULATIONS = [
    "RT-SCIE (Portaria 1532/2008 + 135/2020) — Art. 57 evacuation distance",
    "DL 163/2006 — Anexo 2.5.1 ramp accessibility",
]

# Warning check_id -> human label, shared by the TXT section headers, the
# JSON key names and the BCF TopicType.
_WARNING_GROUPS = [
    ("IDS-05",       "missing_long_name", "Missing Long Name"),
    ("IDS-06",       "non_occupiable",    "Non-occupiable Spaces"),
    ("COVERAGE-GAP", "coverage_gap",      "Coverage Gaps (no space assigned)"),
    ("NO-FIRE-EXIT", "no_fire_exit",      "Floors With No Fire Exit"),
]


def _violation_category(v: dict) -> str:
    """
    Classify one violation for report grouping: which top-level section
    (TXT/JSON) and TopicType (BCF) it belongs to.

    Args:
        v: one violation dict (rule_checker.check_all / check_ramps).
    Returns:
        str: "Ramp Accessibility", "Evacuation Distance — Unreachable", or
            "Evacuation Distance — Exceeds Limit".
    Called by: generate_bcf_issues, _write_bcf_file.
    """
    if v.get("domain") == "DL163":
        return "Ramp Accessibility"
    if v.get("type") == "UNREACHABLE":
        return "Evacuation Distance — Unreachable"
    return "Evacuation Distance — Exceeds Limit"


def generate_bcf_issues(check_results: dict, output_dir: str,
                        ifc_data: dict = None, model=None,
                        coordinate_base: str = "ifc",
                        path_results: list = None,
                        warnings: list = None,
                        fire_exits_by_floor: list = None) -> str:
    """
    Write the JSON, TXT and BCF violation reports.

    Args:
        check_results: output of rule_checker.check_all().
        output_dir: folder for the three report files.
        ifc_data: full ifc_data dict, for viewpoint camera locations;
            optional — without it the .bcf still selects the violating
            component, just with no camera.
        model: loaded ifcopenshell model; only needed when
            coordinate_base="local" (map offset lookup).
        coordinate_base: "ifc" — viewpoint coordinates exactly as in the IFC
            file. "local" — subtract the shared-coordinates map offset
            (IfcMapConversion / IfcSite placement) for plugins that expect
            internal-origin coordinates.
        path_results: route results, for the violation snapshot images.
        warnings: data-quality warnings that never fail a compliance rule
            (e.g. spaces missing LongName or tagged non_occupiable, a
            floor area with no IfcSpace at all, or a floor with no fire
            exit door — see main.py) — list of dicts {check_id, space_id,
            space_name, storey_name, message}, already fully worded by the
            caller; footprint and storey_elevation_m are optional extras
            used only to aim a BCF topic's camera (see _write_bcf_file).
            Listed in their own, per-check-id section of the
            JSON/TXT/BCF reports, kept separate from violations so they
            never affect the compliant/violations counts.
        fire_exits_by_floor: one entry per IfcBuildingStorey — {storey_name,
            elevation_m, fire_exit_count} (see main.py) — printed as a
            quick per-floor overview; a 0-count floor is also covered by
            its own NO-FIRE-EXIT warning above, in `warnings`.
    Returns:
        str: path to the JSON report.
    Called by: main.py (run).
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    violations           = check_results["violation_list"]
    compliant_list       = check_results.get("compliant_list", [])
    warnings             = warnings or []
    fire_exits_by_floor  = fire_exits_by_floor or []

    # ── Split violations into the three report sections, keeping each
    # topic's guid stable and independent of which bucket it lands in. ──
    unreachable_topics, exceeds_topics, ramp_topics = [], [], []
    for i, v in enumerate(violations, start=1):
        topic = {
            "guid": f"SCIE-{i:04d}",
            "type": "Issue",
            "status": "Open",
            "title": f"Violation — {v['space_name']}",
            "floor": v.get("storey_name") or "Unknown",
            "description": (
                f"Rule: {v['rule_id']}\n"
                f"{v['description']}\n\n"
                f"Space: {v['space_name']} (ID: {v['space_id']})\n"
                f"Floor: {v.get('storey_name') or 'Unknown'}\n"
                f"Path type: {v.get('path_type', 'N/A')}\n\n"
                f"Measured distance: {v['measured_m']} m\n"
                f"Maximum allowed:   {v['limit_m']} m\n"
                f"Excess:            +{v['excess_m']} m"
            ),
            "creation_date": timestamp,
            "ifc_guids": [v["space_id"]],
            "labels": [v["rule_id"]]
        }
        category = _violation_category(v)
        if category == "Ramp Accessibility":
            ramp_topics.append(topic)
        elif category == "Evacuation Distance — Unreachable":
            unreachable_topics.append(topic)
        else:
            exceeds_topics.append(topic)

    # ── Split warnings by check_id, in the same grouping used by TXT/BCF ──
    warnings_by_key = {key: [] for _, key, _ in _WARNING_GROUPS}
    for w in warnings:
        for check_id, key, _label in _WARNING_GROUPS:
            if w["check_id"] == check_id:
                warnings_by_key[key].append({
                    "check_id":    w["check_id"],
                    "space_name":  w["space_name"],
                    "space_id":    w["space_id"],
                    "storey_name": w.get("storey_name"),
                    "message":     w["message"]
                })
                break

    # ── BCF-style JSON ────────────────────────────────────────────────────
    bcf_output = {
        "metadata": {
            "generated": timestamp,
            "regulations": _REGULATIONS,
            "total_spaces_checked": check_results["total_spaces"],
            "compliant": check_results["compliant"],
            "violations_found": check_results["violations"],
            "warnings_found": len(warnings),
            "fire_exits_by_floor": fire_exits_by_floor
        },
        "evacuation_distance": {
            "non_compliant": {
                "unreachable":   unreachable_topics,
                "exceeds_limit": exceeds_topics,
            }
        },
        "ramp_accessibility": {
            "violations": ramp_topics
        },
        "warnings": warnings_by_key
    }

    json_path = os.path.join(output_dir, "violations_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bcf_output, f, indent=2)

    # ── Human-readable TXT ────────────────────────────────────────────────
    txt_path = os.path.join(output_dir, "violations_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:

        def _write_violation(v, index):
            f.write(f"  #{index}\n")
            f.write(f"    Rule:     {v['rule_id']}\n")
            f.write(f"    Space:    {v['space_name']}\n")
            f.write(f"    Space ID: {v['space_id']}\n")
            f.write(f"    Floor:    {v.get('storey_name') or 'Unknown'}\n")
            f.write(f"    Path:     {v.get('path_type', 'N/A')}\n")
            f.write(f"    Measured: {v['measured_m']} m\n")
            f.write(f"    Limit:    {v['limit_m']} m\n")
            f.write(f"    Excess:   +{v['excess_m']} m\n\n")

        ramp_violations = [v for v in violations
                           if _violation_category(v) == "Ramp Accessibility"]
        unreachable = [v for v in violations
                      if _violation_category(v) == "Evacuation Distance — Unreachable"]
        exceeds = [v for v in violations
                  if _violation_category(v) == "Evacuation Distance — Exceeds Limit"]

        f.write("=" * 64 + "\n")
        f.write("FIRE SAFETY & ACCESSIBILITY COMPLIANCE REPORT\n")
        f.write("=" * 64 + "\n")
        f.write(f"Generated:   {timestamp}\n")
        f.write(f"Regulations: {_REGULATIONS[0]}\n")
        for reg in _REGULATIONS[1:]:
            f.write(f"             {reg}\n")
        f.write("-" * 64 + "\n")
        f.write(f"Total spaces checked: {check_results['total_spaces']}\n")
        f.write(f"Compliant:            {check_results['compliant']}\n")
        f.write(f"Violations found:     {check_results['violations']} "
                f"(evacuation: {len(unreachable) + len(exceeds)}, "
                f"ramp: {len(ramp_violations)})\n")
        f.write(f"Warnings found:       {len(warnings)}\n")
        f.write("=" * 64 + "\n")

        if fire_exits_by_floor:
            f.write("\nFire exits by floor:\n")
            for f_info in fire_exits_by_floor:
                count = f_info["fire_exit_count"]
                flag  = "   *** NO FIRE EXIT ***" if count == 0 else ""
                f.write(f"  {f_info['storey_name']:<24} "
                        f"{count} fire exit(s){flag}\n")
            f.write("=" * 64 + "\n")

        # ── Section 1: Evacuation Distance ─────────────────────────────
        f.write("\n" + "#" * 64 + "\n")
        f.write("# EVACUATION DISTANCE — RT-SCIE Art. 57\n")
        f.write("#" * 64 + "\n\n")

        f.write(f"---- COMPLIANT ({len(compliant_list)}) "
                + "-" * 38 + "\n")
        if not compliant_list:
            f.write("  (none)\n")
        else:
            for i, r in enumerate(
                    sorted(compliant_list, key=lambda r: r["space_name"]),
                    start=1):
                measured = r.get("measured_m", "N/A")
                limit    = r.get("limit_m", "N/A")
                path_ty  = r.get("path_type", "N/A")
                floor    = r.get("storey_name") or "Unknown"
                f.write(f"  {i:>3}. {r['space_name']:<22} "
                        f"(ID: {r['space_id']})  "
                        f"[floor: {floor}]  "
                        f"{measured} m / {limit} m limit  [{path_ty}]\n")
        f.write("\n")

        f.write(f"---- NON-COMPLIANT ({len(unreachable) + len(exceeds)}) "
                + "-" * 30 + "\n\n")

        f.write(f"  -- Unreachable ({len(unreachable)}) --\n")
        if not unreachable:
            f.write("    (none)\n\n")
        else:
            for i, v in enumerate(unreachable, start=1):
                _write_violation(v, i)

        f.write(f"  -- Exceeds allowed distance ({len(exceeds)}) --\n")
        if not exceeds:
            f.write("    (none)\n\n")
        else:
            for i, v in enumerate(exceeds, start=1):
                _write_violation(v, i)

        # ── Section 2: Ramp Accessibility ──────────────────────────────
        f.write("#" * 64 + "\n")
        f.write("# RAMP ACCESSIBILITY — DL 163/2006 Anexo 2.5.1\n")
        f.write("#" * 64 + "\n\n")
        if not ramp_violations:
            f.write("  (none)\n\n")
        else:
            for i, v in enumerate(ramp_violations, start=1):
                _write_violation(v, i)

        f.write("=" * 64 + "\n")
        f.write("Summary by rule:\n")
        for rid, info in check_results.get("summary_by_rule", {}).items():
            f.write(f"  {rid} ({info['article']}): "
                    f"{info['count']} violation(s)\n")

        # ── Section 3: Warnings ────────────────────────────────────────
        f.write("\n" + "#" * 64 + "\n")
        f.write(f"# WARNINGS (data quality — not a compliance failure): "
                f"{len(warnings)}\n")
        f.write("#" * 64 + "\n\n")
        for check_id, key, label in _WARNING_GROUPS:
            group = warnings_by_key[key]
            f.write(f"  -- {label} ({check_id}) — {len(group)} --\n")
            if not group:
                f.write("    (none)\n\n")
                continue
            for w in group:
                floor = f" (floor: {w['storey_name']})" \
                    if w.get("storey_name") else ""
                f.write(f"    [{w['check_id']}] {w['message']}{floor}\n")
            f.write("\n")

    # ── Standards-compliant BCF 2.1 container ────────────────────────────
    bcf_path = _write_bcf_file(check_results, output_dir,
                               ifc_data=ifc_data, model=model,
                               coordinate_base=coordinate_base,
                               path_results=path_results,
                               warnings=warnings)

    return json_path


# ══════════════════════════════════════════════════════════════════════════
# BCF 2.1 container generation (buildingSMART BCF-XML, release 2.1).
# A .bcf file is a plain ZIP archive:
#
#   violations_report.bcf
#   ├── bcf.version                  declares the BCF version (2.1)
#   └── <TopicGuid>/                 one folder per issue
#       ├── markup.bcf               title, status, description, labels
#       └── viewpoint.bcfv           camera + IfcGuid selection
#
# Uses only the Python standard library.
# ══════════════════════════════════════════════════════════════════════════


def _get_map_offset(model):
    """
    Read the shared-coordinates offset stored in the IFC model.

    Args:
        model: loaded ifcopenshell model, or None.
    Returns:
        tuple: (dx, dy, dz); (0, 0, 0) when no offset is present.
    Called by: _write_bcf_file, when coordinate_base="local".
    """
    if model is None:
        return (0.0, 0.0, 0.0)

    # 1 — IFC4 map conversion entity
    try:
        conversions = model.by_type("IfcMapConversion")
        if conversions:
            c = conversions[0]
            return (float(c.Eastings or 0.0),
                    float(c.Northings or 0.0),
                    float(c.OrthogonalHeight or 0.0))
    except Exception:
        pass

    # 2 — translation baked into the site placement (IFC2x3 fallback)
    try:
        import ifcopenshell.util.placement as _pl
        sites = model.by_type("IfcSite")
        if sites and sites[0].ObjectPlacement:
            mtx = _pl.get_local_placement(sites[0].ObjectPlacement)
            return (float(mtx[0][3]), float(mtx[1][3]), float(mtx[2][3]))
    except Exception:
        pass

    return (0.0, 0.0, 0.0)


def _xml_bytes(root) -> bytes:
    """
    Serialize an ElementTree root to UTF-8 bytes with XML declaration.

    Args:
        root: xml.etree.ElementTree.Element.
    Returns:
        bytes
    Called by: _version_xml, _markup_xml, _viewpoint_xml.
    """
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def _version_xml() -> bytes:
    """
    Build the bcf.version file declaring BCF 2.1.

    Returns:
        bytes
    Called by: _write_bcf_file.
    """
    root = ET.Element("Version", {"VersionId": "2.1"})
    ET.SubElement(root, "DetailedVersion").text = "2.1"
    return _xml_bytes(root)


def _fmt_m(value) -> str:
    """
    Format a metre value; None becomes 'N/A'.

    Args:
        value: float or None.
    Returns:
        str
    Called by: _markup_xml.
    """
    return "N/A" if value is None else f"{value} m"


def _markup_xml(v: dict, topic_guid: str, viewpoint_guid: str,
                timestamp: str, has_snapshot: bool = False,
                storey_name: str = None) -> bytes:
    """
    Build markup.bcf for one violation.

    Element order inside <Topic> follows markup.xsd:
    Title -> Labels -> CreationDate -> CreationAuthor -> Description.

    storey_name, when known, is included in both the Title (visible in a
    BCF viewer's issue list) and the Description, since a space name alone
    is not unique across a multi-storey building. TopicType carries this
    report's category (_violation_category) — Evacuation Distance /
    Unreachable, Evacuation Distance / Exceeds Limit, or Ramp
    Accessibility — since BCF has no folder/category concept beyond
    per-topic fields; most viewers can group or filter their issue list by
    TopicType and by Labels, which also gets the category added alongside
    the rule id.

    Args:
        v: one violation dict.
        topic_guid: this topic's GUID.
        viewpoint_guid: this topic's viewpoint GUID.
        timestamp: ISO creation timestamp.
        has_snapshot: whether a snapshot.png exists for this topic.
        storey_name: floor name, if known.
    Returns:
        bytes
    Called by: _write_bcf_file.
    """
    root = ET.Element("Markup")

    category = _violation_category(v)
    topic = ET.SubElement(root, "Topic", {
        "Guid":        topic_guid,
        "TopicType":   category,
        "TopicStatus": "Open",
    })
    floor_label = f" ({storey_name})" if storey_name else ""
    ET.SubElement(topic, "Title").text = (
        f"{v['type']} — {v['space_name']}{floor_label}")
    ET.SubElement(topic, "Labels").text = str(v.get("rule_id", ""))
    ET.SubElement(topic, "Labels").text = category
    ET.SubElement(topic, "CreationDate").text = timestamp
    ET.SubElement(topic, "CreationAuthor").text = "SCIE Compliance Checker"
    ET.SubElement(topic, "Description").text = (
        f"Rule: {v['rule_id']}\n"
        f"{v['description']}\n\n"
        f"Space: {v['space_name']} (IFC GUID: {v['space_id']})\n"
        f"Floor:      {storey_name or 'unknown'}\n"
        f"Path type: {v.get('path_type', 'N/A')}\n\n"
        f"Measured distance: {_fmt_m(v['measured_m'])}\n"
        f"Maximum allowed:   {_fmt_m(v['limit_m'])}\n"
        f"Excess:            {_fmt_m(v['excess_m'])}"
    )

    # Register the viewpoint file that belongs to this topic
    vps = ET.SubElement(root, "Viewpoints", {"Guid": viewpoint_guid})
    ET.SubElement(vps, "Viewpoint").text = "viewpoint.bcfv"
    # <Snapshot> must follow <Viewpoint> per the BCF 2.1 markup sequence,
    # and is added only when a PNG was actually rendered.
    if has_snapshot:
        ET.SubElement(vps, "Snapshot").text = "snapshot.png"

    return _xml_bytes(root)


def _warning_markup_xml(w: dict, topic_guid: str, viewpoint_guid: str,
                        timestamp: str, topic_type: str,
                        storey_name: str = None) -> bytes:
    """
    Build markup.bcf for one data-quality warning (never a real BCF
    "Issue" — no measured/limit/excess to report). No <Snapshot>: warnings
    have no routing path or violation geometry for
    visualizer.generate_violation_snapshot to render meaningfully, unlike
    a real compliance violation.

    Args:
        w: one warning dict {check_id, space_id, space_name, storey_name,
            message}.
        topic_guid: this topic's GUID.
        viewpoint_guid: this topic's viewpoint GUID.
        timestamp: ISO creation timestamp.
        topic_type: this warning's category label (see
            bcf_reporter._WARNING_GROUPS in _write_bcf_file).
        storey_name: floor name, if known.
    Returns:
        bytes
    Called by: _write_bcf_file.
    """
    root = ET.Element("Markup")

    topic = ET.SubElement(root, "Topic", {
        "Guid":        topic_guid,
        "TopicType":   topic_type,
        "TopicStatus": "Open",
    })
    floor_label = f" ({storey_name})" if storey_name else ""
    ET.SubElement(topic, "Title").text = (
        f"Warning — {w['space_name']}{floor_label}")
    ET.SubElement(topic, "Labels").text = w["check_id"]
    ET.SubElement(topic, "Labels").text = topic_type
    ET.SubElement(topic, "CreationDate").text = timestamp
    ET.SubElement(topic, "CreationAuthor").text = "SCIE Compliance Checker"
    ET.SubElement(topic, "Description").text = (
        f"{w['message']}\nFloor: {storey_name or 'unknown'}")

    vps = ET.SubElement(root, "Viewpoints", {"Guid": viewpoint_guid})
    ET.SubElement(vps, "Viewpoint").text = "viewpoint.bcfv"

    return _xml_bytes(root)


def _viewpoint_xml(viewpoint_guid: str, ifc_guid: str, cam_target) -> bytes:
    """
    Build viewpoint.bcfv for one violation or warning: a <Selection> of
    the IfcGuid (works with or without a camera), plus an oblique
    perspective camera above the space when its location is known.

    Element order follows visinfo.xsd: Components -> PerspectiveCamera.

    Args:
        viewpoint_guid: this topic's viewpoint GUID.
        ifc_guid: the violating/warned element's IFC GlobalId, or None —
            a coverage-gap warning has no real IFC entity behind it (it is
            a computed area with no GlobalId, see ifc_parser
            check_space_coverage), so its topic gets a camera only, no
            Selection, rather than pointing a <Selection> at a GUID no
            element in the model actually has.
        cam_target: (x, y, z) camera target point, or None.
    Returns:
        bytes
    Called by: _write_bcf_file.
    """
    root = ET.Element("VisualizationInfo", {"Guid": viewpoint_guid})

    comps = ET.SubElement(root, "Components")
    if ifc_guid:
        sel = ET.SubElement(comps, "Selection")
        ET.SubElement(sel, "Component", {"IfcGuid": ifc_guid})
    # Everything stays visible; the selection alone marks the space
    ET.SubElement(comps, "Visibility", {"DefaultVisibility": "true"})

    if cam_target is not None:
        tx, ty, tz = cam_target
        # Oblique bird's-eye position: 7 m south-east, 9 m above
        px, py, pz = tx + 7.0, ty - 7.0, tz + 9.0

        # Direction = normalized vector from camera to the space centre
        dx, dy, dz = tx - px, ty - py, tz - pz
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        dx, dy, dz = dx / norm, dy / norm, dz / norm

        # Up vector = world Z orthogonalized against the view direction
        wz = (0.0, 0.0, 1.0)
        d_dot_w = dx * wz[0] + dy * wz[1] + dz * wz[2]
        ux, uy, uz = (wz[0] - d_dot_w * dx,
                      wz[1] - d_dot_w * dy,
                      wz[2] - d_dot_w * dz)
        unorm = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / unorm, uy / unorm, uz / unorm

        cam = ET.SubElement(root, "PerspectiveCamera")
        pt = ET.SubElement(cam, "CameraViewPoint")
        for tag, val in (("X", px), ("Y", py), ("Z", pz)):
            ET.SubElement(pt, tag).text = f"{val:.6f}"
        dr = ET.SubElement(cam, "CameraDirection")
        for tag, val in (("X", dx), ("Y", dy), ("Z", dz)):
            ET.SubElement(dr, tag).text = f"{val:.6f}"
        up = ET.SubElement(cam, "CameraUpVector")
        for tag, val in (("X", ux), ("Y", uy), ("Z", uz)):
            ET.SubElement(up, tag).text = f"{val:.6f}"
        ET.SubElement(cam, "FieldOfView").text = "60"

    return _xml_bytes(root)


def _centroid_of(footprint):
    """
    Cheap vertex-average centroid of a footprint — good enough to aim a
    BCF topic's camera, not a substitute for a real polygon centroid.

    Args:
        footprint: list of (x, y) points, or None/empty.
    Returns:
        tuple: (x, y), or None if footprint is empty.
    Called by: _write_bcf_file.
    """
    if not footprint:
        return None
    xs = [p[0] for p in footprint]
    ys = [p[1] for p in footprint]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _write_bcf_file(check_results: dict, output_dir: str,
                    ifc_data: dict = None, model=None,
                    coordinate_base: str = "ifc",
                    path_results: list = None,
                    warnings: list = None) -> str:
    """
    Write the BCF 2.1 ZIP container with one Topic per violation, plus one
    per warning (see _warning_markup_xml — no snapshot, and no <Selection>
    for a coverage-gap warning, which has no real IFC entity behind it).

    Args:
        check_results: output of rule_checker.check_all().
        output_dir: folder for the .bcf file.
        ifc_data: full ifc_data dict, for viewpoint camera locations.
        model: loaded ifcopenshell model (coordinate_base="local" only).
        coordinate_base: see generate_bcf_issues.
        path_results: route results, for the violation snapshot images.
        warnings: see generate_bcf_issues.
    Returns:
        str: path to the created .bcf file.
    Called by: generate_bcf_issues.
    """
    violations = check_results["violation_list"]
    warnings   = warnings or []
    bcf_path = os.path.join(output_dir, "violations_report.bcf")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Space locations for the viewpoint cameras. non_occupiable_spaces is
    # merged in only for warning topics below (IDS-06 fires exactly on
    # spaces ifc_parser.extract_all excluded from "spaces") — violation
    # topics never reference one, since a non-occupiable space never
    # enters routing/checks in the first place.
    spaces_by_id = {}
    non_occ_by_id = {}
    ramps_by_id  = {}
    if ifc_data:
        spaces_by_id  = {s["id"]: s for s in ifc_data.get("spaces", [])}
        non_occ_by_id = {s["id"]: s
                         for s in ifc_data.get("non_occupiable_spaces", [])}
        ramps_by_id   = {r["id"]: r for r in ifc_data.get("ramps", [])}

    # Shared-coordinates correction (only for coordinate_base="local")
    offset = (0.0, 0.0, 0.0)
    if coordinate_base == "local":
        offset = _get_map_offset(model)

    # Scratch folder for the per-topic snapshot PNGs before they are zipped.
    import tempfile
    snap_dir = tempfile.mkdtemp(prefix="bcf_snap_")

    # Import the visualizer here (not at module top), so this module has no
    # hard matplotlib dependency when snapshots are not being produced.
    try:
        import visualizer
    except Exception:
        visualizer = None

    with zipfile.ZipFile(bcf_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bcf.version", _version_xml())

        for v in violations:
            topic_guid     = str(uuid.uuid4())
            viewpoint_guid = str(uuid.uuid4())

            # Camera target = space centre point (if known)
            cam_target = None
            sp = spaces_by_id.get(v["space_id"])
            if sp and sp.get("location"):
                loc = sp["location"]
                z = loc[2] if len(loc) > 2 else sp.get("storey_elevation_m", 0.0)
                cam_target = (loc[0] - offset[0],
                              loc[1] - offset[1],
                              z      - offset[2])

            # The element the viewpoint's <Selection> highlights. For a
            # room violation, v["space_id"] IS the real IfcSpace GlobalId.
            # For a ramp violation it is a synthetic composite id (ramp
            # GlobalId + flight index, see ifc_parser.get_ramps) that does
            # not exist as a model element — selecting by it silently
            # fails in the BIM viewer, landing the camera nowhere near the
            # ramp. Resolved below to the flight's real entity_id instead.
            selection_guid = v["space_id"]

            # Floor name for the Title/Description. Exactly one of the two
            # lookups matches: a SCIE issue matches a space, a DL163 ramp
            # issue matches a ramp flight.
            storey_name = None
            if sp:
                storey_name = sp.get("storey_name")
            else:
                rp = ramps_by_id.get(v["space_id"])
                if rp:
                    storey_name = rp.get("storey_name")
                    selection_guid = rp.get("entity_id") or rp.get("ramp_id") or selection_guid
                    if not cam_target and rp.get("location"):
                        loc = rp["location"]
                        cam_target = (loc[0] - offset[0],
                                      loc[1] - offset[1],
                                      loc[2] - offset[2])

            # ── Render the annotated snapshot for this violation ──────────
            # A successful render returns the PNG path; a failure returns
            # None and the topic is written without a preview image.
            snapshot_path = None
            if (visualizer is not None
                    and hasattr(visualizer, "generate_violation_snapshot")
                    and ifc_data is not None):
                candidate_png = os.path.join(snap_dir, f"{topic_guid}.png")
                snapshot_path = visualizer.generate_violation_snapshot(
                    v, path_results, ifc_data, candidate_png)

            has_snapshot = bool(snapshot_path and os.path.exists(snapshot_path))

            # Markup is written after the snapshot attempt, so it can
            # correctly declare whether a <Snapshot> element is present.
            zf.writestr(f"{topic_guid}/markup.bcf",
                        _markup_xml(v, topic_guid, viewpoint_guid, timestamp,
                                    has_snapshot=has_snapshot,
                                    storey_name=storey_name))
            zf.writestr(f"{topic_guid}/viewpoint.bcfv",
                        _viewpoint_xml(viewpoint_guid, selection_guid,
                                       cam_target))

            if has_snapshot:
                zf.write(snapshot_path, f"{topic_guid}/snapshot.png")

        # ── One topic per warning — no snapshot (see _warning_markup_xml),
        # and no <Selection> for a coverage-gap warning (no real entity).
        warning_topic_types = {check_id: label
                               for check_id, _key, label in _WARNING_GROUPS}
        for w in warnings:
            topic_guid     = str(uuid.uuid4())
            viewpoint_guid = str(uuid.uuid4())

            is_coverage_gap = w["check_id"] == "COVERAGE-GAP"
            selection_guid = None if is_coverage_gap else w.get("space_id")

            # Camera target: a real space's own "location" first (matches
            # the violation loop above), else a centroid of its footprint
            # — needed for non_occupiable_spaces/coverage-gap entries,
            # which carry a footprint but no "location" field.
            cam_target = None
            sp = (spaces_by_id.get(w.get("space_id"))
                 or non_occ_by_id.get(w.get("space_id")))
            elevation = w.get("storey_elevation_m")
            if sp:
                elevation = sp.get("storey_elevation_m", elevation)
            if sp and sp.get("location"):
                loc = sp["location"]
                z = loc[2] if len(loc) > 2 else (elevation or 0.0)
                cam_target = (loc[0] - offset[0], loc[1] - offset[1],
                             z - offset[2])
            else:
                centroid = _centroid_of(w.get("footprint"))
                if centroid:
                    cam_target = (centroid[0] - offset[0],
                                 centroid[1] - offset[1],
                                 (elevation or 0.0) - offset[2])

            topic_type = warning_topic_types.get(w["check_id"], "Warning")
            zf.writestr(f"{topic_guid}/markup.bcf",
                        _warning_markup_xml(w, topic_guid, viewpoint_guid,
                                            timestamp, topic_type,
                                            storey_name=w.get("storey_name")))
            zf.writestr(f"{topic_guid}/viewpoint.bcfv",
                        _viewpoint_xml(viewpoint_guid, selection_guid,
                                       cam_target))

    # Remove the scratch folder now that everything is inside the .bcf
    try:
        import shutil
        shutil.rmtree(snap_dir, ignore_errors=True)
    except Exception:
        pass

    return bcf_path
