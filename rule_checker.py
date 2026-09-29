"""
rule_checker.py
Applies regulation thresholds to already-measured data — the one place in
the codebase that decides "is this a violation, and what does its report
record look like", for both domains below. Neither domain's functions
call the other's; they share this module only because both are "compare a
measurement to a rule's limit":

  - SCIE Art.57 (evacuation travel distance): every occupiable space's
    farthest walkable point to its nearest fire exit door, checked against
    dead_end_max_m (default 15 m) or two_exits_max_m (default 30 m).
    Spaces tagged non_occupiable=Yes are excluded upstream by ifc_parser.
    Use-type refinements and sprinkler relaxations are out of scope.
    Limits are read from rules/scie_rules.json. The measurement itself
    (routing, travel distance) is RT_SCIE_checker's job, not this
    module's — check_space/check_all only read its already-measured
    path_results.

  - DL163/2006 Anexo 2.5.1 (accessibility ramp slope): every accessible
    ramp flight's slope, rise, horizontal projection and width, checked
    against rules/dl163_rules.json (paired slope/rise/projection limits,
    interpolated between two anchors, plus an independent width minimum).
    Landing size (Anexo 2.5.5-2.5.6) and curved-flight radius (2.5.3) are
    out of scope. The interpolation/limit-working-out itself is
    accessibility_checker's job, not this module's — check_ramps only
    reads its already-computed per-flight checks (mirrors the SCIE split
    above: accessibility_checker is to check_ramps what RT_SCIE_checker
    is to check_space/check_all).

This module makes no IFC calls and does no geometry — it only reads the
plain measurement dicts ifc_parser, RT_SCIE_checker and
accessibility_checker already produced.
"""

import json

import accessibility_checker


def load_rules(rules_path: str) -> dict:
    """
    Load a rules JSON file.

    Args:
        rules_path: path to a rules JSON file.
    Returns:
        dict: the parsed rules.
    Called by: main.py (run), check_all.
    """
    with open(rules_path, "r") as f:
        return json.load(f)


def get_applicable_rule(rules: dict) -> dict:
    """
    Return the SCIE GENERAL rule entry, which applies to all occupiable
    spaces regardless of use type.

    Args:
        rules: parsed rules dict (see load_rules).
    Returns:
        dict: the GENERAL rule entry, or None if not found.
    Called by: check_space.
    """
    scie = rules.get("SCIE", rules)
    return scie.get("GENERAL") or next(iter(scie.values()), None)


def check_space(path_result: dict, rules: dict) -> list:
    """
    Check a single space's route result against the SCIE Art.57 baseline.

    Args:
        path_result: one entry from RT_SCIE_checker.build_path_results.
        rules: parsed rules dict.
    Returns:
        list[dict]: violation dicts (empty if compliant).
    Called by: check_all.
    """
    violations = []

    # Unreachable space — no path to any fire exit door.
    if not path_result.get("reachable"):
        violations.append({
            "type":        "UNREACHABLE",
            "space_id":    path_result["space_id"],
            "space_name":  path_result["space_name"],
            "storey_name": path_result.get("storey_name"),
            "rule_id":     "SCIE-UNREACHABLE",
            "article":     "Art. 57 RT-SCIE",
            "description": "Space has no path to any fire exit door",
            "measured_m":  None,
            "limit_m":     None,
            "excess_m":    None,
            "two_exits":   False,
        })
        return violations

    rule = get_applicable_rule(rules)
    if not rule:
        return violations

    # Select the applicable distance limit.
    two_exits = path_result.get("has_two_exits", False)
    limit     = rule["two_exits_max_m"] if two_exits else rule["dead_end_max_m"]
    path_type = "two exits" if two_exits else "dead-end"

    # Single travel distance: farthest walkable point -> nearest fire exit.
    distance = path_result.get("travel_distance_m", 0.0) or 0.0

    if distance > limit:
        excess = round(distance - limit, 3)
        detail = (
            f"farthest point to fire exit: {distance:.2f} m "
            f"(limit {limit} m, excess {excess} m, {path_type})"
        )
        violations.append({
            "type":        "ART57_VIOLATION",
            "space_id":    path_result["space_id"],
            "space_name":  path_result["space_name"],
            "storey_name": path_result.get("storey_name"),
            "rule_id":     rule["id"],
            "article":     rule["article"],
            "description": rule["description"],
            "measured_m":  round(distance, 3),
            "limit_m":     limit,
            "excess_m":    excess,
            "two_exits":   two_exits,
            "path_type":   path_type,
            "detail":      detail,
        })

    return violations


def check_all(path_results: list, rules_path: str,
              building_ut: str = None) -> dict:
    """
    Run Art.57 checks on every routed space.

    Args:
        path_results: list of route result dicts (RT_SCIE_checker.build_path_results).
        rules_path: path to the SCIE rules JSON file.
        building_ut: unused; kept for forward-compatibility.
    Returns:
        dict: {total_spaces, compliant, compliant_list, violations,
            violation_list, summary_by_rule}. compliant_list is a record
            per reachable, non-violating space — {space_id, space_name,
            storey_name, rule_id, article, measured_m, limit_m,
            path_type} — for the TEXT report's full compliant listing
            (bcf_reporter); the JSON and BCF reports only ever carry the
            aggregate `compliant` count, never this list.
    Called by: main.py (run).
    """
    rules = load_rules(rules_path)
    all_violations = []

    for result in path_results:
        all_violations.extend(check_space(result, rules))

    violating_ids = {v["space_id"] for v in all_violations}
    rule = get_applicable_rule(rules)

    compliant_list = []
    for r in path_results:
        if not r.get("reachable") or r["space_id"] in violating_ids:
            continue
        record = {"space_id": r["space_id"], "space_name": r["space_name"],
                  "storey_name": r.get("storey_name")}
        if rule:
            two_exits = r.get("has_two_exits", False)
            limit     = rule["two_exits_max_m"] if two_exits else rule["dead_end_max_m"]
            path_type = "two exits" if two_exits else "dead-end"
            distance  = r.get("travel_distance_m", 0.0) or 0.0
            record.update({
                "rule_id":    rule["id"],
                "article":    rule["article"],
                "measured_m": round(distance, 3),
                "limit_m":    limit,
                "path_type":  path_type,
            })
        compliant_list.append(record)

    summary = {}
    for v in all_violations:
        rid = v["rule_id"]
        if rid not in summary:
            summary[rid] = {"count": 0, "article": v["article"],
                            "description": v["description"]}
        summary[rid]["count"] += 1

    return {
        "total_spaces":    len(path_results),
        "compliant":       len(compliant_list),
        "compliant_list":  compliant_list,
        "violations":      len(all_violations),
        "violation_list":  all_violations,
        "summary_by_rule": summary,
    }


# ══════════════════════════════════════════════════════════════════════
# DL163/2006 — accessibility ramp slope (Anexo 2.5.1)
# ══════════════════════════════════════════════════════════════════════

def _make_ramp_violation(ramp: dict, rule_id: str, article: str,
                         description: str, measured_m, limit_m,
                         detail: str) -> dict:
    """
    Build a ramp violation dict in the shape the BCF reporter and TXT
    report consume. The flight's composite id and name (e.g. "Ramp A
    [flight 1/2]") occupy the space_id / space_name fields.

    Args:
        ramp: one flight dict from ifc_parser.get_ramps.
        rule_id, article, description: rule metadata.
        measured_m, limit_m: measured value and its limit.
        detail: human-readable explanation.
    Returns:
        dict: violation record.
    Called by: check_ramps.
    """
    if measured_m is not None and limit_m is not None:
        excess = round(measured_m - limit_m, 3)
    else:
        excess = None

    return {
        "domain":      "DL163",
        "type":        "RAMP_SLOPE_VIOLATION",
        "space_id":    ramp["id"],
        "space_name":  ramp.get("name", "Ramp"),
        "storey_name": ramp.get("storey_name"),
        "rule_id":     rule_id,
        "article":     article,
        "description": description,
        "measured_m":  measured_m,
        "limit_m":     limit_m,
        "excess_m":    excess,
        "path_type":   "ramp",
        "detail":      detail,
    }


def check_ramps(ifc_data: dict, rules: dict) -> list:
    """
    Decide which ramp flights violate DL 163/2006 Seccao 2.5.1 (slope,
    rise, projection) or 2.5.4 (width), from the limits
    accessibility_checker.compute_ramp_checks already worked out for each
    flight. This function's only job is the "measured vs limit ->
    violation?" decision and building the violation record — the same
    role check_space plays for SCIE Art.57.

    Args:
        ifc_data: parsed data dict (uses ifc_data['ramps'], already
            restricted by ifc_parser to HandicapAccessible=True flights).
        rules: full namespaced rules dict (uses rules['DL163']['ramp']).
    Returns:
        list[dict]: violation dicts (empty if all flights comply).
    Called by: main.py (run).
    """
    violations = []

    for c in accessibility_checker.compute_ramp_checks(ifc_data, rules):
        ramp = c["ramp"]

        # ── Slope check (Seccao 2.5.1) ─────────────────────────────────
        if c["exceeds_abs_max_slope"]:
            violations.append(_make_ramp_violation(
                ramp, c["rule_id"], c["article"], c["description"],
                c["slope_pct"], c["abs_max_slope"],
                f"slope {c['slope_pct']}% exceeds absolute maximum "
                f"{c['abs_max_slope']}%.{c['landing_note']}"))
        else:
            if c["rise_m"] > c["max_rise_m"]:
                violations.append(_make_ramp_violation(
                    ramp, c["rule_id"], c["article"], c["description"],
                    c["rise_m"], c["max_rise_m"],
                    f"rise {c['rise_m']} m exceeds {c['max_rise_m']} m "
                    f"allowed at slope {c['slope_pct']}%.{c['landing_note']}"))

            if c["projection_m"] > c["max_projection_m"]:
                violations.append(_make_ramp_violation(
                    ramp, c["rule_id"], c["article"], c["description"],
                    c["projection_m"], c["max_projection_m"],
                    f"horizontal projection {c['projection_m']} m exceeds "
                    f"{c['max_projection_m']} m allowed at slope "
                    f"{c['slope_pct']}%.{c['landing_note']}"))

        # ── Width check (Seccao 2.5.4) — independent of slope, so it
        # runs regardless of the slope outcome above. ───────────────────
        if c["width_check_enabled"] and c["width_m"] < c["effective_min_width_m"]:
            violations.append(_make_ramp_violation(
                ramp, c["rule_id"] + "-WIDTH", c["width_article"],
                "Minimum clear width of an accessible ramp.",
                c["width_m"], c["effective_min_width_m"],
                f"width {c['width_m']} m below "
                f"{c['effective_min_width_m']} m minimum"))

    return violations
