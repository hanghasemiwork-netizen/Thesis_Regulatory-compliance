"""
accessibility_checker.py
Measurement/processing layer for DL163/2006 Anexo 2.5.1 ramp
accessibility — the DL163 counterpart to RT_SCIE_checker.py's role on the
SCIE side.

Takes the raw per-flight geometry ifc_parser.get_ramps already measured
(rise, projection, slope, width) and the DL163 rules, and works out what
each flight is ALLOWED to measure: interpolating the paired
slope/rise/projection limits between the rule's two anchors (Seccao
2.5.1), and picking the applicable minimum width for that flight's own
projection (Seccao 2.5.4). It never decides pass/fail and never builds a
violation record — that decision belongs to rule_checker.check_ramps
alone, the same way RT_SCIE_checker only measures travel distance and
rule_checker.check_space alone decides whether it's a violation. Keeping
that line in one place (rule_checker) means there is exactly one spot in
the codebase that knows what a "violation" looks like, for both domains.

This module makes no IFC calls and does no geometry — it only reads the
plain flight-measurement dicts ifc_parser.get_ramps already produced.
"""


def interpolated_limits(slope_pct: float, anchors: list) -> tuple:
    """
    Linearly interpolate the (max_rise_m, max_projection_m) limits allowed
    by Seccao 2.5.1 for a given slope, between two anchor points.

    Args:
        slope_pct: measured flight slope, percent.
        anchors: two dicts, each with slope_pct, max_rise_m,
            max_projection_m. Anchor A is the gentler slope (6%), anchor B
            the steeper (8%).
    Returns:
        tuple: (max_rise_m, max_projection_m)
    Called by: compute_ramp_checks.
    """
    a, b = sorted(anchors, key=lambda x: x["slope_pct"])

    if slope_pct <= a["slope_pct"]:
        return (a["max_rise_m"], a["max_projection_m"])

    if slope_pct >= b["slope_pct"]:
        return (b["max_rise_m"], b["max_projection_m"])

    span = b["slope_pct"] - a["slope_pct"]
    fraction = (slope_pct - a["slope_pct"]) / span if span else 0.0

    max_rise = a["max_rise_m"] + fraction * (b["max_rise_m"] - a["max_rise_m"])
    max_proj = (a["max_projection_m"]
                + fraction * (b["max_projection_m"] - a["max_projection_m"]))
    return (round(max_rise, 3), round(max_proj, 3))


def compute_ramp_checks(ifc_data: dict, rules: dict) -> list:
    """
    Work out the applicable DL163 Seccao 2.5.1 / 2.5.4 limits for every
    accessible ramp flight, without deciding whether any of them is a
    violation — that is rule_checker.check_ramps's job, run on this
    function's output.

    Args:
        ifc_data: parsed data dict (uses ifc_data['ramps'], already
            restricted by ifc_parser to HandicapAccessible=True flights).
        rules: full namespaced rules dict (uses rules['DL163']['ramp']).
    Returns:
        list[dict]: one record per flight — {ramp, rule_id, article,
            description, slope_pct, rise_m, projection_m, width_m,
            landing_note, abs_max_slope, exceeds_abs_max_slope,
            max_rise_m, max_projection_m (both None when
            exceeds_abs_max_slope, since Seccao 2.5.1's paired limits
            don't apply past the absolute maximum), width_check_enabled,
            effective_min_width_m, width_article}. Empty if the DL163
            ramp rule or the model's ramps are missing.
    Called by: rule_checker.check_ramps.
    """
    checks = []

    dl163 = rules.get("DL163", {})
    ramp_rule = dl163.get("ramp")
    if not ramp_rule:
        return checks

    ramps = ifc_data.get("ramps", [])
    if not ramps:
        return checks

    rule_id       = ramp_rule.get("id", "DL163-RAMP-2.5.1")
    article       = ramp_rule.get("article", "Anexo 2.5.1")
    description   = ramp_rule.get("description", "")
    anchors       = ramp_rule.get("anchors", [])
    abs_max_slope = ramp_rule.get("absolute_max_slope_pct", 8.0)
    width_cfg     = ramp_rule.get("width_check", {"enabled": False})

    width_enabled = width_cfg.get("enabled", False)
    min_width     = width_cfg.get("min_width_m", 1.20)
    min_width_sr  = width_cfg.get("min_width_short_ramp_m", 0.90)
    short_proj    = width_cfg.get("short_ramp_projection_m", 5.0)
    width_article = width_cfg.get("article", "Anexo 2.5.4")

    for ramp in ramps:
        slope_pct    = ramp.get("slope_pct", 0.0)
        rise_m       = ramp.get("rise_m", 0.0)
        projection_m = ramp.get("projection_m", 0.0)
        width_m      = ramp.get("width_m", 0.0)
        landing_note = (" A landing was detected between flights."
                        if ramp.get("has_landing") else "")

        exceeds_abs_max_slope = slope_pct > abs_max_slope
        if exceeds_abs_max_slope:
            # Past the absolute maximum, Seccao 2.5.1's interpolated
            # rise/projection limits no longer apply — only the slope
            # itself is checked (see rule_checker.check_ramps).
            max_rise_m, max_projection_m = None, None
        else:
            max_rise_m, max_projection_m = interpolated_limits(slope_pct, anchors)

        # Width (Seccao 2.5.4) is independent of slope, so it is computed
        # for every flight regardless of the slope outcome above.
        effective_min_width_m = min_width_sr if projection_m <= short_proj \
            else min_width

        checks.append({
            "ramp":                  ramp,
            "rule_id":               rule_id,
            "article":               article,
            "description":           description,
            "slope_pct":             slope_pct,
            "rise_m":                rise_m,
            "projection_m":          projection_m,
            "width_m":               width_m,
            "landing_note":          landing_note,
            "abs_max_slope":         abs_max_slope,
            "exceeds_abs_max_slope": exceeds_abs_max_slope,
            "max_rise_m":            max_rise_m,
            "max_projection_m":      max_projection_m,
            "width_check_enabled":   width_enabled,
            "effective_min_width_m": effective_min_width_m,
            "width_article":         width_article,
        })

    return checks
