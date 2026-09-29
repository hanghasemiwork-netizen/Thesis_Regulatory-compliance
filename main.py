"""
main.py
SCIE Fire Evacuation Travel Distance and Accessibility Ramp Compliance Checker
RT-SCIE (Portaria 1532/2008) | Accessibility Ramp (DL 163/2006)

Usage:
    python main.py              auto-detects a .ifc file in the project folder
    python main.py model.ifc    uses the given path
"""

import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ifc_parser
import ids_validator
import rule_checker
import bcf_reporter
import RT_SCIE_checker as visibility_engine
import visualizer

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IDS_PATH   = os.path.join(BASE_DIR, "ids",   "scie_ids.ids")
RULES_PATH = os.path.join(BASE_DIR, "rules", "scie_rules.json")
DL163_PATH = os.path.join(BASE_DIR, "rules", "dl163_rules.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def run(ifc_path: str) -> dict:
    """
    Run the full pipeline: load IFC, validate against IDS, extract data,
    route evacuation paths, check SCIE + DL163 rules, write the BCF report
    and floor-plan PNGs.

    Args:
        ifc_path: path to the .ifc model file.
    Returns:
        check_results dict (see rule_checker.check_all), or None if the
        model has no spaces.
    Called by: __main__ (script entry point).
    """
    # ── Step 1: Load IFC ──────────────────────────────────────────
    print("Uploading IFC file...")
    model = ifc_parser.load_ifc(ifc_path)

    # ── Step 2: IDS Validation ────────────────────────────────────
    print("Running IDS validation...")
    ids_result = ids_validator.validate(model, IDS_PATH)

    # A failed prerequisite stops only the domain it belongs to: SCIE checks
    # guard the evacuation routing; accessibility checks are independent, so
    # a missing fire-exit or entrance marking must not skip the ramp check.
    scie_ok = ids_result.get("domain_passed", {}).get("SCIE",
                                                      ids_result["passed"])
    if not scie_ok:
        print("  Fire-safety prerequisites not met — evacuation check skipped.")

    # Warning-severity IDS checks (e.g. IDS-05 missing space LongName) never
    # gate the pipeline — just surfaced here so they are not silently lost.
    for w in ids_result.get("warnings", []):
        print(f"  IDS warning [{w['check_id']}]: {w['message']}")

    # ── Step 3: Extract IFC data ──────────────────────────────────
    print("Parsing IFC model...")
    rules    = rule_checker.load_rules(RULES_PATH)
    ifc_data = ifc_parser.extract_all(model, rules)

    if len(ifc_data["spaces"]) == 0:
        print("No spaces found in the model. Stopping.")
        return None

    exit_door_count = sum(1 for d in ifc_data["doors"] if d.get("is_fire_exit"))
    print(f"  {len(ifc_data['spaces'])} spaces, {exit_door_count} exit doors found")

    if len(ifc_data["space_boundaries"]) == 0:
        print("  Warning: no room-door relationships found in the model.")

    # Fire exits per floor — every IfcBuildingStorey, not just the ones
    # with doors, so a floor missing an exit entirely still shows up (it
    # would otherwise never appear by iterating doors alone). Matched by
    # elevation with the same 1.0 m tolerance used everywhere else in the
    # pipeline for "is this door/space/obstacle on this floor".
    exit_doors = [d for d in ifc_data["doors"] if d.get("is_fire_exit")]
    fire_exits_by_floor = []
    no_fire_exit_warnings = []
    for storey in ifc_data.get("storeys", []):
        elev  = storey["elevation_m"]
        count = sum(1 for d in exit_doors
                   if abs(d.get("storey_elevation_m", 0.0) - elev) <= 1.0)
        fire_exits_by_floor.append({
            "storey_name":     storey["name"],
            "elevation_m":     elev,
            "fire_exit_count": count,
        })
        if count == 0:
            no_fire_exit_warnings.append({
                "check_id":    "NO-FIRE-EXIT",
                "space_id":    storey["id"],
                "space_name":  f"Floor {storey['name']}",
                "storey_name": storey["name"],
                "message": (f"Floor '{storey['name']}' has no fire exit "
                            f"door — every space on it will show as "
                            f"unreachable (SCIE-UNREACHABLE) until a door "
                            f"is tagged FireExit = Yes.")
            })
    if no_fire_exit_warnings:
        print(f"  Warning: {len(no_fire_exit_warnings)} floor(s) have no "
              f"fire exit door — see report for details.")

    # Floor area with no IfcSpace covering it — a possible missing room.
    # Data-quality warning only; never blocks the pipeline (a gap can also
    # be intentional: a void, an unenclosed balcony, a non-occupiable slab
    # edge).
    coverage_warnings = ifc_parser.check_space_coverage(model)
    coverage_report_warnings = [
        {"check_id":    "COVERAGE-GAP",
         "space_id":    c["id"],
         "space_name":  f"(uncovered area, {c['residual_area_m2']} m²)",
         "storey_name": c.get("storey_name"),
         "message": (f"Uncovered area of {c['residual_area_m2']} m² with "
                     f"no IfcSpace assigned — a room may be missing from "
                     f"the export."),
         # Not a real IFC entity — no GUID a BCF viewer can select. Kept
         # only so bcf_reporter can still aim the topic's camera near the
         # gap; footprint/elevation, never a selection.
         "footprint":          c.get("footprint"),
         "storey_elevation_m": c.get("storey_elevation_m")}
        for c in coverage_warnings
    ]
    if coverage_warnings:
        print(f"  Warning: {len(coverage_warnings)} floor area(s) have no "
              f"room assigned — see report for details.")

    # Spaces with no LongName (IDS-05): a naming/identification warning,
    # not a compliance failure — helps the architect find and name the
    # space, so it is reported and drawn but never blocks the pipeline.
    missing_long_name = [
        {"check_id":    "IDS-05",
         "space_id":    s["id"],
         "space_name":  s["name"],
         "storey_name": s.get("storey_name"),
         "message": (f"Space '{s['name']}' has no Long Name — assign one "
                     f"so it can be identified in reports and floor "
                     f"plans."),
         # For the BCF topic's viewpoint camera (real space -> a proper
         # <Selection> too; see bcf_reporter._write_bcf_file).
         "footprint":          s.get("footprint"),
         "storey_elevation_m": s.get("storey_elevation_m")}
        for s in ifc_data["spaces"] if not s.get("long_name")
    ]
    if missing_long_name:
        print(f"  Warning: {len(missing_long_name)} space(s) have no Long "
              f"Name — see report for details.")

    # Spaces tagged non_occupiable = Yes (IDS-06): excluded from the Art.57
    # evacuation-distance check (see ifc_parser.extract_all) — flagged so
    # that exclusion is a reviewed decision, not silently missed. Never
    # blocks the pipeline.
    non_occupiable_warnings = [
        {"check_id":    "IDS-06",
         "space_id":    s["id"],
         "space_name":  s["name"],
         "storey_name": s.get("storey_name"),
         "message": (f"Space '{s['name']}' (ID: {s['id']}) is tagged "
                     f"non_occupiable = Yes and excluded from the "
                     f"evacuation-distance check."),
         "footprint":          s.get("footprint"),
         "storey_elevation_m": s.get("storey_elevation_m")}
        for s in ifc_data.get("non_occupiable_spaces", [])
    ]
    if non_occupiable_warnings:
        print(f"  Warning: {len(non_occupiable_warnings)} space(s) are "
              f"tagged non_occupiable — see report for details.")

    report_warnings = (missing_long_name + non_occupiable_warnings
                       + coverage_report_warnings + no_fire_exit_warnings)

    # ── Step 4-5: Fire safety block (SCIE) ────────────────────────
    # Runs only when its own IDS prerequisites are satisfied.
    path_results, building = [], {}
    check_results = {"total_spaces": 0, "compliant": 0, "violations": 0,
                     "violation_list": []}

    if scie_ok:
        print("Generating visibility graph...")
        print("Finding evacuation paths...")
        path_results, building = visibility_engine.build_path_results(ifc_data)

        if path_results:
            print("Checking compliance...")
            check_results = rule_checker.check_all(path_results, RULES_PATH)

    # ── Accessibility block (DL 163/2006) ─────────────────────────
    # Separate domain from the fire-safety block; only checks ramp slope.
    # Its violations are merged into the same violation list so they appear
    # in the same BCF and report, but the two checks never share logic.
    ramp_violations = []
    dl163_ok = ids_result.get("domain_passed", {}).get("DL163", True)
    if not dl163_ok:
        print("  Accessibility prerequisites not met — ramp check skipped.")
    else:
        try:
            # Each regulation has its own rules file: SCIE limits in
            # rules/scie_rules.json, DL 163/2006 limits in rules/dl163_rules.json.
            dl163_rules = rule_checker.load_rules(DL163_PATH)
            ramp_violations = rule_checker.check_ramps(
                ifc_data, dl163_rules)
        except Exception as e:
            print(f"  Accessibility check could not run — skipped ({e}).")

    # Merge ramp violations into the shared results so BCF + report include them
    if ramp_violations:
        check_results["violation_list"].extend(ramp_violations)
        check_results["violations"] += len(ramp_violations)

    # ── Step 6: BCF report ────────────────────────────────────────
    print("Generating report...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear floor/ramp PNGs left over from a previous run before writing
    # this run's own. Step 7 below only writes a floor_*.png for a floor
    # that currently has a fire exit — a floor that had one on a previous
    # run (and got a PNG) but lost it since (e.g. a re-exported model) would
    # otherwise leave that old PNG sitting in the folder, still showing its
    # old exit star and looking exactly like a fresh, current result.
    for stale_png in (glob.glob(os.path.join(OUTPUT_DIR, "floor_*.png"))
                      + glob.glob(os.path.join(OUTPUT_DIR, "ramp_*.png"))):
        try:
            os.remove(stale_png)
        except OSError:
            pass

    bcf_reporter.generate_bcf_issues(check_results, OUTPUT_DIR,
                                     ifc_data=ifc_data, model=model,
                                     warnings=report_warnings,
                                     fire_exits_by_floor=fire_exits_by_floor)

    # ── Step 7: Floor plan visualisation ─────────────────────────
    # Draws exactly the engine's own waypoints (single source of truth).
    print("Visualising floor plans...")
    # Ids of every space that failed any compliance check; the visualizer
    # fills these rooms red. Ramp violations carry a ramp GlobalId that
    # never matches a room id, so they correctly colour no room.
    violating_ids = {v.get("space_id")
                     for v in check_results["violation_list"]}
    # No-path spaces get their own colour (purple), separate from a route
    # that merely exceeds the distance limit.
    unreachable_ids = {v.get("space_id")
                       for v in check_results["violation_list"]
                       if v.get("type") == "UNREACHABLE"}
    # Spaces with no LongName get a yellow fill (see missing_long_name
    # above) so the architect can spot them at a glance on the floor plan.
    missing_long_name_ids = {w["space_id"] for w in missing_long_name}
    # Non-occupiable spaces never enter routing (excluded upstream in
    # ifc_parser.extract_all), so they are not in floor["area"]["spaces"]
    # at all — passed separately below and drawn green as their own layer.
    non_occupiable_spaces = ifc_data.get("non_occupiable_spaces", [])
    for elevation, floor in building.items():
        if floor["exit"] is None:
            continue
        png_path = os.path.join(OUTPUT_DIR, f"floor_{elevation:.2f}m.png")
        visualizer.draw_all_paths(
            floor["area"], floor["results"], floor["exit"],
            elevation, png_path, violating_ids=violating_ids,
            unreachable_ids=unreachable_ids,
            missing_long_name_ids=missing_long_name_ids,
            non_occupiable_spaces=non_occupiable_spaces,
            coverage_gaps=coverage_warnings,
            exit_points=floor.get("exits"))

    # ── Step 7b: Ramp overview diagrams (DL 163/2006) ──────────────
    # One PNG per ramp entity: the same floor-plan-crop rendering used for
    # its BCF snapshots, but showing every currently-known violating
    # flight on the ramp at once rather than one topic's own flight.
    if ifc_data.get("ramps"):
        violating_flight_ids = {v["space_id"] for v in ramp_violations}
        ramp_ids = {f["ramp_id"]: f for f in ifc_data["ramps"]}.items()
        for ramp_id, sample_flight in ramp_ids:
            ramp_name = sample_flight.get("name", "Ramp").split(" [flight")[0]
            safe_name = "".join(c if c.isalnum() else "_" for c in ramp_name)
            ramp_png = os.path.join(OUTPUT_DIR, f"ramp_{safe_name}.png")
            visualizer.draw_ramp_snapshot(ifc_data, ramp_id, violating_flight_ids,
                                          ramp_png, ramp_name=ramp_name)

    # ── Final summary ─────────────────────────────────────────────
    scie_viol  = check_results['violations'] - len(ramp_violations)
    total_viol = check_results["violations"]
    print()
    if total_viol == 0:
        print(f"Done — {check_results['total_spaces']} spaces checked, all compliant.")
    else:
        print(f"Done — {check_results['total_spaces']} spaces checked, "
              f"{check_results['compliant']} compliant, {total_viol} "
              f"violation(s) ({scie_viol} fire, {len(ramp_violations)} ramp).")
    print(f"See {OUTPUT_DIR}/ for the full report.")

    return check_results


if __name__ == "__main__":
    # `python main.py model.ifc` uses sys.argv[1] as the IFC path;
    # `python main.py` alone auto-detects the .ifc file in BASE_DIR.
    if len(sys.argv) >= 2:
        ifc_path = sys.argv[1]
    else:
        ifc_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".ifc")]
        if not ifc_files:
            print("No .ifc file found in the project folder.")
            sys.exit(1)
        if len(ifc_files) > 1:
            print(f"Multiple IFC files found — using {ifc_files[0]}")
        ifc_path = os.path.join(BASE_DIR, ifc_files[0])

    run(ifc_path)
