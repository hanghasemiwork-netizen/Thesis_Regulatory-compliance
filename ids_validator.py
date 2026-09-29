"""
ids_validator.py
Validates IFC model completeness against a buildingSMART IDS 1.0 XML file
(ids/scie_ids.ids), using the ifctester library (part of IfcOpenShell).

Domain routing (SCIE vs DL163) and hard-stop/warning severity are not
concepts standard IDS carries, so they are encoded in this project's own
naming convention and read back out here: each
<specification name="IDS-XX: ..."> starts with its check ID, extracted here
to look up domain and severity in CHECK_META.

Known library limitation: the installed ifctester version does not parse
the standard `identifier` XML attribute on <specification>, even though the
IDS 1.0 XSD defines it. This module reads the check ID from the
specification's `name` attribute instead (convention: "IDS-XX: ...").
"""

import ifctester.ids as ids_module


# Each check's domain and severity — the two things standard IDS cannot
# express on its own.
CHECK_META = {
    "IDS-01": {"domain": "SCIE",  "severity": "hard_stop"},
    "IDS-02": {"domain": "SCIE",  "severity": "hard_stop"},
    "IDS-03": {"domain": "SCIE",  "severity": "hard_stop"},
    "IDS-04": {"domain": "DL163", "severity": "hard_stop"},
    "IDS-05": {"domain": "SCIE",  "severity": "warning"},
    "IDS-06": {"domain": "SCIE",  "severity": "warning"},
}


def load_ids(ids_path: str):
    """
    Load an IDS XML file.

    Args:
        ids_path: path to the IDS XML file.
    Returns:
        ifctester.ids.Ids
    Called by: validate.
    """
    return ids_module.open(ids_path)


def validate(model, ids_path: str) -> dict:
    """
    Validate an IFC model against the IDS file at ids_path.

    Args:
        model: loaded ifcopenshell model.
        ids_path: path to the IDS XML file.
    Returns:
        dict: {passed, errors, warnings, domain_passed}
    Called by: main.py (run).
    """
    my_ids = load_ids(ids_path)
    my_ids.validate(model)

    errors   = []
    warnings = []

    for spec in my_ids.specifications:
        # Recover this project's check ID from the specification name (see
        # module docstring: ifctester does not parse the `identifier` attr).
        code = spec.name.split(":")[0].strip()
        meta = CHECK_META.get(code, {"domain": "SCIE", "severity": "hard_stop"})
        domain   = meta["domain"]
        severity = meta["severity"]

        if not spec.is_ifc_version:
            msg = (f"{spec.name} — model schema does not match the "
                   f"required IFC version ({spec.ifcVersion}).")
            (errors if severity == "hard_stop" else warnings).append({
                "check_id": code, "domain": domain, "message": msg})
            continue

        if spec.status:
            pass
        else:
            n_app = len(spec.applicable_entities)
            n_fail = len(spec.failed_entities)
            if n_app == 0:
                msg = f"{spec.description or spec.name} — no applicable element found."
            else:
                msg = (f"{spec.description or spec.name} — "
                       f"{n_fail}/{n_app} elements do not satisfy the requirement.")
            (errors if severity == "hard_stop" else warnings).append({
                "check_id": code, "domain": domain, "message": msg})

    # ── Summary ───────────────────────────────────────────────────────────
    failed_domains = set(e["domain"] for e in errors)
    all_domains    = set(meta["domain"] for meta in CHECK_META.values())
    domain_passed  = {d: (d not in failed_domains) for d in all_domains}

    return {
        "passed":        len(errors) == 0,
        "errors":        errors,
        "warnings":      warnings,
        "domain_passed": domain_passed,
    }
