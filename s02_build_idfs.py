"""
It generates one modified IDF per design variant.

Reads datasets/design_matrix.csv, applies each row to the prototype IDF,
and writes runs/idfs/d###.idf. Also injects the hourly Output:Meter
objects needed for weekly energy extraction.

Run `python s02_build_idfs.py --inspect` first: it lists Material,
Material:NoMass, glazing and setpoint-schedule names so the patterns in
config.IDF_PATTERNS can be verified.
"""

import argparse
import sys

import pandas as pd
from eppy.modeleditor import IDF

from config import (BASE_IDF, IDD_FILE, IDF_PATTERNS, OUTPUT_DIR,
                    DATASET_DIR, INSULATION_CONDUCTIVITY)


# helpers

def _as_tuple(pattern):
    return pattern if isinstance(pattern, (tuple, list)) else (pattern,)


def find_objects(idf, idf_type, patterns):
    """Case-insensitive substring match on object Name; any pattern counts."""
    needles = [p.lower() for p in _as_tuple(patterns)]
    return [o for o in idf.idfobjects[idf_type]
            if any(n in o.Name.lower() for n in needles)]


def require(objs, what):
    if not objs:
        sys.exit(f"ERROR: no IDF object matched pattern for '{what}'. "
                 f"Run with --inspect and fix config.IDF_PATTERNS.")
    return objs


def set_insulation(idf, patterns, thickness_m, what):
    """Set insulation 'thickness': directly for Material; as equivalent
    R-value (t / k) for Material:NoMass layers."""
    mats = find_objects(idf, "MATERIAL", patterns)
    if mats:
        for m in mats:
            m.Thickness = thickness_m
        return [m.Name for m in mats]
    nomass = require(find_objects(idf, "MATERIAL:NOMASS", patterns), what)
    r_value = thickness_m / INSULATION_CONDUCTIVITY
    for m in nomass:
        m.Thermal_Resistance = round(r_value, 4)
    return [m.Name for m in nomass]


def set_schedule_setpoints(idf, prefix, old_value, new_value):
    """Replace the occupied setpoint value in EVERY Schedule:Compact whose
    name starts with the given prefix (covers all prototype variants)."""
    scheds = [s for s in idf.idfobjects["SCHEDULE:COMPACT"]
              if s.Name.upper().startswith(prefix.upper())]
    require(scheds, f"schedules starting with {prefix}")
    hits = 0
    for sched in scheds:
        for i, field in enumerate(sched.fieldvalues):
            try:
                if abs(float(field) - old_value) < 0.05:
                    sched.fieldvalues[i] = round(new_value, 2)
                    hits += 1
            except (TypeError, ValueError):
                continue
    if hits == 0:
        print(f"  WARNING: value {old_value} not found in any {prefix}* "
              f"schedule; setpoint not applied.")


def scale_windows_vertically(idf, factor):
    """Scale each window's height about its own vertical centroid."""
    for w in idf.idfobjects["FENESTRATIONSURFACE:DETAILED"]:
        zs = [w[f"Vertex_{i}_Zcoordinate"] for i in range(1, 5)]
        zc = sum(zs) / len(zs)
        for i in range(1, 5):
            z = w[f"Vertex_{i}_Zcoordinate"]
            w[f"Vertex_{i}_Zcoordinate"] = zc + (z - zc) * factor


def apply_design(idf, row):
    p = IDF_PATTERNS

    # --- insulation (Material OR Material:NoMass)
    set_insulation(idf, p["wall_insulation_material"],
                   row.wall_insulation_thickness, "wall insulation")
    set_insulation(idf, p["roof_insulation_material"],
                   row.roof_insulation_thickness, "roof insulation")

    # --- mass layer thicknesses (non-fatal if the prototype has none active)
    for m in find_objects(idf, "MATERIAL", p["wall_mass_material"]):
        m.Thickness = row.wall_thickness
    for m in find_objects(idf, "MATERIAL", p["roof_mass_material"]):
        m.Thickness = row.roof_thickness

    # --- exterior absorptances
    for m in require(find_objects(idf, "MATERIAL", p["wall_exterior_material"]),
                     "wall exterior layer"):
        m.Solar_Absorptance = row.wall_absorptance
    for m in require(find_objects(idf, "MATERIAL", p["roof_exterior_material"]),
                     "roof exterior layer"):
        m.Solar_Absorptance = row.roof_absorptance

    # --- glazing (SimpleGlazingSystem covers U, SHGC, VT in one object)
    glz = require(idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"],
                  "WindowMaterial:SimpleGlazingSystem")
    for g in glz:
        g.UFactor = row.window_u_factor
        g.Solar_Heat_Gain_Coefficient = row.window_shgc
        g.Visible_Transmittance = row.visible_transmittance

    # --- orientation
    idf.idfobjects["BUILDING"][0].North_Axis = row.north_axis

    # --- internal equipment gains (W/m2 design level)
    for eq in idf.idfobjects["ELECTRICEQUIPMENT"]:
        method = str(eq.Design_Level_Calculation_Method).lower()
        if "area" in method:
            eq.Watts_per_Zone_Floor_Area = row.equipment_gain

    # --- window geometry
    scale_windows_vertically(idf, row.window_scale_factor)

    # --- setpoints (all HTGSETP_SCH* / CLGSETP_SCH* variants)
    set_schedule_setpoints(idf, p["heating_schedule_prefix"],
                           p["default_heating_occupied"], row.heating_setpoint)
    set_schedule_setpoints(idf, p["cooling_schedule_prefix"],
                           p["default_cooling_occupied"], row.cooling_setpoint)


def ensure_output_meters(idf):
    """Hourly whole-facility meters; eplusout.csv is produced via readvars (-r)."""
    wanted = ["Electricity:Facility", "NaturalGas:Facility"]
    existing = {m.Key_Name for m in idf.idfobjects["OUTPUT:METER"]}
    for key in wanted:
        if key not in existing:
            idf.newidfobject("OUTPUT:METER", Key_Name=key,
                             Reporting_Frequency="Hourly")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true",
                    help="List material/schedule names in the base IDF and exit.")
    args = ap.parse_args()

    if IDD_FILE:
        IDF.setiddname(str(IDD_FILE))
    else:
        import shutil, os
        exe = shutil.which("energyplus")
        if exe:
            IDF.setiddname(os.path.join(os.path.dirname(os.path.realpath(exe)),
                                        "Energy+.idd"))
        else:
            sys.exit("Set config.IDD_FILE to your Energy+.idd path.")

    idf = IDF(str(BASE_IDF))

    if args.inspect:
        print("== MATERIAL objects ==")
        for m in idf.idfobjects["MATERIAL"]:
            print("  ", m.Name)
        print("== MATERIAL:NOMASS objects ==")
        for m in idf.idfobjects["MATERIAL:NOMASS"]:
            print("  ", m.Name)
        print("== WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM ==")
        for m in idf.idfobjects["WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM"]:
            print("  ", m.Name)
        print("== SCHEDULE:COMPACT (setpoint-like) ==")
        for s in idf.idfobjects["SCHEDULE:COMPACT"]:
            if "SETP" in s.Name.upper():
                print("  ", s.Name)
        return

    designs = pd.read_csv(DATASET_DIR / "design_matrix.csv")
    idf_dir = OUTPUT_DIR / "idfs"
    idf_dir.mkdir(parents=True, exist_ok=True)

    for row in designs.itertuples(index=False):
        idf = IDF(str(BASE_IDF))          # fresh copy each time
        apply_design(idf, row)
        ensure_output_meters(idf)
        out = idf_dir / f"{row.design_id}.idf"
        idf.saveas(str(out))
        print(f"wrote {out.name}")

    print(f"\nDone: {len(designs)} IDFs in {idf_dir}")


if __name__ == "__main__":
    main()
