"""
Central configuration for the Nepal surrogate-modeling pipeline.
Parameter ranges follow Manmatharasan et al. (2026), Energy & Buildings 359, 117251.
"""

from pathlib import Path


PROJECT_DIR = Path(__file__).parent
ENERGYPLUS_EXE = "C:/EnergyPlusV26-2-0/energyplus.exe"
BASE_IDF = "G:/Aashish Documents/Papers/building energy/baseline files and weather files/model/Baseline Building_denver.idf"
IDD_FILE = "C:/EnergyPlusV26-2-0/Energy+.idd"
WEATHER_DIR = Path("G:/Aashish Documents/Papers/building energy/baseline files and weather files/weather file")     # put EPW files here, named <location>.epw
OUTPUT_DIR = PROJECT_DIR / "runs"
DATASET_DIR = PROJECT_DIR / "datasets"

N_DESIGNS = 50          
RANDOM_SEED = 42
WEEKS_PER_YEAR = 52     

# Assumed conductivity used to convert sampled insulation THICKNESS (m) into an
# equivalent R-value for Material:NoMass insulation layers:  R = t / k
INSULATION_CONDUCTIVITY = 0.045   # W/m-K, typical board insulation

# Design parameters: (min, max) — exactly the 14 ranges used in the paper

PARAM_RANGES = {
    "wall_insulation_thickness": (0.02, 0.10),   # m
    "roof_insulation_thickness": (0.02, 0.10),   # m
    "window_u_factor":           (1.2, 2.0),     # W/m2K
    "window_shgc":               (0.3, 0.7),     # -
    "visible_transmittance":     (0.5, 1.0),     # -
    "wall_thickness":            (0.1, 0.5),     # m  (mass wall layers, if present)
    "roof_thickness":            (0.1, 0.5),     # m  (mass roof layer, if present)
    "north_axis":                (0.0, 360.0),   # deg
    "wall_absorptance":          (0.5, 0.9),     # solar absorptance
    "roof_absorptance":          (0.5, 0.9),
    "equipment_gain":            (5.0, 15.0),    # W/m2
    "window_scale_factor":       (0.5, 1.0),     # vertical scaling of glazing
    "heating_setpoint":          (18.0, 22.0),   # degC (occupied)
    "cooling_setpoint":          (24.0, 28.0),   # degC (occupied)
}

# IDF object-name patterns, aligned to the names found in this prototype
# via `python s02_build_idfs.py --inspect`. Each entry may be a single
# substring or a tuple of substrings (any match counts, case-insensitive).

IDF_PATTERNS = {
    # Insulation: searched in MATERIAL first, then MATERIAL:NOMASS
    "wall_insulation_material": ("wall insulation", "wall_insulation"),
    "roof_insulation_material": ("roof insulation", "roof_insulation"),
    # Mass layers whose Thickness is varied (non-fatal if absent)
    "wall_mass_material":       "concrete wall",
    "roof_mass_material":       "concrete roof",
    # Exterior-most layers whose Solar Absorptance is varied
    "wall_exterior_material":   "stucco",
    "roof_exterior_material":   ("built-up roofing", "metal roof surface", "asphalt shingles"),
    # Setpoint schedules: PREFIX match — updates every HTGSETP_SCH_* / CLGSETP_SCH_* variant
    "heating_schedule_prefix":  "HTGSETP_SCH",
    "cooling_schedule_prefix":  "CLGSETP_SCH",
    # Occupied setpoint values in the stock schedules, replaced by samples
    "default_heating_occupied": 21.0,
    "default_cooling_occupied": 24.0,
}

# TRAINING locations (Nepal). These are the only locations ever used for
# training, --grid rows, holdout pools, AE pretraining, and the
# --weather-scaler all reference set.

LOCATIONS = {
    "kathmandu":   {"zone": "temperate_midhill",  "epw": "kathmandu.epw"},
    "biratnagar":  {"zone": "hot_terai_east",     "epw": "biratnagar.epw"},
    "simara":      {"zone": "hot_terai_central",  "epw": "simara.epw"},
    "bhairahawa":  {"zone": "hot_terai_west",     "epw": "bhairahawa.epw"},
    "nepalgunj":   {"zone": "hot_terai_farwest",  "epw": "nepalgunj.epw"},
    "surkhet":     {"zone": "inner_valley",       "epw": "surkhet.epw"},
    "pokhara":     {"zone": "humid_midhill",      "epw": "pokhara.epw"},
    "okhaldhunga": {"zone": "cool_hill_east",     "epw": "okhaldhunga.epw"},
    "dadeldhura":  {"zone": "cool_hill_west",     "epw": "dadeldhura.epw"},
    "jumla":       {"zone": "cold_mountain",      "epw": "jumla.epw"},
}


# v4: fixed physical bounds for weather scaling. Keys and ORDER must
# match EPW_FEATURES in s04_extract_weekly.py. EPW units: degC, %, Pa,
# Wh/m2, m/s. Values slightly outside bounds map slightly outside [0,1]
# (no clipping) - harmless; the point is a fixed, location-independent
# scale that never changes when locations are added.

WEATHER_BOUNDS = {
    "drybulb":  (-40.0,    50.0),
    "dewpoint": (-45.0,    35.0),
    "rel_hum":  (  0.0,   100.0),
    "pressure": (45000.0, 105000.0),
    "glob_rad": (  0.0,  1250.0),
    "dir_rad":  (  0.0,  1100.0),
    "dif_rad":  (  0.0,   700.0),
    "wind_spd": (  0.0,    40.0),
}


# EVALUATION-ONLY cities (Asia): simulated by s03/s04 and evaluated by
# s06, but NEVER trained on. Excluded from --grid, holdout pools, AE
# pretraining, and the --weather-scaler all reference set.
# Uncomment each entry once its EPW is downloaded (lowercase filename).

EVAL_LOCATIONS = {
    # --- South Asia (claimed region) ---
    "patna":       {"zone": "indo_gangetic",    "epw": "patna.epw"},
    "lahore":      {"zone": "indo_gangetic",    "epw": "lahore.epw"},
    "dhaka":       {"zone": "monsoon_delta",    "epw": "dhaka.epw"},
    "chennai":     {"zone": "tropical",         "epw": "chennai.epw"},
    "colombo":     {"zone": "equatorial",       "epw": "colombo.epw"},
    "male":        {"zone": "oceanic_equator",  "epw": "male.epw"},
    "karachi":     {"zone": "hot_semiarid",     "epw": "karachi.epw"},
    "jodhpur":     {"zone": "hot_desert",       "epw": "jodhpur.epw"},
    "kabul":       {"zone": "continental_cold", "epw": "kabul.epw"},
    "thimphu":     {"zone": "himalayan_cold",   "epw": "thimphu.epw"},
    # --- Boundary tests (outside claimed region) ---
    "seoul":       {"zone": "boundary_continental", "epw": "seoul.epw"},
    "ulaanbaatar": {"zone": "boundary_extreme",     "epw": "ulaanbaatar.epw"},
}

ALL_LOCATIONS = {**LOCATIONS, **EVAL_LOCATIONS}