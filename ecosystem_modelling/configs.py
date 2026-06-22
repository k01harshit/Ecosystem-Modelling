"""
configs.py — v5

Validation protocol: 5-fold stratified cross-validation only.
No fixed train/test split. Every event is test exactly once.

All 21 invasion events are in INVASION_EVENTS.
The CV protocol in metrics.py handles stratified splitting per fold.

Event composition (21 total):
  collapse  (6): Python, Brown Tree Snake, Nile Perch, Smallpox,
                 Stoat, Chytrid Fungus
  disrupted (9): Cane Toad, Lionfish, Zebra Mussel, Rabbit,
                 Asian Carp, Parthenium Weed, Ash Borer,
                 Water Hyacinth, Red Fox
  stable    (6): Honeybee, Pheasant, Common Carp, Fallow Deer,
                 Canada Goose, Mallard Duck
  Spans: 8 biomes, 6 invader taxonomic groups, 3 continents
"""

from typing import List

# ── Node feature dimensions ───────────────────────────────────────────────────
NODE_FEATURE_DIM = 16
HIDDEN_DIM       = 64
EMBED_DIM        = 32
NUM_HEADS        = 4
NUM_GNN_LAYERS   = 3

# ── Edge / relation types ─────────────────────────────────────────────────────
EDGE_TYPES     = ["predation", "competition", "mutualism", "parasitism"]
NUM_EDGE_TYPES = len(EDGE_TYPES)

# ── Neural ODE ────────────────────────────────────────────────────────────────
ODE_TIME_STEPS = 200
ODE_T_END      = 50.0
ODE_SOLVER     = "rk4"

# ── SSL pretraining ───────────────────────────────────────────────────────────
SSL_MASK_RATE       = 0.20
SSL_CONTRASTIVE_TAU = 0.3
SSL_PRETRAIN_EPOCHS = 150
SSL_LR              = 8e-4
SSL_TROPHIC_WEIGHT  = 1.5
SSL_WARMUP_EPOCHS   = 10

# ── Fine-tuning ───────────────────────────────────────────────────────────────

# ── Cross-validation ──────────────────────────────────────────────────────────
                 # producing far more stable AUC/accuracy estimates with 21 events

# ── LV physics losses ─────────────────────────────────────────────────────────


# ── EdgePredictor ─────────────────────────────────────────────────────────────
EDGE_PRED_EPOCHS = 20
EDGE_PRED_LR     = 5e-4

# ── Data augmentation ─────────────────────────────────────────────────────────
AUG_EDGE_DROP_RATE   = 0.15
AUG_WEIGHT_NOISE_STD = 0.08
NUM_SYNTHETIC_GRAPHS = 200

# ── Validation ────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# ALL INVASION EVENTS — used exclusively via 5-fold cross-validation
# ══════════════════════════════════════════════════════════════════════════════
# 6 collapse + 9 disrupted + 6 stable
# Biomes: Everglades, Guam, Lake Victoria, NA forest, Australia (multiple),
#         Caribbean reef, Great Lakes, grassland, meadow, wetlands, NZ forest,
#         global wetlands, African lakes
INVASION_EVENTS = [

    # collapses (4)
    {
        "name": "Burmese Python — Florida Everglades",
        "invader": "Python bivittatus",
        "ecosystem": "Florida Everglades",
        "outcome": "collapse", "severity": 0.95,
        "at_risk_groups": ["mammals", "birds", "reptiles"],
        "notes": "Mammal populations declined 90-99%.",
    },
    {
        "name": "Brown Tree Snake — Guam",
        "invader": "Boiga irregularis",
        "ecosystem": "Guam forest",
        "outcome": "collapse", "severity": 0.98,
        "at_risk_groups": ["birds", "lizards"],
        "notes": "Nearly all native forest birds extinct.",
    },
    {
        "name": "Nile Perch — Lake Victoria",
        "invader": "Lates niloticus",
        "ecosystem": "Lake Victoria",
        "outcome": "collapse", "severity": 0.92,
        "at_risk_groups": ["cichlids", "fish"],
        "notes": "~200 cichlid species extinct.",
    },
    {
        "name": "Smallpox — Native American Ecosystems",
        "invader": "Variola major",
        "ecosystem": "North American forest",
        "outcome": "collapse", "severity": 0.88,
        "at_risk_groups": ["humans", "large_mammals"],
        "notes": "Cascading ecosystem effects from loss of keystone management.",
    },

    # disrupted (6)
    {
        "name": "Cane Toad — Australia",
        "invader": "Rhinella marina",
        "ecosystem": "Australian wetlands",
        "outcome": "disrupted", "severity": 0.70,
        "at_risk_groups": ["reptiles", "mammals"],
        "notes": "Toxin kills predators.",
    },
    {
        "name": "Lionfish — Caribbean Reef",
        "invader": "Pterois volitans",
        "ecosystem": "Caribbean reef",
        "outcome": "disrupted", "severity": 0.65,
        "at_risk_groups": ["reef_fish"],
        "notes": "Reef fish declined ~65% locally.",
    },
    {
        "name": "Zebra Mussel — Great Lakes",
        "invader": "Dreissena polymorpha",
        "ecosystem": "Great Lakes",
        "outcome": "disrupted", "severity": 0.60,
        "at_risk_groups": ["filter_feeders", "plankton"],
        "notes": "Outcompetes native mussels.",
    },
    {
        "name": "Rabbit — Australia",
        "invader": "Oryctolagus cuniculus",
        "ecosystem": "Australian grassland",
        "outcome": "disrupted", "severity": 0.55,
        "at_risk_groups": ["vegetation", "small_mammals"],
        "notes": "Vegetation loss and soil erosion.",
    },
    {
        "name": "Asian Carp — Mississippi River",
        "invader": "Hypophthalmichthys molitrix",
        "ecosystem": "Mississippi river",
        "outcome": "disrupted", "severity": 0.62,
        "at_risk_groups": ["native_fish", "plankton"],
        "notes": "Outcompetes native filter feeders.",
    },
    {
        "name": "Parthenium Weed — India/Australia",
        "invader": "Parthenium hysterophorus",
        "ecosystem": "Grassland",
        "outcome": "disrupted", "severity": 0.50,
        "at_risk_groups": ["vegetation", "insects"],
        "notes": "Allelopathic suppression of native plants.",
    },
    {
        "name": "Emerald Ash Borer — North America",
        "invader": "Agrilus planipennis",
        "ecosystem": "North American forest",
        "outcome": "disrupted", "severity": 0.68,
        "at_risk_groups": ["ash_trees", "forest_birds"],
        "notes": "Killed billions of ash trees. Acts via tree mortality "
                 "not direct predation — indirect cascade mechanism.",
    },

    # stable (5)
    {
        "name": "European Honeybee — North America",
        "invader": "Apis mellifera",
        "ecosystem": "North American meadow",
        "outcome": "stable", "severity": 0.10,
        "at_risk_groups": [],
        "notes": "Integrated as pollinator. Minor competition with native bees.",
    },
    {
        "name": "Ring-necked Pheasant — North America",
        "invader": "Phasianus colchicus",
        "ecosystem": "North American grassland",
        "outcome": "stable", "severity": 0.12,
        "at_risk_groups": [],
        "notes": "Occupies farmland niche. Minimal native impact.",
    },
    {
        "name": "Common Carp — Australia (managed)",
        "invader": "Cyprinus carpio",
        "ecosystem": "Australian river",
        "outcome": "stable", "severity": 0.18,
        "at_risk_groups": ["aquatic_plants"],
        "notes": "Impacts controlled by fishing pressure.",
    },
    {
        "name": "Fallow Deer — Australia",
        "invader": "Dama dama",
        "ecosystem": "Australian forest",
        "outcome": "stable", "severity": 0.15,
        "at_risk_groups": [],
        "notes": "Fills vacant large herbivore niche.",
    },
    {
        "name": "Canada Goose — UK",
        "invader": "Branta canadensis",
        "ecosystem": "UK wetlands",
        "outcome": "stable", "severity": 0.20,
        "at_risk_groups": ["wetland_plants"],
        "notes": "Grazing impacts localised. Stability maintained.",
    },

    # collapse (continued)
    {
        "name": "Stoat — New Zealand Forest",
        "invader": "Mustela erminea",
        "ecosystem": "New Zealand forest",
        "outcome": "collapse", "severity": 0.85,
        "at_risk_groups": ["birds", "lizards", "invertebrates"],
        "notes": "Caused collapse of native bird populations on NZ mainland "
                 "and offshore islands. No natural predators.",
    },
    {
        "name": "Chytrid Fungus — Global Amphibians",
        "invader": "Batrachochytrium dendrobatidis",
        "ecosystem": "Global wetlands",
        "outcome": "collapse", "severity": 0.90,
        "at_risk_groups": ["amphibians", "frogs"],
        "notes": "Caused extinction of 90+ amphibian species.",
    },

    # disrupted (continued)
    {
        "name": "Water Hyacinth — African Lakes",
        "invader": "Eichhornia crassipes",
        "ecosystem": "African lakes",
        "outcome": "disrupted", "severity": 0.58,
        "at_risk_groups": ["aquatic_plants", "fish"],
        "notes": "Covers lake surface, reduces oxygen.",
    },
    {
        "name": "Red Fox — Australia",
        "invader": "Vulpes vulpes",
        "ecosystem": "Australian shrubland",
        "outcome": "disrupted", "severity": 0.72,
        "at_risk_groups": ["small_mammals", "ground_birds"],
        "notes": "Predation on small marsupials and ground-nesting birds.",
    },

    # stable (continued)
    {
        "name": "Mallard Duck — New Zealand",
        "invader": "Anas platyrhynchos",
        "ecosystem": "New Zealand wetlands",
        "outcome": "stable", "severity": 0.22,
        "at_risk_groups": [],
        "notes": "Hybridises with grey duck but wetland function preserved.",
    },
]

# ── Random seed ───────────────────────────────────────────────────────────────
SEED = 42


# ── Acquired events (from data/acquire.py) ────────────────────────────────────
# Run:  python data/acquire.py  to populate data/acquired_events.json
# This adds GIDIAS-sourced events on top of the hand-curated INVASION_EVENTS.
try:
    from data.acquire import load_acquired_events as _load_acq
    INVASION_EVENTS = INVASION_EVENTS + _load_acq()
    _n = len(INVASION_EVENTS)
except Exception:
    pass   # acquire.py not run yet — that's fine, use base events only
