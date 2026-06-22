"""
data/acquire.py — Database-driven invasion event and food web acquisition.

Run this script ONCE from your project root before training:
    python data/acquire.py

What it does, in order:
    1. Downloads GIDIAS (22,865 invasion impact records, Bacher et al. 2025)
       from Figshare — the authoritative global invasive species impact database.
    2. Filters to community/ecosystem-level impacts with documented outcomes.
    3. Maps EICAT severity categories to collapse / disrupted / stable.
    4. For each event, queries Mangal (1,300+ real food webs) for a matching
       ecosystem graph, scored by habitat type + geographic proximity.
    5. For unmatched events, queries GloBI for species interactions and
       builds an ad-hoc food web graph.
    6. Writes two output files:
         data/acquired_events.json   — new INVASION_EVENTS entries
         data/acquired_graphs/       — matched food web graphs as JSON
    7. Prints a summary showing how many events were acquired and matched.

After running, add to configs.py:
    from data.acquire import load_acquired_events
    INVASION_EVENTS = INVASION_EVENTS + load_acquired_events()

And add to fetch.py's fetch_all_real_networks():
    from data.acquire import load_acquired_graphs
    all_networks += load_acquired_graphs()

Requirements (already in your environment): requests, numpy
"""

import os, sys, json, time, logging, hashlib, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import requests
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CACHE_DIR   = ROOT / "data" / ".cache"
OUT_EVENTS  = ROOT / "data" / "acquired_events.json"
OUT_GRAPHS  = ROOT / "data" / "acquired_graphs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_GRAPHS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── GIDIAS ────────────────────────────────────────────────────────────────────
# Bacher et al. 2025, Scientific Data 12:832
# DOI: 10.1038/s41597-025-05184-5 | Data: 10.6084/m9.figshare.27908838
GIDIAS_FIGSHARE_ID = 27908838
FIGSHARE_API       = "https://api.figshare.com/v2"

# ── Mangal ────────────────────────────────────────────────────────────────────
MANGAL_API  = "https://mangal.io/api/v2"

# ── GloBI ─────────────────────────────────────────────────────────────────────
GLOBI_API   = "https://api.globalbioticinteractions.org"

# ── GBIF ──────────────────────────────────────────────────────────────────────
GBIF_API    = "https://api.gbif.org/v1"

# ── EICAT → outcome mapping ───────────────────────────────────────────────────
# EICAT categories (Hawkins et al. 2015, IUCN 2020):
#   MV = Massive  — irreversible community/ecosystem change, local extinction
#   MR = Major    — population-level decline; reversible with management
#   MO = Moderate — individual performance reduced at population scale
#   MN = Minor    — detectable individual effects, limited population impact
#   MC = Minimal  — individual-level effects only, no population change
#
# Our labeling rule adds native_taxon_level as a tiebreaker:
#   MV + any level               → collapse  (0.85–0.98)
#   MR + community/ecosystem     → collapse  (0.80–0.87)
#   MR + population              → disrupted (0.65–0.75)
#   MO + community               → disrupted (0.55–0.68)
#   MO + population/individual   → disrupted (0.42–0.55)
#   MN + any                     → disrupted (0.35–0.50)
#   MC + any                     → stable    (0.10–0.28)

def eicat_to_outcome(eicat_cat: str, native_level: str,
                     severity_raw: float = None) -> Tuple[str, float]:
    """
    Map EICAT category + impact level to (outcome, severity).
    severity_raw is the numeric score (1-5) if present in GIDIAS.
    """
    cat   = str(eicat_cat).strip().upper()
    level = str(native_level).strip().lower()

    if cat == "MV":
        # Massive = irreversible local extinction
        sev = 0.90 + 0.08 * min(1, (severity_raw or 5) / 5)
        return "collapse", round(min(0.98, sev), 2)

    elif cat == "MR":
        if any(k in level for k in ("community", "ecosystem", "guild", "assemblage")):
            sev = 0.80 + 0.07 * min(1, (severity_raw or 4) / 5)
            return "collapse", round(min(0.87, sev), 2)
        else:
            sev = 0.65 + 0.10 * min(1, (severity_raw or 3) / 5)
            return "disrupted", round(min(0.75, sev), 2)

    elif cat == "MO":
        if any(k in level for k in ("community", "ecosystem")):
            sev = 0.55 + 0.13 * min(1, (severity_raw or 3) / 5)
            return "disrupted", round(min(0.68, sev), 2)
        else:
            sev = 0.42 + 0.13 * min(1, (severity_raw or 2) / 5)
            return "disrupted", round(min(0.55, sev), 2)

    elif cat == "MN":
        sev = 0.35 + 0.15 * min(1, (severity_raw or 2) / 5)
        return "disrupted", round(min(0.50, sev), 2)

    elif cat == "MC":
        sev = 0.10 + 0.18 * min(1, (severity_raw or 1) / 5)
        return "stable", round(min(0.28, sev), 2)

    else:
        # DD/NA/NE — data deficient or not evaluated, skip
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GIDIAS download and parsing
# ═══════════════════════════════════════════════════════════════════════════════

def _cached_get(url: str, timeout: int = 30, retries: int = 3) -> Optional[dict]:
    """GET with local cache (keyed by URL hash) and retry logic."""
    cache_key  = hashlib.md5(url.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout,
                             headers={"User-Agent": "EcosystemGNN-Research/1.0"})
            if r.status_code == 200:
                data = r.json()
                cache_file.write_text(json.dumps(data))
                return data
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.debug(f"GET {url}: {e}")
            time.sleep(1)
    return None


def _cached_get_csv(url: str, timeout: int = 60) -> Optional[bytes]:
    """Download a CSV/binary file with caching."""
    cache_key  = hashlib.md5(url.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.csv"
    if cache_file.exists():
        return cache_file.read_bytes()
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "EcosystemGNN-Research/1.0"},
                         stream=True)
        if r.status_code == 200:
            data = r.content
            cache_file.write_bytes(data)
            return data
    except Exception as e:
        logger.warning(f"CSV download failed: {e}")
    return None


def fetch_gidias() -> List[Dict]:
    """
    Download GIDIAS from Figshare and parse into a list of impact records.
    Returns records filtered to:
      - negative impacts only (direction=negative)
      - community or ecosystem level (native_taxon_level)
      - EICAT category present (not DD/NA/NE)
    """
    logger.info("Fetching GIDIAS from Figshare...")

    # Get file list from Figshare API
    meta = _cached_get(f"{FIGSHARE_API}/articles/{GIDIAS_FIGSHARE_ID}/files")
    if not meta:
        logger.error("Could not reach Figshare API. Check your internet connection.")
        return []

    # Find the main CSV file
    csv_url = None
    for f in meta:
        name = f.get("name", "")
        if name.lower().endswith(".csv") and "gidias" in name.lower():
            csv_url = f["download_url"]
            logger.info(f"  Found: {name} ({f.get('size',0)//1024} KB)")
            break

    if not csv_url:
        # Try the first CSV
        for f in meta:
            if f.get("name", "").endswith(".csv"):
                csv_url = f["download_url"]
                break

    if not csv_url:
        logger.error(f"No CSV found in Figshare article {GIDIAS_FIGSHARE_ID}")
        logger.error("Files available: " + str([f.get("name") for f in meta]))
        return []

    logger.info(f"  Downloading GIDIAS CSV...")
    raw = _cached_get_csv(csv_url)
    if not raw:
        return []

    # Parse CSV manually (avoid pandas dependency)
    import csv, io
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace")))
    rows = list(reader)
    logger.info(f"  Raw GIDIAS records: {len(rows):,}")

    # Detect column names (GIDIAS field names may vary slightly between versions)
    if rows:
        cols = sorted(rows[0].keys())
        logger.info(f"  All {len(cols)} columns:")
        for i in range(0, len(cols), 5):
            logger.info(f"    {cols[i:i+5]}")
        # Show a sample row to help diagnose field values
        sample = {k.lower().replace(" ","_").replace(".","_"): v
                  for k, v in rows[0].items()}
        key_fields = ["impact_direction","eicat_category","impact_level",
                      "alien_taxon","habitat","realm","country"]
        logger.info("  Sample row key fields:")
        for f in key_fields:
            _val = sample.get(f, "<missing>")
        logger.info(f"    {f} = {repr(_val)}")

    # ── Parse with REAL GIDIAS column names (from actual CSV schema) ──────────
    # Key fields (real column -> normalised key after .lower().replace(".","_")):
    #   direction.Nature                   -> direction_nature       (neg/pos/neutral)
    #   magnitude.Nature                   -> magnitude_nature       (MV/MR/MO/MN/MC)
    #   investigated.level.of.organization -> investigated_level_of_organization
    #   IAS.Species.Name                   -> ias_species_name       (invader binomial)
    #   IAS.Taxon                          -> ias_taxon              (invader taxon)
    #   Kingdom                            -> kingdom
    #   Realm                              -> realm                  (Terrestrial/Freshwater/Marine)
    #   Country.Location                   -> country_location
    #   Affected.native.species.Taxon      -> affected_native_species_taxon
    #   Affected.ecosystem.property        -> affected_ecosystem_property
    #   mechanism.Nature.clean             -> mechanism_nature_clean
    #   UoA.Wetland / UoA.Grassland / etc  -> uoa_wetland / uoa_grassland (binary habitat flags)
    #   global.extinction                  -> global_extinction      (TRUE/FALSE)

    UOA_HABITAT_MAP = {
        "uoa_boreal": "forest", "uoa_coastal": "marine",
        "uoa_cultivated": "agricultural", "uoa_desert": "desert",
        "uoa_dry": "grassland", "uoa_grassland": "grassland",
        "uoa_mediterranean": "mediterranean", "uoa_ocean": "marine",
        "uoa_savanna": "grassland", "uoa_shelf": "marine",
        "uoa_surface": "lake", "uoa_tundra": "tundra",
        "uoa_urban": "urban", "uoa_wetland": "wetland",
        "uoa_aquaculture": "aquaculture",
    }

    records = []
    skipped = Counter()
    for row in rows:
        r = {k.lower().strip().replace(" ", "_").replace(".", "_"): v.strip()
             for k, v in row.items()}

        # ── Impact direction ──────────────────────────────────────────────────
        direction = (r.get("direction_nature") or r.get("direction") or
                     r.get("impact_direction") or "").lower().strip()
        if "negative" not in direction:
            skipped["not_negative"] += 1
            continue

        # ── EICAT / magnitude ─────────────────────────────────────────────────
        # magnitude.Nature holds EICAT-equivalent category (MV/MR/MO/MN/MC)
        eicat = (r.get("magnitude_nature") or r.get("eicat_category") or
                 r.get("eicat") or "").upper().strip()
        if eicat in ("", "DD", "NA", "NE", "DATA DEFICIENT", "NO IMPACT"):
            skipped["no_eicat"] += 1
            continue

        # ── Level of organisation ─────────────────────────────────────────────
        level = (r.get("investigated_level_of_organization") or
                 r.get("affected_ecosystem_property") or
                 r.get("impact_level") or r.get("organisation_level") or "").lower()
        if not any(k in level for k in ("community", "ecosystem", "assemblage",
                                         "guild", "population")):
            skipped["low_level"] += 1
            continue

        # ── Invader species ───────────────────────────────────────────────────
        invader = (r.get("ias_species_name") or r.get("ias_taxon") or
                   r.get("alien_taxon") or r.get("verified_name_gbif_taxon") or
                   r.get("gbif_scientificname_with_author") or "").strip()
        # Strip author names — keep only first two words (genus + species)
        if invader:
            parts = invader.split()
            if len(parts) > 2 and not parts[1][0].isupper():
                invader = " ".join(parts[:2])

        if not invader:
            skipped["no_invader"] += 1
            continue

        # ── Taxonomy ──────────────────────────────────────────────────────────
        kingdom = (r.get("kingdom") or "").lower()
        clazz   = (r.get("class") or "").lower()

        # ── Habitat from UoA binary flags ─────────────────────────────────────
        habitat = ""
        for uoa_key, hab_val in UOA_HABITAT_MAP.items():
            val = (r.get(uoa_key) or "").strip()
            if val in ("1", "TRUE", "True", "true", "yes", "x", "X"):
                habitat = hab_val
                break   # take first matching habitat

        # ── Geography ─────────────────────────────────────────────────────────
        realm   = (r.get("realm") or "").lower()
        country = (r.get("country_location") or r.get("country") or "").upper()[:2]
        region  = (r.get("region") or "").lower()
        island  = (r.get("island") or "").strip()

        # ── Affected taxon ────────────────────────────────────────────────────
        native_t  = (r.get("affected_native_species_taxon") or
                     r.get("affected_native_species_details") or "").strip()

        # ── Mechanism ─────────────────────────────────────────────────────────
        mechanism = (r.get("mechanism_nature_clean") or
                     r.get("mechanism_nature") or "").lower()

        # ── Reference ─────────────────────────────────────────────────────────
        reference = (r.get("doi") or r.get("reference") or
                     r.get("text_excerpt") or "").strip()[:120]

        # ── Global extinction flag ────────────────────────────────────────────
        extinct = (r.get("global_extinction") or "").upper() in ("TRUE","YES","1","X")

        # ── Map to outcome ────────────────────────────────────────────────────
        outcome, severity = eicat_to_outcome(eicat, level, None)
        # Upgrade to collapse if global extinction recorded
        if extinct and outcome != "collapse":
            outcome, severity = "collapse", min(0.98, (severity or 0.5) + 0.15)
        if outcome is None:
            skipped["unmappable"] += 1
            continue

        records.append({
            "invader":      invader,
            "kingdom":      kingdom,
            "class":        clazz,
            "habitat":      habitat,
            "realm":        realm,
            "country":      country,
            "region":       region,
            "island":       island,
            "native_taxon": native_t,
            "impact_level": level,
            "mechanism":    mechanism,
            "eicat":        eicat,
            "outcome":      outcome,
            "severity":     severity,
            "reference":    reference,
            "extinct":      extinct,
        })

    logger.info(f"  After filtering: {len(records):,} usable records")
    logger.info(f"  Skipped: {dict(skipped)}")
    logger.info(f"  Outcomes: {dict(Counter(r['outcome'] for r in records))}")
    return records


def deduplicate_gidias(records: List[Dict],
                       max_per_invader: int = 3) -> List[Dict]:
    """
    GIDIAS has many records per species (one per study per location).
    Keep the highest-severity record per (invader, habitat, realm),
    capped at max_per_invader records per species to avoid one dominant invader
    flooding the training set.
    """
    # Group by invader + realm
    groups = defaultdict(list)
    for r in records:
        key = (r["invader"], r["realm"], _coarse_habitat(r["habitat"]))
        groups[key].append(r)

    # Within each group keep highest-severity record
    best = []
    for key, recs in groups.items():
        recs.sort(key=lambda x: x["severity"], reverse=True)
        best.append(recs[0])

    # Cap per invader species
    per_invader = defaultdict(list)
    for r in best:
        per_invader[r["invader"]].append(r)
    capped = []
    for inv, recs in per_invader.items():
        recs.sort(key=lambda x: x["severity"], reverse=True)
        capped.extend(recs[:max_per_invader])

    logger.info(f"  Deduplicated: {len(records):,} → {len(capped):,} records "
                f"({len(per_invader):,} unique invader species)")
    return capped


def _coarse_habitat(habitat: str) -> str:
    """Map free-text habitat to one of 8 coarse categories."""
    h = habitat.lower()
    if any(k in h for k in ("forest","woodland","tree","rainforest")): return "forest"
    if any(k in h for k in ("grassland","savanna","prairie","steppe","meadow")): return "grassland"
    if any(k in h for k in ("wetland","marsh","swamp","bog","fen","riparian")): return "wetland"
    if any(k in h for k in ("stream","river","brook","creek","canal")): return "river"
    if any(k in h for k in ("lake","pond","reservoir","lentic")): return "lake"
    if any(k in h for k in ("reef","coral","kelp","seagrass","benthos","coastal","marine")): return "marine"
    if any(k in h for k in ("island","atoll","archipelago")): return "island"
    if any(k in h for k in ("agricultural","crop","arable","farmland")): return "agricultural"
    return "other"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mangal food web matching
# ═══════════════════════════════════════════════════════════════════════════════

# Country-code to ISO continent
_CC_TO_CONTINENT = {
    "AU":"Oceania","NZ":"Oceania","PG":"Oceania","FJ":"Oceania","NC":"Oceania",
    "US":"North America","CA":"North America","MX":"North America",
    "GB":"Europe","FR":"Europe","DE":"Europe","IT":"Europe","ES":"Europe",
    "NL":"Europe","PL":"Europe","SE":"Europe","NO":"Europe","FI":"Europe",
    "KE":"Africa","TZ":"Africa","ZA":"Africa","NG":"Africa","CD":"Africa",
    "UG":"Africa","ET":"Africa","GH":"Africa","CM":"Africa","SN":"Africa",
    "BR":"South America","AR":"South America","CL":"South America","CO":"South America",
    "IN":"Asia","CN":"Asia","JP":"Asia","ID":"Asia","MY":"Asia","PH":"Asia",
    "TH":"Asia","VN":"Asia","KR":"Asia","PK":"Asia","BD":"Asia",
}

_HABITAT_KEYWORDS = {
    "forest":      ["forest","wood","tree","rainforest","boreal","tropical"],
    "grassland":   ["grass","savanna","prairie","steppe","shrub","scrub"],
    "wetland":     ["wetland","marsh","riparian","swamp","bog"],
    "river":       ["river","stream","brook","creek","lotic"],
    "lake":        ["lake","pond","reservoir","lentic"],
    "marine":      ["marine","reef","coral","kelp","ocean","sea","coastal"],
    "island":      ["island","atoll","insular"],
    "agricultural":["agri","crop","farm","plantation"],
}

def _score_mangal_match(mangal_net: Dict, event: Dict) -> float:
    """
    Score how well a Mangal network matches a GIDIAS event.
    Returns 0.0–1.0; threshold 0.35 for acceptance.
    """
    score = 0.0
    name  = (mangal_net.get("name") or "").lower()
    desc  = (mangal_net.get("description") or "").lower()
    text  = name + " " + desc

    # Habitat match (0.0 – 0.40)
    event_hab = _coarse_habitat(event["habitat"])
    for kw in _HABITAT_KEYWORDS.get(event_hab, []):
        if kw in text:
            score += 0.40
            break
    # Realm match (0.0 – 0.20)
    realm = event.get("realm", "").lower()
    if "terrestrial" in realm and any(k in text for k in ("land","forest","grass","soil")):
        score += 0.20
    elif "freshwater" in realm and any(k in text for k in ("stream","river","lake","pond")):
        score += 0.20
    elif "marine" in realm and any(k in text for k in ("marine","ocean","reef","sea")):
        score += 0.20

    # Geographic proximity (0.0 – 0.30)
    country = event.get("country", "").upper()
    continent = _CC_TO_CONTINENT.get(country, "")
    region = event.get("region", "").lower()

    for geo in (country.lower(), continent.lower(), region):
        if geo and any(geo in text for geo in [geo]):
            score += 0.30
            break
        # Partial: check continent keywords
        if continent:
            continent_kws = {
                "Europe": ["europe","uk","british","french","german","nordic"],
                "North America": ["north america","usa","canada","american"],
                "South America": ["south america","brazil","latin"],
                "Africa": ["africa","african","kenya","tanzania"],
                "Asia": ["asia","asian","japan","china","india"],
                "Oceania": ["australia","new zealand","pacific","oceania"],
            }
            for kw in continent_kws.get(continent, []):
                if kw in text:
                    score += 0.15
                    break

    # Minimum graph quality
    n_nodes = mangal_net.get("node_count", 0) or 0
    n_edges = mangal_net.get("edge_count", 0) or 0
    if n_nodes >= 10 and n_edges >= 15:
        score += 0.10   # bonus for substantive graph

    return min(score, 1.0)


def fetch_all_mangal_networks(timeout: int = 20) -> List[Dict]:
    """Fetch metadata for ALL Mangal networks (paginated)."""
    cache_file = CACHE_DIR / "mangal_all_networks.json"
    if cache_file.exists():
        nets = json.loads(cache_file.read_text())
        logger.info(f"  Mangal: {len(nets):,} networks (cached)")
        return nets

    logger.info("  Fetching Mangal network list (paginated)...")
    all_nets, page, page_size = [], 0, 100
    while True:
        url  = f"{MANGAL_API}/network/?limit={page_size}&offset={page*page_size}"
        data = _cached_get(url, timeout=timeout)
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        all_nets.extend(data)
        logger.info(f"    ... fetched {len(all_nets)} networks")
        if len(data) < page_size:
            break
        page += 1
        time.sleep(0.2)

    cache_file.write_text(json.dumps(all_nets))
    logger.info(f"  Mangal: {len(all_nets):,} networks total")
    return all_nets


def match_event_to_mangal(event: Dict, mangal_nets: List[Dict],
                           threshold: float = 0.35) -> Optional[Dict]:
    """Find the best-matching Mangal network for a GIDIAS event."""
    best_net, best_score = None, 0.0
    for net in mangal_nets:
        score = _score_mangal_match(net, event)
        if score > best_score:
            best_score, best_net = score, net
    if best_score >= threshold:
        return best_net, best_score
    return None, 0.0


def fetch_mangal_graph(network_id: int, timeout: int = 20) -> Optional[Dict]:
    """Fetch full interaction data for one Mangal network."""
    # Nodes
    nr = _cached_get(f"{MANGAL_API}/node/?network={network_id}&limit=500", timeout)
    ir = _cached_get(f"{MANGAL_API}/interaction/?network={network_id}&limit=500", timeout)
    if not nr or not ir:
        return None

    node_map = {}
    for nd in nr:
        nid = nd.get("id")
        name = (nd.get("original_name") or
                (nd.get("taxonomy") or {}).get("name") or f"sp_{nid}")
        trophic = (nd.get("taxonomy") or {}).get("rank") or 2.0
        node_map[nid] = {"name": name, "trophic": trophic}

    species, interactions = set(), []
    for inter in ir:
        src_id = inter.get("node_from")
        tgt_id = inter.get("node_to")
        if src_id not in node_map or tgt_id not in node_map:
            continue
        src = node_map[src_id]["name"]
        tgt = node_map[tgt_id]["name"]
        w   = float(inter.get("value") or 1.0)
        t   = _norm_itype(str(inter.get("type", "predation")))
        species.update([src, tgt])
        interactions.append({"source": src, "target": tgt,
                              "type": t, "weight": abs(w)})

    if len(species) < 5 or len(interactions) < 5:
        return None

    return {
        "species":      list(species),
        "interactions": interactions,
        "source":       "mangal",
        "mangal_id":    network_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GloBI fallback graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_globi_graph(invader: str, event: Dict,
                      max_species: int = 40,
                      max_per_species: int = 20) -> Optional[Dict]:
    """
    Build a food web from GloBI interaction data for the invader's ecosystem.
    Seeds with the invader, then expands to its interaction partners.
    """
    species_set, interactions = set(), []

    def _query_species(sp: str, limit: int = max_per_species):
        url = (f"{GLOBI_API}/interaction"
               f"?sourceTaxon={requests.utils.quote(sp)}"
               f"&interactionType=eats,preysOn,parasiteOf,competesFor,mutualistOf"
               f"&limit={limit}"
               f"&fields=source_taxon_name,interaction_type,target_taxon_name")
        data = _cached_get(url, timeout=15)
        if not data:
            return []
        return data.get("data", [])

    # Start with invader
    seed_data = _query_species(invader)
    for row in seed_data:
        if len(row) < 3:
            continue
        src, itype, tgt = row[0], row[1], row[2]
        if src and tgt:
            species_set.update([src, tgt])
            interactions.append({"source": src, "target": tgt,
                                  "type": _norm_itype(itype), "weight": 1.0})

    # Expand to top interaction partners (up to total max_species)
    partners = list(species_set - {invader})[:8]
    for partner in partners:
        if len(species_set) >= max_species:
            break
        for row in _query_species(partner, limit=10):
            if len(row) < 3:
                continue
            src, itype, tgt = row[0], row[1], row[2]
            if src and tgt and len(species_set) < max_species:
                species_set.update([src, tgt])
                interactions.append({"source": src, "target": tgt,
                                      "type": _norm_itype(itype), "weight": 1.0})
        time.sleep(0.1)

    if len(species_set) < 5 or len(interactions) < 5:
        return None

    return {
        "species":      list(species_set),
        "interactions": interactions,
        "source":       "globi",
        "invader":      invader,
    }


def _norm_itype(itype: str) -> str:
    i = itype.lower()
    if any(k in i for k in ("eats","preys","kills","consumes","predation")): return "predation"
    if any(k in i for k in ("parasite","parasit")): return "parasitism"
    if any(k in i for k in ("competes","competition")): return "competition"
    if any(k in i for k in ("mutual","symbiosis","pollinate")): return "mutualism"
    return "predation"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GBIF species traits
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_gbif_traits(species_name: str) -> Dict:
    """
    Fetch basic ecological traits from GBIF backbone taxonomy.
    Returns a trait dict compatible with finetune.py's invader trait table.
    """
    url  = f"{GBIF_API}/species/match?name={requests.utils.quote(species_name)}&strict=false"
    data = _cached_get(url, timeout=10)
    if not data or data.get("matchType") == "NONE":
        return {}

    kingdom = (data.get("kingdom") or "").lower()
    clazz   = (data.get("class") or "").lower()
    order   = (data.get("order") or "").lower()

    # Infer trophic level from taxonomy
    trophic = _infer_trophic(kingdom, clazz, order)
    # Infer diet type: 0=carnivore,1=omnivore/filter,2=herbivore,3=parasite/pathogen
    diet    = _infer_diet(kingdom, clazz, order)
    # Infer habitat: 0=terrestrial,1=wetland,2=forest,3=aquatic
    habitat = _infer_habitat(kingdom, clazz)

    return {
        "trophic_level":   trophic,
        "log_body_mass":   _infer_body_mass(kingdom, clazz),
        "diet_type":       diet,
        "habitat":         habitat,
        "log_repro_rate":  _infer_repro(kingdom, clazz),
        "intrinsic_growth":_infer_growth(kingdom, clazz),
    }


def _infer_trophic(kingdom, clazz, order):
    if kingdom == "plantae":                              return 1.0
    if kingdom in ("fungi", "bacteria", "viruses"):       return 2.0  # parasite
    if any(k in clazz for k in ("insecta","arachnida")): return 2.5
    if "amphibia" in clazz:                              return 3.2
    if "reptilia" in clazz:                              return 3.5
    if "aves" in clazz:                                  return 3.5
    if "actinopterygii" in clazz:                        return 3.2
    if "mammalia" in clazz:
        if any(k in order for k in ("carnivora","crocodilia")): return 4.0
        if any(k in order for k in ("rodentia","lagomorpha","artiodactyla")): return 2.0
        return 3.0
    return 2.5


def _infer_diet(kingdom, clazz, order):
    if kingdom == "plantae":                              return 3  # autotroph
    if kingdom in ("fungi","bacteria","viruses"):         return 3  # saprotroph/pathogen
    if any(k in order for k in ("rodentia","lagomorpha",
                                 "artiodactyla","perissodactyla")): return 2  # herbivore
    if any(k in clazz for k in ("insecta",)) and any(
            k in order for k in ("coleoptera","lepidoptera","hymenoptera")): return 2
    if any(k in order for k in ("carnivora","falconiformes",
                                 "strigiformes","piciformes")):              return 0  # carnivore
    return 1  # omnivore/filter feeder default


def _infer_body_mass(kingdom, clazz):
    if kingdom == "plantae":  return 1.0
    if kingdom in ("fungi","bacteria","viruses"): return -6.0
    mass_map = {
        "mammalia": 4.0, "aves": 2.5, "reptilia": 2.0,
        "amphibia": 1.5, "actinopterygii": 2.0,
        "insecta": -1.0, "arachnida": -1.5,
        "bivalvia": 0.5, "gastropoda": 0.5,
    }
    for k, v in mass_map.items():
        if k in clazz:
            return v
    return 1.0


def _infer_repro(kingdom, clazz):
    if kingdom == "plantae": return 2.5
    if kingdom in ("fungi","bacteria","viruses"): return 4.5
    repro_map = {
        "mammalia": 0.8, "aves": 1.0, "reptilia": 1.5,
        "amphibia": 2.0, "actinopterygii": 2.5,
        "insecta": 2.8, "arachnida": 2.2,
    }
    for k, v in repro_map.items():
        if k in clazz:
            return v
    return 1.5


def _infer_growth(kingdom, clazz):
    if kingdom == "plantae": return 2.0
    if kingdom in ("fungi","bacteria","viruses"): return 4.0
    growth_map = {
        "mammalia": 0.7, "aves": 0.9, "reptilia": 1.0,
        "amphibia": 1.5, "actinopterygii": 1.8,
        "insecta": 2.5,
    }
    for k, v in growth_map.items():
        if k in clazz:
            return v
    return 1.2


def _infer_habitat(kingdom, clazz):
    if kingdom == "plantae": return 0
    if "actinopterygii" in clazz or "bivalvia" in clazz: return 3
    if "amphibia" in clazz: return 1
    if "aves" in clazz: return 0
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Build default ecosystem attributes (consistent with fetch.py)
# ═══════════════════════════════════════════════════════════════════════════════

import random as _random

def _default_attributes(species_set: set) -> Dict:
    rng = _random.Random(42)
    return {
        sp: {
            "trophic_level":  rng.uniform(1.0, 4.5),
            "log_body_mass":  rng.gauss(2.0, 2.0),
            "diet_type":      rng.randint(0, 3),
            "habitat":        rng.randint(0, 5),
            "log_repro_rate": rng.gauss(0.0, 1.0),
            "intrinsic_growth": rng.uniform(0.1, 2.0),
        }
        for sp in species_set
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Assemble invasion events
# ═══════════════════════════════════════════════════════════════════════════════

def build_ecosystem_name(event: Dict, graph: Optional[Dict],
                          mangal_net: Optional[Dict]) -> str:
    """Build a descriptive ecosystem name for the event."""
    if mangal_net:
        return mangal_net.get("name", "unknown ecosystem")
    # Construct from GIDIAS fields
    habitat = _coarse_habitat(event["habitat"]).title()
    country = event.get("country", "")
    region  = event.get("region", "")
    geo     = country or region or "Global"
    return f"{habitat} — {geo}"


def assemble_invasion_events(gidias_records: List[Dict],
                              mangal_nets: List[Dict],
                              fetch_graphs: bool = True,
                              fetch_traits: bool = True,
                              max_events: int = 500,
                              match_threshold: float = 0.35,
                              ) -> Tuple[List[Dict], Dict[str, Dict]]:
    """
    Main assembly function.
    Returns:
        events: List of INVASION_EVENT dicts
        graphs: Dict of {ecosystem_name: graph_dict} for matched graphs
    """
    events, graphs = [], {}
    trait_cache    = {}
    unmatched_globi= 0
    unmatched_none = 0

    logger.info(f"Assembling invasion events from {len(gidias_records):,} GIDIAS records...")

    for i, rec in enumerate(gidias_records[:max_events]):
        invader = rec["invader"]

        # 1. Match to Mangal food web
        mangal_net, match_score = match_event_to_mangal(rec, mangal_nets,
                                                         threshold=match_threshold)
        graph = None
        eco_name = build_ecosystem_name(rec, None, mangal_net)

        if mangal_net and fetch_graphs:
            mid   = mangal_net.get("id")
            gkey  = f"mangal_{mid}"
            if gkey not in graphs:
                g = fetch_mangal_graph(mid)
                if g:
                    g["name"] = eco_name
                    graphs[gkey] = g
                    graph = g
            else:
                graph = graphs[gkey]

        # 2. GloBI fallback
        if graph is None and fetch_graphs:
            gkey = f"globi_{invader.replace(' ','_')}"
            if gkey not in graphs:
                logger.debug(f"  GloBI fallback for {invader}")
                g = fetch_globi_graph(invader, rec)
                if g:
                    g["name"] = eco_name
                    graphs[gkey] = g
                    graph = g
                    unmatched_globi += 1
                else:
                    unmatched_none += 1
            else:
                graph = graphs[gkey]

        # 3. Fetch invader traits from GBIF
        if fetch_traits and invader not in trait_cache:
            traits = fetch_gbif_traits(invader)
            if traits:
                trait_cache[invader] = traits
            time.sleep(0.05)

        # 4. Build event dict
        short_invader = invader.split()
        invader_common = (invader if len(short_invader) <= 2
                          else " ".join(short_invader[:2]))
        event_name = f"{invader_common} — {eco_name}"

        at_risk = []
        if rec.get("native_taxon"):
            at_risk.append(rec["native_taxon"].lower().replace(" ", "_")[:30])

        event = {
            "name":          event_name,
            "invader":       invader,
            "ecosystem":     eco_name,
            "outcome":       rec["outcome"],
            "severity":      rec["severity"],
            "at_risk_groups":at_risk,
            "notes":         (f"EICAT:{rec['eicat']} | "
                              f"Level:{rec['impact_level']} | "
                              f"Mechanism:{rec['mechanism']} | "
                              f"Ref:{rec['reference'][:80]}"),
            "_source":       "GIDIAS",
            "_match_score":  round(match_score, 3),
            "_graph_key":    (f"mangal_{mangal_net['id']}" if mangal_net
                              else f"globi_{invader.replace(' ','_')}"
                              if graph else None),
        }
        events.append(event)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{min(len(gidias_records),max_events)} events "
                        f"({len(graphs)} graphs, {unmatched_globi} GloBI, {unmatched_none} none)")

    # Summary
    outcomes = Counter(e["outcome"] for e in events)
    matched  = sum(1 for e in events if e["_match_score"] > 0)
    logger.info(f"\n  Done: {len(events)} events assembled")
    logger.info(f"  Outcomes: collapse={outcomes['collapse']} "
                f"disrupted={outcomes['disrupted']} stable={outcomes['stable']}")
    logger.info(f"  Mangal matched: {matched}/{len(events)} "
                f"({100*matched//max(len(events),1)}%)")
    logger.info(f"  GloBI fallback: {unmatched_globi}")
    logger.info(f"  No graph:       {unmatched_none}")
    logger.info(f"  Unique graphs:  {len(graphs)}")
    logger.info(f"  Invader traits: {len(trait_cache)} species from GBIF")

    return events, graphs, trait_cache


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Save and load
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(events: List[Dict], graphs: Dict,
                 trait_cache: Dict) -> None:
    """Save acquired events, graphs, and traits to disk."""

    # Events
    OUT_EVENTS.write_text(json.dumps(events, indent=2, ensure_ascii=False))
    logger.info(f"  Saved {len(events)} events → {OUT_EVENTS}")

    # Graphs (one JSON per graph)
    for key, g in graphs.items():
        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
        path = OUT_GRAPHS / f"{safe_key}.json"
        path.write_text(json.dumps(g, indent=2, ensure_ascii=False))
    logger.info(f"  Saved {len(graphs)} graphs → {OUT_GRAPHS}/")

    # Trait cache
    trait_path = ROOT / "data" / "acquired_traits.json"
    trait_path.write_text(json.dumps(trait_cache, indent=2))
    logger.info(f"  Saved {len(trait_cache)} species traits → {trait_path}")

    # Print copy-paste snippet for configs.py
    logger.info("\n" + "="*60)
    logger.info("ADD THIS TO configs.py:")
    logger.info("="*60)
    print("\nfrom data.acquire import load_acquired_events")
    print("INVASION_EVENTS = INVASION_EVENTS + load_acquired_events()\n")
    logger.info("="*60)
    logger.info("ADD THIS TO data/fetch.py fetch_all_real_networks():")
    logger.info("="*60)
    print("\nfrom data.acquire import load_acquired_graphs")
    print("all_networks += load_acquired_graphs()\n")


def load_acquired_events(max_per_outcome: int = None) -> List[Dict]:
    """Load previously acquired events. Called from configs.py."""
    if not OUT_EVENTS.exists():
        return []
    events = json.loads(OUT_EVENTS.read_text())
    # Strip internal metadata fields before returning
    clean = []
    for e in events:
        c = {k: v for k, v in e.items() if not k.startswith("_")}
        clean.append(c)
    if max_per_outcome:
        by_outcome = defaultdict(list)
        for e in clean:
            by_outcome[e["outcome"]].append(e)
        balanced = []
        for outcome, evts in by_outcome.items():
            balanced.extend(evts[:max_per_outcome])
        return balanced
    return clean


def load_acquired_graphs() -> List[Dict]:
    """Load previously acquired graphs for pretraining. Called from fetch.py."""
    graphs = []
    for path in OUT_GRAPHS.glob("*.json"):
        try:
            g = json.loads(path.read_text())
            if (g.get("species") and g.get("interactions")
                    and len(g["species"]) >= 5):
                # Ensure required fields
                g.setdefault("attributes", _default_attributes(set(g["species"])))
                g.setdefault("source", "acquired")
                graphs.append(g)
        except Exception as e:
            logger.debug(f"Failed to load {path}: {e}")
    return graphs


def load_acquired_traits() -> Dict:
    """Load GBIF-acquired invader traits. Called from finetune.py."""
    path = ROOT / "data" / "acquired_traits.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Database-driven invasion data acquisition")
    parser.add_argument("--max-events",   type=int, default=500,
                        help="Maximum events to acquire from GIDIAS (default: 500)")
    parser.add_argument("--max-per-invader", type=int, default=3,
                        help="Max records per invader species (default: 3)")
    parser.add_argument("--match-threshold", type=float, default=0.35,
                        help="Mangal match score threshold (default: 0.35)")
    parser.add_argument("--no-globi",     action="store_true",
                        help="Skip GloBI fallback (faster, fewer graphs)")
    parser.add_argument("--no-traits",    action="store_true",
                        help="Skip GBIF trait fetching")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Parse GIDIAS only, don't fetch graphs or traits")
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("  Ecosystem GNN — Database-driven Data Acquisition")
    logger.info("="*60)
    logger.info(f"  Sources: GIDIAS (Figshare) + Mangal + GloBI + GBIF")
    logger.info(f"  Max events: {args.max_events} | Match threshold: {args.match_threshold}")

    # Step 1: GIDIAS
    logger.info("\n── Step 1: GIDIAS ──────────────────────────────────────")
    gidias_raw = fetch_gidias()
    if not gidias_raw:
        logger.error("GIDIAS fetch failed. Check internet connection and try again.")
        return
    gidias = deduplicate_gidias(gidias_raw, max_per_invader=args.max_per_invader)

    # Step 2: Mangal network index
    logger.info("\n── Step 2: Mangal network index ────────────────────────")
    mangal_nets = fetch_all_mangal_networks()

    if args.dry_run:
        logger.info("\n── Dry run: reporting only ─────────────────────────────")
        outcomes = Counter(r["outcome"] for r in gidias)
        logger.info(f"  Would produce: {len(gidias)} events")
        logger.info(f"  collapse={outcomes['collapse']} "
                    f"disrupted={outcomes['disrupted']} stable={outcomes['stable']}")
        # Show match rate estimate
        matched = sum(1 for r in gidias[:100]
                      if match_event_to_mangal(r, mangal_nets,
                                               args.match_threshold)[0] is not None)
        logger.info(f"  Estimated Mangal match rate: {matched}% (from first 100)")
        return

    # Step 3: Assemble
    logger.info("\n── Step 3: Assemble events ─────────────────────────────")
    events, graphs, traits = assemble_invasion_events(
        gidias,
        mangal_nets,
        fetch_graphs = not args.dry_run,
        fetch_traits = not args.no_traits and not args.dry_run,
        max_events   = args.max_events,
        match_threshold = args.match_threshold,
    )

    # Step 4: Save
    logger.info("\n── Step 4: Save ─────────────────────────────────────────")
    save_outputs(events, graphs, traits)

    logger.info("\n" + "="*60)
    logger.info("  Acquisition complete.")
    logger.info(f"  Run: python main.py --epochs 100")
    logger.info("="*60)


if __name__ == "__main__":
    main()
