"""
data/fetch.py

Network sources (in priority order):
  1. Web of Life API  — real food webs with ecosystem name mapping
  2. GlobalWeb API    — 290 published food webs (Brose et al. 2019)
  3. GloBI API        — species interaction records by focal ecosystem
  4. Local files      — any CSV/TSV/JSON files in the networks/ folder
  5. Synthetic LV     — 200 random stable ecosystems
  6. Diverse archetypes — 10 structural types × n_per_archetype each

Mangal removed: it contains mutualistic/parasitic networks that produce
incorrect signs for Lotka-Volterra dynamics.
"""

import os, csv, json, time, random, logging, requests
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ── Web of Life ────────────────────────────────────────────────────────────────

WOL_BASE = "https://www.web-of-life.es"
KNOWN_WOL_NETWORKS = [
    "FW_001","FW_002","FW_003","FW_004","FW_005",
    "FW_006","FW_007","FW_010","FW_014","FW_017",
]
WOL_ECOSYSTEM_NAMES = {
    "FW_001": "Neotropical river",
    "FW_002": "Caribbean reef",
    "FW_003": "Australian grassland",
    "FW_004": "North American forest",
    "FW_005": "African lakes",
    "FW_006": "Great Lakes",
    "FW_007": "Australian wetlands",
    "FW_010": "Grassland",
}

def fetch_web_of_life(network_ids=None, timeout=15):
    if network_ids is None:
        network_ids = KNOWN_WOL_NETWORKS
    ecosystems = []
    for nid in network_ids:
        try:
            resp = requests.get(
                f"{WOL_BASE}/get_networks.php?network_name={nid}", timeout=timeout)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not data:
                continue
            records = data if isinstance(data, list) else [data]
            species_set, interactions = set(), []
            for rec in records:
                s1 = rec.get("species1", rec.get("node1", ""))
                s2 = rec.get("species2", rec.get("node2", ""))
                w  = float(rec.get("connection_strength", rec.get("weight", 1.0)))
                if s1 and s2:
                    species_set.update([s1, s2])
                    interactions.append({"source": s1, "target": s2,
                                         "type": "predation", "weight": abs(w)})
            if len(species_set) < 3:
                continue
            eco_name = WOL_ECOSYSTEM_NAMES.get(nid, nid)
            ecosystems.append({"name": eco_name, "species": list(species_set),
                "attributes": _default_attributes(species_set),
                "interactions": interactions, "source": "web_of_life", "wol_id": nid})
            logger.info(f"WoL {nid} ({eco_name}): {len(species_set)} sp, "
                        f"{len(interactions)} edges")
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"WoL {nid}: {e}")
    return ecosystems


# ── GlobalWeb ──────────────────────────────────────────────────────────────────

GW_BASE = "https://globalwebdb.com/api"
GLOBALWEB_NETWORK_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
]

def fetch_globalweb(network_ids=None, timeout=20):
    if network_ids is None:
        network_ids = GLOBALWEB_NETWORK_IDS
    ecosystems = []
    for nid in network_ids:
        try:
            mr = requests.get(f"{GW_BASE}/networks/{nid}/", timeout=timeout)
            if mr.status_code != 200:
                continue
            meta  = mr.json()
            lr    = requests.get(f"{GW_BASE}/networks/{nid}/links/", timeout=timeout)
            if lr.status_code != 200:
                continue
            links = lr.json()
            if isinstance(links, dict):
                links = links.get("results", links.get("data", []))
            if not links:
                continue
            species_set, interactions = set(), []
            for lk in links:
                src = (lk.get("resource_name") or str(lk.get("resource", ""))).strip()
                tgt = (lk.get("consumer_name") or str(lk.get("consumer", ""))).strip()
                w   = float(lk.get("interaction_strength") or lk.get("weight") or 1.0)
                if src and tgt:
                    species_set.update([src, tgt])
                    interactions.append({"source": src, "target": tgt,
                                         "type": "predation", "weight": abs(w)})
            if len(species_set) < 4 or not interactions:
                continue
            eco_name = meta.get("name") or meta.get("ecosystem") or f"globalweb_{nid}"
            ecosystems.append({"name": eco_name, "species": list(species_set),
                "attributes": _default_attributes(species_set),
                "interactions": interactions, "source": "globalweb", "gw_id": nid})
            logger.info(f"GlobalWeb {nid} ({eco_name}): {len(species_set)} sp")
            time.sleep(0.25)
        except Exception as e:
            logger.debug(f"GlobalWeb {nid}: {e}")
    logger.info(f"GlobalWeb: {len(ecosystems)} networks")
    return ecosystems


# ── GloBI ──────────────────────────────────────────────────────────────────────

GLOBI_BASE = "https://api.globalbioticinteractions.org/interaction"
FOCAL_ECOSYSTEMS = {
    "Florida Everglades":   ["Python bivittatus", "Odocoileus virginianus",
                             "Alligator mississippiensis", "Ardea herodias",
                             "Procyon lotor", "Sylvilagus palustris"],
    "Australian wetlands":  ["Rhinella marina", "Varanus varius",
                             "Pseudechis australis", "Dacelo novaeguineae",
                             "Macropus giganteus"],
    "Caribbean reef":       ["Pterois volitans", "Epinephelus striatus",
                             "Acanthurus chirurgus", "Diadema antillarum"],
    "Great Lakes":          ["Dreissena polymorpha", "Alewife",
                             "Coregonus clupeaformis", "Mysis relicta"],
    "Global wetlands":      ["Rana temporaria", "Bufo bufo", "Triturus cristatus",
                             "Bombina variegata", "Pelophylax ridibundus"],
    "New Zealand wetlands": ["Anas chlorotis", "Porphyrio melanotus",
                             "Anas rhynchotis", "Gallirallus australis",
                             "Hemiphaga novaeseelandiae"],
    "New Zealand forest":   ["Nestor meridionalis", "Mohoua ochrocephala",
                             "Petroica longipes", "Apteryx mantelli",
                             "Falco novaeseelandiae"],
    "Guam forest":          ["Corvus kubaryi", "Halcyon cinnamomina",
                             "Aplonis opaca", "Boiga irregularis", "Gekko gecko"],
    "Lake Victoria":        ["Lates niloticus", "Oreochromis niloticus",
                             "Rastrineobola argentea", "Haplochromis sp",
                             "Bagrus docmak"],
    "Mississippi river":    ["Hypophthalmichthys molitrix", "Polyodon spathula",
                             "Sander vitreus", "Pylodictis olivaris",
                             "Morone mississippiensis"],
}

def fetch_globi(ecosystems=None, max_per_species=30, timeout=20):
    if ecosystems is None:
        ecosystems = FOCAL_ECOSYSTEMS
    results = []
    for eco_name, species_list in ecosystems.items():
        species_set, interactions = set(species_list), []
        for sp in species_list:
            try:
                params = {"sourceTaxon": sp,
                          "interactionType": "eats,preysOn,parasiteOf,competesFor",
                          "limit": max_per_species,
                          "fields": "source_taxon_name,interaction_type,target_taxon_name"}
                resp = requests.get(GLOBI_BASE, params=params, timeout=timeout)
                if resp.status_code != 200:
                    continue
                for item in resp.json().get("data", []):
                    if len(item) < 3:
                        continue
                    src, itype, tgt = item[0], item[1], item[2]
                    if src and tgt:
                        species_set.update([src, tgt])
                        interactions.append({"source": src, "target": tgt,
                            "type": _normalize_interaction(itype), "weight": 1.0})
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"GloBI {sp}: {e}")
        if interactions:
            results.append({"name": eco_name, "species": list(species_set),
                "attributes": _default_attributes(species_set),
                "interactions": interactions, "source": "globi"})
            logger.info(f"GloBI {eco_name}: {len(species_set)} sp, "
                        f"{len(interactions)} edges")
    return results


# ── Local file loader ──────────────────────────────────────────────────────────

def load_networks_from_files(directory: str) -> List[Dict]:
    """
    Load food web networks from local CSV/TSV/JSON files.
    Place files in the networks/ folder next to main.py.
    CSV columns: source/resource, target/consumer, weight (opt), type (opt)
    JSON format: {"nodes":[...], "links":[{"source":..,"target":..,"weight":..}]}
    """
    if not directory or not os.path.isdir(directory):
        return []
    ecosystems = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        name  = os.path.splitext(fname)[0]
        try:
            if fname.endswith(".json"):
                eco = _load_json_network(fpath, name)
            elif fname.endswith((".csv", ".tsv", ".txt")):
                eco = _load_csv_network(fpath, name)
            else:
                continue
            if eco:
                ecosystems.append(eco)
                logger.info(f"Local {fname}: {len(eco['species'])} sp")
        except Exception as e:
            logger.warning(f"Local {fname}: {e}")
    logger.info(f"Local files: {len(ecosystems)} networks from {directory}")
    return ecosystems

def _load_json_network(fpath, name):
    with open(fpath) as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    links = data.get("links", data.get("edges", data.get("interactions", [])))
    if not links:
        return None
    node_map = {}
    for nd in nodes:
        nid = nd.get("id", nd.get("node_id", ""))
        node_map[str(nid)] = (nd.get("name") or nd.get("label") or
                               nd.get("species") or str(nid))
    species_set, interactions = set(), []
    for lk in links:
        src = node_map.get(str(lk.get("source", lk.get("from", ""))),
                           str(lk.get("source", "")))
        tgt = node_map.get(str(lk.get("target", lk.get("to", ""))),
                           str(lk.get("target", "")))
        w   = float(lk.get("weight", lk.get("value",
              lk.get("interaction_strength", 1.0))) or 1.0)
        t   = _normalize_interaction(str(lk.get("type", "predation")))
        if src and tgt:
            species_set.update([src, tgt])
            interactions.append({"source": src, "target": tgt,
                                  "type": t, "weight": abs(w)})
    if len(species_set) < 4:
        return None
    return {"name": name, "species": list(species_set),
            "attributes": _default_attributes(species_set),
            "interactions": interactions, "source": "local_file"}

def _load_csv_network(fpath, name):
    delim = "\t" if fpath.endswith((".tsv", ".txt")) else ","
    species_set, interactions = set(), []
    with open(fpath, newline="", encoding="utf-8-sig") as f:
        reader  = csv.DictReader(f, delimiter=delim)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]
        src_col = next((h for h in headers if h in
            ["source","resource","predator","from","sp1","species1"]),
            headers[0] if headers else None)
        tgt_col = next((h for h in headers if h in
            ["target","consumer","prey","to","sp2","species2"]),
            headers[1] if len(headers) > 1 else None)
        w_col = next((h for h in headers if h in
            ["weight","value","strength","interaction_strength","frequency"]), None)
        t_col = next((h for h in headers if h in
            ["type","interaction_type","link_type","interaction"]), None)
        if not src_col or not tgt_col:
            return None
        for row in reader:
            src = row.get(src_col, "").strip()
            tgt = row.get(tgt_col, "").strip()
            w   = float(row.get(w_col, 1.0) or 1.0) if w_col else 1.0
            t   = (_normalize_interaction(row.get(t_col, "predation") or "predation")
                   if t_col else "predation")
            if src and tgt and src != tgt:
                species_set.update([src, tgt])
                interactions.append({"source": src, "target": tgt,
                                      "type": t, "weight": abs(w)})
    if len(species_set) < 4 or not interactions:
        return None
    return {"name": name, "species": list(species_set),
            "attributes": _default_attributes(species_set),
            "interactions": interactions, "source": "local_file"}


# ── Combined real-network loader ───────────────────────────────────────────────

def fetch_all_real_networks(local_dir="networks",
                             use_globalweb=True, use_globi=True) -> List[Dict]:
    """Fetch from all sources, deduplicated by name."""
    all_networks, seen = [], set()

    def _add(nets, label):
        added = sum(
            1 for eco in nets
            if eco["name"] not in seen
            and not seen.add(eco["name"])
            and all_networks.append(eco) is None
        )
        logger.info(f"  {label}: +{added} (total: {len(all_networks)})")

    if local_dir:      _add(load_networks_from_files(local_dir), "Local")
    if use_globalweb:  _add(fetch_globalweb(),                   "GlobalWeb")
    if use_globi:      _add(fetch_globi(),                       "GloBI")

    try:
        from data.acquire import load_acquired_graphs as _lag
        _add(_lag(), "Acquired")
    except Exception:
        pass

    logger.info(f"Total real networks: {len(all_networks)}")
    return all_networks


# ── Synthetic LV (random) ──────────────────────────────────────────────────────

def generate_synthetic_ecosystems(n=200, n_species_range=(8, 25), seed=42):
    rng, ecosystems = np.random.default_rng(seed), []
    for i in range(n):
        eco = _generate_one_lv_ecosystem(rng.integers(*n_species_range), i, rng)
        if eco:
            ecosystems.append(eco)
    logger.info(f"Generated {len(ecosystems)} random LV ecosystems")
    return ecosystems

def _generate_one_lv_ecosystem(n_sp, idx, rng):
    A = rng.normal(0, 0.3, (n_sp, n_sp)) * (rng.random((n_sp, n_sp)) < 0.3)
    np.fill_diagonal(A, -rng.uniform(0.5, 2.0, n_sp))
    off = A.copy()
    np.fill_diagonal(off, 0)
    sr = np.max(np.abs(np.linalg.eigvals(off)))
    if sr > 0.8:
        off = off * (0.7 / sr)
        np.fill_diagonal(off, np.diag(A))
        A = off
    r         = rng.uniform(0.5, 2.0, n_sp)
    x         = simulate_lv(A, r)
    surviving = np.where(x > 0.01)[0]
    if len(surviving) < 4:
        return None
    A_s      = A[np.ix_(surviving, surviving)]
    species  = [f"species_{idx}_{j}" for j in surviving]
    n_s      = len(surviving)
    interactions = [
        {"source": species[si], "target": species[sj],
         "type": _lv_to_edge_type(A_s[si, sj], A_s[sj, si]),
         "weight": float(abs(A_s[si, sj]))}
        for si in range(n_s) for sj in range(n_s)
        if si != sj and abs(A_s[si, sj]) > 0.05
    ]
    trophic    = _estimate_trophic_levels(n_s, A_s)
    attributes = {
        sp: {"trophic_level":    float(trophic[k]),
             "log_body_mass":    float(rng.normal(trophic[k] * 2, 1.0)),
             "diet_type":        int(rng.integers(0, 4)),
             "habitat":          int(rng.integers(0, 6)),
             "log_repro_rate":   float(rng.normal(-trophic[k] * 0.5, 0.5)),
             "intrinsic_growth": float(r[surviving[k]])}
        for k, sp in enumerate(species)
    }
    return {"name": f"synthetic_{idx}", "species": species,
            "attributes": attributes, "interactions": interactions,
            "lv_matrix": A_s.tolist(), "source": "synthetic", "stable": True}


# ── LV simulators ─────────────────────────────────────────────────────────────

def simulate_lv(A, r, x0=None, t_end=50.0, n_steps=500):
    x  = np.ones(len(r)) * 0.5 if x0 is None else x0.copy()
    dt = t_end / n_steps
    for _ in range(n_steps):
        x = np.clip(x + dt * x * (r + A @ x), 0.0, 1e4)
    return x

def simulate_lv_stochastic(A, r, x0=None, t_end=50.0, n_steps=500,
                             noise_scale=0.05, rng=None):
    """Stochastic LV: dx_i = x_i*(r_i + ΣA_ij*x_j)*dt + σ*x_i*dW_i"""
    if rng is None:
        rng = np.random.default_rng()
    x      = np.ones(len(r)) * 0.5 if x0 is None else x0.copy()
    dt     = t_end / n_steps
    sq_dt  = np.sqrt(dt)
    for _ in range(n_steps):
        x = np.clip(
            x + dt * x * (r + A @ x) + sq_dt * noise_scale * x * rng.standard_normal(len(r)),
            0.0, 1e4)
    return x

def _add_observation_noise(A, noise_fraction=0.15, rng=None):
    """Lognormal multiplicative noise on interaction strengths (Berlow 2004)."""
    if rng is None:
        rng = np.random.default_rng()
    A_n = A.copy()
    nz  = A != 0
    sig = np.sqrt(np.log(1 + noise_fraction ** 2))
    A_n[nz] = A[nz] * rng.lognormal(-sig ** 2 / 2, sig, A.shape)[nz]
    return np.sign(A) * np.abs(A_n)


# ── Diverse structured synthetic ──────────────────────────────────────────────

def generate_diverse_synthetic_ecosystems(n_per_archetype=80, seed=42):
    """
    10 structural archetypes × n_per_archetype ecosystems each.
    Stochastic integration + observation noise for realism.
    """
    rng = np.random.default_rng(seed)
    archetypes = [
        # name,             n_sp, connectance, n_tl, linear_chain
        ("simple_chain",      8,  0.15, 3, True),
        ("complex_web",      22,  0.40, 5, False),
        ("island_sparse",     6,  0.10, 3, True),
        ("marine_dense",     24,  0.45, 4, False),
        ("plant_dominated",  14,  0.20, 2, False),
        ("large_forest",     40,  0.18, 5, False),
        ("stream_web",       15,  0.25, 4, True),
        ("tropical_dense",   35,  0.35, 4, False),
        ("sparse_island",     8,  0.08, 3, True),
        ("wetland_web",      28,  0.22, 3, False),
    ]
    all_eco = []
    for arch, n_sp, conn, n_tl, linear in archetypes:
        for i in range(n_per_archetype):
            eco = _generate_structured_ecosystem(
                n_sp, conn, n_tl, linear, f"{arch}_{i}", arch, rng)
            if eco:
                all_eco.append(eco)
    logger.info(f"Generated {len(all_eco)} diverse structured ecosystems")
    return all_eco

def _generate_structured_ecosystem(n_sp, connectance, n_tl, linear_chain,
                                    name, archetype, rng):
    tl = np.linspace(1.0, n_tl, n_sp)
    A  = np.zeros((n_sp, n_sp))
    if linear_chain:
        for i in range(1, n_sp):
            s = rng.uniform(0.2, 0.7)
            A[i, i-1] =  s
            A[i-1, i] = -rng.uniform(0.1, s * 0.6)
    else:
        for i in range(n_sp):
            for j in range(n_sp):
                if i == j:
                    continue
                d = tl[i] - tl[j]
                if 0.3 < d < 2.5 and rng.random() < connectance:
                    s = rng.uniform(0.1, 0.6)
                    A[i, j] =  s
                    A[j, i] = -rng.uniform(0.05, s * 0.5)
                elif abs(d) < 0.4 and rng.random() < connectance * 0.3:
                    s = rng.uniform(0.05, 0.2)
                    A[i, j] = -s
                    A[j, i] = -s
    margin = rng.uniform(0.3, 0.85)
    np.fill_diagonal(A, -rng.uniform(0.5, 2.0, n_sp))
    off = A.copy()
    np.fill_diagonal(off, 0)
    sr = np.max(np.abs(np.linalg.eigvals(off)))
    if sr > margin:
        off = off * (margin / sr)
        np.fill_diagonal(off, np.diag(A))
        A = off
    A = _add_observation_noise(A, 0.12, rng)
    r = rng.uniform(0.3, 2.0, n_sp)
    x = simulate_lv_stochastic(A, r, t_end=80.0, n_steps=800,
                                 noise_scale=0.04, rng=rng)
    surviving = np.where(x > 0.01)[0]
    if len(surviving) < 4:
        return None
    A_s     = A[np.ix_(surviving, surviving)]
    species = [f"{name}_sp{j}" for j in surviving]
    n_s     = len(surviving)
    interactions = [
        {"source": species[si], "target": species[sj],
         "type": _lv_to_edge_type(A_s[si, sj], A_s[sj, si]),
         "weight": float(abs(A_s[si, sj]))}
        for si in range(n_s) for sj in range(n_s)
        if si != sj and abs(A_s[si, sj]) > 0.04
    ]
    trophic    = _estimate_trophic_levels(n_s, A_s)
    attributes = {
        sp: {"trophic_level":    float(trophic[k]),
             "log_body_mass":    float(trophic[k] * 1.8 + rng.normal(0, 0.5)),
             "diet_type":        int(rng.integers(0, 4)),
             "habitat":          int(rng.integers(0, 6)),
             "log_repro_rate":   float(-trophic[k] * 0.4 + rng.normal(0, 0.3)),
             "intrinsic_growth": float(r[surviving[k]])}
        for k, sp in enumerate(species)
    }
    return {"name": name, "species": species, "attributes": attributes,
            "interactions": interactions, "lv_matrix": A_s.tolist(),
            "source": "synthetic_structured", "archetype": archetype, "stable": True}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _default_attributes(species_set):
    rng = random.Random(42)
    return {sp: {"trophic_level":    rng.uniform(1.0, 4.5),
                 "log_body_mass":    rng.gauss(2.0, 2.0),
                 "diet_type":        rng.randint(0, 3),
                 "habitat":          rng.randint(0, 5),
                 "log_repro_rate":   rng.gauss(0.0, 1.0),
                 "intrinsic_growth": rng.uniform(0.1, 2.0)}
            for sp in species_set}

def _normalize_interaction(itype):
    itype = itype.lower()
    if any(k in itype for k in ["eats", "preys", "kills", "consumes"]):
        return "predation"
    if "parasite" in itype or "parasit" in itype:
        return "parasitism"
    if "competes" in itype or "competition" in itype:
        return "competition"
    if any(k in itype for k in ["mutual", "symbiosis", "pollinate"]):
        return "mutualism"
    return "predation"

def _lv_to_edge_type(a_ij, a_ji):
    if a_ij > 0 and a_ji < 0: return "predation"
    if a_ij < 0 and a_ji > 0: return "predation"
    if a_ij < 0 and a_ji < 0: return "competition"
    if a_ij > 0 and a_ji > 0: return "mutualism"
    return "competition"

def _estimate_trophic_levels(n, A):
    tl = np.ones(n)
    for _ in range(20):
        new_tl = np.ones(n)
        for i in range(n):
            prey = np.where(A[i, :] > 0)[0]
            if len(prey) > 0:
                new_tl[i] = 1.0 + np.mean(tl[prey])
        tl = new_tl
    return tl
