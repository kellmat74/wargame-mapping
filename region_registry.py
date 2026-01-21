#!/usr/bin/env python3
"""
Region Registry - Auto-discovery of available Geofabrik PBF files.

Scans the cache directory for downloaded PBF files and extracts metadata
(bounding boxes, display names) automatically. No hardcoded region lists needed.

Usage:
    from region_registry import get_available_regions, detect_region_for_coords

    # Get all available regions
    regions = get_available_regions()
    # Returns: {'ukraine': {'bounds': (22.1, 44.2, 40.2, 52.4), 'display_name': 'Ukraine', ...}, ...}

    # Detect region for coordinates
    region = detect_region_for_coords(lat=50.0, lon=36.5)
    # Returns: 'ukraine' or None
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

# Directories where PBF files may be stored
# Primary location used by download_mgrs_data_osmium.py
GEOFABRIK_DIR = Path(__file__).parent / "data" / "geofabrik"
# Secondary location for manual downloads
CACHE_DIR = Path(__file__).parent / "cache"
# Registry file stored in the geofabrik directory
REGISTRY_FILE = GEOFABRIK_DIR / "regions.json"

# Geofabrik base URLs by continent (flat lookup for backward compatibility)
# This is auto-generated from GEOFABRIK_REGIONS - keep them in sync
GEOFABRIK_CONTINENTS = {
    # Europe
    "albania": "europe", "andorra": "europe", "austria": "europe", "azores": "europe",
    "belarus": "europe", "belgium": "europe", "bosnia-herzegovina": "europe",
    "bulgaria": "europe", "croatia": "europe", "cyprus": "europe", "czech-republic": "europe",
    "denmark": "europe", "estonia": "europe", "faroe-islands": "europe", "finland": "europe",
    "france": "europe", "georgia": "europe", "germany": "europe", "great-britain": "europe",
    "greece": "europe", "guernsey-jersey": "europe", "hungary": "europe", "iceland": "europe",
    "ireland-and-northern-ireland": "europe", "isle-of-man": "europe", "italy": "europe",
    "kosovo": "europe", "latvia": "europe", "liechtenstein": "europe", "lithuania": "europe",
    "luxembourg": "europe", "macedonia": "europe", "malta": "europe", "moldova": "europe",
    "monaco": "europe", "montenegro": "europe", "netherlands": "europe", "norway": "europe",
    "poland": "europe", "portugal": "europe", "romania": "europe", "russia": "europe",
    "serbia": "europe", "slovakia": "europe", "slovenia": "europe", "spain": "europe",
    "sweden": "europe", "switzerland": "europe", "turkey": "europe", "ukraine": "europe",
    # North America
    "us": "north-america", "canada": "north-america", "mexico": "north-america",
    "greenland": "north-america",
    # South America
    "argentina": "south-america", "bolivia": "south-america", "brazil": "south-america",
    "chile": "south-america", "colombia": "south-america", "ecuador": "south-america",
    "guyana": "south-america", "paraguay": "south-america", "peru": "south-america",
    "suriname": "south-america", "uruguay": "south-america", "venezuela": "south-america",
    # Asia
    "afghanistan": "asia", "armenia": "asia", "azerbaijan": "asia", "bangladesh": "asia",
    "bhutan": "asia", "cambodia": "asia", "china": "asia", "east-timor": "asia",
    "gcc-states": "asia", "india": "asia", "indonesia": "asia", "iran": "asia",
    "iraq": "asia", "israel-and-palestine": "asia", "japan": "asia", "jordan": "asia",
    "kazakhstan": "asia", "kyrgyzstan": "asia", "laos": "asia", "lebanon": "asia",
    "malaysia-singapore-brunei": "asia", "maldives": "asia", "mongolia": "asia",
    "myanmar": "asia", "nepal": "asia", "north-korea": "asia", "pakistan": "asia",
    "philippines": "asia", "south-korea": "asia", "sri-lanka": "asia", "syria": "asia",
    "taiwan": "asia", "tajikistan": "asia", "thailand": "asia", "turkmenistan": "asia",
    "uzbekistan": "asia", "vietnam": "asia", "yemen": "asia",
    # Africa
    "algeria": "africa", "angola": "africa", "benin": "africa", "botswana": "africa",
    "burkina-faso": "africa", "burundi": "africa", "cameroon": "africa",
    "canary-islands": "africa", "cape-verde": "africa", "central-african-republic": "africa",
    "chad": "africa", "comores": "africa", "congo-brazzaville": "africa",
    "congo-democratic-republic": "africa", "djibouti": "africa", "egypt": "africa",
    "equatorial-guinea": "africa", "eritrea": "africa", "ethiopia": "africa",
    "gabon": "africa", "ghana": "africa", "guinea": "africa", "guinea-bissau": "africa",
    "ivory-coast": "africa", "kenya": "africa", "lesotho": "africa", "liberia": "africa",
    "libya": "africa", "madagascar": "africa", "malawi": "africa", "mali": "africa",
    "mauritania": "africa", "mauritius": "africa", "morocco": "africa",
    "mozambique": "africa", "namibia": "africa", "niger": "africa", "nigeria": "africa",
    "rwanda": "africa", "saint-helena-ascension-and-tristan-da-cunha": "africa",
    "sao-tome-and-principe": "africa", "senegal-and-gambia": "africa", "seychelles": "africa",
    "sierra-leone": "africa", "somalia": "africa", "south-africa": "africa",
    "south-sudan": "africa", "sudan": "africa", "swaziland": "africa", "tanzania": "africa",
    "togo": "africa", "tunisia": "africa", "uganda": "africa", "zambia": "africa",
    "zimbabwe": "africa",
    # Oceania
    "american-oceania": "australia-oceania", "australia": "australia-oceania",
    "cook-islands": "australia-oceania", "fiji": "australia-oceania",
    "ile-de-clipperton": "australia-oceania", "kiribati": "australia-oceania",
    "marshall-islands": "australia-oceania", "micronesia": "australia-oceania",
    "nauru": "australia-oceania", "new-caledonia": "australia-oceania",
    "new-zealand": "australia-oceania", "niue": "australia-oceania",
    "palau": "australia-oceania", "papua-new-guinea": "australia-oceania",
    "pitcairn-islands": "australia-oceania", "polynesie-francaise": "australia-oceania",
    "samoa": "australia-oceania", "solomon-islands": "australia-oceania",
    "tokelau": "australia-oceania", "tonga": "australia-oceania",
    "tuvalu": "australia-oceania", "vanuatu": "australia-oceania",
    "wallis-et-futuna": "australia-oceania",
}

# Hierarchical region structure with subregions
# Format: continent -> country -> {subregions: [...]} or {} if no subregions
# Complete list from https://download.geofabrik.de/
GEOFABRIK_REGIONS = {
    "europe": {
        # Countries with subregions
        "germany": {
            "subregions": [
                "baden-wuerttemberg", "bayern", "berlin", "brandenburg", "bremen",
                "hamburg", "hessen", "mecklenburg-vorpommern", "niedersachsen",
                "nordrhein-westfalen", "rheinland-pfalz", "saarland", "sachsen",
                "sachsen-anhalt", "schleswig-holstein", "thueringen"
            ]
        },
        "france": {
            "subregions": [
                "alsace", "aquitaine", "auvergne", "basse-normandie", "bourgogne",
                "bretagne", "centre", "champagne-ardenne", "corse", "franche-comte",
                "haute-normandie", "ile-de-france", "languedoc-roussillon", "limousin",
                "lorraine", "midi-pyrenees", "nord-pas-de-calais", "pays-de-la-loire",
                "picardie", "poitou-charentes", "provence-alpes-cote-d-azur", "rhone-alpes"
            ]
        },
        "italy": {
            "subregions": [
                "centro", "isole", "nord-est", "nord-ovest", "sud"
            ]
        },
        "poland": {
            "subregions": [
                "dolnoslaskie", "kujawsko-pomorskie", "lodzkie", "lubelskie",
                "lubuskie", "malopolskie", "mazowieckie", "opolskie", "podkarpackie",
                "podlaskie", "pomorskie", "slaskie", "swietokrzyskie",
                "warminsko-mazurskie", "wielkopolskie", "zachodniopomorskie"
            ]
        },
        "great-britain": {
            "subregions": [
                "england", "scotland", "wales"
            ]
        },
        "russia": {
            "subregions": [
                "central-fed-district", "crimean-fed-district", "far-eastern-fed-district",
                "kaliningrad", "north-caucasus-fed-district", "northwestern-fed-district",
                "siberian-fed-district", "south-fed-district", "ural-fed-district",
                "volga-fed-district"
            ]
        },
        # Countries without subregions (complete list from Geofabrik)
        "albania": {},
        "andorra": {},
        "austria": {},
        "azores": {},
        "belarus": {},
        "belgium": {},
        "bosnia-herzegovina": {},
        "bulgaria": {},
        "croatia": {},
        "cyprus": {},
        "czech-republic": {},
        "denmark": {},
        "estonia": {},
        "faroe-islands": {},
        "finland": {},
        "georgia": {},
        "greece": {},
        "guernsey-jersey": {},
        "hungary": {},
        "iceland": {},
        "ireland-and-northern-ireland": {},
        "isle-of-man": {},
        "kosovo": {},
        "latvia": {},
        "liechtenstein": {},
        "lithuania": {},
        "luxembourg": {},
        "macedonia": {},
        "malta": {},
        "moldova": {},
        "monaco": {},
        "montenegro": {},
        "netherlands": {},
        "norway": {},
        "portugal": {},
        "romania": {},
        "serbia": {},
        "slovakia": {},
        "slovenia": {},
        "spain": {},
        "sweden": {},
        "switzerland": {},
        "turkey": {},
        "ukraine": {},
    },
    "north-america": {
        "us": {
            "subregions": [
                "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
                "connecticut", "delaware", "district-of-columbia", "florida", "georgia",
                "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
                "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
                "mississippi", "missouri", "montana", "nebraska", "nevada", "new-hampshire",
                "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
                "ohio", "oklahoma", "oregon", "pennsylvania", "puerto-rico", "rhode-island",
                "south-carolina", "south-dakota", "tennessee", "texas", "us-virgin-islands",
                "utah", "vermont", "virginia", "washington", "west-virginia", "wisconsin", "wyoming"
            ]
        },
        "canada": {
            "subregions": [
                "alberta", "british-columbia", "manitoba", "new-brunswick",
                "newfoundland-and-labrador", "northwest-territories", "nova-scotia",
                "nunavut", "ontario", "prince-edward-island", "quebec", "saskatchewan", "yukon"
            ]
        },
        "mexico": {},
        "greenland": {},
    },
    "south-america": {
        "brazil": {
            "subregions": [
                "centro-oeste", "nordeste", "norte", "sudeste", "sul"
            ]
        },
        # All South American countries
        "argentina": {},
        "bolivia": {},
        "chile": {},
        "colombia": {},
        "ecuador": {},
        "guyana": {},
        "paraguay": {},
        "peru": {},
        "suriname": {},
        "uruguay": {},
        "venezuela": {},
    },
    "asia": {
        "japan": {
            "subregions": [
                "chubu", "chugoku", "hokkaido", "kansai", "kanto", "kyushu", "shikoku", "tohoku"
            ]
        },
        "india": {
            "subregions": [
                "central-zone", "eastern-zone", "north-eastern-zone", "northern-zone",
                "southern-zone", "western-zone"
            ]
        },
        "indonesia": {
            "subregions": [
                "java", "kalimantan", "maluku-and-papua", "nusa-tenggara", "sulawesi", "sumatra"
            ]
        },
        # All Asian countries (complete list from Geofabrik)
        "afghanistan": {},
        "armenia": {},
        "azerbaijan": {},
        "bangladesh": {},
        "bhutan": {},
        "cambodia": {},
        "china": {},
        "east-timor": {},
        "gcc-states": {},
        "iran": {},
        "iraq": {},
        "israel-and-palestine": {},
        "jordan": {},
        "kazakhstan": {},
        "kyrgyzstan": {},
        "laos": {},
        "lebanon": {},
        "malaysia-singapore-brunei": {},
        "maldives": {},
        "mongolia": {},
        "myanmar": {},
        "nepal": {},
        "north-korea": {},
        "pakistan": {},
        "philippines": {},
        "south-korea": {},
        "sri-lanka": {},
        "syria": {},
        "taiwan": {},
        "tajikistan": {},
        "thailand": {},
        "turkmenistan": {},
        "uzbekistan": {},
        "vietnam": {},
        "yemen": {},
    },
    "africa": {
        "south-africa": {
            "subregions": [
                "eastern-cape", "free-state", "gauteng", "kwazulu-natal", "limpopo",
                "mpumalanga", "north-west", "northern-cape", "western-cape"
            ]
        },
        # All African countries (complete list from Geofabrik)
        "algeria": {},
        "angola": {},
        "benin": {},
        "botswana": {},
        "burkina-faso": {},
        "burundi": {},
        "cameroon": {},
        "canary-islands": {},
        "cape-verde": {},
        "central-african-republic": {},
        "chad": {},
        "comores": {},
        "congo-brazzaville": {},
        "congo-democratic-republic": {},
        "djibouti": {},
        "egypt": {},
        "equatorial-guinea": {},
        "eritrea": {},
        "ethiopia": {},
        "gabon": {},
        "ghana": {},
        "guinea": {},
        "guinea-bissau": {},
        "ivory-coast": {},
        "kenya": {},
        "lesotho": {},
        "liberia": {},
        "libya": {},
        "madagascar": {},
        "malawi": {},
        "mali": {},
        "mauritania": {},
        "mauritius": {},
        "morocco": {},
        "mozambique": {},
        "namibia": {},
        "niger": {},
        "nigeria": {},
        "rwanda": {},
        "saint-helena-ascension-and-tristan-da-cunha": {},
        "sao-tome-and-principe": {},
        "senegal-and-gambia": {},
        "seychelles": {},
        "sierra-leone": {},
        "somalia": {},
        "south-sudan": {},
        "sudan": {},
        "swaziland": {},
        "tanzania": {},
        "togo": {},
        "tunisia": {},
        "uganda": {},
        "zambia": {},
        "zimbabwe": {},
    },
    "australia-oceania": {
        "australia": {
            "subregions": [
                "australian-capital-territory", "new-south-wales", "northern-territory",
                "queensland", "south-australia", "tasmania", "victoria", "western-australia"
            ]
        },
        # All Oceania countries (complete list from Geofabrik)
        "american-oceania": {},
        "cook-islands": {},
        "fiji": {},
        "ile-de-clipperton": {},
        "kiribati": {},
        "marshall-islands": {},
        "micronesia": {},
        "nauru": {},
        "new-caledonia": {},
        "new-zealand": {},
        "niue": {},
        "palau": {},
        "papua-new-guinea": {},
        "pitcairn-islands": {},
        "polynesie-francaise": {},
        "samoa": {},
        "solomon-islands": {},
        "tokelau": {},
        "tonga": {},
        "tuvalu": {},
        "vanuatu": {},
        "wallis-et-futuna": {},
    },
}


def get_pbf_bounds(pbf_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract bounding box from a PBF file using osmium fileinfo.

    Returns: (min_lon, min_lat, max_lon, max_lat) or None if extraction fails.
    """
    try:
        result = subprocess.run(
            ["osmium", "fileinfo", "-e", str(pbf_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"  Warning: osmium fileinfo failed for {pbf_path.name}")
            return None

        # Parse output for bounding box
        # Format: "    (min_lon,min_lat,max_lon,max_lat)"
        for line in result.stdout.split('\n'):
            match = re.search(r'\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\)', line)
            if match:
                min_lon = float(match.group(1))
                min_lat = float(match.group(2))
                max_lon = float(match.group(3))
                max_lat = float(match.group(4))
                return (min_lon, min_lat, max_lon, max_lat)

        print(f"  Warning: No bounding box found in osmium output for {pbf_path.name}")
        return None

    except subprocess.TimeoutExpired:
        print(f"  Warning: Timeout extracting bounds from {pbf_path.name}")
        return None
    except FileNotFoundError:
        print("  Warning: osmium not found. Install with: brew install osmium-tool")
        return None
    except Exception as e:
        print(f"  Warning: Error extracting bounds: {e}")
        return None


def derive_display_name(region_name: str) -> str:
    """Convert region filename to display name."""
    # Handle special cases
    special_names = {
        "malaysia-singapore-brunei": "Malaysia",
        "south-korea": "South Korea",
        "united-kingdom": "United Kingdom",
        "new-zealand": "New Zealand",
        "south-africa": "South Africa",
        "czech-republic": "Czech Republic",
    }

    if region_name in special_names:
        return special_names[region_name]

    # Default: capitalize and replace hyphens with spaces
    return region_name.replace("-", " ").title()


def get_geofabrik_url(region_name: str, continent: str = None, country: str = None) -> str:
    """Get the Geofabrik download URL for a region.

    Args:
        region_name: The region/subregion name (e.g., 'california', 'ukraine')
        continent: Optional continent for subregion URLs (e.g., 'north-america')
        country: Optional country for subregion URLs (e.g., 'us')

    Returns:
        Full Geofabrik download URL

    Examples:
        get_geofabrik_url('ukraine') -> .../europe/ukraine-latest.osm.pbf
        get_geofabrik_url('california', 'north-america', 'us') -> .../north-america/us/california-latest.osm.pbf
    """
    # If continent and country provided, this is a subregion
    if continent and country:
        return f"https://download.geofabrik.de/{continent}/{country}/{region_name}-latest.osm.pbf"

    # Check if it's a known country in the new hierarchy
    for cont_name, countries in GEOFABRIK_REGIONS.items():
        if region_name in countries:
            return f"https://download.geofabrik.de/{cont_name}/{region_name}-latest.osm.pbf"

    # Fall back to old GEOFABRIK_CONTINENTS lookup
    continent = GEOFABRIK_CONTINENTS.get(region_name, "asia")
    return f"https://download.geofabrik.de/{continent}/{region_name}-latest.osm.pbf"


def get_regions_by_continent() -> Dict[str, dict]:
    """
    Return regions grouped by continent for UI dropdowns.

    Returns hierarchical dict:
    {
        "continent_name": {
            "display_name": "Continent Name",
            "countries": {
                "country_name": {
                    "display_name": "Country Name",
                    "subregions": [
                        {"name": "subregion", "display_name": "Subregion"},
                        ...
                    ]  # Empty list if no subregions
                }
            }
        }
    }
    """
    continent_display = {
        "europe": "Europe",
        "asia": "Asia",
        "africa": "Africa",
        "north-america": "North America",
        "south-america": "South America",
        "australia-oceania": "Oceania",
    }

    # Build hierarchical structure from GEOFABRIK_REGIONS
    result = {}
    continent_order = ["europe", "north-america", "asia", "south-america", "africa", "australia-oceania"]

    for continent in continent_order:
        if continent not in GEOFABRIK_REGIONS:
            continue

        countries_data = {}
        for country_name, country_info in GEOFABRIK_REGIONS[continent].items():
            subregions = country_info.get("subregions", [])
            subregion_list = [
                {"name": sr, "display_name": derive_display_name(sr)}
                for sr in sorted(subregions, key=lambda x: derive_display_name(x))
            ]

            countries_data[country_name] = {
                "display_name": derive_display_name(country_name),
                "subregions": subregion_list,
            }

        # Sort countries by display name
        sorted_countries = dict(sorted(
            countries_data.items(),
            key=lambda x: x[1]["display_name"]
        ))

        result[continent] = {
            "display_name": continent_display.get(continent, continent.title()),
            "countries": sorted_countries,
        }

    return result


def scan_pbf_directories() -> Dict[str, dict]:
    """
    Scan both geofabrik and cache directories for PBF files and extract metadata.

    Prefers geofabrik directory (primary) over cache directory (secondary).
    Returns dict of region_name -> {bounds, display_name, pbf_file, pbf_path, url}
    """
    regions = {}

    # Scan both directories, preferring geofabrik over cache
    directories = []
    if GEOFABRIK_DIR.exists():
        directories.append(("geofabrik", GEOFABRIK_DIR))
    if CACHE_DIR.exists():
        directories.append(("cache", CACHE_DIR))

    if not directories:
        GEOFABRIK_DIR.mkdir(parents=True, exist_ok=True)
        return regions

    # Find all PBF files from both directories
    pbf_files = []
    for dir_name, dir_path in directories:
        for pbf_path in dir_path.glob("*-latest.osm.pbf"):
            region_name = pbf_path.stem.replace("-latest.osm", "")
            # Only add if not already found (geofabrik takes priority)
            if region_name not in [r[0] for r in pbf_files]:
                pbf_files.append((region_name, pbf_path, dir_name))

    if not pbf_files:
        return regions

    print(f"Scanning {len(pbf_files)} PBF file(s)...")

    for region_name, pbf_path, dir_name in pbf_files:
        print(f"  Processing {region_name} (from {dir_name})...")

        # Get bounding box
        bounds = get_pbf_bounds(pbf_path)
        if bounds is None:
            print(f"    Skipping {region_name} - could not extract bounds")
            continue

        regions[region_name] = {
            "bounds": bounds,
            "display_name": derive_display_name(region_name),
            "pbf_file": pbf_path.name,
            "pbf_path": str(pbf_path),
            "url": get_geofabrik_url(region_name),
        }

        print(f"    Bounds: {bounds}")

    return regions


def save_registry(regions: Dict[str, dict]) -> None:
    """Save region registry to JSON file."""
    GEOFABRIK_DIR.mkdir(parents=True, exist_ok=True)

    # Convert tuples to lists for JSON serialization
    serializable = {}
    for name, data in regions.items():
        serializable[name] = {
            "bounds": list(data["bounds"]),
            "display_name": data["display_name"],
            "pbf_file": data["pbf_file"],
            "pbf_path": data.get("pbf_path", ""),
            "url": data["url"],
        }

    with open(REGISTRY_FILE, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved registry to {REGISTRY_FILE}")


def load_registry() -> Dict[str, dict]:
    """Load region registry from JSON file."""
    if not REGISTRY_FILE.exists():
        return {}

    try:
        with open(REGISTRY_FILE) as f:
            data = json.load(f)

        # Convert lists back to tuples for bounds
        regions = {}
        for name, info in data.items():
            regions[name] = {
                "bounds": tuple(info["bounds"]),
                "display_name": info["display_name"],
                "pbf_file": info["pbf_file"],
                "pbf_path": info.get("pbf_path", ""),
                "url": info["url"],
            }
        return regions
    except Exception as e:
        print(f"Warning: Could not load registry: {e}")
        return {}


def get_available_regions(force_rescan: bool = False) -> Dict[str, dict]:
    """
    Get all available regions with their metadata.

    Uses cached registry if available, otherwise scans PBF directories.
    Set force_rescan=True to always rescan.

    Returns dict of region_name -> {bounds, display_name, pbf_file, pbf_path, url}
    """
    # Check if any new PBF files exist that aren't in registry
    registry = load_registry() if not force_rescan else {}

    # Get list of PBF files from both directories
    pbf_files = set()
    for dir_path in [GEOFABRIK_DIR, CACHE_DIR]:
        if dir_path.exists():
            for p in dir_path.glob("*-latest.osm.pbf"):
                pbf_files.add(p.stem.replace("-latest.osm", ""))

    # Check if registry is up to date
    registry_regions = set(registry.keys())

    if pbf_files != registry_regions or force_rescan:
        # Rescan needed
        registry = scan_pbf_directories()
        if registry:
            save_registry(registry)

    return registry


def detect_region_for_coords(lat: float, lon: float) -> Optional[str]:
    """
    Detect which available region contains the given coordinates.

    Returns region name or None if not found.
    """
    regions = get_available_regions()

    for region_name, info in regions.items():
        min_lon, min_lat, max_lon, max_lat = info["bounds"]
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return region_name

    return None


def get_region_url(region_name: str) -> Optional[str]:
    """Get the download URL for a region (from registry or generated)."""
    regions = get_available_regions()

    if region_name in regions:
        return regions[region_name]["url"]

    # Not in registry, generate URL
    return get_geofabrik_url(region_name)


def get_region_display_name(region_name: str) -> str:
    """Get the display name for a region."""
    regions = get_available_regions()

    if region_name in regions:
        return regions[region_name]["display_name"]

    return derive_display_name(region_name)


def get_region_pbf_path(region_name: str) -> Optional[Path]:
    """Get the path to the cached PBF file for a region."""
    regions = get_available_regions()

    if region_name in regions:
        pbf_path = regions[region_name].get("pbf_path")
        if pbf_path:
            return Path(pbf_path)
        # Fallback to geofabrik directory
        return GEOFABRIK_DIR / regions[region_name]["pbf_file"]

    # Check if file exists even if not in registry (check both directories)
    for dir_path in [GEOFABRIK_DIR, CACHE_DIR]:
        expected_path = dir_path / f"{region_name}-latest.osm.pbf"
        if expected_path.exists():
            return expected_path

    return None


# CLI for testing/manual operations
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Region Registry - manage available map regions")
    parser.add_argument("--scan", action="store_true", help="Force rescan of cache directory")
    parser.add_argument("--list", action="store_true", help="List available regions")
    parser.add_argument("--detect", nargs=2, type=float, metavar=("LAT", "LON"),
                        help="Detect region for coordinates")

    args = parser.parse_args()

    if args.scan:
        print("Forcing rescan of cache directory...")
        regions = get_available_regions(force_rescan=True)
        print(f"\nFound {len(regions)} region(s)")

    if args.list or (not args.scan and not args.detect):
        regions = get_available_regions()
        if regions:
            print("\nAvailable regions:")
            for name, info in sorted(regions.items()):
                bounds = info["bounds"]
                print(f"  {name}:")
                print(f"    Display name: {info['display_name']}")
                print(f"    Bounds: {bounds[0]:.2f}°E to {bounds[2]:.2f}°E, {bounds[1]:.2f}°N to {bounds[3]:.2f}°N")
                print(f"    PBF file: {info['pbf_file']}")
        else:
            print("\nNo regions available. Download PBF files to cache/ directory.")
            print("Example: curl -o cache/ukraine-latest.osm.pbf https://download.geofabrik.de/europe/ukraine-latest.osm.pbf")

    if args.detect:
        lat, lon = args.detect
        region = detect_region_for_coords(lat, lon)
        if region:
            print(f"\nCoordinates ({lat}, {lon}) are in region: {region}")
        else:
            print(f"\nNo available region contains coordinates ({lat}, {lon})")
