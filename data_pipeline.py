"""
Data collection and preprocessing pipeline for TESS/Kepler light curves.
Handles pulling target lists, downloading light curve files, and basic cleaning.
"""

import random
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import lightkurve as lk
from tqdm import tqdm
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive


def in_prime_mission(sector_str, sector_max=21):
    if pd.isna(sector_str):
        return False

    sectors = [int(s.strip()) for s in str(sector_str).split(',') if s.strip().isdigit()]
    return any(sec <= sector_max for sec in sectors)


def pull_TOI_stars():
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?output=csv"
    df = pd.read_csv(url)

    df_prime = df[df["Sectors"].apply(in_prime_mission)]

    positives = df_prime[df_prime['TFOPWG Disposition'].isin(['CP', 'PC'])]
    negatives = df_prime[df_prime['TFOPWG Disposition'] == 'FP']

    positives[['TIC ID', 'TFOPWG Disposition', 'Sectors']].to_csv("positives.csv", index=False)
    negatives[['TIC ID', 'TFOPWG Disposition', 'Sectors']].to_csv("negatives.csv", index=False)


def expand_tic_sector_pairs(df, label):
    rows = []
    for _, row in df.iterrows():
        tic_id = int(row['TIC ID'])
        if pd.isna(row['Sectors']):
            continue
        for sec in str(row['Sectors']).split(','):
            sec = sec.strip()
            if sec.isdigit():
                rows.append((tic_id, int(sec), label))
    return rows


def filter_and_balance_sectors():
    pos = pd.read_csv("positives.csv")
    neg = pd.read_csv("negatives.csv")
    pairs_pos = expand_tic_sector_pairs(pos, "positive")
    pairs_neg = expand_tic_sector_pairs(neg, "negative")

    pairs_pos = [p for p in pairs_pos if p[1] <= 21]
    pairs_neg = [p for p in pairs_neg if p[1] <= 21]

    n = min(len(pairs_pos), len(pairs_neg))
    random.seed(42)
    pairs_pos = random.sample(pairs_pos, n)
    pairs_neg = random.sample(pairs_neg, n)

    balanced_pairs = pairs_neg + pairs_pos
    print(f"Balanced Dataset: {len(pairs_pos)} positives and {len(pairs_neg)} negatives")
    return balanced_pairs


def parse_file_name(filename):
    fname = Path(filename).stem
    tic_part, sector_part = fname.split("_")
    tic_id = tic_part.replace("tic", "")
    sector = int(sector_part.replace("s", ""))
    return tic_id, sector


def build_metadata_df(pos_dir, neg_dir):
    rows = []
    for f in Path(pos_dir).glob("*.fits"):
        tic_id, sector = parse_file_name(f)
        rows.append({"tic_id": tic_id, "sector": sector, "label": 1, "file_path": str(f)})
    for f in Path(neg_dir).glob("*.fits"):
        tic_id, sector = parse_file_name(f)
        rows.append({"tic_id": tic_id, "sector": sector, "label": 0, "file_path": str(f)})
    df = pd.DataFrame(rows)
    print(df.head())
    df.to_csv("metadata.csv", index=False)


def download_lightcurve(tic_id, sector, label, max_retries=3):
    for attempt in range(max_retries):
        outdir = Path("data") / label
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector, author="QLP", exptime=1800)

            if len(search) == 0:
                return None

            lc = search.download(download_dir=str(outdir), cache=False)

            if lc is None:
                return None

            return "OK"
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"Failed to download TIC {tic_id} sector {sector} after {max_retries} attempts: {e}")
            return None


def download_all(balanced_pairs, max_workers=16):
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for tic_id, sec, label in balanced_pairs:
            futures.append(executor.submit(download_lightcurve, tic_id, sec, label))
        results = []
        success_count = 0
        with tqdm(as_completed(futures), total=len(futures), desc="Downloading TESS LC") as pbar:
            for fut in pbar:
                r = fut.result()
                results.append(r)
                if r is not None:
                    success_count += 1
                pbar.set_postfix({"success_count": success_count})
    print(f"Downloaded {success_count} / {len(balanced_pairs)} lightcurves", file=sys.__stdout__)


def choose_stars():
    # We retrieve a table of every kepler object of interest
    koi_table = NasaExoplanetArchive.query_criteria(table="cumulative", select="kepid,koi_disposition")
    # Convert into pandas dataframe for easier access
    koi_df = koi_table.to_pandas()
    # Filter the table with the labels we are interested in: confirmed planets and false positive planets
    confirmed = koi_df[koi_df["koi_disposition"] == "CONFIRMED"]
    false_pos = koi_df[koi_df["koi_disposition"] == "FALSE POSITIVE"]
    # Delete duplicates so that we dont get multiple planets from the same star
    confirmed_unique = confirmed.drop_duplicates(subset="kepid")
    false_pos_unique = false_pos.drop_duplicates(subset="kepid")

    # Combine confirmed and false positive stars into one set
    subset = pd.concat([confirmed_unique, false_pos_unique])
    # Shuffle the set in a random order and clean the index
    subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)

    return subset


def clean_lightcurve(lc):
    clean_lc = lc.remove_outliers(sigma=5)
    clean_lc = clean_lc.normalize()
    return clean_lc


def extract_and_clean_lightcurve(fits_path):
    lc = lk.read(fits_path)

    # Drop NaNs
    lc = lc.remove_nans()

    # Remove strong outliers (5-sigma clipping) and normalise flux (so median ~ 1.0)
    lc = clean_lightcurve(lc)

    # Extract just the flux values as numpy array
    flux = lc.flux.value
    print(len(flux))
    return flux


def load_lightcurves_data():
    df = pd.read_csv("all_global.csv")
    X = df.drop(columns=["label"]).values
    y = (df["label"] == "CONFIRMED").astype(int).values
    return X, y


def check_files_are_valid(path):
    import astropy.io.fits as fits

    cache_dir = Path(path)
    fits_files = list(cache_dir.glob("*.fits"))

    good_files, bad_files = [], []

    for f in fits_files:
        try:
            with fits.open(f, mode="readonly") as hdul:
                hdul.verify('exception')  # check internal consistency
            good_files.append(f)
        except Exception as e:
            print(f"Bad file: {f.name} ({e})")
            bad_files.append(f)

    print(f"Total files: {len(fits_files)}")
    print(f"Good files: {len(good_files)}")
    print(f"Bad files: {len(bad_files)}")


def test_lightkurve_open(path):
    try:
        lc = lk.read(path)
        print(f"✅ {path} OK → {len(lc.flux)} points")
        return True
    except Exception as e:
        print(f"❌ {path} failed: {e}")
        return False
