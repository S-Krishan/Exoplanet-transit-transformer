"""
Entry point for the exoplanet transit detection project.

This ties together three stages of the project that were run at different
points during development:
  1. run_download()        - pull TOI/TESS targets and download their light curves
  2. run_star_diff_check() - find new Kepler stars not already in the working dataset
  3. run_training()        - train and evaluate the transformer on processed data

Only one stage runs by default in main() - swap it out depending on which
part of the pipeline you're working on.
"""

import torch
import pandas as pd

from data_pipeline import filter_and_balance_sectors, download_all, choose_stars, load_lightcurves_data
from model import (
    LightCurveDataset,
    LightCurveTransformer,
    train_validation_split,
    train,
    evaluate_performance,
)


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    X, y = load_lightcurves_data()
    dataset = LightCurveDataset(X, y)
    train_loader, val_loader = train_validation_split(dataset)
    model = LightCurveTransformer().to(device)
    train(model, 19, train_loader, val_loader, device)
    evaluate_performance(model, val_loader, device)


def run_star_diff_check():
    all_stars = choose_stars()
    mend = pd.read_csv("all_global.csv")
    print(mend.columns)

    all_stars["kepid"] = all_stars["kepid"].astype(int)
    mend["kepid"] = mend["kepid"].astype(int)
    new_koi = all_stars[~all_stars["kepid"].isin(mend["kepid"])]
    confirmed = new_koi[new_koi["koi_disposition"] == "CONFIRMED"]
    false_pos = new_koi[new_koi["koi_disposition"] == "FALSE POSITIVE"]
    print("Confirmed planets:", len(confirmed))
    print("False positives:", len(false_pos))


def run_download():
    balanced_sectors = filter_and_balance_sectors()
    download_all(balanced_sectors[:20])


def main():
    run_training()


if __name__ == "__main__":
    main()
