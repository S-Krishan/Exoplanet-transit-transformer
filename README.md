## Exoplanet Transit Detection using a Transformer Architecture

A transformer encoder that classifies light curves as containing a genuine planetary transit or not — trained first on preprocessed Kepler data, then extended to raw TESS observations pulled directly from NASA's MAST archive.

## Results

- **0.8134 ROC-AUC** distinguishing confirmed planets from false positives on held-out, folded and preprocessed Kepler light curve data.
- Extended the experiment to **raw TESS light curves**, built a custom pipeline to download and preprocess them directly from MAST, and found the model collapsed to predicting a single class on this harder, noisier dataset.
- Diagnosed the cause by plotting labelled samples directly: many of the labelled positive transits were weak and buried in noise relative to the negative class, which limited the model's ability to learn a clean decision boundary. Identifying *why* it failed, rather than a working raw-TESS model, was the main output of this stage.

## Architecture

- Each light curve is treated as a sequence of flux values. Every timestep is projected from 1D into d_model-dimensional space with a linear embedding layer.
- Standard sinusoidal positional encoding (as in the original Transformer paper) is added so the model retains information about each value's position in the sequence.
- A TransformerEncoder (2 layers, 4 attention heads, d_model=128) processes the sequence.
- The output sequence is mean-pooled across time into a single vector per light curve, then passed through a linear layer producing one logit for binary classification (transit / no transit).

## Data pipeline

**Kepler (folded, preprocessed).** Confirmed and false-positive KOIs are pulled from NASA's Exoplanet Archive; the folded, normalised light curve data is used directly for training.

**TESS (raw, extended experiment).**
- Target list pulled from ExoFOP-TESS's TOI catalogue, filtered to TESS's prime mission (sectors ≤ 21) and to confirmed/false-positive labels only.
- Positive and negative classes are balanced by random sampling.
- Light curves are downloaded in parallel (ThreadPoolExecutor, with retry logic) directly from MAST via lightkurve, then cleaned — NaN removal, 5σ outlier clipping, normalisation.
- Train/val/test splits are stratified **by star** (tic_id), not by individual light curve, so the same star's data can't leak across splits.

## Project structure


.
├── data_pipeline.py   # Fetching, downloading, cleaning, and preparing light curve data
├── model.py           # Dataset class, positional encoding, transformer model, training/eval loop
├── main.py            # Entry point - swap between download / dataset-diff / training stages
└── .gitignore


## Setup


pip install torch pandas scikit-learn lightkurve astroquery tqdm


## Usage

This project has three separate stages, all driven from main.py. Only one stage runs by default — open main.py and change which function main() calls:

- run_download() — pulls the TOI target list, balances classes, and downloads TESS light curves.
- run_star_diff_check() — checks for Kepler stars not already represented in the working dataset.
- run_training() — trains and evaluates the transformer.


## Challenges and lessons learned

- The bigger challenge here wasn't the model architecture — it was data quality. Raw TESS light curves are far noisier than the folded, preprocessed Kepler data typically used in published benchmarks, and that surfaced directly as a model-collapse failure rather than a small accuracy drop.
- Moving the MAST download step from serial to threaded downloads was necessary to make working with a large, balanced TESS sample practical.

## Possible next steps
- Filter TESS positives by transit strength (e.g. signal-to-noise ratio or transit depth) rather than only balancing class counts. The model-collapse diagnosis suggests many labelled TESS positives are too weak or noisy to teach a clean decision boundary — filtering on signal strength, not just label balance, may address this directly.
- Explore data augmentation that preserves transit shape, to give the model more robust positive examples from noisy TESS data.
