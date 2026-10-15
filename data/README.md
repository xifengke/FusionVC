# Initial VCC 2025 assets

The public source snapshot includes this README only, without matrices or processed caches.
This file began as an initial-stage record; the full test set was later downloaded and used for all final evaluations.
Current facts and resource requirements are in [the data notes](../docs/resources-and-data.md).

The initial download contains the training and validation splits under raw/vcc2025/, the official gene_names.csv and per-split pert_counts CSVs.

The test split was deferred during the initial stage, then downloaded for the frozen final evaluation. FASTQ reads and pretraining corpora are outside this project download.

Files are obtained directly from the Arc Institute's public bucket, pinned to the recorded GCS object generation. The downloader verifies full file length and the official CRC32C (plus MD5 when provided), and records a local SHA256.

See ../docs/initial-download-manifest.json for sources and ../docs/download-status.json for completion and checksums. A .part file is incomplete even if its logical length looks correct; only files marked verified are ready to use.

Do not split randomly by cell: the official training and validation sets hold out perturbation targets. Validation labels are for evaluation/model selection, not fitting. Preserve original files and gene order.

`processed/scgpt-frozen-v1/` is a rebuildable feature cache extracted only from
2,048 training non-targeting cells and the verified whole-human pretrained
checkpoint. It stores individual control embeddings, their fixed mean context,
target features, control indices, hashes and coverage. It is shared across seeds
and losses; no held-out perturbation expression is used to fit these features.
See [the scGPT protocol](../docs/scgpt-protocol.md).
