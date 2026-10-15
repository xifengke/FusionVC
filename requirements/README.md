# Environments

`scripts/setup_training.ps1` creates the GPU environment at `.venv/`.
The tested installation freeze is saved as `windows-training.lock.txt`.
The official cell-eval source must be downloaded first; its fixed revision is in
`docs/source-provenance.json`. PyTorch is installed from the official cu128 wheel index.

The absolute editable path in a local freeze is provenance, not a portable installation command.
Recreate with the setup script and `pyproject.toml`, then compare versions to the lock.
The `.venv-download` environment and `docs/download-tools.lock.txt` are separate download tools.

The frozen scGPT experiment uses the existing GPU environment without extra
packages. The dependency-light adapter strictly loads the pinned checkpoint's
encoder and is numerically checked against the downloaded official implementation.
It does not require the upstream legacy torchtext or FlashAttention packages.
Run `scripts/check_scgpt_parity.py` before extracting features on a new machine;
see [the protocol](../docs/scgpt-protocol.md).

Frozen Geneformer V1-10M adds `transformers==4.57.1` and `safetensors==0.6.2`
through the `geneformer` extra. `setup_training.ps1` installs this extra, and the
tested environment freeze/constraints include its transitive dependencies.
Only local pinned weights and Genecorpus-30M dictionaries are loaded; extraction
does not contact Hugging Face or install the unrelated upstream fine-tuning,
plotting, loom or dataset packages. Run `scripts/check_geneformer_parity.py` first;
see [the Geneformer protocol](../docs/geneformer-protocol.md).

On sandboxed Windows, if the default pytest temporary directory cannot be created,
use a fresh workspace path, for example `python -m pytest -q --basetemp tmp/pytest-local`.

The `final_test` extra adds the locally verified requests 2.34.2 and
google-crc32c 1.9.0 for the pinned, resumable final-test download. These versions
are already in the recorded environment lock/constraints; no cloud account or
billing project is needed for the currently public official objects.

The historical `fusion_features.py` v1 extractor is excluded only from automatic
formatting because existing feature manifests bind its exact source bytes. It
is still linted and tested; changes to the extractor require a new cache version.
