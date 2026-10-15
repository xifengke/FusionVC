# Frozen model assets

The source release contains this README, not pretrained weights. Obtain the pinned assets through the documented download workflow and check their terms.

- scgpt/whole-human/: best_model.pt, args.json and vocab.json from the official scGPT whole-human download folder. The original config contains the authors' training-machine paths and distributed settings; these are provenance, not instructions to use those paths or launch distributed jobs locally.
- geneformer-v1/: the original V1-10M model as config.json and model.safetensors. Version-matched 30M dictionaries are in dictionaries/.

Both encoders remained frozen in the completed experiments. Do not silently substitute a different Geneformer version or vocabulary when rebuilding these comparisons.

The official competition output has 18,080 genes. scGPT's checkpoint vocabulary covers 17,840 of them; missing input tokens must be handled explicitly while retaining every required output gene.

Geneformer V1 uses the original training file's `var/gene_id` Ensembl IDs and
the version-matched 30M mapping, token and median dictionaries. Its vocabulary
covers 17,915/18,080 output genes and 254/256 fixed input candidates; all 150
training and 50 validation targets are covered. The two missing input genes
are C13orf42 and APELA. Every required output remains in the decoder.

docs/download-status.json records source checksums where available, and local SHA256 hashes. scGPT's weight archive has also been tested and read on CPU using torch.load(weights_only=True).
