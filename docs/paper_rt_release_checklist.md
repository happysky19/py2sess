# py2sess paper release checklist

Use this checklist after the manuscript, paper assets, and benchmark scripts are final.
It is intentionally separate from the working benchmark outputs under `outputs/`.

1. Choose a release tag name for the paper artifact, for example `paper-rt-v1.0`.
2. Replace `<final-paper-tag-or-commit>` in the Colab snippets with that tag name.
3. Commit the final paper state, including the release tag name in the rerun instructions.
4. Create the release tag on that commit.
5. From the clean tagged tree, generate release manifests to temporary paths so the manifest
   commands do not dirty the repository before the second clean-git check:

   ```bash
   PYTHONPATH=src python scripts/generate_paper_artifact_manifest.py \
     --output /tmp/paper_rt_artifact_manifest.csv \
     --require-clean-git
   PYTHONPATH=src python scripts/prepare_paper_archive_manifest.py \
     --output /tmp/paper_rt_archive_manifest.csv \
     --readme /tmp/paper_rt_archive_README.md \
     --include-raw-outputs \
     --require-clean-git
   ```

6. Confirm that `/tmp/paper_rt_artifact_manifest.csv` reports `git_dirty=false` and the release commit.
7. Copy the files listed by `/tmp/paper_rt_archive_manifest.csv` into the persistent archive.
8. Copy `/tmp/paper_rt_artifact_manifest.csv`, `/tmp/paper_rt_archive_manifest.csv`, and
   `/tmp/paper_rt_archive_README.md` into the persistent archive.
9. Include or cite the external 2S-ESS source/build provenance used for the Fortran timing rows.
10. Mint the archive DOI and replace draft DOI language in the manuscript with the final citation.
11. If submitting to GMD, convert the final text to the Copernicus/GMD template and keep the same figure/table assets.
