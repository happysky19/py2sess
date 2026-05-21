This directory is the local staging area for RtRetrievalFramework static inputs
used by the OCO-3 paper replay scripts.  The large HDF5 inputs are intentionally
ignored by git.

- `l2_aerosol_combined.h5`: L2FP aerosol optical-property table used by
  `--aerosol-treatment oco-l2fp`.
- `l2_oco3_eof.h5`: OCO-3 static EOF basis used only when
  `--eof-treatment oco3-static` is enabled.

They were copied from the local RtRetrievalFramework checkout so the replay
scripts do not depend on `/tmp/RtRetrievalFramework`.
