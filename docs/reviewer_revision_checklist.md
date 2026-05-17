# Reviewer Revision Checklist

This file records the review package in `/Users/thl/Downloads/opus_r1.pdf` and tracks
how each issue is addressed in the manuscript and benchmark workflow.  The working
target is a GMD-style model-development paper: py2sess is a Python platform for
prepared-input 2S-ESS forward and differentiable RT benchmarks, not a new RT
approximation or an operational OCO-3 aerosol product.

Status key:

- `done`: addressed in code, manuscript, or figures.
- `partial`: partly addressed; remaining work is explicit.
- `blocked`: requires external input or data not currently in this repository.
- `waived`: intentionally not addressed in this revision.

## Blue-Ink Author Decisions

These items are transcribed from the handwritten notes in `opus_r1.pdf`.  They override
the synthetic reviewer/editor priorities where they conflict.

| ID | Blue-ink direction | Decision |
| --- | --- | --- |
| B1 | Report A100 GPU results where available; do not over-emphasize cloud-provider context. | done: abstract now reports the representative A100 VJP times; Tesla T4 rows remain only as secondary figure/table context. |
| B2 | Check whether a Fortran Jacobian speed was already collected. | partial: checked-in compact Fortran Jacobian fixtures contain radiance/Jacobian references but no timing field; matched Fortran analytic-Jacobian timing remains external-blocked, and the manuscript now adds hand-coded RT Jacobian context rather than implying a matched speed comparison. |
| B3 | Do not make the thermal FO source convention confusing. | done: reduced to one explicit Fortran-reference convention sentence plus the small diagnostic impact. |
| B4 | Clarify that fixed delta-M in VJPs is a benchmark design choice, not a py2sess AD limitation. | done: manuscript now frames gradients as prepared-input derivatives and reports the fixed-`f` versus differentiable-`f(g)` diagnostic. |
| B5 | Rename all paper-facing O$_2$ A-band labels to `Solar`, not `UV`. | done: manuscript and plot labels use `Solar`; internal case codes can remain `UV` for compatibility. |
| B6 | Remove the `Timed Python quantity` and `Timed Fortran quantity` rows from the full-spectrum settings table. | done: timing-scope differences are described in prose instead. |
| B7 | Avoid deep thread/process discussion. | done: only the single-threaded NumPy/BLAS setting is stated. |
| B8 | Evaluate JAX only if the project is not too large; otherwise stay with PyTorch and give a fair reason. | done: no JAX backend is added; manuscript explains that a fair JAX benchmark requires a separate functional solver implementation. |
| B9 | Do not add Jacobian memory profiling. | waived: explicitly excluded by author instruction. |
| B10 | For `torch.compile`, mention only if relevant; do not force timing if it is not ready. | done: benchmark driver now has optional `--torch-compile`; thermal compiled rows can be rerun, while Solar compiled rows are explicitly skipped until the NumPy cached geometry precompute is removed. |
| B11 | Improve gradient validation with a better/mid-scale case. | done: added sampled 1000-wavelength, 50-layer gradient checks. |
| B12 | Fit the empirical timing model only if the numbers are reasonable; otherwise keep qualitative discussion. | done: removed the Eq. 16 fit table and now uses the equation only to define the swept dimensions. |
| B13 | Do not increase timing repeats to 20+. | waived: current paper keeps five-run timing assets and reports ranges/standard deviations. |
| B14 | Do not report FLOPs. | waived: FLOP accounting is not useful for this mixed Python/BLAS/PyTorch/CUDA benchmark scope. |
| B15 | OCO-3 single-case result is not enough; continue improving/expanding rather than presenting weak validation. | partial: manuscript currently demotes the existing case and removes product-level claims; stronger multi-granule OCO results remain separate follow-up work before making aerosol conclusions. |
| B16 | Research EOF use, averaging-kernel/DFS, and aerosol model choices before strengthening OCO-3 claims. | partial: recorded as OCO follow-up; not mixed into the RT benchmark revision until the real-data case is scientifically stable. |

## Editor Consensus

| ID | Required revision | Status | Action |
| --- | --- | --- | --- |
| E1 | Add Fortran analytic Jacobian timing context. | blocked | Keep the paper from claiming a practical speed comparison with analytic Fortran Jacobians until an external 2S-ESS analytic-linearization timing is available. Add a paper-facing limitation and follow-up protocol. |
| E2 | Reframe or expand OCO-3 demonstration. | partial | Current manuscript demotes the existing case and removes product-level claims. Author preference is to expand to stronger multi-granule results before using OCO-3 as a real retrieval result. |
| E3 | Add Jacobian memory profiling. | waived | Waived by author instruction. Do not add memory profiling to this revision. |
| E4 | Add gradient validation at medium scale. | done | Extend gradient-validation summary to sampled 1000-wavelength, 50-layer solar and thermal cases. |
| E5 | Add qualitative comparison with alternative Python/differentiable RT tools. | waived | Removed the broad tool-comparison table per author blue-ink preference; manuscript now discusses only analytic RT Jacobian context needed for the benchmark claims. |
| E6 | Rename the misleading `UV` benchmark label. | done | Use `Solar` or solar-scattering labels in manuscript/figures; keep internal case codes only where needed for compatibility. |
| E7 | Discuss delta-M chain rule limitation. | done | Add a representative fixed-`f` versus differentiable-`f(g)` sensitivity diagnostic and manuscript text. |

## Reviewer 1: Atmospheric RT Physics

| ID | Comment | Status | Action |
| --- | --- | --- | --- |
| R1-M1 | Missing analytic Fortran Jacobian comparison. | blocked | External Fortran source/driver is not checked into this repo. Manuscript must explicitly state this is not a matched analytic-Jacobian speed comparison and provide a follow-up timing protocol. |
| R1-M2 | Thermal FO source delta-M convention needs clarification. | done | Clarify that the paper follows the exported Fortran benchmark convention and add a sensitivity/limitation sentence rather than implying universal convention. |
| R1-M3 | Delta-M truncation factor held fixed in VJP benchmarks. | done | Prominently state fixed prepared-input convention; add quantitative fixed-`f` versus differentiable-`f(g)` diagnostic. |
| R1-M4 | OCO-3 spatial correlations are alarming. | done | Demote OCO-3 to workflow demonstration and remove validation/skill language from abstract/conclusion. |
| R1-m1 | `UV` label is misleading for O2 A-band. | done | Rename paper-facing labels to `Solar` or solar-scattering. |
| R1-m2 | Add residual panel to spectrum validation figure. | done | Regenerated the full-spectrum comparison figure with compact Python-minus-Fortran relative-difference panels. |
| R1-m3 | Clarify Fortran timed quantity. | done | State Fortran rows are external driver RT-module timings with different scope from Python component timings. |
| R1-m4 | Justify spectral relative-error floor. | done | Explain the numerical floor prevents line-center/small-radiance singular ratios. |
| R1-m5 | Describe pentadiagonal BVP solver. | done | Add concise algorithmic description: banded layer-neighbor system solved by an autograd-compatible pentadiagonal path without pivoting in the normal benchmark path. |
| R1-m6 | Acknowledge HG aerosol phase limitation. | done | OCO-3 workflow text now treats HG as a simple effective model and not a dust validation model. |
| R1-m7 | Discuss NumPy thread/process context. | done | State single-threaded BLAS/OpenMP settings for NumPy benchmark rows. |

## Reviewer 2: Scientific Computing and AD

| ID | Comment | Status | Action |
| --- | --- | --- | --- |
| R2-M1 | Add JAX backend or explain PyTorch choice. | done | Add discussion that PyTorch is the implemented backend; JAX is future work because it would require a separate functional solver implementation. |
| R2-M2 | Add memory profiling. | waived | Waived by author instruction. |
| R2-M3 | Evaluate `torch.compile` or explain. | done | Added `--torch-compile`; thermal rows can be timed, and Solar rows are skipped with a concrete NumPy geometry-precompute reason. |
| R2-M4 | Gradient validation scale is insufficient. | done | Add sampled 1000-wavelength, 50-layer gradient checks. |
| R2-M5 | Empirical scaling equation is decorative. | done | Removed the forward-scaling fit table; keep only qualitative interpretation and endpoint VJP slopes. |
| R2-m1 | Compare with PythonicDISORT, pydisort, and DART. | waived | Removed the comparison table per author preference; keep the paper focused on 2S-ESS and hand-coded Jacobian context. |
| R2-m2 | Timing statistics from only five repeats. | waived | Author note says not to increase repeats. Keep five-run paper assets and report ranges/standard deviations without over-precision. |
| R2-m3 | No profiling of VJP cost components. | waived | Author note prefers not to do this. The VJP decomposition is framed as bookkeeping, not profiler attribution. |
| R2-m4 | Specify versions and hardware. | partial | Add hardware/threading text; exact package/CUDA versions should be pulled from final artifact manifest before submission. |
| R2-m5 | Clarify M2 Pro/MPS. | done | State MPS is out of scope; requested paper backends are NumPy, Torch CPU, and Torch CUDA. |
| R2-m6 | Describe pentadiagonal solver. | done | Covered with R1-m5. |
| R2-m7 | Describe vectorized wavelength batching. | done | Add memory-layout/batch-dimension description. |
| R2-m8 | Consider FLOPs. | waived | FLOP accounting is not needed for the current GMD benchmark and would be solver-path dependent. |

## Reviewer 3: Remote Sensing and Retrieval

| ID | Comment | Status | Action |
| --- | --- | --- | --- |
| R3-M1 | Nine OCO-3 soundings are insufficient. | partial | Existing weak case is demoted, but author preference is to expand to multiple granules rather than relying on the nine-sounding result. |
| R3-M2 | ABSCO-lite spectroscopy limitation is fundamental. | done | State sensitivity/retrieval results are conditional on open ABSCO-lite; strong-CO2 band is diagnostic only. |
| R3-M3 | Fixed aerosol profile and phase function dominate retrieval. | done | Present fixed aerosol assumptions as limitations; do not claim aerosol product validation. |
| R3-M4 | AERONET comparison is not validation. | done | Remove AERONET quantitative validation language unless strict collocation is available. |
| R3-m1 | EOF nuisance degeneracy. | done | Discuss empirical spectral nuisance terms as degenerate with AOD. |
| R3-m2 | Add DFS/averaging-kernel information for retrieval. | done | OCO-3 was demoted to workflow demonstration, so product-level DFS/AK claims are removed rather than introduced with incomplete diagnostics. |
| R3-m3 | Clarify retrieval algorithm. | done | Add compact cost-function/optimizer description for the workflow demonstration. |
| R3-m4 | Band residual context. | done | State residuals combine spectroscopy/model/noise and are not product-quality residuals. |
| R3-m5 | State OCO-3 granule date/location. | done | Keep 24 June 2022 and approximate location in the workflow section. |
| R3-m6 | Error bars on fixed-AOD scan curves. | waived | OCO-3 fixed-AOD scan is not central after demotion. |
| R3-m7 | Flag MERRA-2 prior/reference circularity. | done | State MERRA-2 profile use makes MERRA-2 AOD a consistency check, not independent validation. |
| R3-m8 | Define quality screening. | done | The OCO-3 subsection now lists the finite-data, quality-flag, land-fraction, O2-band RMS, L1B-L2 matching, and patch-distance screens. |
| R3-m9 | Report OCO-3 retrieval computational cost. | partial | Optional after OCO-3 demotion; can be added from retrieval summary CSV if retained. |

## Remaining Blocking Work Before Submission

1. Obtain or explicitly defer matched external 2S-ESS Fortran analytic-Jacobian timing.
2. Fill final software/hardware version manifest and archive DOI.
3. Decide whether OCO-3 remains in the main paper as a short workflow subsection or moves to an appendix.
