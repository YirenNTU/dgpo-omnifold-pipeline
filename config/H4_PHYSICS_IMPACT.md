# Physics impact: production-calibration diagnostic and full-closure prerequisites

Status: production calibration diagnostic implemented and locally tested.
Full B_i/C_ij, response and unfolding closure is NOT implemented or claimed.

The inspected production exporter `export_evenet_qi_inputs.py:evenet_tau_pair`
constructs tau vectors at E=91.2/2, m=1.777, then calls
`common.post_calibrate_tau_tau`. That function uses CM_ENERGY=91.25 and
TAU_MASS=1.77 and forces the two reconstructed momenta to be exactly opposite.
These existing conventions are preserved, not silently repaired. Post-
calibration opening-angle closure is therefore imposed, not learned.

## Run now, no GPU or fresh predictions required

```bash
shifter python3 -u scripts/diagnose_h4_calibration_impact.py \
  --pretrained10pct /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_pretrain10pct_raw \
  --step1110 /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_1110 \
  --pretrainedfull /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_fullpretrain_raw \
  --output /pscratch/sd/y/yiren/Ztautau/h4_calibration_impact
```

This calls the actual production calibration on all saved candidates with
equal weights. Panels, event indices and sampling settings must match exactly.
Reports pre/post opening deficits, calibration angular displacement, direction
errors against the SAME saved angular target, separate errors to a projected
target, direction-component mean bias, and event-cluster SE. It preserves all
events and rejects invalid inputs; no best-of-K selection. An exactly matched
unprojected target can get worse after calibration; tests explicitly verify
that this effect is not hidden by replacing the reference with its projection.

Outputs: report.json and event_direction_errors.npz (one value/event/arm/stage).
No W&B run or production outputs are altered. Reports explicitly contain
physics_closure_complete=false. Angular reconstruction error is not a proper
distribution score or final spin-observable bias. A successful calibration
diagnostic cannot establish full physics closure.

## Required to finish the requested full test

The local checkout does not contain the `quantum.observables_builder` imported
by the production exporter or the QIProcessor/unfolding project. Saved coverage
panels contain direction targets, but not complete truth tau p4, channels,
analyzing powers, event weights or original source event IDs. Pool row numbers
must NOT be used as Parquet source row numbers.

Need the NERSC quantum/QIProcessor source path, active analysis config, and an
evaluation Parquet with truth tau p4, visible p4/decay inputs, channel labels,
analyzing powers, source IDs and event weights. Verify pretraining overlap.
Use the actual production observable builder, selection and channel conventions;
do not substitute +/-3 or +/-9 moment formulas without analyzing-power checks.

Then run all three raw checkpoints on a common evaluation set and export pre-
and post-calibration observables. Build responses on event-disjoint simulation
from pseudo-data; all candidates from one event stay in one split. Follow the
existing production binning/unfolding settings without tuning to pseudo-data.
Report B_i/C_ij and derived QI quantities, full covariance, weighted selection
efficiencies, migration, closure bias and event-bootstrap uncertainty, including
alternative truth-shape stress tests. MC closure is a test of that simulation
and method, not proof of robustness on data. Compare bias to a predeclared
physics systematic budget; no arbitrary threshold is selected here.
