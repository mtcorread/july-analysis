# july-analysis

Analysis and visualisation toolkit for outputs of the JUNE agent-based model.

Reads `simulation_events.h5` and renders epidemiological summaries: incidence curves, encounter-channel contribution, R_eff, network structure, profile-facet breakdowns, and a self-contained HTML report.

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## CLI

```bash
july-analyze inspect    --events runs/<id>/simulation_events.h5
july-analyze report     --events runs/<id>/simulation_events.h5 [--world worlds/world.h5] -o report.html
july-analyze to-parquet --events runs/<id>/simulation_events.h5 -o parquet_dir/
july-analyze collect-runs runs/ -o sweep.csv
```

`--events` accepts either a merged `simulation_events.h5` file or the run directory containing it.

## Library

```python
from july_analysis import EventStore

store = EventStore("runs/<id>/simulation_events.h5")
infections = store.infections()
encounters = store.coordinated_encounters()
```

See `src/july_analysis/io.py` for the full `EventStore` API.

## Tests

The smoke test runs end-to-end against a real events file:

```bash
JULY_TEST_HDF5=/path/to/simulation_events.h5 pytest
```

Without `JULY_TEST_HDF5` set, it skips.

## License

MIT — see [LICENSE](LICENSE).
