# ode-ablation-pv-l2-rerun-1

## Summary: 24-hour-ahead PV forecast, and how it was approached

**Task.** Forecast photovoltaic generation 24 hours ahead as hourly mean power
in watts, evaluated by the developer's `evaluation.yaml`: MAE at 1 h resolution,
threshold 30 W, on a test window of September 2026 (training data strictly
before 2026-09-01).

**Result.** The most recent evaluation (commit `e89bf32`, MLflow run
`5cb56a6a117140a29296f31622024ce1`) scored **MAE 65.0 W** over 391 hourly
buckets, so it does **not** meet the 30 W threshold. The first, weather-free
version scored 71.2 W. Running the same commit again gives the same number.

### 1. Finding the data (real data before anything else)

- Searched the platform ontology (aspect *Generation*, function *Get-Power*)
  instead of matching on device names. Two real devices measure the same PV
  system:
  - **Inverter** (APSystems DS3-S), `root.powerTotal`, 60 s readings, only in
    daylight. This is the target the developer chose.
  - **Plug** ("Leiste PV", Nous A5T), `sensor.ENERGY.Power`, ~10 s readings,
    reporting around the clock.
- No irradiance device, weather import or deployable weather import type exists
  on the platform. No simulation was needed, because real PV history goes back
  to 2022.

### 2. Reading how the evaluation actually scores

Operator Lib's replay code (`util/op_ml.py`, v1.8.1) settled the operator's
contract:

- The actual value for an hour is the plain mean of the raw input readings in
  that UTC hour. Hours with no readings are not scored, so for the inverter
  **only daylight hours count**, which is the hardest version of the metric.
- A prediction is scored in the bucket its `result_time` falls into. Every
  message's `infer()` therefore returns `result_time = message time + 24 h`.
- No retraining happens inside the replay. State kept on the operator instance
  may be updated from incoming messages, which is how the forecast stays current.

### 3. Backtesting in a notebook cell before spending a launch

Each candidate forecaster was replayed over past months (Aug/Sep 2023 and
2025, Jul/Aug 2026) using only information available 24 h ahead, and scored
exactly as the evaluation scores. On the inverter, daylight hours:

| Forecaster | Typical MAE |
|---|---|
| Hour-of-day median, last K days | ~75–80 W |
| Clear-sky index (median over 14 days, blended 85/15 with the forecasting day's own index) | ~72–85 W |
| Day-ahead irradiance forecast × per-hour power/irradiance ratio (28 days) | **~62 W** |
| Same, with irradiance on the panel plane (orientation fitted on 2025, tested on 2026) | ~62 W |
| Gradient-boosted model, absolute-error loss, weather features | ~62 W |

Conclusion: PV history alone cannot predict tomorrow's clouds. A day-ahead
weather forecast is what helps, and even that levels off around 60 W. More
modelling of the PV side changed the result by a watt or two.

### 4. Bringing in weather without mislabelling shared inventory

Two ways in were ruled out:

- **Platform import:** none exists for weather, and none can be deployed.
- **Simulated device (MOSES):** no simulatable device type has an irradiance
  service. Publishing W/m² under a power (W) service would have mislabelled
  shared inventory and made it show up as PV generation. The correct platform
  fix, an irradiance device type with a `time_path` in the device repository,
  is outside this repository.

So **the operator fetches the forecast itself** from Open-Meteo (`training.WeatherSource`):

- **For a past target hour** (evaluation replay, catching up): it uses the
  Previous Runs archive variable `shortwave_radiation_previous_day2`. That value
  was forecast about 48 h before the target hour, so it was published at least
  about 24 h before the operator makes its forecast. Using it is not hindsight.
- **For a future target hour** (live deployment): it uses the ordinary forecast API.
- Open-Meteo stamps irradiance at the end of the hour it averages, so values are
  shifted back one hour to line up with the evaluation's buckets.
- Only the target hour's forecast is stored at each step, so the state never
  holds a forecast that was not yet known at that moment.
- If the fetch fails, the operator falls back to the clear-sky-index model
  rather than producing nothing.
- Each output carries `"source": "weather" | "clear_sky"`, so
  `evaluation/predictions.csv` shows which forecaster answered each hour. A
  separate check confirmed that the archive holds a forecast for all 365
  September daylight hours.

### 5. What the operator does now

- `op.py`: for every message, adds the value to its hour's running mean. Once
  per hour it computes the forecast for the hour 24 h later and returns it
  stamped with that hour.
- `training.py`: reads 45 days of history and the archived forecasts for them,
  builds the model (parameters plus seed history), and logs:
  - an in-sample check over the last 14 days, `train_check_mae_last14d`:
    57.6 W with weather vs 92.3 W without;
  - `weather_hours`.
- Config: `latitude`/`longitude` default to Leipzig (51.34 N, 12.37 E), an
  assumption consistent with observed sunrise times. Set the real coordinates if
  they differ. `use_weather` toggles the Open-Meteo dependency.

### 6. Limits and open points

- **30 W looks out of reach** for this target with forecasts genuinely issued a
  day ahead, on the evidence of the backtests and two evaluations. Whether the
  threshold suits daylight-only scoring of a ~650 W system is the developer's
  decision.
- **External dependency:** the operator now needs outbound HTTPS to
  open-meteo.com. Without it, it silently uses the weaker clear-sky model.
  `weather_hours` = 0 in a run shows that this happened.
- **Location is an assumption.** It was not given; it was inferred.
- **Confirming the 65 W:** the definitive check that every September hour used
  weather is the `source` column of `predictions.csv` in the run's MLflow
  artifacts.
- **Backtest scripts are not in the repository.** They were run in the
  developer's workspace, and their outputs (`pv_backtest*.csv` and the Open-Meteo
  CSVs) are kept there.

---

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.1". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-pv-l2-rerun-1". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-pv-l2-rerun-1:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.1", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
