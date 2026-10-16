"""Training, on Ray, for a 24-hour-ahead PV forecast of hourly mean power.

The model is a clear-sky-index forecaster:

* every complete hour of history is divided by the clear-sky irradiance of that
  hour (Haurwitz model at the configured location), which removes the daily
  and seasonal shape and leaves a cloudiness index;
* the forecast for hour T is the median index of the same UTC hour over the
  last `lookback_days` complete days, blended with the index the forecasting
  day has shown so far, and multiplied back by the clear-sky value of hour T.

It was chosen by a backtest over past Septembers and August 2026 (ODE
workspace, pv_backtest.csv) against plain hour-of-day medians and quantiles.
It uses no weather data, because the platform carries none: day-to-day
cloudiness is not predictable from PV history alone, and that bounds how good
this can get.

The model object carries the parameters and a seed of recent hourly means. The
operator keeps the hourly means rolling from the messages it receives, so a
forecast always uses everything available at the moment it is made.
"""

import datetime
import functools
import math
import typing

import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads: the lookback plus margin, and enough
# for the in-sample check over the last two weeks.
TRAINING_WINDOW = datetime.timedelta(days=45)

LOOKBACK_DAYS = 14
QUANTILE = 0.5
BLEND = 0.85
# Below this clear-sky irradiance (W/m2) an hour is treated as night: its index
# is not used and its forecast is zero.
MIN_CLEAR_SKY = 20.0

HOUR = datetime.timedelta(hours=1)
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def hour_index(ts: datetime.datetime) -> int:
    """Whole UTC hours since the epoch: the same bucketing the evaluation uses."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=datetime.timezone.utc)
    return int((ts - EPOCH) // HOUR)


@functools.lru_cache(maxsize=8192)
def clear_sky(hour: int, lat_deg: float, lon_deg: float) -> float:
    """Mean Haurwitz clear-sky GHI over one UTC hour, in W/m2 (six sub-samples)."""
    lat = math.radians(lat_deg)
    start = EPOCH + hour * HOUR
    total = 0.0
    for k in range(6):
        t = start + datetime.timedelta(minutes=5 + 10 * k)
        doy = t.timetuple().tm_yday
        hr = t.hour + t.minute / 60.0
        g = 2 * math.pi / 365 * (doy - 1 + (hr - 12) / 24)
        decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
                - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
                - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
        eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                        - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
        ha = math.radians((hr * 60 + eqt + 4 * lon_deg) / 4 - 180)
        cz = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
        if cz > 0:
            total += 1098.0 * cz * math.exp(-0.057 / max(cz, 1e-3))
    return total / 6


def _quantile(values: typing.List[float], q: float) -> float:
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class PVDayAheadModel(PythonModel):
    """What MLflow registers and op.py loads.

    forecast() is the contract op.py uses; predict() wraps it for pyfunc callers
    with a payload of {"hourly": {hour_index: mean_W}, "now": datetime}.
    """

    def __init__(self, hourly: typing.Dict[int, float], latitude: float, longitude: float,
                 lookback_days: int = LOOKBACK_DAYS, quantile: float = QUANTILE,
                 blend: float = BLEND, min_clear_sky: float = MIN_CLEAR_SKY) -> None:
        self.hourly = {int(k): float(v) for k, v in hourly.items()}
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.lookback_days = int(lookback_days)
        self.quantile = float(quantile)
        self.blend = float(blend)
        self.min_clear_sky = float(min_clear_sky)

    def _cs(self, hour: int) -> float:
        return clear_sky(hour, self.latitude, self.longitude)

    def forecast(self, hourly: typing.Dict[int, float], now_hour: int) -> float:
        """Hourly mean power for the hour 24 h after now_hour, from complete hours only."""
        target = now_hour + 24
        cs_target = self._cs(target)
        if cs_target <= self.min_clear_sky:
            return 0.0

        # Same UTC hour on the last lookback_days complete days before today.
        indices, levels = [], []
        for j in range(2, self.lookback_days + 2):
            h = target - 24 * j
            v = hourly.get(h)
            if v is None:
                continue
            levels.append(v)
            cs = self._cs(h)
            if cs > self.min_clear_sky:
                indices.append(v / cs)
        if not indices:
            return max(_quantile(levels, self.quantile), 0.0) if levels else 0.0
        kt = _quantile(indices, self.quantile)

        # Today's index so far: complete hours of the forecasting day before now.
        day_start = now_hour - (now_hour % 24)
        num = den = 0.0
        for h in range(day_start, now_hour):
            v = hourly.get(h)
            cs = self._cs(h)
            if v is None or cs <= self.min_clear_sky:
                continue
            num += v
            den += cs
        if den > 0:
            kt = self.blend * kt + (1 - self.blend) * (num / den)
        return max(kt * cs_target, 0.0)

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        hourly = payload.get("hourly", self.hourly)
        return self.forecast(hourly, hour_index(payload["now"]))


def _hourly_means(datasets) -> typing.Dict[int, float]:
    sums: typing.Dict[int, float] = {}
    counts: typing.Dict[int, int] = {}
    for ref in datasets:
        dataset = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
        for batch in dataset.iter_batches(batch_size=65536, batch_format="pandas"):
            if "value" not in batch or "time" not in batch:
                continue
            for t, v in zip(batch["time"], batch["value"]):
                if v is None or v != v:
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                ts = t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
                h = hour_index(ts)
                sums[h] = sums.get(h, 0.0) + v
                counts[h] = counts.get(h, 0) + 1
    return {h: sums[h] / counts[h] for h in sums}


def train_model(logger: TrainMlflowLogger, latitude: float, longitude: float) -> typing.Optional[PythonModel]:
    """Read the history, check the forecaster on its own last two weeks, hand back the model."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        return None

    with logger.trace("hourly means"):
        hourly = _hourly_means(datasets)
    if not hourly:
        return None

    model = PVDayAheadModel(hourly, latitude, longitude)

    # In-sample check, honest in time: each hour of the last 14 days is forecast
    # from the hours that were complete 24 h before it, as the operator would.
    with logger.trace("check"):
        last = max(hourly)
        errors = []
        for target in range(last - 14 * 24 + 1, last + 1):
            if target not in hourly:
                continue
            past = {h: v for h, v in hourly.items() if h < target - 24}
            errors.append(abs(model.forecast(past, target - 24) - hourly[target]))

    # Keep only what the forecaster can reach.
    keep_from = last - 24 * (LOOKBACK_DAYS + 3)
    model.hourly = {h: v for h, v in hourly.items() if h >= keep_from}

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "lookback_days": LOOKBACK_DAYS,
        "quantile": QUANTILE,
        "blend": BLEND,
        "latitude": latitude,
        "longitude": longitude,
    })
    metrics = {"history_hours": float(len(hourly))}
    if errors:
        metrics["train_check_mae_last14d"] = sum(errors) / len(errors)
    logger.log_metrics(metrics)
    return model
