"""Training, on Ray, for a 24-hour-ahead PV forecast of hourly mean power.

Two forecasters, the first used whenever it can answer:

1. Weather: the day-ahead irradiance forecast for the target hour (Open-Meteo,
   global horizontal irradiance) times the median ratio of measured power to
   forecast irradiance for the same UTC hour over the last 28 complete days.
   The per-hour ratio absorbs the panels' orientation and shading.
2. Clear-sky index (fallback, no weather): the median ratio of measured power
   to clear-sky irradiance for the same hour over the last 14 complete days,
   blended with the forecasting day's own ratio so far.

Backtest (ODE workspace, pv_backtest_weather*.csv, inverter, daylight hours as
the evaluation scores them): about 63 W MAE for (1) against about 75-85 W for
(2), over Aug/Sep 2025 and Jul/Aug 2026.

Which forecast is used, and why that is honest in time:

* For a target hour already past (an evaluation replay, or catching up), the
  forecast comes from Open-Meteo's Previous Runs archive as
  `shortwave_radiation_previous_day2` -- the value forecast about 48 h before
  the target hour, so at least ~24 h before the moment the operator forecasts.
* For a target hour still in the future (a live deployment), it comes from the
  ordinary forecast API, which is by construction issued before the operator
  forecasts.

The operator needs outbound HTTPS to open-meteo.com for this. Where that fails,
every forecast falls back to (2) rather than to nothing.
"""

import datetime
import functools
import math
import typing

import ray
import requests
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger
from operator_lib.util.logger import logger


# How much history one training pass reads: the weather lookback plus margin.
TRAINING_WINDOW = datetime.timedelta(days=45)

LOOKBACK_DAYS = 14          # clear-sky fallback
WX_LOOKBACK_DAYS = 28       # weather ratio
RETAIN_DAYS = max(LOOKBACK_DAYS, WX_LOOKBACK_DAYS) + 3
QUANTILE = 0.5
BLEND = 0.85
# Below this clear-sky irradiance (W/m2) an hour is night: forecast zero.
MIN_CLEAR_SKY = 20.0
# Below this forecast irradiance (W/m2) an hour does not enter the weather ratio.
MIN_GHI = 50.0

ARCHIVE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LIVE_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_VARIABLE = "shortwave_radiation_previous_day2"
LIVE_VARIABLE = "shortwave_radiation"
HTTP_TIMEOUT_S = 60

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


# ---------------------------------------------------------------- weather ----

def _parse_ghi(payload: dict, variable: str) -> typing.Dict[int, float]:
    """Open-Meteo stamps irradiance at the end of the hour it averages, so the
    value at HH:00 belongs to the bucket starting at HH-1."""
    out: typing.Dict[int, float] = {}
    hourly = payload.get("hourly", {})
    for t, v in zip(hourly.get("time", []), hourly.get(variable, [])):
        if v is None:
            continue
        ts = datetime.datetime.fromisoformat(t).replace(tzinfo=datetime.timezone.utc)
        out[hour_index(ts) - 1] = float(v)
    return out


def fetch_archive_ghi(lat: float, lon: float, start: datetime.date, end: datetime.date) -> typing.Dict[int, float]:
    response = requests.get(ARCHIVE_URL, timeout=HTTP_TIMEOUT_S, params={
        "latitude": lat, "longitude": lon, "timezone": "GMT", "hourly": ARCHIVE_VARIABLE,
        "start_date": start.isoformat(), "end_date": end.isoformat()})
    response.raise_for_status()
    return _parse_ghi(response.json(), ARCHIVE_VARIABLE)


def fetch_live_ghi(lat: float, lon: float) -> typing.Dict[int, float]:
    response = requests.get(LIVE_URL, timeout=HTTP_TIMEOUT_S, params={
        "latitude": lat, "longitude": lon, "timezone": "GMT", "hourly": LIVE_VARIABLE,
        "forecast_days": 3})
    response.raise_for_status()
    return _parse_ghi(response.json(), LIVE_VARIABLE)


class WeatherSource:
    """The irradiance forecast for one target hour, as it was known 24 h before it.

    Not part of the model: it holds caches and does network I/O. A failed fetch
    is remembered per month (archive) or per hour (live), so a missing network
    costs one attempt rather than one per message.
    """

    def __init__(self, latitude: float, longitude: float) -> None:
        self.latitude, self.longitude = float(latitude), float(longitude)
        self._archive: typing.Dict[int, float] = {}
        self._archive_tried: typing.Set[typing.Tuple[int, int]] = set()
        self._live: typing.Dict[int, float] = {}
        self._live_fetched_for: typing.Optional[int] = None

    def ghi_for(self, target_hour: int) -> typing.Optional[float]:
        wall = datetime.datetime.now(datetime.timezone.utc)
        target_dt = EPOCH + target_hour * HOUR
        if target_dt < wall - datetime.timedelta(days=3):
            # A past target: the archived forecast issued ~48 h before it.
            if target_hour not in self._archive:
                key = (target_dt.year, target_dt.month)
                if key not in self._archive_tried:
                    self._archive_tried.add(key)
                    first = datetime.date(target_dt.year, target_dt.month, 1)
                    last = (first + datetime.timedelta(days=32)).replace(day=1) - datetime.timedelta(days=1)
                    last = min(last, (wall - datetime.timedelta(days=3)).date())
                    try:
                        self._archive.update(fetch_archive_ghi(
                            self.latitude, self.longitude, first - datetime.timedelta(days=1), last))
                    except Exception as ex:
                        logger.warning(f"weather archive fetch for {key} failed, falling back: {ex}")
            return self._archive.get(target_hour)
        # A future target: the current forecast, refreshed at most once an hour.
        now_hour = hour_index(wall)
        if self._live_fetched_for != now_hour:
            self._live_fetched_for = now_hour
            try:
                self._live.update(fetch_live_ghi(self.latitude, self.longitude))
            except Exception as ex:
                logger.warning(f"live weather fetch failed, falling back: {ex}")
        return self._live.get(target_hour)


# ------------------------------------------------------------------ model ----

class PVDayAheadModel(PythonModel):
    """What MLflow registers and op.py loads.

    forecast_combined() is the contract op.py uses; predict() wraps it for pyfunc
    callers with a payload of {"hourly": {...}, "ghi": {...}, "now": datetime}.
    """

    def __init__(self, hourly: typing.Dict[int, float], latitude: float, longitude: float,
                 ghi: typing.Optional[typing.Dict[int, float]] = None,
                 lookback_days: int = LOOKBACK_DAYS, wx_lookback_days: int = WX_LOOKBACK_DAYS,
                 quantile: float = QUANTILE, blend: float = BLEND,
                 min_clear_sky: float = MIN_CLEAR_SKY, min_ghi: float = MIN_GHI) -> None:
        self.hourly = {int(k): float(v) for k, v in hourly.items()}
        self.ghi = {int(k): float(v) for k, v in (ghi or {}).items()}
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.lookback_days = int(lookback_days)
        self.wx_lookback_days = int(wx_lookback_days)
        self.quantile = float(quantile)
        self.blend = float(blend)
        self.min_clear_sky = float(min_clear_sky)
        self.min_ghi = float(min_ghi)

    def _cs(self, hour: int) -> float:
        return clear_sky(hour, self.latitude, self.longitude)

    def forecast_weather(self, hourly: typing.Dict[int, float], ghi: typing.Dict[int, float],
                         now_hour: int) -> typing.Optional[float]:
        """Per-hour ratio of power to forecast irradiance x the target's forecast; None if it cannot answer."""
        target = now_hour + 24
        if self._cs(target) <= self.min_clear_sky:
            return 0.0
        g_target = ghi.get(target)
        if g_target is None:
            return None
        ratios = []
        for j in range(2, self.wx_lookback_days + 2):
            h = target - 24 * j
            v, g = hourly.get(h), ghi.get(h)
            if v is not None and g is not None and g > self.min_ghi:
                ratios.append(v / g)
        if not ratios:
            return None
        return max(_quantile(ratios, self.quantile) * g_target, 0.0)

    def forecast(self, hourly: typing.Dict[int, float], now_hour: int) -> float:
        """Clear-sky-index fallback for the hour 24 h after now_hour, from complete hours only."""
        target = now_hour + 24
        cs_target = self._cs(target)
        if cs_target <= self.min_clear_sky:
            return 0.0
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

    def forecast_combined(self, hourly: typing.Dict[int, float], ghi: typing.Dict[int, float],
                          now_hour: int) -> typing.Tuple[float, str]:
        p = self.forecast_weather(hourly, ghi, now_hour)
        if p is not None:
            return p, "weather"
        return self.forecast(hourly, now_hour), "clear_sky"

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        value, _ = self.forecast_combined(payload.get("hourly", self.hourly), payload.get("ghi", self.ghi),
                                          hour_index(payload["now"]))
        return value


# --------------------------------------------------------------- training ----

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


def train_model(logger_: TrainMlflowLogger, latitude: float, longitude: float) -> typing.Optional[PythonModel]:
    """Read the history and the archived forecasts for it, check both forecasters, hand back the model."""
    with logger_.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        return None
    with logger_.trace("hourly means"):
        hourly = _hourly_means(datasets)
    if not hourly:
        return None

    first, last = min(hourly), max(hourly)
    ghi: typing.Dict[int, float] = {}
    with logger_.trace("weather"):
        try:
            ghi = fetch_archive_ghi(latitude, longitude,
                                    (EPOCH + first * HOUR).date(), (EPOCH + last * HOUR).date())
        except Exception as ex:
            logger.warning(f"weather archive fetch for training failed, model will use the clear-sky fallback: {ex}")
    ghi = {h: g for h, g in ghi.items() if h <= last}

    model = PVDayAheadModel(hourly, latitude, longitude, ghi=ghi)

    # In-sample check, honest in time: each hour of the last 14 days is forecast
    # from the hours complete 24 h before it, as the operator would.
    with logger_.trace("check"):
        err_combined, err_cs, used_weather = [], [], 0
        for target in range(last - 14 * 24 + 1, last + 1):
            if target not in hourly:
                continue
            past = {h: v for h, v in hourly.items() if h < target - 24}
            p, source = model.forecast_combined(past, ghi, target - 24)
            used_weather += source == "weather"
            err_combined.append(abs(p - hourly[target]))
            err_cs.append(abs(model.forecast(past, target - 24) - hourly[target]))

    keep_from = last - 24 * RETAIN_DAYS
    model.hourly = {h: v for h, v in hourly.items() if h >= keep_from}
    model.ghi = {h: g for h, g in ghi.items() if h >= keep_from}

    logger_.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "lookback_days": LOOKBACK_DAYS,
        "wx_lookback_days": WX_LOOKBACK_DAYS,
        "weather_variable": ARCHIVE_VARIABLE,
        "quantile": QUANTILE,
        "blend": BLEND,
        "latitude": latitude,
        "longitude": longitude,
    })
    metrics = {"history_hours": float(len(hourly)), "weather_hours": float(len(ghi))}
    if err_combined:
        metrics["train_check_mae_last14d"] = sum(err_combined) / len(err_combined)
        metrics["train_check_mae_last14d_clearsky"] = sum(err_cs) / len(err_cs)
        metrics["train_check_weather_share"] = used_weather / len(err_combined)
    logger_.log_metrics(metrics)
    return model
