"""The operator: a 24-hour-ahead forecast of hourly mean PV power, in watts.

For every message it adds the value to the hourly mean of the hour it falls in,
and answers with the forecast for the hour 24 h later, stamped with that hour's
time (result timestamp = message time + 24 h). The forecast only changes when
an hour completes, so it is computed once per hour and reused.

The forecast uses the day-ahead irradiance forecast for the target hour when
it can get one (see training.WeatherSource) and the clear-sky-index model
otherwise; the output says which in "source". The hourly means and the
forecasts used live on the operator instance and are seeded from the model, so
the forecast keeps using fresh data between retrains.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import RETAIN_DAYS, WeatherSource, hour_index, train_model


HORIZON = datetime.timedelta(hours=24)


class CustomConfig(Config):
    # Retrain at most this often, in seconds.
    retrain_after_s = 86400
    # Location of the PV system, for the clear-sky curve and the weather
    # forecast. The default is Leipzig, an assumption consistent with the
    # observed sunrise times; set the real coordinates if they differ.
    latitude = 51.34
    longitude = 12.37
    # Fetch the day-ahead irradiance forecast from open-meteo.com. Off means the
    # clear-sky-index model only.
    use_weather = True


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._seeded_from: typing.Optional[int] = None
        self._sums: typing.Dict[int, float] = {}
        self._counts: typing.Dict[int, int] = {}
        self._hourly: typing.Dict[int, float] = {}
        self._ghi: typing.Dict[int, float] = {}
        self._current_hour: typing.Optional[int] = None
        self._cached_for: typing.Optional[int] = None
        self._cached_prediction: typing.Optional[float] = None
        self._cached_source: typing.Optional[str] = None
        self._weather: typing.Optional[WeatherSource] = None
        super().init(*args, **kwargs)

    @staticmethod
    def _python_model(model):
        if model is None:
            return None
        try:
            return model.unwrap_python_model()
        except AttributeError:
            return model

    @staticmethod
    def _truthy(value) -> bool:
        return str(value).strip().lower() not in ("false", "0", "no", "off", "")

    def _weather_source(self) -> typing.Optional[WeatherSource]:
        if self._weather is None and self._truthy(self.config.use_weather):
            self._weather = WeatherSource(float(self.config.latitude), float(self.config.longitude))
        return self._weather

    def _seed(self, pm) -> None:
        """Take the model's history on first use and whenever the model changes."""
        if self._seeded_from == id(pm):
            return
        self._seeded_from = id(pm)
        for h, v in pm.hourly.items():
            if h not in self._counts:
                self._hourly.setdefault(h, v)
        for h, g in getattr(pm, "ghi", {}).items():
            self._ghi.setdefault(h, g)
        self._cached_for = None

    def _close_hours(self, now_hour: int) -> None:
        """Move every accumulated hour before now_hour into the complete means."""
        for h in [h for h in self._sums if h < now_hour]:
            self._hourly[h] = self._sums.pop(h) / self._counts.pop(h)
        horizon = now_hour - 24 * RETAIN_DAYS
        for store in (self._hourly, self._ghi):
            for h in [h for h in store if h < horizon]:
                del store[h]

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        pm = self._python_model(model)
        value = data.get("value")
        if pm is None or value is None:
            return None, None, None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, None, None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

        self._seed(pm)
        now_hour = hour_index(timestamp)
        if self._current_hour is None or now_hour != self._current_hour:
            self._close_hours(now_hour)
            self._current_hour = now_hour
        self._sums[now_hour] = self._sums.get(now_hour, 0.0) + value
        self._counts[now_hour] = self._counts.get(now_hour, 0) + 1

        if self._cached_for != now_hour:
            target = now_hour + 24
            source = self._weather_source()
            if source is not None and target not in self._ghi:
                g = source.ghi_for(target)
                if g is not None:
                    # Only the target hour is kept: the forecast as it was known now.
                    self._ghi[target] = g
            self._cached_prediction, self._cached_source = pm.forecast_combined(
                self._hourly, self._ghi, now_hour)
            self._cached_for = now_hour

        return timestamp + HORIZON, {
            "prediction": self._cached_prediction,
            # "weather" or "clear_sky": which forecaster produced this value.
            "source": self._cached_source,
        }, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger, float(self.config.latitude), float(self.config.longitude))

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= float(self.config.retrain_after_s)
