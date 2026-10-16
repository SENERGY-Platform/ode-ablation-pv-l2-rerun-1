"""The operator: a 24-hour-ahead forecast of hourly mean PV power, in watts.

For every message it adds the value to the hourly mean of the hour it falls in,
and answers with the forecast for the hour 24 h later, stamped with that hour's
time (result timestamp = message time + 24 h). The forecast only changes when
an hour completes, so it is computed once per hour and reused.

The hourly means live on the operator instance and are seeded from the model's
own recent history, so the forecast keeps using fresh data between retrains.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import LOOKBACK_DAYS, hour_index, train_model


HORIZON = datetime.timedelta(hours=24)


class CustomConfig(Config):
    # Retrain at most this often, in seconds.
    retrain_after_s = 86400
    # Location of the PV system, for the clear-sky curve. The default is
    # Leipzig, an assumption consistent with the observed sunrise times; set the
    # real coordinates in the deployment config if they differ.
    latitude = 51.34
    longitude = 12.37


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
        self._current_hour: typing.Optional[int] = None
        self._cached_for: typing.Optional[int] = None
        self._cached_prediction: typing.Optional[float] = None
        super().init(*args, **kwargs)

    @staticmethod
    def _python_model(model):
        if model is None:
            return None
        try:
            return model.unwrap_python_model()
        except AttributeError:
            return model

    def _seed(self, pm) -> None:
        """Take the model's history on first use and whenever the model changes."""
        if self._seeded_from == id(pm):
            return
        self._seeded_from = id(pm)
        for h, v in pm.hourly.items():
            # Hours the stream has already measured win over the seed.
            if h not in self._counts:
                self._hourly.setdefault(h, v)
        self._cached_for = None

    def _close_hours(self, now_hour: int) -> None:
        """Move every accumulated hour before now_hour into the complete means."""
        for h in [h for h in self._sums if h < now_hour]:
            self._hourly[h] = self._sums.pop(h) / self._counts.pop(h)
        horizon = now_hour - 24 * (LOOKBACK_DAYS + 3)
        for h in [h for h in self._hourly if h < horizon]:
            del self._hourly[h]

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
            self._cached_prediction = pm.forecast(self._hourly, now_hour)
            self._cached_for = now_hour

        return timestamp + HORIZON, {"prediction": self._cached_prediction}, None

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
