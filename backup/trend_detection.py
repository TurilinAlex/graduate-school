from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol, Callable

import numpy as np
from TradingMath.extremal import extremal_max, extremal_min
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from .extremum import merge_extremum, TypeCoincident
from .merge_arg_sort import merge_arg_sort

# Activating text rendering by LaTex
plt.rcParams.update(
    {
        "figure.figsize": [20, 10],
        "text.usetex": True,
        "font.family": "monospace",
        "font.monospace": "Computer Modern Typewriter",
        "font.size": 12,
    }
)


class TrendDetection(Protocol):
    # region Other getter

    def get_split_num(self) -> int:
        ...

    def get_all_values(self) -> np.ndarray[np.float32]:
        ...

    def get_current_iter(self) -> int:
        ...

    # endregion Other getter

    # region Extremum getter
    def get_min_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    def get_max_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    def get_min_values(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        ...

    def get_max_values(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        ...

    def get_min_eps(self, after_iter: int | None = None) -> int:
        ...

    def get_max_eps(self, after_iter: int | None = None) -> int:
        ...

    def get_combined_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    def get_combined_values(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    # endregion Extremum getter

    # region Trend getter

    def get_down2up_trend_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    def get_up2down_trend_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    def get_down2up_trend_values(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        ...

    def get_up2down_trend_values(self, after_iter: int | None = None) -> np.ndarray[np.float32]:
        ...

    def get_down2up_trend_eps(self, after_iter: int | None = None) -> int:
        ...

    def get_up2down_trend_eps(self, after_iter: int | None = None) -> int:
        ...

    def get_trend_combined_indexes(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    def get_trend_combined_values(self, after_iter: int | None = None) -> np.ndarray[np.uint32]:
        ...

    # endregion Trend getter


class Visualisation:
    def __init__(self, model: TrendDetection):
        self._fig, (self._ax_plot, self._ax_legend) = plt.subplots(
            nrows=1, ncols=2, gridspec_kw={"width_ratios": [7, 1]}
        )
        self._fig.subplots_adjust(
            left=0.05, right=0.99, top=0.95, bottom=0.05, wspace=0, hspace=0
        )
        self._model: TrendDetection = model
        self._ax_legend.set_axis_off()
        y_min = np.min(self._model.get_all_values())
        y_max = np.max(self._model.get_all_values())
        offset = 0.01
        self._ax_plot.set_ylim(y_min - y_min * offset, y_max + y_max * offset)

        self._legend_handles: list[tuple[...]] = []
        self._tx = 0.0
        self._ty = 1.5

        self._plot(
            plot_func=self._ax_plot.annotate,
            args=("",),
            kwargs={
                "xy": (self._model.get_split_num(), y_max + y_max * offset / 2),
                "xytext": (self._model.get_split_num() - self._model.get_split_num() * 0.1, y_max + y_max * offset / 2),
                "arrowprops": {
                    "facecolor": "black",
                    "arrowstyle": "<|-",
                    "lw": 0.7
                },
                "ha": "left",
                "va": "center",
            }
        )
        self._plot(
            plot_func=self._ax_plot.annotate,
            args=(
                "History",
            ),
            kwargs={
                "xy": (self._model.get_split_num(), y_max + y_max * offset / 2),
                "xytext": (self._model.get_split_num() - self._model.get_split_num() * 0.2, y_max + y_max * offset / 2),
                "ha": "left",
                "va": "center",
            }
        )

        self._plot(
            plot_func=self._ax_plot.annotate,
            args=(
                "",
            ),
            kwargs={
                "xy": (self._model.get_split_num(), y_max + y_max * offset / 2),
                "xytext": (self._model.get_split_num() + self._model.get_split_num() * 0.1, y_max + y_max * offset / 2),
                "arrowprops": {
                    "facecolor": "black",
                    "arrowstyle": "<|-",
                    "lw": 0.7
                },
                "ha": "right",
                "va": "center",
            }
        )
        self._plot(
            plot_func=self._ax_plot.annotate,
            args=(
                "Future",
            ),
            kwargs={
                "xy": (self._model.get_split_num(), y_max + y_max * offset / 2),
                "xytext": (self._model.get_split_num() + self._model.get_split_num() * 0.2, y_max + y_max * offset / 2),
                "ha": "right",
                "va": "center",
            }
        )

    def plot_all_values(self):

        self._plot(
            plot_func=self._ax_plot.plot,
            args=(
                self._model.get_all_values(),
            ),
            kwargs={
                "color": "black",
                "linestyle": "-",
                "linewidth": 0.7,
                "alpha": 0.7
            },
        )
        self._add_legend(
            plot_func=self._ax_legend.plot,
            args=(
                [],
                [],
            ),
            kwargs={
                "color": "black",
                "linestyle": "-",
                "linewidth": 0.7,
                "label": "Close value",
            },
        )
        self._plot(
            plot_func=self._ax_plot.axvline,
            args=(
                self._model.get_split_num(),
            ),
            kwargs={
                "color": "black",
                "linestyle": "-.",
                "linewidth": 1,
            },
        )

        self._add_legend(
            plot_func=self._ax_plot.plot,
            args=(
                [],
            ),
            kwargs={
                "color": "black",
                "linestyle": "-.",
                "linewidth": 1,
                "label": "Split value",
            },
        )

    def plot_combined_extremum(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self._model.get_current_iter()

        self._plot(
            plot_func=self._ax_plot.plot,
            args=(
                self._model.get_combined_indexes(after_iter=after_iter),
                self._model.get_combined_values(after_iter=after_iter)
            ),
            kwargs={
                # "color": "black",
                "linestyle": "-",
                "linewidth": 0.6 + after_iter / 10,
                "alpha": 0.6
            },
        )
        self._add_legend(
            plot_func=self._ax_legend.plot,
            args=(
                [],
                [],
            ),
            kwargs={
                # "color": "black",
                "linestyle": "-",
                "linewidth": 0.6 + after_iter / 10,
                "label": f"Extr after iter={after_iter}",
            },
        )

    def plot_extremum(self, after_iter: int | None = None):
        self.plot_max_extremum(after_iter=after_iter)
        self.plot_min_extremum(after_iter=after_iter)

    def plot_max_extremum(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self._model.get_current_iter()

        affine = Affine2D()
        affine.translate(tx=self._tx, ty=self._ty)
        marker = MarkerStyle(marker=r"$\bigwedge$", transform=affine)

        self._plot(
            plot_func=self._ax_plot.scatter,
            args=(
                self._model.get_max_indexes(after_iter),
                self._model.get_max_values(after_iter),
            ),
            kwargs={
                "marker": marker,
                "s": 25 * after_iter * (1.1 * (after_iter - 1) + 1),
                "edgecolors": "black",
                "facecolors": "none",
                "linewidth": 0.5,
            },
        )

        marker = marker.transformed(Affine2D().translate(tx=self._tx, ty=-self._ty))
        self._add_legend(
            plot_func=self._ax_plot.scatter,
            args=(
                [],
                [],
            ),
            kwargs={
                "marker": marker,
                "s": 25 * after_iter * (1.1 * (after_iter - 1) + 1),
                "edgecolors": "black",
                "facecolors": "none",
                "linewidth": 0.5,
                "label": f"Extr max:\t"
                         f"iter={after_iter}\n\t"
                         f"eps={self._model.get_max_eps(after_iter):3d}\t"
                         f"len={len(self._model.get_max_indexes(after_iter)):3d}",
            },
        )

    def plot_min_extremum(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self._model.get_current_iter()

        affine = Affine2D()
        affine.translate(tx=self._tx, ty=-self._ty)
        marker = MarkerStyle(marker=r"$\bigvee$", transform=affine)

        self._plot(
            plot_func=self._ax_plot.scatter,
            args=(
                self._model.get_min_indexes(after_iter),
                self._model.get_min_values(after_iter),
            ),
            kwargs={
                "marker": marker,
                "s": 25 * after_iter * (1.1 * (after_iter - 1) + 1),
                "edgecolors": "black",
                "facecolors": "none",
                "linewidth": 0.5,
            },
        )

        marker = marker.transformed(Affine2D().translate(tx=self._tx, ty=self._ty))
        self._add_legend(
            plot_func=self._ax_plot.scatter,
            args=(
                [],
                [],
            ),
            kwargs={
                "marker": marker,
                "s": 25 * after_iter * (1.1 * (after_iter - 1) + 1),
                "edgecolors": "black",
                "facecolors": "none",
                "linewidth": 0.5,
                "label": f"Extr min:\t"
                         f"iter={after_iter}\n\t"
                         f"eps={self._model.get_min_eps(after_iter):3d}\t"
                         f"len={len(self._model.get_min_indexes(after_iter)):3d}",
            },
        )

    def plot_change_trend(self, after_iter: int | None = None):
        self.plot_up2down_trend_point(after_iter=after_iter)
        self.plot_down2up_trend_point(after_iter=after_iter)

    def plot_down2up_trend_point(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self._model.get_current_iter()

        affine = Affine2D()
        affine.translate(tx=self._tx, ty=-self._ty - 0.1)
        marker = MarkerStyle(marker=r"$\underline{\bigvee}$", transform=affine)

        self._plot(
            plot_func=self._ax_plot.scatter,
            args=(
                self._model.get_down2up_trend_indexes(after_iter),
                self._model.get_down2up_trend_values(after_iter),
            ),
            kwargs={
                "marker": marker,
                "edgecolors": "black",
                "facecolors": "none",
                "s": 25 * (after_iter + 1) * (1.1 * ((after_iter + 1) - 1) + 1),
                "linewidths": 1.2,
            },
        )

        marker = marker.transformed(Affine2D().translate(tx=self._tx, ty=self._ty))
        self._add_legend(
            plot_func=self._ax_legend.scatter,
            args=(
                [],
                [],
            ),
            kwargs={
                "marker": marker,
                "edgecolors": "black",
                "facecolors": "none",
                "s": 25 * (after_iter + 1) * (1.1 * ((after_iter + 1) - 1) + 1),
                "linewidths": 1.2,
                "label": f"Down->Up trend:\n\t"
                         f"after iter={after_iter}\n\t"
                         f"eps={self._model.get_down2up_trend_eps(after_iter):3d}\t"
                         f"len={len(self._model.get_down2up_trend_indexes(after_iter)):3d}",
            },
        )

    def plot_up2down_trend_point(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self._model.get_current_iter()

        affine = Affine2D()
        affine.translate(tx=self._tx, ty=self._ty + 0.1)
        marker = MarkerStyle(marker=r"$\overline{\bigwedge}$", transform=affine)

        self._plot(
            plot_func=self._ax_plot.scatter,
            args=(
                self._model.get_up2down_trend_indexes(after_iter),
                self._model.get_up2down_trend_values(after_iter),
            ),
            kwargs={
                "marker": marker,
                "edgecolors": "black",
                "facecolors": "none",
                "s": 25 * (after_iter + 1) * (1.1 * ((after_iter + 1) - 1) + 1),
                "linewidths": 1.2,
            },
        )

        marker = marker.transformed(Affine2D().translate(tx=self._tx, ty=-self._ty))
        self._add_legend(
            plot_func=self._ax_legend.scatter,
            args=(
                [],
                [],
            ),
            kwargs={
                "marker": marker,
                "edgecolors": "black",
                "facecolors": "none",
                "s": 25 * (after_iter + 1) * (1.1 * ((after_iter + 1) - 1) + 1),
                "linewidths": 1.2,
                "label": f"Up->Down trend:\n\t"
                         f"after iter={after_iter}\n\t"
                         f"eps={self._model.get_up2down_trend_eps(after_iter):3d}\t"
                         f"len={len(self._model.get_up2down_trend_indexes(after_iter)):3d}",
            },
        )

    def show(self, title: str, from_date: str, to_date: str, split_date: str, timeframe: str):
        self._render(
            title=title,
            from_date=from_date,
            to_date=to_date,
            split_date=split_date,
            timeframe=timeframe
        )
        plt.show()
        plt.close(self._fig)

    def save(self, title: str, from_date: str, to_date: str, split_date: str, timeframe: str, name: str):
        self._render(
            title=title,
            from_date=from_date,
            to_date=to_date,
            split_date=split_date,
            timeframe=timeframe
        )
        plt.savefig(name)
        plt.close(fig="all")

    @staticmethod
    def _plot(plot_func: Callable, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        plot_result = plot_func(*args, **kwargs)
        plot_result = plot_result[0] if isinstance(plot_result, list) else plot_result
        return plot_result

    def _add_legend(self, plot_func: Callable, args=None, kwargs=None):
        plot_result = self._plot(plot_func=plot_func, args=args, kwargs=kwargs)
        self._legend_handles.append(plot_result)

    def _render(self, title: str, from_date: str, to_date: str, split_date: str, timeframe: str):
        legend_1 = self._ax_legend.legend(
            handles=self._legend_handles,
            scatteryoffsets=[0.5],
            loc="center left",
            labelspacing=1,
            markerscale=1,
            fontsize="medium",
        )
        self._fig.gca().add_artist(legend_1)
        self._fig.suptitle(
            rf"{self._model.__class__.__name__}. {title.replace('_', '/')}. Split data - {split_date}"
        )
        self._ax_plot.minorticks_on()
        self._ax_plot.grid(
            color="gray",
            linestyle="-",
            linewidth=0.2,
        )
        self._ax_plot.set_xlabel(
            rf"From {from_date} to {to_date} with timeframe {timeframe}", fontsize=15
        )
        self._ax_plot.set_ylabel(r"Price \$", fontsize=15)
        for x_min in self._ax_plot.xaxis.get_minorticklocs():
            self._ax_plot.axvline(
                x=x_min,
                ls="--",
                linewidth=0.1,
                color="gray",
                alpha=0.7,
            )

        for y_min in self._ax_plot.yaxis.get_minorticklocs():
            self._ax_plot.axhline(
                y=y_min,
                ls="--",
                linewidth=0.1,
                color="gray",
                alpha=0.7,
            )


class BaseTrendDetection(ABC):
    class ExtrName(Enum):
        min_indexes = "min_indexes_iter_{after_iter}_extr"
        max_indexes = "max_indexes_iter_{after_iter}_extr"
        max_values = "max_values_iter_{after_iter}_extr"
        min_values = "min_values_iter_{after_iter}_extr"
        max_eps = "max_eps_iter_{after_iter}_extr"
        min_eps = "min_eps_iter_{after_iter}_extr"
        combined_indexes = "combined_indexes_iter_{after_iter}_extr"
        combined_values = "combined_values_iter_{after_iter}_extr"

    class TrendName(Enum):
        min_indexes = "min_indexes_iter_{after_iter}_trend"
        max_indexes = "max_indexes_iter_{after_iter}_trend"
        max_values = "max_values_iter_{after_iter}_trend"
        min_values = "min_values_iter_{after_iter}_trend"
        max_eps = "max_eps_iter_{after_iter}_trend"
        min_eps = "min_eps_iter_{after_iter}_trend"
        combined_indexes = "combined_indexes_iter_{after_iter}_trend"
        combined_values = "combined_values_iter_{after_iter}_trend"

    def __init__(self, values: np.ndarray[np.float32], test_size: int, coincident: TypeCoincident):
        self._coincident = coincident
        self._all_values = values
        self._values = values[:test_size]
        self._current_iter: int = 0
        self._split_num: int = test_size
        self._last_values = None
        self._last_indexes = None

    # region Other getter

    def get_split_num(self):
        return self._split_num

    def get_all_values(self):
        return self._all_values

    def get_current_iter(self):
        return self._current_iter

    # endregion Other getter

    # region Extremum getter
    def get_min_indexes(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.min_indexes, after_iter)

    def get_max_indexes(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.max_indexes, after_iter)

    def get_min_values(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.min_values, after_iter)

    def get_max_values(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.max_values, after_iter)

    def get_min_eps(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.min_eps, after_iter)

    def get_max_eps(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.max_eps, after_iter)

    def get_combined_indexes(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.combined_indexes, after_iter)

    def get_combined_values(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.ExtrName.combined_values, after_iter)

    # endregion Extremum getter

    # region Trend getter

    def get_down2up_trend_indexes(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.min_indexes, after_iter)

    def get_up2down_trend_indexes(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.max_indexes, after_iter)

    def get_down2up_trend_values(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.min_values, after_iter)

    def get_up2down_trend_values(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.max_values, after_iter)

    def get_down2up_trend_eps(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.min_eps, after_iter)

    def get_up2down_trend_eps(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.max_eps, after_iter)

    def get_trend_combined_indexes(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.combined_indexes, after_iter)

    def get_trend_combined_values(self, after_iter: int | None = None):

        if after_iter is None:
            after_iter = self.get_current_iter()

        return self.getter(self.TrendName.combined_values, after_iter)

    # endregion Trend getter

    # region Calculation Function

    def search_extremum(self, num_coincident: int, start_eps: int):
        self._current_iter += 1
        if self._last_values is None and self._last_indexes is None:
            self._start_iteration(num_coincident, start_eps)
        else:
            self._continuation_iterations(num_coincident, start_eps)

        return self

    def search_change_trend_point(self, eps: int, after_iter: int | None = None):
        self.search_down2up_trend_point(after_iter=after_iter, eps=eps)
        self.search_up2down_trend_point(after_iter=after_iter, eps=eps)

    def search_down2up_trend_point(self, eps: int, after_iter: int | None = None):
        if after_iter is None:
            after_iter = self.get_current_iter()
        values = np.array(self.get_min_values(after_iter))
        indexes = np.array(self.get_min_indexes(after_iter))
        if values is not None and indexes is not None:
            self._search_down2up_trend_point(
                after_iter=after_iter,
                indexes=indexes,
                values=values,
                eps=eps,
            )

    def search_up2down_trend_point(self, eps: int, after_iter: int | None = None):
        if after_iter is None:
            after_iter = self.get_current_iter()
        values = np.array(self.get_max_values(after_iter))
        indexes = np.array(self.get_max_indexes(after_iter))
        if values is not None and indexes is not None:
            self._search_up2down_trend_point(
                after_iter=after_iter,
                indexes=indexes,
                values=values,
                eps=eps,
            )

    def getter(self, name: ExtrName | TrendName, after_iter: int | None = None):
        result = None

        if after_iter is None:
            after_iter = self.get_current_iter()
        if after_iter > self._current_iter:
            return result
        try:
            result = getattr(self, name.value.format(after_iter=after_iter))
        except AttributeError as error:
            print(error)

        return result

    # endregion Calculation Function

    # region Abstract implement
    @abstractmethod
    def _search_down2up_trend_point(
            self, values: np.ndarray, indexes: np.ndarray, eps: int, after_iter: int | None = None
    ):
        pass

    @abstractmethod
    def _search_up2down_trend_point(
            self, values: np.ndarray, indexes: np.ndarray, eps: int, after_iter: int | None = None
    ):
        pass

    @abstractmethod
    def _start_iteration(self, num_coincident: int, start_eps: int):
        pass

    @abstractmethod
    def _continuation_iterations(self, num_coincident: int, start_eps: int):
        pass

    # endregion Abstract implement


class CombinedExtremum(BaseTrendDetection):
    def __init__(self, values: np.ndarray[np.float32], test_size: int, coincident: TypeCoincident):
        super().__init__(values, test_size, coincident)

    # region Abstract implement

    def _search_down2up_trend_point(
            self, values: np.ndarray, indexes: np.ndarray, eps: int, after_iter: int | None = None
    ):
        _index = merge_arg_sort(values)
        __min_index = np.sort(extremal_min(index=_index, eps=eps))
        _min_values = values[__min_index]
        _min_indexes = indexes[__min_index]

        setattr(
            self,
            self.TrendName.min_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.min_values.value.format(after_iter=after_iter),
            _min_values,
        )
        setattr(
            self,
            self.TrendName.min_indexes.value.format(after_iter=after_iter),
            _min_indexes,
        )

    def _search_up2down_trend_point(
            self, values: np.ndarray, indexes: np.ndarray, eps: int, after_iter: int | None = None
    ):
        _index = merge_arg_sort(values)
        __max_index = np.sort(extremal_max(index=_index, eps=eps))
        _max_values = values[__max_index]
        _max_indexes = indexes[__max_index]

        setattr(
            self,
            self.TrendName.max_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.max_values.value.format(after_iter=after_iter),
            _max_values,
        )
        setattr(
            self,
            self.TrendName.max_indexes.value.format(after_iter=after_iter),
            _max_indexes,
        )

    def _start_iteration(self, num_coincident: int, start_eps: int):
        _index = merge_arg_sort(self._values)
        __max_index, _max_eps = self._coincident(
            extremal_max,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __min_index, _min_eps = self._coincident(
            extremal_min,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        _max_values = self._values[__max_index]
        _min_values = self._values[__min_index]

        _index_combined = np.sort(
            np.hstack((__max_index, __min_index)), kind="mergesort"
        )
        _values_combined = self._values[_index_combined]

        setattr(
            self,
            self.ExtrName.min_indexes.value.format(after_iter=self._current_iter),
            __min_index,
        )
        setattr(
            self,
            self.ExtrName.max_indexes.value.format(after_iter=self._current_iter),
            __max_index,
        )

        setattr(
            self,
            self.ExtrName.min_values.value.format(after_iter=self._current_iter),
            _min_values,
        )
        setattr(
            self,
            self.ExtrName.max_values.value.format(after_iter=self._current_iter),
            _max_values,
        )

        setattr(
            self,
            self.ExtrName.min_eps.value.format(after_iter=self._current_iter),
            _min_eps,
        )
        setattr(
            self,
            self.ExtrName.max_eps.value.format(after_iter=self._current_iter),
            _max_eps,
        )

        setattr(
            self,
            self.ExtrName.combined_indexes.value.format(after_iter=self._current_iter),
            _index_combined,
        )
        setattr(
            self,
            self.ExtrName.combined_values.value.format(after_iter=self._current_iter),
            _values_combined,
        )

        self._last_values = _values_combined
        self._last_indexes = _index_combined

    def _continuation_iterations(self, num_coincident: int, start_eps: int):
        _index = merge_arg_sort(self._last_values)
        __max_index, _max_eps = self._coincident(
            extremal_max,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __min_index, _min_eps = self._coincident(
            extremal_min,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        _max_values = self._last_values[__max_index]
        _min_values = self._last_values[__min_index]

        _index_combined = np.sort(
            np.hstack((__max_index, __min_index)), kind="mergesort"
        )

        _min_indexes = self._last_indexes[__min_index]
        _max_indexes = self._last_indexes[__max_index]
        _combined_values = self._last_values[_index_combined]
        _combined_indexes = self._last_indexes[_index_combined]

        setattr(
            self,
            self.ExtrName.min_indexes.value.format(after_iter=self._current_iter),
            _min_indexes,
        )
        setattr(
            self,
            self.ExtrName.max_indexes.value.format(after_iter=self._current_iter),
            _max_indexes,
        )

        setattr(
            self,
            self.ExtrName.min_values.value.format(after_iter=self._current_iter),
            _min_values,
        )
        setattr(
            self,
            self.ExtrName.max_values.value.format(after_iter=self._current_iter),
            _max_values,
        )

        setattr(
            self,
            self.ExtrName.min_eps.value.format(after_iter=self._current_iter),
            _min_eps,
        )
        setattr(
            self,
            self.ExtrName.max_eps.value.format(after_iter=self._current_iter),
            _max_eps,
        )

        setattr(
            self,
            self.ExtrName.combined_values.value.format(after_iter=self._current_iter),
            _combined_values,
        )
        setattr(
            self,
            self.ExtrName.combined_indexes.value.format(after_iter=self._current_iter),
            _combined_indexes,
        )

        self._last_indexes = _combined_indexes
        self._last_values = _combined_values

    # endregion Abstract implement


class MergeExtremum(BaseTrendDetection):
    def __init__(self, values: np.ndarray[np.float32], test_size: int, coincident: TypeCoincident):
        super().__init__(values, test_size, coincident)

    # region Abstract implement
    def _search_down2up_trend_point(
            self, values: np.ndarray, indexes: np.ndarray, eps: int, after_iter: int | None = None
    ):
        _index = merge_arg_sort(values)
        __min_index = np.sort(extremal_min(index=_index, eps=eps))
        _min_values = values[__min_index]
        _min_indexes = indexes[__min_index]

        setattr(
            self,
            self.TrendName.min_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.min_values.value.format(after_iter=after_iter),
            _min_values,
        )
        setattr(
            self,
            self.TrendName.min_indexes.value.format(after_iter=after_iter),
            _min_indexes,
        )

    def _search_up2down_trend_point(
            self, values: np.ndarray[np.float32], indexes: np.ndarray[np.uint32], eps: int,
            after_iter: int | None = None
    ):
        _index = merge_arg_sort(values)
        __max_index = np.sort(extremal_max(index=_index, eps=eps))
        _max_values = values[__max_index]
        _max_indexes = indexes[__max_index]

        setattr(
            self,
            self.TrendName.max_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.max_values.value.format(after_iter=after_iter),
            _max_values,
        )
        setattr(
            self,
            self.TrendName.max_indexes.value.format(after_iter=after_iter),
            _max_indexes,
        )

    def _start_iteration(self, num_coincident: int, start_eps: int):
        _index = merge_arg_sort(self._values)
        __max_index, _max_eps = self._coincident(
            extremal_max,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __min_index, _min_eps = self._coincident(
            extremal_min,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        _index_combined, __min_index, __max_index = merge_extremum(
            extr_max_index=np.sort(__max_index, kind="mergesort"),
            extr_min_index=np.sort(__min_index, kind="mergesort"),
            values=self._values,
        )

        _max_values = self._values[__max_index]
        _min_values = self._values[__min_index]
        _values_combined = self._values[_index_combined]

        setattr(
            self,
            self.ExtrName.min_indexes.value.format(after_iter=self._current_iter),
            __min_index,
        )
        setattr(
            self,
            self.ExtrName.max_indexes.value.format(after_iter=self._current_iter),
            __max_index,
        )

        setattr(
            self,
            self.ExtrName.min_values.value.format(after_iter=self._current_iter),
            _min_values,
        )
        setattr(
            self,
            self.ExtrName.max_values.value.format(after_iter=self._current_iter),
            _max_values,
        )

        setattr(
            self,
            self.ExtrName.min_eps.value.format(after_iter=self._current_iter),
            _min_eps,
        )
        setattr(
            self,
            self.ExtrName.max_eps.value.format(after_iter=self._current_iter),
            _max_eps,
        )

        setattr(
            self,
            self.ExtrName.combined_indexes.value.format(after_iter=self._current_iter),
            _index_combined,
        )
        setattr(
            self,
            self.ExtrName.combined_values.value.format(after_iter=self._current_iter),
            _values_combined,
        )

        self._last_values = _values_combined
        self._last_indexes = _index_combined

    def _continuation_iterations(self, num_coincident: int, start_eps: int):
        _index = merge_arg_sort(self._last_values)
        __max_index, _max_eps = self._coincident(
            extremal_max,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __min_index, _min_eps = self._coincident(
            extremal_min,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        _index_combined, __min_index, __max_index = merge_extremum(
            extr_max_index=np.sort(__max_index, kind="mergesort"),
            extr_min_index=np.sort(__min_index, kind="mergesort"),
            values=self._last_values,
        )

        _max_values = self._last_values[__max_index]
        _min_values = self._last_values[__min_index]

        _min_indexes = self._last_indexes[__min_index]
        _max_indexes = self._last_indexes[__max_index]
        _combined_values = self._last_values[_index_combined]
        _combined_indexes = self._last_indexes[_index_combined]

        setattr(
            self,
            self.ExtrName.min_indexes.value.format(after_iter=self._current_iter),
            _min_indexes,
        )
        setattr(
            self,
            self.ExtrName.max_indexes.value.format(after_iter=self._current_iter),
            _max_indexes,
        )

        setattr(
            self,
            self.ExtrName.min_values.value.format(after_iter=self._current_iter),
            _min_values,
        )
        setattr(
            self,
            self.ExtrName.max_values.value.format(after_iter=self._current_iter),
            _max_values,
        )

        setattr(
            self,
            self.ExtrName.min_eps.value.format(after_iter=self._current_iter),
            _min_eps,
        )
        setattr(
            self,
            self.ExtrName.max_eps.value.format(after_iter=self._current_iter),
            _max_eps,
        )

        setattr(
            self,
            self.ExtrName.combined_values.value.format(after_iter=self._current_iter),
            _combined_values,
        )
        setattr(
            self,
            self.ExtrName.combined_indexes.value.format(after_iter=self._current_iter),
            _combined_indexes,
        )

        self._last_indexes = self._last_indexes[_index_combined]
        self._last_values = _combined_values

    # endregion Abstract implement


class SplitExtremum(BaseTrendDetection):
    def __init__(self, values: np.ndarray[np.float32], test_size: int, coincident: TypeCoincident):
        super().__init__(values, test_size, coincident)

        self._last_values_min = None
        self._last_values_max = None
        self._last_indexes_min = None
        self._last_indexes_max = None

    # region Abstract implement

    def _search_down2up_trend_point(
            self, values: np.ndarray, indexes: np.ndarray, eps: int, after_iter: int | None = None
    ):
        _index = merge_arg_sort(values)
        __min_index = np.sort(extremal_min(index=_index, eps=eps))
        _min_values = values[__min_index]
        _min_indexes = indexes[__min_index]

        setattr(
            self,
            self.TrendName.min_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.min_values.value.format(after_iter=after_iter),
            _min_values,
        )
        setattr(
            self,
            self.TrendName.min_indexes.value.format(after_iter=after_iter),
            _min_indexes,
        )

    def _search_up2down_trend_point(
            self, values: np.ndarray, indexes: np.ndarray, eps: int, after_iter: int | None = None
    ):
        _index = merge_arg_sort(values)
        __max_index = np.sort(extremal_max(index=_index, eps=eps))
        _max_values = values[__max_index]
        _max_indexes = indexes[__max_index]

        setattr(
            self,
            self.TrendName.max_eps.value.format(after_iter=after_iter),
            eps,
        )
        setattr(
            self,
            self.TrendName.max_values.value.format(after_iter=after_iter),
            _max_values,
        )
        setattr(
            self,
            self.TrendName.max_indexes.value.format(after_iter=after_iter),
            _max_indexes,
        )

    def _start_iteration(self, num_coincident: int, start_eps: int):
        _index = merge_arg_sort(self._values)
        __max_index, _max_eps = self._coincident(
            extremal_max,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __min_index, _min_eps = self._coincident(
            extremal_min,
            index=_index,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __max_index = np.sort(__max_index)
        __min_index = np.sort(__min_index)
        _max_values = self._values[__max_index]
        _min_values = self._values[__min_index]

        _index_combined = np.sort(
            np.hstack((__max_index, __min_index)), kind="mergesort"
        )
        _values_combined = self._values[_index_combined]

        setattr(
            self,
            self.ExtrName.min_indexes.value.format(after_iter=self._current_iter),
            __min_index,
        )
        setattr(
            self,
            self.ExtrName.max_indexes.value.format(after_iter=self._current_iter),
            __max_index,
        )

        setattr(
            self,
            self.ExtrName.min_values.value.format(after_iter=self._current_iter),
            _min_values,
        )
        setattr(
            self,
            self.ExtrName.max_values.value.format(after_iter=self._current_iter),
            _max_values,
        )

        setattr(
            self,
            self.ExtrName.min_eps.value.format(after_iter=self._current_iter),
            _min_eps,
        )
        setattr(
            self,
            self.ExtrName.max_eps.value.format(after_iter=self._current_iter),
            _max_eps,
        )

        setattr(
            self,
            self.ExtrName.combined_indexes.value.format(after_iter=self._current_iter),
            _index_combined,
        )
        setattr(
            self,
            self.ExtrName.combined_values.value.format(after_iter=self._current_iter),
            _values_combined,
        )

        self._last_min_values = _min_values
        self._last_max_values = _max_values

        self._last_min_indexes = __min_index
        self._last_max_indexes = __max_index

        self._last_values = _values_combined
        self._last_indexes = _index_combined

    def _continuation_iterations(self, num_coincident: int, start_eps: int):
        _index_min = merge_arg_sort(self._last_min_values)
        _index_max = merge_arg_sort(self._last_max_values)
        __max_index, _max_eps = self._coincident(
            extremal_max,
            index=_index_max,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __min_index, _min_eps = self._coincident(
            extremal_min,
            index=_index_min,
            max_coincident=num_coincident,
            eps=start_eps
        )
        __max_index = np.sort(__max_index)
        __min_index = np.sort(__min_index)

        _max_values = self._last_max_values[__max_index]
        _min_values = self._last_min_values[__min_index]

        _min_indexes = self._last_min_indexes[__min_index]
        _max_indexes = self._last_max_indexes[__max_index]

        _index_combined = np.sort(
            np.hstack((_max_indexes, _min_indexes)), kind="mergesort"
        )

        _combined_values = self._values[_index_combined]
        _combined_indexes = _index_combined

        setattr(
            self,
            self.ExtrName.min_indexes.value.format(after_iter=self._current_iter),
            _min_indexes,
        )
        setattr(
            self,
            self.ExtrName.max_indexes.value.format(after_iter=self._current_iter),
            _max_indexes,
        )

        setattr(
            self,
            self.ExtrName.min_values.value.format(after_iter=self._current_iter),
            _min_values,
        )
        setattr(
            self,
            self.ExtrName.max_values.value.format(after_iter=self._current_iter),
            _max_values,
        )

        setattr(
            self,
            self.ExtrName.min_eps.value.format(after_iter=self._current_iter),
            _min_eps,
        )
        setattr(
            self,
            self.ExtrName.max_eps.value.format(after_iter=self._current_iter),
            _max_eps,
        )

        setattr(
            self,
            self.ExtrName.combined_values.value.format(after_iter=self._current_iter),
            _combined_values,
        )
        setattr(
            self,
            self.ExtrName.combined_indexes.value.format(after_iter=self._current_iter),
            _combined_indexes,
        )

        self._last_min_indexes = _min_indexes
        self._last_max_indexes = _max_indexes

        self._last_min_values = _min_values
        self._last_max_values = _max_values

        self._last_indexes = _combined_indexes
        self._last_values = _combined_values

    # endregion Abstract implement
