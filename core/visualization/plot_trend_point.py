from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from ..trend.base_trend_point import BaseTrendDetection

__all__ = [
    "PlotTrendPoint",
]

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


class PlotTrendPoint:
    def __init__(self, model: BaseTrendDetection):
        self._fig, (self._ax_plot, self._ax_legend) = plt.subplots(
            nrows=1, ncols=2, gridspec_kw={"width_ratios": [7, 1]}
        )
        self._fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05, wspace=0, hspace=0)
        self._model: BaseTrendDetection = model
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
                "xy": (self._model.get_split_index(), y_max + y_max * offset / 2),
                "xytext": (
                    self._model.get_split_index() - self._model.get_split_index() * 0.1,
                    y_max + y_max * offset / 2,
                ),
                "arrowprops": {"facecolor": "black", "arrowstyle": "<|-", "lw": 0.7},
                "ha": "left",
                "va": "center",
            },
        )
        self._plot(
            plot_func=self._ax_plot.annotate,
            args=("History",),
            kwargs={
                "xy": (self._model.get_split_index(), y_max + y_max * offset / 2),
                "xytext": (
                    self._model.get_split_index() - self._model.get_split_index() * 0.2,
                    y_max + y_max * offset / 2,
                ),
                "ha": "left",
                "va": "center",
            },
        )

        self._plot(
            plot_func=self._ax_plot.annotate,
            args=("",),
            kwargs={
                "xy": (self._model.get_split_index(), y_max + y_max * offset / 2),
                "xytext": (
                    self._model.get_split_index() + self._model.get_split_index() * 0.1,
                    y_max + y_max * offset / 2,
                ),
                "arrowprops": {"facecolor": "black", "arrowstyle": "<|-", "lw": 0.7},
                "ha": "right",
                "va": "center",
            },
        )
        self._plot(
            plot_func=self._ax_plot.annotate,
            args=("Future",),
            kwargs={
                "xy": (self._model.get_split_index(), y_max + y_max * offset / 2),
                "xytext": (
                    self._model.get_split_index() + self._model.get_split_index() * 0.2,
                    y_max + y_max * offset / 2,
                ),
                "ha": "right",
                "va": "center",
            },
        )

    def plot_all_values(self):
        self._plot(
            plot_func=self._ax_plot.plot,
            args=(self._model.get_all_values(),),
            kwargs={"color": "black", "linestyle": "-", "linewidth": 0.7, "alpha": 0.7},
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
            args=(self._model.get_split_index(),),
            kwargs={
                "color": "black",
                "linestyle": "-.",
                "linewidth": 1,
            },
        )

        self._add_legend(
            plot_func=self._ax_plot.plot,
            args=([],),
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
                self._model.get_combined_values(after_iter=after_iter),
            ),
            kwargs={
                # "color": "black",
                "linestyle": "-",
                "linewidth": 0.6 + after_iter / 10,
                "alpha": 0.6,
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
                self._model.get_trend_indexes_by_min(after_iter),
                self._model.get_trend_values_by_min(after_iter),
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
                f"eps={self._model.get_trend_eps_by_min(after_iter):3d}\t"
                f"len={len(self._model.get_trend_indexes_by_min(after_iter)):3d}",
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
                self._model.get_trend_indexes_by_max(after_iter),
                self._model.get_trend_values_by_max(after_iter),
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
                f"eps={self._model.get_trend_eps_by_max(after_iter):3d}\t"
                f"len={len(self._model.get_trend_indexes_by_max(after_iter)):3d}",
            },
        )

    def show(self, title: str, from_date: str, to_date: str, split_date: str, timeframe: str):
        self._render(
            title=title,
            from_date=from_date,
            to_date=to_date,
            split_date=split_date,
            timeframe=timeframe,
        )
        plt.show()
        plt.close(self._fig)

    def save(
        self,
        title: str,
        from_date: str,
        to_date: str,
        split_date: str,
        timeframe: str,
        name: str,
    ):
        self._render(
            title=title,
            from_date=from_date,
            to_date=to_date,
            split_date=split_date,
            timeframe=timeframe,
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
