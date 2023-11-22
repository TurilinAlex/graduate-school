import argparse
import os
import sys
from enum import Enum

import pandas as pd
from loguru import logger

import core.matches_extremum as matches
from core import (
    CombinedTrendDetection,
    MergeTrendDetection,
    SplitTrendDetection,
    PlotTrendPoint,
    config,
)


class RunMode(Enum):
    COMBINED = "COMBINED"
    SPLIT = "SPLIT"
    MERGE = "MERGE"


class MatchesMode(Enum):
    BYINPUT = "BYINPUT"
    BYRECALC = "BYRECALC"


class ProgramOptions:
    """
    The class provides an interface for parsing program options form command line
    """

    class StringVerify:
        @classmethod
        def verify(cls, value: str):
            if not value:
                raise ValueError(
                    "Cannot can not be empty string! Run main.py with flag -h (--help) for get more information"
                )

        def __set_name__(self, owner, name):
            self.name = f"_{name}"

        def __get__(self, instance, owner):
            return instance.__dict__[self.name]

        def __set__(self, instance, value):
            try:
                self.verify(value)
            except ValueError as ve:
                print(f"-{self.name[2:]} {ve}")
                sys.exit(1)

            instance.__dict__[self.name] = value

    _run_mode = StringVerify()
    _match_mode = StringVerify()

    def __init__(self):
        _parser = argparse.ArgumentParser(description="Trend detection algorithm")
        _parser.add_argument(
            "-mode",
            help="Determines which algorithm for searching trend reversal points will be launched",
            default="combined",
        )
        _parser.add_argument(
            "-match",
            help="Determines which algorithm for searching trend reversal points will be launched",
            default="byinput",
        )

        args = _parser.parse_args()

        self._run_mode = args.mode
        self._match_mode = args.match

    def get_run_mode(self) -> RunMode:
        """
        :return: Run mode
        """
        return RunMode(self._run_mode.upper())

    def get_match_mode(self) -> MatchesMode:
        """
        :return: Matches mode
        """
        return MatchesMode(self._match_mode.upper())


def main():
    option = ProgramOptions()

    path = config.PATH
    rows = config.ROWS
    test_size = config.TEST_SIZE
    eps = config.EPS
    repeat = config.REPEAT

    file = os.path.basename(path).split(".")[0]

    df = pd.read_csv(path, nrows=rows)
    close = df.Close.values
    dates = df.Date.values

    if option.get_match_mode() == MatchesMode.BYINPUT:
        matches_extremum = matches.MatchesOnInputArray()
    elif option.get_match_mode() == MatchesMode.BYRECALC:
        matches_extremum = matches.MatchesOnRecalculatedArray()
    else:
        logger.error("Error")
        sys.exit(1)

    if option.get_run_mode() == RunMode.COMBINED:
        trend_points = CombinedTrendDetection(
            values=close, test_size=test_size, coincident=matches_extremum
        )
    elif option.get_run_mode() == RunMode.MERGE:
        trend_points = MergeTrendDetection(
            values=close, test_size=test_size, coincident=matches_extremum
        )
    elif option.get_run_mode() == RunMode.SPLIT:
        trend_points = SplitTrendDetection(
            values=close, test_size=test_size, coincident=matches_extremum
        )
    else:
        logger.error("Error")
        sys.exit(1)

    visualisation = PlotTrendPoint(trend_points)
    visualisation.plot_all_values()

    for i, (e, r) in enumerate(zip(eps, repeat), start=1):
        trend_points.search_extremum(num_coincident=r, start_eps=e)
        trend_points.search_change_trend_point(eps=2)
        visualisation.plot_extremum()
        visualisation.plot_combined_extremum()

    visualisation.plot_change_trend()

    visualisation.show(
        title=file,
        from_date=dates[0],
        to_date=dates[-1],
        split_date=dates[test_size],
        timeframe="1 minute",
    )


if __name__ == "__main__":
    main()
