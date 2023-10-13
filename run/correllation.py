import os
from collections import defaultdict

import numpy as np
import pandas as pd
from TradingMath.extremal import extremal_max, extremal_min

from core_math.extremum import coincident
from core_math.merge_arg_sort import merge_arg_sort

P = 1
N_ROW = 10_000


def main_correlation():
    directory_path = "../data/"
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            f = os.path.join(directory_path, file)
            print(f)
            df_all = pd.read_csv(f, chunksize=N_ROW)
            statistic = defaultdict(list)
            df: pd.DataFrame
            for df in df_all:
                close = df.Close.values
                index = merge_arg_sort(close)
                if len(index) != N_ROW:
                    break
                for i in range(1, 15):
                    max_index = extremal_max(index=index, eps=i)
                    min_index = extremal_min(index=index, eps=i)

                    statistic[f"max_len_eps_{i}"].append(len(max_index))
                    statistic[f"min_len_eps_{i}"].append(len(min_index))

                max_index, eps_max, _ = coincident(4)(extremal_max)(index=index, eps=50)
                min_index, eps_min, _ = coincident(4)(extremal_min)(index=index, eps=50)

                statistic["max_len_eps"].append(len(max_index))
                statistic["min_len_eps"].append(len(min_index))

                statistic["element"].append(len(index))
                statistic["noisiness"].append((np.max(close) - np.min(close)) / np.std(close))

                statistic["max_eps_label"].append(eps_max)
                statistic["min_eps_label"].append(eps_min)

            st = pd.DataFrame(statistic)
            st.to_csv(f"eps_{file}_new", index=False)


if __name__ == "__main__":
    main_correlation()
