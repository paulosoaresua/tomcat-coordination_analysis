from typing import Any, List

import matplotlib.pyplot as plt


def plot_vocalic_series(series_a: List[float],
                        series_b: List[float],
                        title: str,
                        marker: str = "o",
                        ax: Any = None):
    if ax is None:
        plt.figure(figsize=(20, 6))
        ax = plt.gca()
    
    plt.scatter(range(len(series_a)), series_a, color="blue", label="A", marker=marker)
    plt.scatter(range(len(series_b)), series_b, color="orange", label="B", marker=marker)

    plt.title(title)
    plt.legend()        

    plt.show()
