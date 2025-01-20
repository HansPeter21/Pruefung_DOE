import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def one_factor_analysis(
    y, level=None, alpha=0.05, postprocess=True, figsize=(12, 6), boxplot_dots_alpha=0.2
):
    """
    Performs a one-way ANOVA, checks the residuals for normality and does the
    tukey-test to check which groups differ.

    The input data y has to be an 2D array in the following form:
    y = [
          [y11, y12, y13, ...],
          [y21, y22, y23, ...],
          [y31, y32, y33, ...],
          ...
    ]

    If the level are specified, they should be a 1D array. These are used to
    label the charts and in the tukey comparison:
    level = [l1, l2, l3, ...]
    """
    # Check input
    if level is None:
        level = np.arange(len(y))
    elif len(level) != len(y):
        print(
            f"\x1b[1;31mError:\x1b[0m The size of the input data y ({len(y)}) "
            f"doesn't match the size of the levels ({len(level)})."
        )
        return

    # Convert to np array, in case it isn't already
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    if not isinstance(level, np.ndarray):
        level = np.asarray(level)

    # Calculate residuals
    avg_yid = np.mean(y, axis=1)
    e = y - avg_yid.reshape(-1, 1)

    # Check residuals
    check_alpha = 0.05
    median_check = st.levene(*e)
    normality_check = st.shapiro(e.flatten())

    # Perform one-way-ANOVA
    ANOVA = st.f_oneway(*y)

    # Postprocessing
    if postprocess:
        if median_check.pvalue < check_alpha:
            print(
                "\x1b[1;33mWarning:\x1b[0m The median of the residuals is not "
                f"equal (levene p-value = {median_check.pvalue:.3f} < {check_alpha})!"
            )
        else:
            print(
                "\x1b[1;32mResidual medians looking good!\x1b[0m Levene "
                f"p-value = {median_check.pvalue:.3f} > {check_alpha}"
            )

        if normality_check.pvalue < check_alpha:
            print(
                "\x1b[1;33mWarning:\x1b[0m It seems, that the residuals aren't "
                f"normal distributed (Shapiro p-value = {normality_check.pvalue:.3f} "
                f"< {check_alpha})! Consider transforming the observation data."
            )
        else:
            print(
                "\x1b[1;32mResiduals seem normal distributed!\x1b[0m Shapiro "
                f"p-value = {normality_check.pvalue:.3f} > {check_alpha}"
            )

        if ANOVA.pvalue < alpha:
            print(
                "\x1b[1;32mAt least one mean should be different!\x1b[0m One-way "
                f"ANOVA p-value = {ANOVA.pvalue:.3f} < {alpha}."
            )
        else:
            print(
                "\x1b[1;33mAll means should be the same!\x1b[0m One-way ANOVA "
                f"p-value = {ANOVA.pvalue:.3f} > {alpha}."
            )

        # Plotting
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        axs[0, 0].grid(alpha=0.4)
        # Add points to boxplot
        if boxplot_dots_alpha > 0:
            for yi, li in zip(y, range(1, len(y) + 1)):
                x = np.random.normal(li, 0.02, size=len(yi))
                axs[0, 0].plot(x, yi, "b.", alpha=boxplot_dots_alpha)
        axs[0, 0].boxplot(y.T)
        axs[0, 0].set_xlabel("Level")
        axs[0, 0].set_title("Observations")
        axs[0, 0].set_xticks(range(1, len(y) + 1), labels=level)

        axs[0, 1].grid(alpha=0.4)
        # Add points to boxplot
        if boxplot_dots_alpha > 0:
            for ei, li in zip(e, range(1, len(e) + 1)):
                x = np.random.normal(li, 0.02, size=len(ei))
                axs[0, 1].plot(x, ei, "b.", alpha=boxplot_dots_alpha)
        axs[0, 1].boxplot(e.T)
        axs[0, 1].set_xlabel("Level")
        if median_check.pvalue < alpha:
            axs[0, 1].set_title(f"Residuals - WARNING: Levene < {alpha}!")
        else:
            axs[0, 1].set_title("Residuals")

        axs[1, 1].grid(alpha=0.4)
        st.probplot(e.flatten(), plot=axs[1, 1])
        if normality_check.pvalue < alpha:
            axs[1, 1].set_title(f"Q-Q-Plot - WARNING: Shapiro < {alpha}!")
        else:
            axs[1, 1].set_title("Q-Q-Plot")

        melted = pd.DataFrame(y)
        melted.index = level
        melted = melted.T.melt()

        tukey_hsd = pairwise_tukeyhsd(
            endog=melted["value"], groups=melted["variable"], alpha=alpha
        )
        print()
        print(tukey_hsd.summary())
        print("\x1b[1mReject True\x1b[0m => Means are different")
        axs[1, 0].grid(alpha=0.4)
        tukey_hsd.plot_simultaneous(ax=axs[1, 0])
        axs[1, 0].set_ylabel("Level")
        axs[1, 0].set_title("Confidence intervals (Tukey)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    y = np.array(
        [
            [575, 542, 530, 539, 570],
            [565, 593, 590, 579, 610],
            [600, 651, 610, 637, 629],
            [725, 700, 715, 685, 710],
        ]
    )

    one_factor_analysis(y, ["a", "b", "c", "d"])
