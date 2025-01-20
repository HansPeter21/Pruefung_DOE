import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula as smf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from math import ceil, floor


class Anova:
    def __init__(self, dataframe: pd.DataFrame, model_formula: str):
        self.df = dataframe

        # Create and fit model
        self.model = smf.api.ols(model_formula, dataframe).fit()
        self.summary = self.model.summary()

        # Perform ANOVA
        self.result = sm.stats.anova_lm(self.model).sort_values(by="PR(>F)")

    def check(self):
        residuals = self.model.resid
        shapiro = st.shapiro(residuals).pvalue

        print(f"Shapiro p-value: {shapiro}")
        if shapiro > 0.05:
            print("\x1b[32mThe residuals seem normal distributed.\x1b[0m")
        else:
            print(
                "\x1b[31mThe residuals are probably \x1b[1mnot\x1b[0;31m normal distributed!\x1b[0m"
            )

        fig, ax = plt.subplots(1, 2)
        st.probplot(residuals, plot=ax[0])
        ax[0].grid(alpha=0.3)

        sig = np.sqrt(self.result.loc["Residual"]["mean_sq"])
        ax[1].set_title(f"Residuals ($\\sigma_R \\approx {sig:.1e}$)")
        ax[1].plot(self.model.predict(self.df), residuals / sig, "o")
        ax[1].set_xlabel("$\\hat{y}_{ijk}$")
        ax[1].set_ylabel("Residual / $\\sigma_R$")
        ax[1].grid(alpha=0.3)
        plt.show()

    def contribution(self) -> pd.DataFrame:
        sum_sq = self.result["sum_sq"].sum()
        self.result["contribution %"] = self.result["sum_sq"] / sum_sq * 100
        return self.result.sort_values(by=["contribution %"], ascending=False)

    def plot_model(
        self, factors, response=None, plt_range=(-1, 1), transform_function=None
    ):
        if len(factors) == 1:
            print("tbd: 2D Plot")
        if len(factors) == 2:
            print("3D Plot")

            # Create data for surface plot
            x1_values = np.linspace(plt_range[0], plt_range[1], 100)
            x2_values = np.linspace(plt_range[0], plt_range[1], 100)
            x1_mesh, x2_mesh = np.meshgrid(x1_values, x2_values)

            df = pd.DataFrame(
                {
                    factors[0]: x1_mesh.flatten(),
                    factors[1]: x2_mesh.flatten(),
                }
            )

            Z = self.model.predict(df)
            Z = np.reshape(Z, x1_mesh.shape)

            if transform_function is not None:
                Z = transform_function(Z)

            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(z=Z, x=x1_values, y=x2_values)])
            if response is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=self.df[factors[0]],
                        y=self.df[factors[1]],
                        z=self.df[response],
                        mode="markers",
                        marker=dict(size=6, color="red"),
                        name="Raw Data Points",
                    )
                )

            # Set axis labels
            fig.update_layout(
                scene=dict(xaxis_title=factors[0], yaxis_title=factors[1])
            )

            # Set layout parameters
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(nticks=4, tickangle=45),
                    yaxis=dict(nticks=4, tickangle=45),
                    zaxis=dict(nticks=4, tickangle=45),
                ),
            )

            # Show the plot
            fig.show()
        else:
            print(
                "Only 2D and 3D plots are possible. Please define 1 or 2 factors to plot."
            )


def plot_linear_effects(df, factor_columns, response_colum, max_ncols=3):
    if isinstance(factor_columns, str):
        factor_columns = [factor_columns]

    nfac = len(factor_columns)
    if nfac <= max_ncols:
        ncols = nfac
    else:
        ncols = max_ncols
    nrows = ceil(nfac / max_ncols)

    df = df[factor_columns + [response_colum]]
    fig, axs = plt.subplots(nrows, ncols, sharey="all", squeeze=False)
    for col, i in zip(factor_columns, range(nfac)):
        ncol = i % ncols
        nrow = floor(i / ncols)
        group = df.groupby(col).mean()
        axs[nrow, ncol].plot(group.index.values, group[response_colum], "o-")
        axs[nrow, ncol].plot(df[col], df[response_colum], ".", alpha=0.5)
        axs[nrow, ncol].set_xlabel(col)
        axs[nrow, ncol].set_ylabel(response_colum)
        axs[nrow, ncol].grid(alpha=0.3)

    plt.show()


def plot_interaction(
    df: pd.DataFrame, main_factor: str, secondary_factor: str, response: str
):
    """
    Plots the interaction between the specified main and secondary factors to
    the response. The main_factor, secondary_factor and response have to be
    columns in the specified DataFrame df!
    """
    main_values = df[main_factor].unique()
    secondary_values = df[secondary_factor].unique()

    # Generate the interaction lines
    lines_x = [main_values for _ in main_values]
    lines_y = []
    for secondary in secondary_values:
        current_y = []
        filter_sec = df[secondary_factor] == secondary
        for main in main_values:
            filter_main = df[main_factor] == main
            values = df[filter_main & filter_sec][response].values
            mean = np.array(values).mean()
            current_y.append(mean)
        lines_y.append(current_y)

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(df[main_factor], df[response], ".", alpha=0.15, label="Samples")
    for i, (x, y, secondary) in enumerate(zip(lines_x, lines_y, secondary_values)):
        alpha = (i + 1) / len(lines_x)
        ax.plot(x, y, "o-", label=f"{secondary_factor} = {secondary}", alpha=alpha)
    ax.set_title(f"Interaction between {main_factor} and {secondary_factor}")
    ax.set_xlabel(main_factor)
    ax.set_ylabel(secondary_factor)
    ax.grid(alpha=0.4)
    ax.legend()
    plt.show()
