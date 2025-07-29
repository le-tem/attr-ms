import glob
import logging
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psychrolib as psyc
import seaborn as sns
from pythermalcomfort.models import two_nodes_gagge, set_tmp, phs
from pythermalcomfort.utilities import v_relative
from scipy import optimize
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    filename="info.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

psyc.SetUnitSystem(psyc.SI)

matplotlib.use("TkAgg")


@dataclass()
class Variables:
    met: float = 1.2
    clo: float = 0.6
    t_min: float = 20
    v: float = float(v_relative(0.3, 1.2))
    mrt_t_delta: float = 0
    core_base_temperature: float = 36.8
    core_temperature_calc_met: float = 37.2
    threshold_core_temperature_rise = [0.2, 0.5, 0.8]
    duration = 60
    position = "standing"
    accl = 0
    col_t = "TTX"
    col_rh = "FFX"
    col_time = "date"
    col_t_core = "t_cr"
    col_t_core_rise = "t_core_rise"


profiles = {
    "active": {"met": 1.8, "clo": 0.5, "v": float(v_relative(Variables.v, 1.8))},
    "exercising": {
        "met": 4,
        "clo": 0.36,
        "v": float(v_relative(Variables.v, 4)),
    },
    "resting": {"met": 1.2, "clo": 0.6, "v": float(v_relative(Variables.v, 1.2))},
}


def calculate_results(file_name, profile, model="two-node", cumulative=False):

    # import weather data
    df_weather = pd.read_pickle(file_name)
    df_weather.drop_duplicates(inplace=True)

    df_weather.rename(columns={"time": Variables.col_time}, inplace=True)

    df_weather = df_weather[
        [Variables.col_time, Variables.col_t, Variables.col_rh]
    ].dropna()
    try:
        df_weather.set_index(
            pd.to_datetime(df_weather[Variables.col_time]), inplace=True
        )
    except ValueError:
        logging.error(f"Error setting index for file: {file_name}", exc_info=True)
        return
    df_weather = df_weather[[Variables.col_t, Variables.col_rh]]

    # removing all the entries below a minimum temperature
    df_weather = df_weather.loc[df_weather[Variables.col_t] > Variables.t_min]
    # df_weather = df_weather.loc[df_weather.index > pd.Timestamp("1993-01-01")]

    df_r = pd.DataFrame()

    if model == "two-node":
        r = {}
        res = two_nodes_gagge(
            tdb=df_weather[Variables.col_t],
            tr=df_weather[Variables.col_t],
            rh=df_weather[Variables.col_rh],
            met=Variables.met,
            clo=Variables.clo,
            v=Variables.v,
            round_output=False,
        )
        r["t_cr"] = res["t_core"]
        df_r = pd.DataFrame(r, index=df_weather.index)["t_cr"].to_frame()

    if model == "met":
        results_met = []

        def function(x):
            return (
                two_nodes_gagge(
                    tdb=row[Variables.col_t],
                    tr=row[Variables.col_t],
                    rh=row[Variables.col_rh],
                    met=x,
                    clo=Variables.clo,
                    v=Variables.v,
                    round_output=False,
                )["t_core"]
                - Variables.core_temperature_calc_met
            )

        for ix, row in tqdm(
            df_weather.iterrows(), total=len(df_weather), desc="Estimating MET"
        ):
            try:
                met_estimated = optimize.brentq(function, 0.8, 5)
            except ValueError:
                met_estimated = np.nan
            results_met.append(met_estimated)
        df_r = pd.DataFrame(results_met, index=df_weather.index, columns=["met"])

    elif model == "phs":
        # loop through each day of the year
        df_weather["doi"] = df_weather.index.date
        results_phs = []
        r_phs = {}
        if cumulative:
            for day in df_weather["doi"].unique():
                df_day = df_weather[df_weather["doi"] == day].reset_index()
                for ix, row in df_day.iterrows():
                    if ix == 0:
                        r_phs = phs(
                            tdb=row[Variables.col_t],
                            tr=row[Variables.col_t],
                            v=Variables.v,
                            rh=row[Variables.col_rh],
                            # todo check which version of pythermalcomfort I am using
                            met=Variables.met * 58,
                            clo=Variables.clo,
                            posture=Variables.position,
                            duration=Variables.duration,
                            acclimatized=Variables.accl,
                            round_output=False,
                        )
                    else:
                        r_phs = phs(
                            tdb=row[Variables.col_t],
                            tr=row[Variables.col_t],
                            v=Variables.v,
                            rh=row[Variables.col_rh],
                            met=Variables.met * 58,
                            clo=Variables.clo,
                            posture=Variables.position,
                            duration=Variables.duration,
                            t_re=r_phs["t_re"],
                            t_sk=r_phs["t_sk"],
                            t_cr=r_phs["t_cr"],
                            t_cr_eq=r_phs["t_cr_eq"],
                            t_sk_t_cr_wg=r_phs["t_sk_t_cr_wg"],
                            sweat_rate=r_phs["water_loss_watt"],
                            acclimatized=Variables.accl,
                            round=False,
                        )
                    results_phs.append(r_phs)

            df_r = pd.DataFrame(results_phs, index=df_weather.index)["t_cr"].to_frame()
        else:
            results_phs = phs(
                tdb=df_weather[Variables.col_t],
                tr=df_weather[Variables.col_t],
                v=Variables.v,
                rh=df_weather[Variables.col_rh],
                met=Variables.met,
                clo=Variables.clo,
                posture=Variables.position,
                duration=Variables.duration,
                acclimatized=Variables.accl,
                round=False,
                limit_inputs=False,
            )
            r_phs["t_cr"] = results_phs.t_cr
            df_r = pd.DataFrame(r_phs, index=df_weather.index)[["t_cr"]]

    df_weather = df_weather.merge(df_r, left_index=True, right_index=True)
    # df_weather[Variables.col_t_core_rise] = (
    #     df_weather[Variables.col_t_core] - Variables.core_base_temperature
    # )

    # df_weather.loc[
    #     df_weather[Variables.col_t_core_rise] < 0, ["TTX", "FFX", "t_core_rise"]
    # ].head()

    results_path = Path("results")

    if cumulative:
        results_path = results_path / "cumulative"

    else:
        results_path = results_path / "not_cumulative"

    results_path = results_path / model[:3]

    if model[:3] != "met":
        results_path = results_path / profile

    results_path.mkdir(exist_ok=True, parents=True)

    df_weather.to_pickle(
        results_path / file_name.name,
        compression="gzip",
    )


def plot_rh_lines(ax, rh_val=1, t_array=np.arange(-10, 40, 0.5)):
    hr_array = []
    for t, rh in zip(t_array, np.ones(len(t_array)) * rh_val):
        hr_array.append(psyc.GetHumRatioFromRelHum(t, rh, 101325) * 1000)
    ax.plot(t_array, hr_array, c="k", lw=0.2)
    ax.text(
        22,
        psyc.GetHumRatioFromRelHum(20, rh_val, 101325) * 1000,
        f"{rh_val*100}%",
        va="center",
        ha="center",
        rotation=20,
        fontsize=6,
    )


def plot_constant_set_lines(
    ax,
    met=2,
    clo=0.5,
):

    still_air_threshold = 0.1
    rh_val = np.arange(0, 100, 1)

    results = []
    for target_set in np.arange(26, 36, 2):
        t_vals = []

        for rh in rh_val:

            def function(x):
                return (
                    set_tmp(
                        x,
                        x,
                        v=still_air_threshold,
                        rh=rh,
                        met=met,
                        clo=clo,
                        wme=0,
                        round=False,
                        limit_inputs=False,
                    )
                    - target_set
                )

            t = optimize.brentq(function, 0.0, 80)
            t_vals.append(t)
            results.append({"t": t, "rh": rh, "set": target_set})

        ax.plot(t_vals, rh_val, c="k", lw=0.2)
        ax.text(t_vals[0], rh_val[0], target_set)


def check_model_output(model="two-node"):
    t_range = [float(x) for x in np.arange(20, 40, 1)]
    rh_range = [float(x) for x in np.arange(0, 105, 5)]
    var_to_plot = {
        # "d_lim_t_re": {"max": duration * 0.5, "min": duration},
        # "water_loss_watt": {"max": 120, "min": 0},
        # "water_loss": {"max": 1900, "min": 1800},
        # "t_re": {"max": 39, "min": 37.3},
        "t_cr": {
            "max": Variables.core_base_temperature
            + Variables.threshold_core_temperature_rise[-1],
            "min": Variables.core_base_temperature
            + Variables.threshold_core_temperature_rise[0],
        },
        # "w": {"max": 1, "min": 0.7},
        # "w_req": {"max": 1.3, "min": 1},
    }

    combinations = list(product(t_range, rh_range))
    df = pd.DataFrame(combinations, columns=["t", "rh"])
    results = []
    for ix, row in df.iterrows():
        r = {}
        if model == "two-node":
            res = two_nodes_gagge(
                tdb=row["t"],
                tr=row["t"] + Variables.mrt_t_delta,
                rh=row["rh"],
                met=Variables.met,
                clo=Variables.clo,
                v=Variables.v,
                round_output=False,
            )
            r["t_cr"] = res["t_core"]
        elif model == "phs":
            res = phs(
                tdb=row.t,
                tr=row.t + Variables.mrt_t_delta,
                v=Variables.v,
                rh=row.rh,
                met=Variables.met,
                clo=Variables.clo,
                posture=Variables.position,
                duration=Variables.duration,
                round_output=False,
                acclimatized=Variables.accl,
            )
            r["t_cr"] = res["t_cr"]
        r["t"] = row.t
        r["rh"] = row.rh
        results.append(r)

    df_results = pd.DataFrame(results)
    for var, limits in var_to_plot.items():
        f, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        glue = (
            df_results.pivot(index="rh", columns="t", values=var)
            .sort_index(ascending=False)
            .astype("float")
        )
        sns.heatmap(
            glue,
            annot=True,
            fmt=".2f",
            vmin=limits["min"],
            vmax=limits["max"],
            mask=glue < limits["min"],
        )
        plt.title(
            f"{var=}; {Variables.met=}; {Variables.clo=}; {Variables.v=}; tr ="
            f" tdb+{Variables.mrt_t_delta}"
        )
        # plt.show()

        plt.savefig(
            f"figures/{model}_{var}_met{Variables.met}_clo{Variables.clo}_v{Variables.v}.png",
            dpi=300,
        )


def process_file(path_file, run_met=True, run_profiles=True):
    path_file = Path(path_file)
    logging.info(f"Processing file: {path_file}")
    try:
        if run_met:
            Variables.met = 1
            Variables.clo = 0.36
            calculate_results(
                file_name=path_file,
                model="met",
                cumulative=False,
                profile="met",
            )
            logging.info(f"Processed met model for file: {path_file}")

        # generate the results for the different profiles
        if run_profiles:
            for profile in tqdm(profiles.keys(), desc="Processing profiles"):
                # for profile in tqdm(profiles.keys()):
                Variables.met = profiles[profile]["met"]
                Variables.clo = profiles[profile]["clo"]
                Variables.v = profiles[profile]["v"]
                calculate_results(
                    file_name=path_file,
                    model="two-node",
                    cumulative=False,
                    profile=profile,
                )
                logging.info(
                    f"Processed two-node model for profile {profile} in file: {path_file}"
                )
                calculate_results(
                    file_name=path_file,
                    model="phs",
                    cumulative=False,
                    profile=profile,
                )
                logging.info(
                    f"Processed phs model for profile {profile} in file: {path_file}"
                )
    except (FileNotFoundError, EOFError, TimeoutError) as e:
        logging.error(f"Error processing file {path_file}: {e}", exc_info=True)


if __name__ == "__main__":
    plt.close("all")

if __name__ == "__pre_analysis__":
    check_model_output(model="two-node")
    check_model_output(model="phs")

if __name__ == "__calculate_results__":
    files = glob.glob("data/**/*.pkl.gz")
    files_analyse = []
    result_paths = [
        Path("results") / "not_cumulative" / "met",
        Path("results") / "not_cumulative" / "phs" / "active",
        Path("results") / "not_cumulative" / "phs" / "resting",
        Path("results") / "not_cumulative" / "two" / "active",
        Path("results") / "not_cumulative" / "two" / "resting",
        Path("results") / "not_cumulative" / "phs" / "exercising",
        Path("results") / "not_cumulative" / "two" / "exercising",
    ]
    for file in files:
        # file = Path("data") / 'Counterfactual_batch3' / "litschau_MPI-ESM1-2-HR_counterfactual05.pkl.gz"
        file = Path(file)
        if not all((result_path / file.name).exists() for result_path in result_paths):
            files_analyse.append(str(file))
        if "data_for_federico_7Mar25" in str(
            file
        ) or "data_for_federico_24Feb25" in str(file):
            files_analyse.append(str(file))

    print(len(files_analyse))
    files_analyse.sort(key=lambda x: os.path.getsize(x))

    for file in tqdm(files_analyse, desc="Processing files"):
        # file = "data/Counterfactual_batch2/linz_FGOALS-g3_counterfactual50.pkl.gz"
        print(file)
        process_file(path_file=file, run_met=True, run_profiles=True)

    # with ThreadPoolExecutor() as executor:
    #     futures = {
    #         executor.submit(process_file, file): file for file in files_analyse
    #     }
    #     for future in tqdm(futures, desc="Processing files"):
    #         file = futures[future]
    #         try:
    #             future.result()  # Set a timeout of 300 seconds (5 minutes)
    #         except TimeoutError:
    #             print(f"Processing file {file} took too long and was terminated.")
    #         except Exception as e:
    #             print(f"Error processing file {file}: {e}")

if __name__ == "__plot__":

    max_condition = {"t": 0, "location": "", "rh": 0}
    median_temp = []
    missing_files = []
    for file in glob.glob("results/*.pkl.gz"):

        # # this function checks in which location the max core temperature was reached
        # print(file)

        # file = "results/not_cumulative_met_met_graz_obs_factual.pkl.gz"

        df_results = pd.read_pickle(file, compression="gzip")

        if df_results.shape[0] == 0:
            print("======={file}")
            missing_files.append(file)
        else:
            print(file)
            print(df_results.shape[0])

        f, axs = plt.subplots(1, 3, constrained_layout=True)
        df_results = df_results[df_results[Variables.col_t] < 21]
        for ix, var in enumerate(df_results.columns):
            sns.boxenplot(data=df_results[var], ax=axs[ix])
            axs[ix].set(title=var)
        f.suptitle(file)

        df_results = pd.read_pickle(file, compression="gzip")
        if df_results[Variables.col_t].max() > max_condition["t"]:
            max_condition["t"] = df_results[Variables.col_t].max()
            max_condition["rh"] = df_results.loc[
                df_results[Variables.col_t] == df_results[Variables.col_t].max(),
                Variables.col_rh,
            ]
            max_condition["location"] = file

        f, axs = plt.subplots(1, 1, constrained_layout=True, sharey=True)
        df_day = df_results.resample("1D").max()
        ax = axs
        for core_rise_threshold in Variables.threshold_core_temperature_rise:
            df_day["exceed"] = 0
            df_day.loc[
                df_day[Variables.col_t_core_rise] > core_rise_threshold, ["exceed"]
            ] = 1
            df_year = df_day.resample("1Y")["exceed"].sum().to_frame()
            sns.regplot(
                data=df_year,
                x=df_year.index.year,
                y="exceed",
                ax=ax,
                label=f"t_core_threshold {core_rise_threshold}",
            )

            ax.set(ylabel="count", xlabel="Year")
            for label in ax.get_xticklabels(which="major"):
                label.set(rotation=90, horizontalalignment="center")
            ax.grid(ls="--", lw=0.5)
            plt.legend(
                bbox_to_anchor=(0, 1.04, 1, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0,
                ncol=3,
                frameon=False,
            )
            sns.despine()

            # calculate the humidity ratio
            hr = []
            for t, rh in zip(
                df_results[Variables.col_t].values,
                (df_results[Variables.col_rh] / 100).values,
            ):
                hr.append(psyc.GetHumRatioFromRelHum(t, rh, 101325))
            df_results["hr"] = hr

            for year in [1995, 2020]:
                f, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7, 3))
                cbar_ax = f.add_axes([0.89, 0.1, 0.015, 0.8])
                cbar_plot = True

                df_field = df_results[df_results.index.year == year]
                df_field[f"hr"] *= 1000
                sns.histplot(
                    df_field,
                    x=Variables.col_t,
                    y="hr",
                    ax=ax,
                    cbar=cbar_plot,
                    cbar_kws={"label": "Hours", "shrink": 0.75},
                    binrange=((-20, 50), (0, 40)),
                    binwidth=(1, 1),
                    stat="count",
                    cbar_ax=cbar_ax,
                    cmap="viridis_r",
                )
                cbar_plot = False
                ax.set(
                    title=f"{year=}",
                    ylim=(0, 40),
                    xlim=(20, 42),
                    ylabel=r"HR $g_{H20}/kg_{dry air}$",
                    xlabel=r"$t_{db}$",
                )
                ax.grid(color="lightgray", ls="--", lw=0.5)
                plot_rh_lines(ax, rh_val=1)
                plot_rh_lines(ax, rh_val=0.75)
                plot_rh_lines(ax, rh_val=0.5)
                plot_rh_lines(ax, rh_val=0.25)
                sns.despine(ax=ax, bottom=True, left=True)
                # f.delaxes(ax[7])
                new_labels = []
                for label in cbar_ax.get_yticklabels():
                    new_labels.append(int(label._text) * 5)
                cbar_ax.set_yticklabels(new_labels)
                f.tight_layout(rect=[0, 0, 0.90, 1])
                # plot_constant_set_lines(
                #     ax,
                #     met=people_profiles["young"]["met"],
                #     clo=people_profiles["young"]["clo"],
                # )
            # plt.savefig(f"figures/psychrometric_comparison.png", dpi=300)

        plt.suptitle(file)

        plt.figure()
        sns.violinplot(df_results[Variables.col_t])
