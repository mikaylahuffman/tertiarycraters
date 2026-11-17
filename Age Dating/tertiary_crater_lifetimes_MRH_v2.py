#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import calibrate_degradation_measurement as cdm


#use a diff initial d/D for each axis
INIT_dD_SHORT = 0.45
INIT_dD_LONG = 0.32


#input & file paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "tertiary_craters_input_v2.csv")

kap = pd.read_csv(
    DATA / "kappa_trask_flat2.csv",
    header=None,
    usecols=[0, 1],
    names=["D", "nominal"],
)


#depth is calced from shadow length and solar incidence angle

depth = df["tert shadow length [m]"] / np.tan(np.deg2rad(55.26))

#detectability threshold
target_dD = 0.04

#uncertainties
PIXEL_SIZE_M = 0.88
SIGMA_SHADOW_M = PIXEL_SIZE_M / 2.0
SIGMA_DIAM_M = PIXEL_SIZE_M / 2.0

#fractional uncertainty in kappa
SIGMA_KAPPA_FRAC = 0.5

#Minton degradation calibration constants
A = 0.1333
P = 2.205

#solar incidence angle term
TAN_SUN = np.tan(np.deg2rad(55.26))


#funcs

def targetKfunc(init_dD):
    """
    Calculate the normalized degradation state K/r^2 at which
    a crater reaches the visibility threshold d/D = target_dD.

    init_dD is the assumed fresh-crater d/D for the relevant axis.
    """

    #ratio of cutoff d/D to initial/fresh d/D
    dd0_cutoff = target_dD / init_dD

    #normalized depth gets limited
    dd0_cutoff = np.clip(dd0_cutoff, 1e-10, 1)

    #normalized depth to normalized degradation state K/r^2
    target_K = cdm.K_vs_depth_over_depth_orig(
        dd0_cutoff,
        A,
        P,
    )

    return target_K


def lifetime(D, init_dD):
    """
    Calculate:
        current d/D,
        current normalized degradation K/r^2,
        remaining lifetime.

    D is crater diameter in meters.
    init_dD is the assumed fresh-crater d/D for the relevant axis.
    """

    #current measured d/D
    dD = depth / D

    #current depth normalized to assumed fresh depth
    dd0_now = dD / init_dD

    #clip
    dd0_now = np.clip(dd0_now, 1e-10, 1)

    #current normalized degradation state K/r^2
    K_now = cdm.K_vs_depth_over_depth_orig(
        dd0_now,
        A,
        P,
    )

    #normalized degradation state at visibility cutoff
    target_K = targetKfunc(init_dD)

    #remaining normalized degradation required to reach cutoff
    remaining_Kr2 = np.maximum(
        target_K - K_now,
        0,
    )

    #convert K/r^2 to dimensional K
    #where r = D/2
    K = remaining_Kr2 * (D / 2) ** 2

    #convert diameter from m to km for the kappa interpolation
    D_km = D / 1000

    #log-log interpolation of degradation rate kappa
    kappa = 10 ** np.interp(
        np.log10(D_km),
        np.log10(kap["D"]),
        np.log10(kap["nominal"]),
    )

    #remaining lifetime = remaining K / kappa
    return dD, K_now, K / kappa


#kappa funcs
def kappa_from_D(D):
    """
    Interpolate kappa as a function of crater diameter.
    """

    D_km = D / 1000

    kappa = 10 ** np.interp(
        np.log10(D_km),
        np.log10(kap["D"]),
        np.log10(kap["nominal"]),
    )

    return kappa


def kappa_slope(D):
    """
    Calculate the local slope d(log kappa) / d(log D)
    used to propagate diameter uncertainty through kappa.
    """

    D_km = np.asarray(D) / 1000

    x = np.log10(kap["D"].to_numpy())
    y = np.log10(kap["nominal"].to_numpy())

    xi = np.log10(D_km)

    i = np.searchsorted(x, xi) - 1
    i = np.clip(i, 0, len(x) - 2)

    slope = (
        (y[i + 1] - y[i])
        / (x[i + 1] - x[i])
    )

    return slope


#derivative of normalized degradation calibration

def gprime(dd0):
    """
    Derivative associated with the inverse degradation calibration.
    """

    return -(A / P) * dd0 ** (-1.0 / P - 1.0)


#unc prop

def sigma_t_classical(D, init_dD):
    """
    Propagate uncertainties in:
        shadow length,
        crater diameter,
        kappa

    into the remaining crater lifetime.

    init_dD is constant for a given axis:
        short = 0.45
        long  = 0.32
    """

    D = np.asarray(D)

    #current measured d/D
    dD = depth.to_numpy() / D

    #current normalized depth relative to fresh crater depth
    dd0_now = dD / init_dD

    #keep w/in calibration range
    dd0_now = np.clip(
        dd0_now,
        1e-10,
        1,
    )

    #current normalized degradation state K/r^2
    K_now = cdm.K_vs_depth_over_depth_orig(
        dd0_now,
        A,
        P,
    )

    #normalized degradation state at visibility cutoff
    target_K = targetKfunc(init_dD)

    #remaining normalized degradation
    remaining_Kr2 = np.maximum(
        target_K - K_now,
        0,
    )

    #dimensional K
    K = remaining_Kr2 * (D / 2) ** 2

    #kappa and its local diameter dependence
    kappa = kappa_from_D(D)
    slope = kappa_slope(D)

    dkappa_dD = kappa * slope / D

    #derivative of normalized degradation calibration
    gp = gprime(dd0_now)


    #partial derivative wrt shadow length

    dt_dL = -(
        D
        / (4.0 * kappa * TAN_SUN * init_dD)
    ) * gp


    #partial derivative wrt diameter


    dt_dD = 0.25 * (
        (
            (D * dd0_now * gp)
            + (2.0 * D * remaining_Kr2)
        )
        / kappa
        -
        (
            (remaining_Kr2 * D**2)
            / kappa**2
        )
        * dkappa_dD
    )


    #partial derivative wrt kappa


    dt_dkappa = -K / kappa**2

    sigma_kappa = SIGMA_KAPPA_FRAC * kappa

    #if the crater has already reached/passed the visibility threshold, remaining lifetime is zero, so set propagated lifetime uncertainties to zero
    dt_dL = np.where(
        remaining_Kr2 > 0,
        dt_dL,
        0.0,
    )

    dt_dD = np.where(
        remaining_Kr2 > 0,
        dt_dD,
        0.0,
    )

    dt_dkappa = np.where(
        remaining_Kr2 > 0,
        dt_dkappa,
        0.0,
    )

    #measurement contribution from shadow length and diameter
    sigma_measurement = np.sqrt(
        (dt_dL * SIGMA_SHADOW_M) ** 2
        + (dt_dD * SIGMA_DIAM_M) ** 2
    )

    #kappa contribution
    sigma_kappa_part = np.abs(
        dt_dkappa * sigma_kappa
    )

    #total propagated uncertainty
    sigma_total = np.sqrt(
        sigma_measurement**2
        + sigma_kappa_part**2
    )

    return (
        sigma_measurement,
        sigma_kappa_part,
        sigma_total,
    )


#calc results

results = {}
uncertainties = {}

init_dD_by_axis = {
    "short": INIT_dD_SHORT,
    "long": INIT_dD_LONG,
}

for axis in ["short", "long"]:

    D = df[f"tert diam {axis} axis [m]"]

    init_dD = init_dD_by_axis[axis]

    results[axis] = lifetime(
        D,
        init_dD,
    )

    uncertainties[axis] = sigma_t_classical(
        D,
        init_dD,
    )


#unpack

dD_short, Kr2_short, t_short = results["short"]
dD_long, Kr2_long, t_long = results["long"]

(
    sig_meas_short,
    sig_kap_short,
    sig_total_short,
) = uncertainties["short"]

(
    sig_meas_long,
    sig_kap_long,
    sig_total_long,
) = uncertainties["long"]


#output df

out = pd.DataFrame(
    {
        "crater ID": df["ID"],
        "depth [m]": depth,

        "d/D short": dD_short,
        "d/D long": dD_long,
        "d/D uncertainty": np.abs(
            dD_short - dD_long
        ),

        "K/r^2 short": Kr2_short,
        "K/r^2 long": Kr2_long,
        "K/r^2 uncertainty": np.abs(
            Kr2_short - Kr2_long
        ),

        "t short [Myr]": t_short,
        "t long [Myr]": t_long,
        "t uncertainty [Myr]": np.abs(
            t_short - t_long
        ),

        "sigma t measurement short [Myr]": sig_meas_short,
        "sigma t kappa short [Myr]": sig_kap_short,
        "sigma t total short [Myr]": sig_total_short,

        "sigma t measurement long [Myr]": sig_meas_long,
        "sigma t kappa long [Myr]": sig_kap_long,
        "sigma t total long [Myr]": sig_total_long,
    }
)


#save

out.to_csv(
    DATA / "tertiary_crater_lifetimes_output_unc_MRH_v2.csv",
    index=False,
)