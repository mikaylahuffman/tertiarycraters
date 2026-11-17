from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cratermaker import Crater, Simulation
from cratermaker.utils.general_utils import format_large_units
from scipy.optimize import curve_fit
from tqdm import tqdm


# These define the fitting functions for the depth-to-diameter vs K relationship, both the original function and its inverse, which are used for fitting and for measuring the degradation state of craters, respectively. This fitting function is somewhat different than the one used in Minton et al. (2019), because instead of using depth-to-diameter, it uses depth-to-original-depth. This reduces the free parameters of the fit from 2 to 1 (in the old one, I had an `a` and `b` parameter, but now I only have an `a` parameter). This also makes the fitting more robust, because the initial depth-to-diameter varies with crater size, while the initial depth-to-original-depth is always 1 by definition.
def depth_over_depth_orig_vs_K(K_r2, a, p):
    return (a / (K_r2 + a)) ** p


def depth_over_depth_orig_vs_K_label(a, p):
    return f"Fit: $\\left[a / (K/r^2 + a)\\right]^p$\n$a$ = ${a:.4f}$\n$p$ = ${p:.4f}$"


def K_vs_depth_over_depth_orig(depth_over_depth_orig, a, p):
    return a * (depth_over_depth_orig ** (-1 / p) - 1.0)


# _DTR is the diameter in km where the complex crater a vs d curve meets up with the simple crater constant curve. I found that this is best set somewhat before the simple-to-complex transition because the craters begin to deviate from geometric similarity above about ~10 km.
_DTR = 10.0e3


def a_vs_diameter_simple(diameter, s0, s1):
    return s0 + s1 * np.log(diameter)


def a_vs_diameter_simple_label(s0, s1):
    first_line = "Fit: $s_0 + s_1 \\ln D$"
    s0line = "$s_0$ = " + f"${s0:.4g}$"
    s1line = "$s_1$ = " + f"${s1:.4g}$"
    return "\n".join([first_line, s0line, s1line])


def p_vs_diameter_simple(diameter, p0, p1):
    return p0 + p1 * np.log(diameter)


def p_vs_diameter_simple_label(p0, p1):
    first_line = "Fit: $p_0 + p_1 \\ln D$"
    p0line = "$p_0$ = " + f"${p0:.4g}$"
    p1line = "$p_1$ = " + f"${p1:.4g}$"
    return "\n".join([first_line, p0line, p1line])


def a_vs_diameter_complex(diameter, c0, c1, d_scale):
    dkm = diameter * 1e-3
    d0km = _DTR * 1e-3
    return c0 + c1 * (1.0 - np.exp((d0km - dkm) / d_scale))


def a_vs_diameter_complex_label(c0, c1, d_scale):
    first_line = "Fit: $c_0 + c_1 \\left[ 1 - \\exp \\left( \\frac{{ D_0-D }} {{ D_{{scale}} }} \\right) \\right]$"
    D0line = "$D_0$ = " + format_large_units(_DTR, quantity="length")
    Dscaleline = "$D_{{scale}}$ = " + format_large_units(
        d_scale * 1e3, quantity="length"
    )
    c0line = "$c_0$ = " + f"${c0:.4g}$"
    c1line = "$c_1$ = " + f"${c1:.4g}$"
    return "\n".join([first_line, D0line, Dscaleline, c0line, c1line])


def measure_degradation_state(crater: Crater, a: float, p: float) -> float:
    depth_over_depth_orig = crater.measured_depth_to_diameter / crater.depth_to_diameter
    K = (
        K_vs_depth_over_depth_orig(depth_over_depth_orig, a, p)
        * crater.measured_radius**2
    )
    crater.degradation_state = K

    return K


def do_diffusion_simulation(
    sim: Simulation,
    diameter: float,
    K_r2_vals: list[float],
    save_intermediate: bool,
    a: float | None = None,
    p: float | None = None,
) -> list[float] | tuple[list[float], list[float]]:
    """
    Perform a diffusion simulation on a crater of a given diameter, applying the specified K/r² values as diffusive degradation steps, and measuring the depth-to-original-depth ratio at each step.

    If `a` is provided, also measure the degradation state using the `measure_degradation_state` function and return it along with the depth_over_original_depth values.

    Parameters
    ----------
    sim : Simulation
        The Simulation object that contains the surface that the crater will be emplaced on and diffused.
    diameter : float
        The diameter of the crater that will be emplaced and diffused.
    K_r2_vals : list[float]
        The list of K/r² values that will be applied as diffusive degradation steps to the crater. The actual K values applied will be computed by multiplying these values by the crater radius squared, so that the degradation state is normalized by crater size.
    save_intermediate : bool
        Whether to save the intermediate results of the Cratermaker simulation for each K/r² value. This is useful for visualizing the evolution of the crater, but will slow down the analysis because it will generating hillshade image of the surface at each intermediate value of K.
    a : float | None, optional
        The fit parameter `a` that will be used to measure the degradation state of the crater at each step using the `measure_degradation_state` function. If None, the degradation state will not be measured and only the depth_over_depth_orig values will be returned. Default is None.
    p : float | None, optional
        The fit parameter `p` that will be used to measure the degradation state of the crater at each step using the `measure_degradation_state` function. If None, the degradation state will not be measured and only the depth_over_depth_orig values will be returned. Default is None.

    Returns
    -------
    list[float]
        The list of depth-to-original-depth values measured at each step of the diffusion simulation.
    list[float], optional
        The list of measured degradation state values normalized by crater radius squared at each step of the diffusion simulation, or None if `a` is None.
    """
    do_K_measure = a is not None and p is not None
    sim.reset()
    crater = sim.emplace(diameter=diameter, location=(0, 0))[0]
    if save_intermediate:
        sim.save()

    # Compute the absolute values of K and the dK values that will be applied to the surface to diffusively degrade the crater
    r2 = crater.radius**2
    K_vals = K_r2_vals * r2
    dK_vals = np.diff(K_vals)
    dK_vals = np.append(np.array([0.0]), dK_vals)

    # On the verification step, a will be supplied so we can measure K with it.
    if do_K_measure:
        K_measured = []
    depth_over_depth_orig_vals = []
    for K, dK in zip(K_vals, dK_vals, strict=True):
        if K > 0.0:  # Allows the saving of initial conditions in the first step
            sim.surface.local.apply_diffusion(dK)
            if save_intermediate:
                sim.interval += 1
                sim.save()
        depth_over_depth_orig_vals.append(
            crater.measured_depth_to_diameter / crater.depth_to_diameter
        )
        if do_K_measure:
            K_measured.append(measure_degradation_state(crater, a, p) / r2)

    if do_K_measure:
        return depth_over_depth_orig_vals, K_measured
    else:
        return depth_over_depth_orig_vals


def fit_depth_over_depth_orig_vs_K(
    crater_type: Literal["simple", "complex"],
    diam_vals: list[float],
    K_r2_vals: list[float],
    pix_vals: list[float],
    old_params: dict[str, float] | None = None,
    save_intermediate: bool = False,
) -> xr.Dataset:
    """
    Compute the depth_over_depth_orig values as a function of K using Cratermaker's apply_diffusion code as part of the first pass to determine fitting parameters.

    For simple craters, which are geometrically self-similar, we will use a fixed value of pix and vary diameter to verify that the fit parameter does not have a strong dependence on resolution. For complex craters, whose geometry is size-dependent, we will fix the ratio of crater diameter to pix so we can find how the fit parameter depends on diameter.

    Parameters
    ----------
    crater_type: Literal["simple", "complex"]
        The type of crater to analyze. For simple craters, it is assumed that the mean of the fit parameter will be found and plotted. For complex craters, each diameter will get its own fit parameter.
    diam_vals : list[float]
        The list of diameter values for which a fitting parameter will be found.
    K_r2_vals : list[float]
        The list of K/r² values used for the diffusion test for each crater diameter.
    pix_vals : list[float]
        The value pix associated with each diameter value.
    old_params : list[float]
        The fit parameters from the simple crater runs that are passed to the complex crater runs so that they can be consistent with each other.
    save_intermediate : bool, optional
        Whether to save the intermediate results of Cratermaker simulations for each diameter and K/r² value. This is useful for visualizing the evolution of each crater, but will slow down the analysis because it will generating hillshade image of the surface at each intermediate value of K. Default is False.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the results of the fit, including the diameter values, the K/r² values, the measured depth_to_diameter values, and the fit parameter `a`.
    """
    if crater_type not in ["simple", "complex"]:
        raise ValueError("crater_type must be either 'simple' or 'complex")
    print(
        f"Starting {crater_type} crater analysis from D = {format_large_units(diam_vals[0], quantity='length')} to {format_large_units(diam_vals[-1], quantity='length')}"
    )
    simdir_prefix = f"{crater_type}_crater_"
    depth_over_depth_orig_runs = []
    avals = []
    pvals = []
    sims = []
    rng = np.random.default_rng(seed=71234325322)
    for num, diameter, pix in tqdm(
        zip(range(len(diam_vals)), diam_vals, pix_vals, strict=True),
        total=len(diam_vals),
        desc="Getting depth_over_depth_orig vs K for each diameter",
    ):
        # Generate a new simulation for this crater size. The simulation parameters are set so that the surfaces have roughly the same number of faces and the craters are roughly the same pixel size.
        local_radius = diameter * 2.0
        simdir = f"{simdir_prefix}{num:03d}"
        sim = Simulation(
            surface="hireslocal",
            simdir=simdir,
            local_location=(0, 0),
            pix=pix,
            local_radius=local_radius,
            ask_overwrite=False,
            reset=True,
            rng=rng,
        )
        sims.append(sim)

        depth_over_depth_orig_vals = do_diffusion_simulation(
            sim, diameter, K_r2_vals, save_intermediate=False
        )

        # Ignore the beginning of the curve because we want the fit to be best near the limit of the crater's visibility, rather than at the beginning of the crater's history
        fit_start_index = len(K_r2_vals) // 2
        if crater_type == "simple":
            d_vs_K_params, _ = curve_fit(
                depth_over_depth_orig_vs_K,
                xdata=K_r2_vals[fit_start_index:],
                ydata=depth_over_depth_orig_vals[fit_start_index:],
                p0=[0.14, 2.0],
            )
            avals.append(d_vs_K_params[0])
            pvals.append(d_vs_K_params[1])
        elif crater_type == "complex":
            p0 = old_params["p0"]
            p1 = old_params["p1"]
            p = p_vs_diameter_simple(
                _DTR, p0, p1
            )  # Connect up the p value curve to the last one from the simple crater run
            d_vs_K_params, _ = curve_fit(
                lambda K_r2, a: depth_over_depth_orig_vs_K(K_r2, a, p),
                xdata=K_r2_vals[fit_start_index:],
                ydata=depth_over_depth_orig_vals[fit_start_index:],
                p0=[p0],
            )
            avals.append(d_vs_K_params[0])
            pvals.append(p)
        depth_over_depth_orig_runs.append(depth_over_depth_orig_vals)

    # Save everything to a Dataset with metadata so we can generate nice plots
    ds = xr.Dataset(
        data_vars={
            "depth_over_depth_orig": (["diameter", "K_r2"], depth_over_depth_orig_runs),
            "a": (["diameter"], avals),
            "p": (["diameter"], pvals),
            "diameter_km": (["diameter"], diam_vals * 1e-3),
        },
        coords={"diameter": diam_vals, "K_r2": K_r2_vals},
    )
    ds = ds.set_coords("diameter_km")
    ds.K_r2.attrs["long_name"] = "Applied degradation"
    ds.K_r2.attrs["unit"] = "$K/r^2$"
    ds.depth_over_depth_orig.attrs["long_name"] = (
        "Ratio of measured depth to original depth"
    )
    ds.depth_over_depth_orig.attrs["unit"] = "$d/d_{orig}$"
    ds.a.attrs["long_name"] = "Fit parameter $a$"
    ds.p.attrs["long_name"] = "Fit parameter $p$"

    ds.diameter.attrs["long_name"] = "Crater diameter"
    ds.diameter.attrs["unit"] = "m"

    ds.diameter_km.attrs["long_name"] = "Crater diameter"
    ds.diameter_km.attrs["unit"] = "km"

    if crater_type == "simple":
        diam_var = "diameter_km"
    else:
        diam_var = "diameter_km"

    if crater_type == "simple":
        a_vs_d_params, _ = curve_fit(
            a_vs_diameter_simple,
            xdata=diam_vals,
            ydata=avals,
            p0=[0.1155, 0.02],
        )
        s0 = a_vs_d_params[0]
        s1 = a_vs_d_params[1]
        p_vs_d_params, _ = curve_fit(
            p_vs_diameter_simple,
            xdata=diam_vals,
            ydata=pvals,
            p0=[2.0, 0.1],
        )
        p0 = p_vs_d_params[0]
        p1 = p_vs_d_params[1]
        fit_params = {"s0": s0, "s1": s1, "p0": p0, "p1": p1}
    elif crater_type == "complex":
        a_vs_d_params, _ = curve_fit(
            lambda diameter, c0, c1, d_scale: a_vs_diameter_complex(
                diameter, c0, c1, d_scale
            ),
            xdata=diam_vals,
            ydata=avals,
            p0=[0.28, 0.103, 180.0],
        )
        c0 = a_vs_d_params[0]
        c1 = a_vs_d_params[1]
        d_scale = a_vs_d_params[2]
        fit_params = {"c0": c0, "c1": c1, "d_scale": d_scale, **old_params}

    ########################################################
    #                     a vs D plot                      #
    ########################################################
    fig, ax = plt.subplots(layout="constrained")
    ds.a.plot.scatter(x=diam_var, ax=ax, edgecolor=None, linewidths=0)
    ax.set_ylim(0.0, 1.5 * ds.a.max())
    xvals = ds[diam_var].data
    if crater_type == "simple":
        yvals = a_vs_diameter_simple(diam_vals, *a_vs_d_params)
    else:
        yvals = a_vs_diameter_complex(diam_vals, *a_vs_d_params)
    if crater_type == "simple":
        ax.semilogx(
            xvals, yvals, "--k", label=a_vs_diameter_simple_label(*a_vs_d_params)
        )
    else:
        ax.semilogx(
            xvals, yvals, "--k", label=a_vs_diameter_complex_label(*a_vs_d_params)
        )
    ax.legend()
    plt.title(f"{crater_type.title()} crater fit $a$ parameter vs diameter")
    plt.savefig(f"a_vs_diameter-{crater_type}.svg", transparent=True)
    plt.savefig(f"a_vs_diameter-{crater_type}.png", dpi=300)

    ########################################################
    #                     p vs D plot                      #
    ########################################################
    if crater_type == "simple":
        fig, ax = plt.subplots(layout="constrained")
        ds.p.plot.scatter(x=diam_var, ax=ax, edgecolor=None, linewidths=0)
        ax.set_ylim(0.0, 1.5 * ds.p.max())
        xvals = ds[diam_var].data
        yvals = p_vs_diameter_simple(diam_vals, *p_vs_d_params)
        ax.semilogx(
            xvals, yvals, "--k", label=p_vs_diameter_simple_label(*p_vs_d_params)
        )
        ax.legend()
        plt.title(f"{crater_type.title()} crater fit $p$ parameter vs diameter")
        plt.savefig(f"p_vs_diameter-{crater_type}.svg", transparent=True)
        plt.savefig(f"p_vs_diameter-{crater_type}.png", dpi=300)

    ########################################################
    #                     d vs K plot                      #
    ########################################################
    fig, ax = plt.subplots(layout="constrained")
    cmap = plt.get_cmap("viridis")
    for diameter in diam_vals:
        color = cmap(
            (np.log(diameter) - np.log(diam_vals.min()))
            / (np.log(diam_vals.max()) - np.log(diam_vals.min()))
        )
        a = ds.a.sel(diameter=diameter).data.item()
        p = ds.p.sel(diameter=diameter).data.item()
        ax.plot(
            ds.K_r2.data,
            depth_over_depth_orig_vs_K(ds.K_r2.data, a, p),
            color=color,
            linestyle="-",
            linewidth=0.2,
        )
    ds["depth_over_depth_orig"].plot.scatter(
        x="K_r2",
        y="depth_over_depth_orig",
        hue=diam_var,
        ax=ax,
        edgecolor=None,
        linewidths=0,
        marker=".",
    )
    plt.title(f"{crater_type.title()} crater depth metric vs degradation state")
    plt.savefig(f"depth_over_depth_orig_vs_K-{crater_type}.svg", transparent=True)
    plt.savefig(f"depth_over_depth_orig_vs_K-{crater_type}.png", dpi=300)

    # Verify fit parameters
    print(f"\nVerifying fit parameters for {crater_type} craters")
    K_measured_runs = []
    diam_vals = ds.diameter.values
    K_r2_vals = ds.K_r2.values
    a_vals = ds.a.values
    if crater_type == "simple":
        a_vals = [
            a_vs_diameter_simple(diameter, *a_vs_d_params) for diameter in diam_vals
        ]
        p_vals = [
            p_vs_diameter_simple(diameter, *p_vs_d_params) for diameter in diam_vals
        ]
    elif crater_type == "complex":
        a_vals = [
            a_vs_diameter_complex(diameter, *a_vs_d_params) for diameter in diam_vals
        ]
        p_vals = ds.p.data
    for diameter, sim, a, p in tqdm(
        zip(diam_vals, sims, a_vals, p_vals, strict=True),
        total=len(diam_vals),
        desc="Verifying fit parameters",
    ):
        depth_over_depth_orig_vals, K_measured_vals = do_diffusion_simulation(
            sim, diameter, K_r2_vals, save_intermediate, a=a, p=p
        )
        K_measured_runs.append(K_measured_vals)
    ds["K_r2_measured"] = xr.DataArray(data=K_measured_runs, dims=["diameter", "K_r2"])
    ds.K_r2_measured.attrs["long_name"] = "Measured degradation"
    ds.K_r2_measured.attrs["unit"] = "$K/r^2$"

    ########################################################
    #          K measured vs K applied plot                #
    ########################################################
    fig, ax = plt.subplots(layout="constrained")
    ax.set_xlim(0, K_r2_vals.max())
    ax.set_ylim(0, K_r2_vals.max())

    # Plot the 1:1 line
    ax.plot(ds.K_r2, ds.K_r2, "--k")

    # Plot the visibility function lines
    _KV1 = 0.30
    maxval = _KV1 / K_r2_vals.max()
    ax.axvline(x=_KV1, ymax=maxval, linestyle=":", color="k")
    ax.axhline(y=_KV1, xmax=maxval, linestyle=":", color="k")
    ax.annotate(
        "Visibility function",
        xy=(0.5 * _KV1, _KV1),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
    )
    ds.K_r2_measured.plot.scatter(
        x="K_r2",
        hue=diam_var,
        ax=ax,
        add_legend=False,
        add_colorbar=True,
        edgecolor=None,
        linewidths=0,
    )

    plt.title(f"{crater_type.title()} crater fit verification.")
    plt.savefig(f"K_measured_vs_K_applied-{crater_type}.svg", transparent=True)
    plt.savefig(f"K_measured_vs_K_applied-{crater_type}.png", dpi=300)
    plt.show()

    # Save and overwrite the ds as well as returning it
    outfile = Path(f"calibrate_degradation_measurement-{crater_type}.nc")
    outfile.unlink(missing_ok=True)
    ds.to_netcdf(outfile)

    return ds, fit_params


if __name__ == "__main__":
    # For each crater of a given size, I scale the total amount of diffusive degradation to be much higher than the visibility function value so that I can get a good fit for empirical functions. I also will be normalizing all degradation state values by crater radius squared, so that we have a unitless measure of degradation that can be compared across crater sizes. The fitting function is then used to fit the relationship between the normalized degradation state (K/r^2) and the ratio of measured depth to original depth (d/d_orig).
    ndiffusion_steps = 50
    ncraters_per_type = 10
    nrepeats_per_diameter = 10
    K_r2_vals = np.linspace(0, 0.5, ndiffusion_steps)

    # The simple crater fits have a different dependence on diameter than the complex crater fits. Therefore we have to separate the analysis into two. The fitting parameters obtained from the largest simple crater
    crater_type = "simple"

    diam_vals = np.logspace(np.log10(40.0), np.log10(_DTR), ncraters_per_type).repeat(
        nrepeats_per_diameter
    )
    rng = np.random.default_rng(seed=122309872)
    diam_vals += rng.normal(
        scale=diam_vals * 0.0001, size=diam_vals.shape
    )  # Add a little bit of noise to the diameters so we can use them as a coordinate dimension in the dataset
    diam_vals = np.sort(diam_vals)
    diameter_over_pix = 50.0
    pix_vals = [diameter / diameter_over_pix for diameter in diam_vals]

    ds, simple_params = fit_depth_over_depth_orig_vs_K(
        crater_type, diam_vals, K_r2_vals, pix_vals=pix_vals, save_intermediate=False
    )

    crater_type = "complex"

    diam_vals = np.logspace(np.log10(_DTR), np.log10(1e6), ncraters_per_type)
    pix_vals = [diameter / diameter_over_pix for diameter in diam_vals]

    ds, complex_params = fit_depth_over_depth_orig_vs_K(
        crater_type,
        diam_vals,
        K_r2_vals,
        pix_vals=pix_vals,
        old_params=simple_params,
        save_intermediate=False,
    )

    print("Complete parameters:")
    for key, value in complex_params.items():
        print(f"{key} = {value}")
