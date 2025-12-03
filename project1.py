import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats


# -------------- Helper functions -------------- #

def parse_manual_data(text: str):
    """Parse numbers from a text box (commas, spaces, newlines allowed)."""
    if not text or text.strip() == "":
        return np.array([])
    tokens = re.split(r"[,\s]+", text.strip())
    vals = []
    for t in tokens:
        if t == "":
            continue
        try:
            vals.append(float(t))
        except ValueError:
            # Ignore non-numeric junk
            continue
    return np.array(vals, dtype=float)


def load_csv_data(file) -> np.ndarray:
    """Load numeric data from a CSV, return 1D numpy array."""
    if file is None:
        return np.array([])

    try:
        df = pd.read_csv(file)
    except Exception:
        return np.array([])

    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        return np.array([])

    # If multiple numeric columns, let user choose
    col_name = st.selectbox(
        "Choose column from CSV",
        num_df.columns,
        key="csv_column_select"
    )
    col = num_df[col_name].dropna().values.astype(float)
    return col


def get_distributions():
    """Return dict: name -> (scipy.stats distribution object, nice label)."""
    return {
        "norm": (stats.norm, "Normal"),
        "gamma": (stats.gamma, "Gamma"),
        "beta": (stats.beta, "Beta"),
        "weibull_min": (stats.weibull_min, "Weibull (min)"),
        "weibull_max": (stats.weibull_max, "Weibull (max)"),
        "lognorm": (stats.lognorm, "Lognormal"),
        "expon": (stats.expon, "Exponential"),
        "uniform": (stats.uniform, "Uniform"),
        "pareto": (stats.pareto, "Pareto"),
        "triang": (stats.triang, "Triangular"),
        "chi2": (stats.chi2, "Chi-square"),
    }


def compute_fit_error(hist_vals, bin_edges, pdf_vals):
    """Return MSE and max error between histogram (density) and pdf."""
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    x_pdf = np.linspace(bin_edges[0], bin_edges[-1], len(pdf_vals))
    pdf_at_centers = np.interp(centers, x_pdf, pdf_vals)

    mse = np.mean((hist_vals - pdf_at_centers) ** 2)
    max_err = np.max(np.abs(hist_vals - pdf_at_centers))
    residuals = hist_vals - pdf_at_centers
    return mse, max_err, centers, residuals


def qq_data(data, frozen_dist):
    """Return x (theoretical) and y (empirical) for a Q–Q plot."""
    data_sorted = np.sort(data)
    n = len(data_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo_quantiles = frozen_dist.ppf(probs)
    return theo_quantiles, data_sorted


# -------------- Streamlit App -------------- #

st.set_page_config(
    page_title="Histogram Fitting Tool",
    layout="wide"
)

# Simple light aesthetic styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1200px;
    }
    .section-box {
        background-color: #ffffff;
        border: 1px solid #d7dde8;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.06);
    }
    .section-title {
        font-weight: 600;
        color: #1f2937;
        border-left: 4px solid #4f46e5;
        padding-left: 0.5rem;
        margin-bottom: 0.75rem;
        font-size: 1.05rem;
    }
    .main-title {
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">Histogram Fitting Webapp</h1>', unsafe_allow_html=True)
st.markdown(
    """
    Fit different probability distributions to your data, 
    compare fits, and manually tweak parameters.

    **Steps:**
    1. Load data (manual or CSV)  
    2. Pick a main distribution and optional comparisons  
    3. Use *Automatic fit* to see best-fit parameters  
    4. Use *Manual fit* to adjust parameters with sliders
    """
)

st.markdown("---")

# -------- Data input (left column) -------- #
col_data, col_options = st.columns([1.2, 1])

with col_data:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1. Data Input</div>', unsafe_allow_html=True)

    source = st.radio(
        "Choose data source:",
        ["Manual entry", "Upload CSV"],
        horizontal=True
    )

    data = np.array([])

    if source == "Manual entry":
        sample_hint = "Example: 1.2, 1.5, 2.7  3.0 (commas / spaces / newlines)."
        text = st.text_area(
            "Enter numeric data:",
            value="",
            height=150,
            help=sample_hint,
        )
        data = parse_manual_data(text)
    else:
        uploaded = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            accept_multiple_files=False
        )
        if uploaded is not None:
            data = load_csv_data(uploaded)

    if data.size == 0:
        st.warning("No valid data yet. Enter numbers or upload a valid CSV.")
    else:
        st.success(f"Loaded {data.size} data points.")
        st.write(
            f"Min: {np.min(data):.3f} | "
            f"Max: {np.max(data):.3f} | "
            f"Mean: {np.mean(data):.3f} | "
            f"Std: {np.std(data, ddof=1):.3f}"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Distribution + plotting options (right) -------- #
with col_options:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2. Distribution & Plot Settings</div>', unsafe_allow_html=True)

    dist_dict = get_distributions()
    dist_keys = list(dist_dict.keys())

    # main distribution (used for auto + manual + diagnostics)
    dist_key = st.selectbox(
        "Main distribution",
        dist_keys,
        format_func=lambda k: dist_dict[k][1]
    )
    dist_obj, dist_label = dist_dict[dist_key]

    # extra dists to overlay and rank
    extra_keys = st.multiselect(
        "Extra distributions to compare (overlay, optional)",
        [k for k in dist_keys if k != dist_key],
        default=[],
        format_func=lambda k: dist_dict[k][1],
        help="These will be auto-fitted and drawn on the same plot, with a ranking table."
    )

    bins = st.slider("Number of histogram bins", 5, 100, 30, step=1)

    # Plot range
    if data.size > 0:
        data_min, data_max = float(np.min(data)), float(np.max(data))
    else:
        data_min, data_max = 0.0, 1.0

    padding = 0.1 * (data_max - data_min + 1e-9)
    x_min = st.number_input(
        "Plot min (x-axis)",
        value=data_min - padding
    )
    x_max = st.number_input(
        "Plot max (x-axis)",
        value=data_max + padding
    )
    x_min, x_max = float(x_min), float(x_max)
    if x_max <= x_min:
        x_max = x_min + 1.0

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Tabs: Auto-fit and Manual-fit
tab_auto, tab_manual = st.tabs(["Automatic fit", "Manual fit"])

# -------- Automatic fit tab -------- #
with tab_auto:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">3. Automatic Fitting</div>', unsafe_allow_html=True)

    if data.size == 0:
        st.info("Provide data first to run the automatic fit.")
    else:
        # histogram (common for all fits)
        hist_vals, bin_edges = np.histogram(
            data,
            bins=bins,
            range=[x_min, x_max],
            density=True
        )

        x_grid = np.linspace(x_min, x_max, 400)

        results = []

        # ---- main distribution fit ---- #
        try:
            params = dist_obj.fit(data)
        except Exception as e:
            st.error(f"Fitting failed for {dist_label}: {e}")
            params = None

        if params is not None:
            shape_names = []
            if dist_obj.shapes is not None:
                shape_names = [s.strip() for s in dist_obj.shapes.split(",") if s.strip()]
            n_shapes = len(shape_names)
            shape_params = params[:n_shapes]
            loc = params[n_shapes]
            scale = params[n_shapes + 1]

            frozen = dist_obj(*shape_params, loc=loc, scale=scale)
            pdf_vals = frozen.pdf(x_grid)

            mse, max_err, centers, residuals = compute_fit_error(
                hist_vals, bin_edges, pdf_vals
            )
            results.append({
                "key": dist_key,
                "label": dist_label,
                "params": params,
                "shape_names": shape_names,
                "mse": mse,
                "max_err": max_err,
                "pdf": pdf_vals,
            })

            # plot main + extras
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                data,
                bins=bins,
                range=[x_min, x_max],
                density=True,
                alpha=0.4,
                edgecolor="black",
                label="Data (histogram)"
            )
            ax.plot(x_grid, pdf_vals, linewidth=2, label=f"{dist_label} (main)")

            # ---- fit and draw extra distributions ---- #
            for key in extra_keys:
                obj, label = dist_dict[key]
                try:
                    p = obj.fit(data)
                except Exception as e:
                    st.warning(f"Could not fit {label}: {e}")
                    continue

                # interpret params
                s_names = []
                if obj.shapes is not None:
                    s_names = [s.strip() for s in obj.shapes.split(",") if s.strip()]
                n_s = len(s_names)
                shp = p[:n_s]
                loc2 = p[n_s]
                scale2 = p[n_s + 1]

                frozen2 = obj(*shp, loc=loc2, scale=scale2)
                pdf2 = frozen2.pdf(x_grid)

                mse2, max_err2, _, _ = compute_fit_error(
                    hist_vals, bin_edges, pdf2
                )

                results.append({
                    "key": key,
                    "label": label,
                    "params": p,
                    "shape_names": s_names,
                    "mse": mse2,
                    "max_err": max_err2,
                    "pdf": pdf2,
                })

                ax.plot(
                    x_grid,
                    pdf2,
                    linewidth=1.5,
                    linestyle="--",
                    label=f"{label} (MSE={mse2:.2e})"
                )

            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title("Automatic Fits: histogram + fitted PDFs")
            ax.legend()
            st.pyplot(fig)

            # ---- parameters + ranking for main dist ---- #
            param_names = shape_names + ["loc", "scale"]
            param_values = list(shape_params) + [loc, scale]
            df_params = pd.DataFrame(
                {"Parameter": param_names, "Value": [f"{v:.4g}" for v in param_values]}
            )
            st.markdown("**Main distribution parameters**")
            st.table(df_params)

            # ---- ranking table for all fitted dists ---- #
            if results:
                df_rank = pd.DataFrame(
                    {
                        "Distribution": [r["label"] for r in results],
                        "MSE": [r["mse"] for r in results],
                        "Max error": [r["max_err"] for r in results],
                    }
                ).sort_values("MSE")
                st.markdown("**Fit comparison (lower MSE is better)**")
                st.table(df_rank.reset_index(drop=True))
                best_label = df_rank.iloc[0]["Distribution"]
                st.success(f"Best fit (by MSE): **{best_label}**")

            # ---- diagnostics for main distribution ---- #
            with st.expander("Advanced diagnostics for main distribution"):
                c1, c2 = st.columns(2)

                # Residual plot
                with c1:
                    fig_res, ax_res = plt.subplots(figsize=(4, 3))
                    ax_res.axhline(0, linewidth=1)
                    ax_res.stem(centers, residuals)
                    ax_res.set_xlabel("Bin center")
                    ax_res.set_ylabel("Residual (hist - pdf)")
                    ax_res.set_title("Residuals")
                    st.pyplot(fig_res)

                # Q–Q plot
                with c2:
                    qq_x, qq_y = qq_data(data, frozen)
                    fig_qq, ax_qq = plt.subplots(figsize=(4, 3))
                    ax_qq.plot(qq_x, qq_y, ".", markersize=4)
                    min_q = min(np.min(qq_x), np.min(qq_y))
                    max_q = max(np.max(qq_x), np.max(qq_y))
                    ax_qq.plot([min_q, max_q], [min_q, max_q], linewidth=1)
                    ax_qq.set_xlabel("Theoretical quantiles")
                    ax_qq.set_ylabel("Empirical quantiles")
                    ax_qq.set_title("Q–Q plot")
                    st.pyplot(fig_qq)

        else:
            st.info("Main distribution could not be fitted; try a different one.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------- Manual fit tab -------- #
with tab_manual:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">4. Manual Fitting (Sliders)</div>', unsafe_allow_html=True)

    if data.size == 0:
        st.info("Provide data first to use manual fitting.")
    else:
        # Base stats to choose slider ranges
        dmin, dmax = float(np.min(data)), float(np.max(data))
        drange = dmax - dmin if dmax > dmin else 1.0
        dstd = float(np.std(data, ddof=1)) if data.size > 1 else drange / 4

        # Try to get auto-fit params as defaults (if possible)
        default_params = None
        try:
            default_params = dist_obj.fit(data)
        except Exception:
            pass

        shape_names = []
        if dist_obj.shapes is not None:
            shape_names = [s.strip() for s in dist_obj.shapes.split(",") if s.strip()]
        n_shapes = len(shape_names)

        # Defaults
        if default_params is not None:
            def_shapes = default_params[:n_shapes]
            def_loc = default_params[n_shapes]
            def_scale = default_params[n_shapes + 1]
        else:
            def_shapes = [1.0] * n_shapes
            def_loc = dmin
            def_scale = dstd if dstd > 0 else drange / 4

        st.markdown(
            f"**Manual parameters for {dist_label}**  "
            "(adjust and see how the fit quality changes)."
        )

        # Shape parameters
        manual_shape_params = []
        for i, name in enumerate(shape_names):
            label = name if name else f"shape{i+1}"
            default_val = float(def_shapes[i]) if i < len(def_shapes) else 1.0
            low = 0.01
            high = max(10.0, default_val * 3.0) if default_val > 0 else 10.0
            manual_val = st.slider(
                f"{label}",
                min_value=low,
                max_value=high,
                value=float(np.clip(default_val, low, high)),
                step=(high - low) / 200.0,
            )
            manual_shape_params.append(manual_val)

        # loc and scale sliders
        loc_val = st.slider(
            "loc",
            min_value=dmin - drange,
            max_value=dmax + drange,
            value=float(np.clip(def_loc, dmin - drange, dmax + drange)),
        )

        scale_min = max(drange / 1000.0, 1e-6)
        scale_max = drange * 5.0
        scale_default = float(np.clip(def_scale, scale_min, scale_max))
        scale_val = st.slider(
            "scale",
            min_value=scale_min,
            max_value=scale_max,
            value=scale_default,
        )

        # Build frozen dist and pdf
        frozen_manual = dist_obj(*manual_shape_params, loc=loc_val, scale=scale_val)
        x_grid_m = np.linspace(x_min, x_max, 400)
        pdf_manual = frozen_manual.pdf(x_grid_m)

        hist_vals_m, bin_edges_m = np.histogram(
            data,
            bins=bins,
            range=[x_min, x_max],
            density=True
        )

        mse_m, max_err_m, _, _ = compute_fit_error(
            hist_vals_m, bin_edges_m, pdf_manual
        )

        # Plot
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(
            data,
            bins=bins,
            range=[x_min, x_max],
            density=True,
            alpha=0.4,
            edgecolor="black",
            label="Data (histogram)"
        )
        ax2.plot(
            x_grid_m,
            pdf_manual,
            linewidth=2,
            label=f"{dist_label} PDF (manual)"
        )
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Manual Fit: {dist_label}")
        ax2.legend()
        st.pyplot(fig2)

        # Show parameters and error
        param_names_m = shape_names + ["loc", "scale"]
        param_vals_m = manual_shape_params + [loc_val, scale_val]
        df_params_m = pd.DataFrame(
            {"Parameter": param_names_m, "Value": [f"{v:.4g}" for v in param_vals_m]}
        )
        st.markdown("**Manual parameters**")
        st.table(df_params_m)

        st.markdown("**Fit Quality (manual params)**")
        st.write(f"- Mean Squared Error (MSE): `{mse_m:.4e}`")
        st.write(f"- Maximum Absolute Error: `{max_err_m:.4e}`")

    st.markdown('</div>', unsafe_allow_html=True)
