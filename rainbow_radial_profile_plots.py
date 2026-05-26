import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

data_dir = r""  # add your desired directory here

output_dir = os.path.join(data_dir, "radial_profiles")
os.makedirs(output_dir, exist_ok=True)

times = ["0000", "0010", "0020", "0060", "0080", "0094"]

colors = [
    "red",
    "orange",
    "gold",
    "green",
    "blue",
    "purple",
]

regions = ["footpoint", "apex"]

variables = {
    "Jz": "Jz_local",
    "Jr": "Jr",
    "Jphi": "Jphi",
    "Bz": "Bz_local",
    "Bphi": "Bphi",
}

r_limits = {
    "footpoint": 0.5,
    "apex": None,
}

n_bins_default = {
    "footpoint": 25,
    "apex": 80,
}


def radial_bin_average(R, F, n_bins, r_max=None):
    r_vals = R.flatten()
    f_vals = F.flatten()

    mask = np.isfinite(r_vals) & np.isfinite(f_vals)

    if r_max is not None:
        mask &= r_vals <= r_max

    r_vals = r_vals[mask]
    f_vals = f_vals[mask]

    if len(r_vals) == 0:
        return np.array([]), np.array([])

    r_bins = np.linspace(r_vals.min(), r_vals.max(), n_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    f_mean = np.full(n_bins, np.nan)

    for i in range(n_bins):
        in_bin = (
            (r_vals >= r_bins[i]) &
            (r_vals < r_bins[i + 1])
        )

        if np.any(in_bin):
            f_mean[i] = np.mean(f_vals[in_bin])

    valid = np.isfinite(f_mean)

    return r_centers[valid], f_mean[valid]


def smooth_profile(y, window_length=11, polyorder=3):
    n = len(y)

    if n < 5:
        return y

    window_length = min(window_length, n)

    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    if window_length > n:
        return y

    return savgol_filter(
        y,
        window_length=window_length,
        polyorder=polyorder,
    )


for region in regions:

    for var_label, dataset_name in variables.items():

        plt.figure(figsize=(10, 7))

        plotted_anything = False

        for t, color in zip(times, colors):

            filename = os.path.join(data_dir, f"analysis_t{t}.h5")

            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue

            with h5py.File(filename, "r") as f:

                field_path = f"{region}/maps/{dataset_name}"
                r_path = f"{region}/maps/R"

                if field_path not in f:
                    print(f"Missing: {field_path} in {filename}")
                    continue

                if r_path not in f:
                    print(f"Missing: {r_path} in {filename}")
                    continue

                F = f[field_path][:]
                R = f[r_path][:]

            r_profile, f_profile = radial_bin_average(
                R,
                F,
                n_bins=n_bins_default[region],
                r_max=r_limits[region],
            )

            if len(r_profile) == 0:
                print(f"No valid data: {region}, {var_label}, t={t}")
                continue

            f_smooth = smooth_profile(
                f_profile,
                window_length=11,
                polyorder=3,
            )

            plt.plot(
                r_profile,
                f_smooth,
                color=color,
                linewidth=2.5,
                label=f"t={int(t)}",
            )

            plotted_anything = True

        if not plotted_anything:
            plt.close()
            print(f"Skipped empty plot: {region}, {var_label}")
            continue

        plt.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
        )

        plt.xlabel(r"$r$", fontsize=18)
        plt.ylabel(rf"${var_label}$", fontsize=18)

        plt.title(
            rf"${var_label}(r)$ radial bin-averaged profiles ({region})",
            fontsize=20,
        )

        plt.legend(fontsize=13)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_name = f"{region}_{var_label}_radial_profile.png"
        save_path = os.path.join(output_dir, save_name)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved: {save_path}")
