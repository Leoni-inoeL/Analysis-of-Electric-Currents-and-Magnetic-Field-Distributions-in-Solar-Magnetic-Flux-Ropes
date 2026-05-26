import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import j0, j1
from pathlib import Path

DATA_DIR = Path(r"C:/Users/user/Desktop/SummerPractice/AnalysisData")
OUTPUT_DIR = DATA_DIR / "Fitting_Rep"
OUTPUT_DIR.mkdir(exist_ok=True)

EPS = 1e-12
ALPHA_MIN = 0.5
ALPHA_MAX = 10.0
WZ = 0.5
WPHI = 0.5

REGIONS = ["footpoint", "apex"]

ALL_FILES = sorted(DATA_DIR.glob("analysis_t*.h5"))
timesteps_all = sorted([f.stem.split("analysis_t")[-1] for f in ALL_FILES])
timesteps_exclude = ["t0095"]
timesteps = [t for t in timesteps_all if t not in timesteps_exclude]


def load_profile(file_path_, region_name_):
    with h5py.File(file_path_, "r") as h5_file_:
        if region_name_ not in h5_file_:
            return None, None, None
        if "profiles" not in h5_file_[region_name_]:
            return None, None, None
        profiles_group_ = h5_file_[region_name_]["profiles"]
        r_ = profiles_group_["r"][:]
        bz_ = profiles_group_["Bz_mean"][:]
        bphi_ = profiles_group_["Bphi_mean"][:]
    return r_, bz_, bphi_


def detect_rope_boundary_current_inversion(r_arr_, bphi_arr_):
    valid_ = np.isfinite(bphi_arr_)
    if not np.any(valid_):
        return 0.3
    r_valid_ = r_arr_[valid_]
    bphi_valid_ = bphi_arr_[valid_]
    sign_changes_ = np.where(np.diff(np.sign(bphi_valid_)))[0]
    if len(sign_changes_) > 0:
        idx_ = sign_changes_[0]
        if idx_ + 1 < len(r_valid_):
            return (r_valid_[idx_] + r_valid_[idx_ + 1]) / 2
    idx_max_ = np.argmax(np.abs(bphi_valid_))
    threshold_ = 0.05 * np.abs(bphi_valid_[idx_max_])
    for i_ in range(idx_max_, len(r_valid_)):
        if np.abs(bphi_valid_[i_]) < threshold_:
            return r_valid_[i_]
    return r_valid_[-1]


def normalize_fields_same_scale(r_arr_, bz_arr_, bphi_arr_, rope_radius_):
    mask_ = (r_arr_ <= rope_radius_) & np.isfinite(bz_arr_) & np.isfinite(bphi_arr_)
    if np.sum(mask_) < 5:
        return None, None, None
    r_norm_ = r_arr_[mask_] / rope_radius_
    r_norm_ = np.maximum(r_norm_, EPS)
    b0_scale_ = max(np.max(np.abs(bz_arr_[mask_])), EPS)
    bz_norm_ = bz_arr_[mask_] / b0_scale_
    bphi_norm_ = bphi_arr_[mask_] / b0_scale_
    return r_norm_, bz_norm_, bphi_norm_


def compute_nrmse(bz_data_, bphi_data_, bz_model_, bphi_model_):
    mse_total_ = np.mean((bz_data_ - bz_model_) ** 2 + (bphi_data_ - bphi_model_) ** 2)
    variance_total_ = np.mean(bz_data_ ** 2 + bphi_data_ ** 2)
    return np.sqrt(mse_total_ / (variance_total_ + EPS))


def compute_aic(mse_val_, n_points_, n_params_):
    if mse_val_ <= 0:
        return np.inf
    n_ = 2 * n_points_
    return n_ * np.log(mse_val_ + EPS) + 2 * n_params_


def model_gold_hoyle(r_norm_, b0_, q_val_, sign_phi_=1):
    q_val_ = np.abs(q_val_)
    denom_ = 1.0 + q_val_ ** 2 * r_norm_ ** 2
    denom_ = np.maximum(denom_, EPS)
    bz_ = b0_ / denom_
    bphi_ = sign_phi_ * b0_ * q_val_ * r_norm_ / denom_
    return bz_, bphi_


def model_lundquist(r_norm_, b0_, alpha_, sign_phi_=1):
    alpha_ = np.abs(alpha_)
    with np.errstate(invalid='ignore'):
        bz_ = b0_ * j0(alpha_ * r_norm_)
        bphi_ = sign_phi_ * b0_ * j1(alpha_ * r_norm_)
    bz_ = np.where(np.isfinite(bz_), bz_, 0)
    bphi_ = np.where(np.isfinite(bphi_), bphi_, 0)
    return bz_, bphi_


def model_exponential(r_norm_, b0_, k_val_, g_val_, sign_phi_=1):
    k_val_ = np.abs(k_val_)
    g_val_ = np.clip(g_val_, 0.1, 0.99)
    exp_term_ = np.exp(-k_val_ ** 2 * r_norm_ ** 2)
    term1_ = (1 - g_val_) * (1 - k_val_ ** 2 * r_norm_ ** 2) * exp_term_
    arg1_ = g_val_ + term1_
    arg2_ = (1 - g_val_) * k_val_ ** 2 * r_norm_ ** 2 * exp_term_
    arg1_ = np.maximum(arg1_, EPS)
    arg2_ = np.maximum(arg2_, EPS)
    bz_ = b0_ * np.sqrt(arg1_)
    bphi_ = sign_phi_ * b0_ * np.sqrt(arg2_)
    return bz_, bphi_


def model_step_like(r_norm_, b0_, g_val_, r0_, b_val_, sign_phi_=1):
    g_val_ = np.clip(g_val_, 0.1, 0.99)
    r0_ = np.maximum(r0_, EPS)
    b_val_ = np.maximum(b_val_, EPS)
    denominator_ = b_val_ ** 2 * r0_ ** 2
    exponent_ = (r_norm_ ** 2 - r0_ ** 2) / (denominator_ + EPS)
    exponent_ = np.clip(exponent_, -50, 50)
    expx_ = np.exp(exponent_)
    s_ = 1.0 / (1.0 + expx_ + EPS)
    f_ = b0_ ** 2 * (g_val_ + (1 - g_val_) * s_)
    ds_dr_ = -2 * r_norm_ * expx_ / (denominator_ * (1 + expx_) ** 2 + EPS)
    df_dr_ = b0_ ** 2 * (1 - g_val_) * ds_dr_
    arg_bz_ = f_ + 0.5 * r_norm_ * df_dr_
    arg_bphi_ = -0.5 * r_norm_ * df_dr_
    arg_bz_ = np.maximum(arg_bz_, EPS)
    arg_bphi_ = np.maximum(arg_bphi_, EPS)
    bz_ = np.sqrt(arg_bz_)
    bphi_ = sign_phi_ * np.sqrt(arg_bphi_)
    return bz_, bphi_


def weighted_mse_mse(bz_data_, bphi_data_, bz_model_, bphi_model_, wz_, wphi_):
    mse_bz_ = np.mean((bz_data_ - bz_model_) ** 2)
    mse_bphi_ = np.mean((bphi_data_ - bphi_model_) ** 2)
    return wz_ * mse_bz_ + wphi_ * mse_bphi_


def weighted_mse_mae(bz_data_, bphi_data_, bz_model_, bphi_model_, wz_, wphi_):
    mae_bz_ = np.mean(np.abs(bz_data_ - bz_model_))
    mae_bphi_ = np.mean(np.abs(bphi_data_ - bphi_model_))
    return wz_ * mae_bz_ + wphi_ * mae_bphi_


def compute_normalized_residual(bz_data_, bphi_data_, bz_model_, bphi_model_, n_params_):
    n_points_ = 2 * len(bz_data_)
    chi2_ = np.sum((bz_data_ - bz_model_) ** 2 + (bphi_data_ - bphi_model_) ** 2)
    return chi2_ / (n_points_ - n_params_) if n_points_ > n_params_ else np.inf


def objective_gold_hoyle(params_, r_norm_, bz_data_, bphi_data_, sign_phi_, wz_, wphi_):
    b0_, q_val_ = params_
    b0_ = np.abs(b0_)
    q_val_ = np.abs(q_val_)
    bz_model_, bphi_model_ = model_gold_hoyle(r_norm_, b0_, q_val_, sign_phi_)
    if not (np.all(np.isfinite(bz_model_)) and np.all(np.isfinite(bphi_model_))):
        return 1e10
    return weighted_mse_mse(bz_data_, bphi_data_, bz_model_, bphi_model_, wz_, wphi_)


def objective_lundquist(params_, r_norm_, bz_data_, bphi_data_, sign_phi_, wz_, wphi_):
    b0_, alpha_ = params_
    b0_ = np.abs(b0_)
    alpha_ = np.abs(alpha_)
    bz_model_, bphi_model_ = model_lundquist(r_norm_, b0_, alpha_, sign_phi_)
    if not (np.all(np.isfinite(bz_model_)) and np.all(np.isfinite(bphi_model_))):
        return 1e10
    return weighted_mse_mse(bz_data_, bphi_data_, bz_model_, bphi_model_, wz_, wphi_)


def objective_exponential(params_, r_norm_, bz_data_, bphi_data_, sign_phi_, wz_, wphi_):
    b0_, k_val_, g_val_ = params_
    b0_ = np.abs(b0_)
    k_val_ = np.abs(k_val_)
    g_val_ = np.clip(g_val_, 0.1, 0.99)
    bz_model_, bphi_model_ = model_exponential(r_norm_, b0_, k_val_, g_val_, sign_phi_)
    if not (np.all(np.isfinite(bz_model_)) and np.all(np.isfinite(bphi_model_))):
        return 1e10
    return weighted_mse_mse(bz_data_, bphi_data_, bz_model_, bphi_model_, wz_, wphi_)


def objective_step_like(params_, r_norm_, bz_data_, bphi_data_, sign_phi_, wz_, wphi_):
    b0_, g_val_, r0_, b_val_ = params_
    b0_ = np.abs(b0_)
    g_val_ = np.clip(g_val_, 0.1, 0.99)
    r0_ = np.abs(r0_)
    b_val_ = np.abs(b_val_)
    bz_model_, bphi_model_ = model_step_like(r_norm_, b0_, g_val_, r0_, b_val_, sign_phi_)
    if not (np.all(np.isfinite(bz_model_)) and np.all(np.isfinite(bphi_model_))):
        return 1e10
    return weighted_mse_mse(bz_data_, bphi_data_, bz_model_, bphi_model_, wz_, wphi_)


def fit_gold_hoyle(r_norm_, bz_norm_, bphi_norm_, sign_phi_):
    b0_guess_ = 1.0
    q_guess_ = 0.1
    x0_ = [b0_guess_, q_guess_]
    bounds_ = [(0.1, 10), (0.001, 10)]
    try:
        result_ = minimize(objective_gold_hoyle, x0_,
                           args=(r_norm_, bz_norm_, bphi_norm_, sign_phi_, WZ, WPHI),
                           bounds=bounds_, method='L-BFGS-B', options={'maxiter': 500})
        if not result_.success:
            return None, None, None, None, None, None, None, None
        b0_, q_val_ = result_.x
        bz_fit_, bphi_fit_ = model_gold_hoyle(r_norm_, b0_, q_val_, sign_phi_)
        if not (np.all(np.isfinite(bz_fit_)) and np.all(np.isfinite(bphi_fit_))):
            return None, None, None, None, None, None, None, None
        mse_ = weighted_mse_mse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        mae_ = weighted_mse_mae(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        nrmse_ = compute_nrmse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_)
        nres_ = compute_normalized_residual(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, 2)
        aic_ = compute_aic(mse_, len(r_norm_), 2)
        return b0_, q_val_, mse_, mae_, nrmse_, nres_, aic_, bz_fit_, bphi_fit_
    except:
        return None, None, None, None, None, None, None, None


def fit_lundquist(r_norm_, bz_norm_, bphi_norm_, sign_phi_):
    b0_guess_ = 1.0
    alpha_guess_ = 1.0
    x0_ = [b0_guess_, alpha_guess_]
    bounds_ = [(0.1, 10), (ALPHA_MIN, ALPHA_MAX)]
    try:
        result_ = minimize(objective_lundquist, x0_,
                           args=(r_norm_, bz_norm_, bphi_norm_, sign_phi_, WZ, WPHI),
                           bounds=bounds_, method='L-BFGS-B', options={'maxiter': 500})
        if not result_.success:
            return None, None, None, None, None, None, None, None
        b0_, alpha_ = result_.x
        bz_fit_, bphi_fit_ = model_lundquist(r_norm_, b0_, alpha_, sign_phi_)
        if not (np.all(np.isfinite(bz_fit_)) and np.all(np.isfinite(bphi_fit_))):
            return None, None, None, None, None, None, None, None
        mse_ = weighted_mse_mse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        mae_ = weighted_mse_mae(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        nrmse_ = compute_nrmse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_)
        nres_ = compute_normalized_residual(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, 2)
        aic_ = compute_aic(mse_, len(r_norm_), 2)
        return b0_, alpha_, mse_, mae_, nrmse_, nres_, aic_, bz_fit_, bphi_fit_
    except:
        return None, None, None, None, None, None, None, None


def fit_exponential(r_norm_, bz_norm_, bphi_norm_, sign_phi_):
    b0_guess_ = 1.0
    k_guess_ = 0.5
    g_guess_ = 0.5
    x0_ = [b0_guess_, k_guess_, g_guess_]
    bounds_ = [(0.1, 10), (0.01, 10), (0.1, 0.99)]
    try:
        result_ = minimize(objective_exponential, x0_,
                           args=(r_norm_, bz_norm_, bphi_norm_, sign_phi_, WZ, WPHI),
                           bounds=bounds_, method='L-BFGS-B', options={'maxiter': 500})
        if not result_.success:
            return None, None, None, None, None, None, None, None, None
        b0_, k_val_, g_val_ = result_.x
        bz_fit_, bphi_fit_ = model_exponential(r_norm_, b0_, k_val_, g_val_, sign_phi_)
        if not (np.all(np.isfinite(bz_fit_)) and np.all(np.isfinite(bphi_fit_))):
            return None, None, None, None, None, None, None, None, None
        mse_ = weighted_mse_mse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        mae_ = weighted_mse_mae(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        nrmse_ = compute_nrmse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_)
        nres_ = compute_normalized_residual(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, 3)
        aic_ = compute_aic(mse_, len(r_norm_), 3)
        return b0_, k_val_, g_val_, mse_, mae_, nrmse_, nres_, aic_, bz_fit_, bphi_fit_
    except:
        return None, None, None, None, None, None, None, None, None


def fit_step_like(r_norm_, bz_norm_, bphi_norm_, sign_phi_):
    b0_guess_ = 1.0
    g_guess_ = 0.5
    r0_guess_ = 0.5
    b_guess_ = 0.2
    x0_ = [b0_guess_, g_guess_, r0_guess_, b_guess_]
    bounds_ = [(0.1, 10), (0.1, 0.99), (0.1, 1.0), (0.05, 0.5)]
    try:
        result_ = minimize(objective_step_like, x0_,
                           args=(r_norm_, bz_norm_, bphi_norm_, sign_phi_, WZ, WPHI),
                           bounds=bounds_, method='L-BFGS-B', options={'maxiter': 500})
        if not result_.success:
            return None, None, None, None, None, None, None, None, None, None
        b0_, g_val_, r0_, b_val_ = result_.x
        bz_fit_, bphi_fit_ = model_step_like(r_norm_, b0_, g_val_, r0_, b_val_, sign_phi_)
        if not (np.all(np.isfinite(bz_fit_)) and np.all(np.isfinite(bphi_fit_))):
            return None, None, None, None, None, None, None, None, None, None
        mse_ = weighted_mse_mse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        mae_ = weighted_mse_mae(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, WZ, WPHI)
        nrmse_ = compute_nrmse(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_)
        nres_ = compute_normalized_residual(bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, 4)
        aic_ = compute_aic(mse_, len(r_norm_), 4)
        return b0_, g_val_, r0_, b_val_, mse_, mae_, nrmse_, nres_, aic_, bz_fit_, bphi_fit_
    except:
        return None, None, None, None, None, None, None, None, None, None


def plot_fit(r_norm_, bz_norm_, bphi_norm_, bz_fit_, bphi_fit_, model_name_, region_name_, timestep_, mse_, mae_,
             nrmse_, nres_, aic_):
    fig_, (ax1_, ax2_) = plt.subplots(1, 2, figsize=(12, 5))

    ax1_.plot(r_norm_, bz_norm_, 'bo', markersize=4, label='Simulation')
    ax1_.plot(r_norm_, bz_fit_, 'r-', linewidth=2, label='Fit')
    ax1_.set_xlabel(r'$r / R_{rope}$')
    ax1_.set_ylabel(r'$\tilde{B}_z$')
    ax1_.set_title(f'{region_name_} t={timestep_} {model_name_}: Bz')
    ax1_.legend()
    ax1_.grid(True, alpha=0.3)

    ax2_.plot(r_norm_, bphi_norm_, 'bo', markersize=4, label='Simulation')
    ax2_.plot(r_norm_, bphi_fit_, 'r-', linewidth=2, label='Fit')
    ax2_.set_xlabel(r'$r / R_{rope}$')
    ax2_.set_ylabel(r'$\tilde{B}_{\phi}$')
    ax2_.set_title(
        f'{region_name_} t={timestep_} {model_name_}\nMSE={mse_:.4f} MAE={mae_:.4f} NRMSE={nrmse_:.4f} AIC={aic_:.1f}')
    ax2_.legend()
    ax2_.grid(True, alpha=0.3)

    plt.tight_layout()
    filename_ = f"{region_name_}_t{timestep_}_{model_name_}.png"
    plt.savefig(OUTPUT_DIR / filename_, dpi=150)
    plt.close()


def plot_evolution(time_vals_, data_dict_, ylabel_, title_, filename_):
    plt.figure(figsize=(10, 6))
    for region_name_, values_ in data_dict_.items():
        valid_pairs_ = [(t_, v_) for t_, v_ in zip(time_vals_, values_) if v_ is not None]
        if len(valid_pairs_) > 0:
            t_valid_, v_valid_ = zip(*valid_pairs_)
            plt.plot(t_valid_, v_valid_, 'o-', linewidth=2, label=region_name_)
    plt.xlabel('Time')
    plt.ylabel(ylabel_)
    plt.title(title_)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename_, dpi=150)
    plt.close()


def plot_evolution_ratio(time_vals_, values_foot_, values_apex_, ylabel_, title_, filename_):
    plt.figure(figsize=(10, 6))
    ratio_ = []
    t_valid_ = []
    for i_, t_ in enumerate(time_vals_):
        if values_foot_[i_] is not None and values_apex_[i_] is not None and values_apex_[i_] != 0:
            ratio_.append(values_foot_[i_] / values_apex_[i_])
            t_valid_.append(t_)
    if len(ratio_) > 0:
        plt.plot(t_valid_, ratio_, 'o-', linewidth=2, color='black')
        plt.axhline(y=1, color='r', linestyle='--', label='Footpoint/Apex = 1')
    plt.xlabel('Time')
    plt.ylabel(ylabel_)
    plt.title(title_)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename_, dpi=150)
    plt.close()


print(f"Processing {len(timesteps)} timesteps")

time_labels = []
time_values = []
n_points_foot = []
n_points_apex = []

gold_hoyle_q = {'footpoint': [], 'apex': []}
gold_hoyle_nrmse = {'footpoint': [], 'apex': []}
gold_hoyle_aic = {'footpoint': [], 'apex': []}
lundquist_alpha = {'footpoint': [], 'apex': []}
lundquist_nrmse = {'footpoint': [], 'apex': []}
lundquist_aic = {'footpoint': [], 'apex': []}
exponential_k = {'footpoint': [], 'apex': []}
exponential_g = {'footpoint': [], 'apex': []}
exponential_nrmse = {'footpoint': [], 'apex': []}
exponential_aic = {'footpoint': [], 'apex': []}
step_like_r0 = {'footpoint': [], 'apex': []}
step_like_nrmse = {'footpoint': [], 'apex': []}
step_like_aic = {'footpoint': [], 'apex': []}
rope_radius_detected = {'footpoint': [], 'apex': []}
best_model_per_timestep = {'footpoint': [], 'apex': []}

fit_results = []

for timestep in timesteps:
    file_path = DATA_DIR / f"analysis_t{timestep}.h5"

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    with h5py.File(file_path, "r") as h5_file:
        physical_time = h5_file.attrs.get("physical_time", float(timestep))
    time_labels.append(timestep)
    time_values.append(physical_time)

    print(f"\nt={physical_time:.2f} ({timestep})")

    for region_name in REGIONS:
        r_arr, bz_arr, bphi_arr = load_profile(file_path, region_name)

        if r_arr is None or len(r_arr) == 0:
            print(f"  {region_name}: No profile data")
            for key in gold_hoyle_q:
                gold_hoyle_q[region_name].append(None)
                gold_hoyle_nrmse[region_name].append(None)
                gold_hoyle_aic[region_name].append(None)
                lundquist_alpha[region_name].append(None)
                lundquist_nrmse[region_name].append(None)
                lundquist_aic[region_name].append(None)
                exponential_k[region_name].append(None)
                exponential_g[region_name].append(None)
                exponential_nrmse[region_name].append(None)
                exponential_aic[region_name].append(None)
                step_like_r0[region_name].append(None)
                step_like_nrmse[region_name].append(None)
                step_like_aic[region_name].append(None)
                rope_radius_detected[region_name].append(None)
                best_model_per_timestep[region_name].append(None)
                if region_name == 'footpoint':
                    n_points_foot.append(None)
                else:
                    n_points_apex.append(None)
            continue

        if region_name == 'footpoint':
            rope_radius = 0.3
        else:
            rope_radius = detect_rope_boundary_current_inversion(r_arr, bphi_arr)
            rope_radius = max(rope_radius, 0.3)

        rope_radius_detected[region_name].append(rope_radius)
        n_inside = (r_arr <= rope_radius).sum()
        print(f"  {region_name}: Rrope={rope_radius:.4f}, Npoints={n_inside}")

        if region_name == 'footpoint':
            n_points_foot.append(n_inside)
        else:
            n_points_apex.append(n_inside)

        norm_result = normalize_fields_same_scale(r_arr, bz_arr, bphi_arr, rope_radius)

        if norm_result[0] is None:
            print(f"  {region_name}: Not enough points inside rope (Rrope={rope_radius:.3f})")
            for key in gold_hoyle_q:
                gold_hoyle_q[region_name].append(None)
                gold_hoyle_nrmse[region_name].append(None)
                gold_hoyle_aic[region_name].append(None)
                lundquist_alpha[region_name].append(None)
                lundquist_nrmse[region_name].append(None)
                lundquist_aic[region_name].append(None)
                exponential_k[region_name].append(None)
                exponential_g[region_name].append(None)
                exponential_nrmse[region_name].append(None)
                exponential_aic[region_name].append(None)
                step_like_r0[region_name].append(None)
                step_like_nrmse[region_name].append(None)
                step_like_aic[region_name].append(None)
                best_model_per_timestep[region_name].append(None)
            continue

        r_norm, bz_norm, bphi_norm = norm_result

        if np.max(np.abs(bphi_norm)) < 0.01:
            print(
                f"  {region_name}: Skipping fit (insufficient azimuthal field, max|Bphi|={np.max(np.abs(bphi_norm)):.4f})")
            gold_hoyle_q[region_name].append(None)
            gold_hoyle_nrmse[region_name].append(None)
            gold_hoyle_aic[region_name].append(None)
            lundquist_alpha[region_name].append(None)
            lundquist_nrmse[region_name].append(None)
            lundquist_aic[region_name].append(None)
            exponential_k[region_name].append(None)
            exponential_g[region_name].append(None)
            exponential_nrmse[region_name].append(None)
            exponential_aic[region_name].append(None)
            step_like_r0[region_name].append(None)
            step_like_nrmse[region_name].append(None)
            step_like_aic[region_name].append(None)
            best_model_per_timestep[region_name].append(None)
            continue

        sign_phi = np.sign(np.mean(bphi_norm[1:5])) if len(bphi_norm) > 5 else 1

        result = fit_gold_hoyle(r_norm, bz_norm, bphi_norm, sign_phi)
        if result[0] is not None:
            b0, q_val, mse, mae, nrmse, nres, aic, bz_fit, bphi_fit = result
            gold_hoyle_q[region_name].append(q_val)
            gold_hoyle_nrmse[region_name].append(nrmse)
            gold_hoyle_aic[region_name].append(aic)
            print(f"  {region_name} Gold-Hoyle: q={q_val:.3f}, NRMSE={nrmse:.4f}, AIC={aic:.1f}")
            plot_fit(r_norm, bz_norm, bphi_norm, bz_fit, bphi_fit, "Gold-Hoyle", region_name, timestep, mse, mae, nrmse,
                     nres, aic)
            fit_results.append({
                'timestep': timestep, 'time': physical_time, 'region': region_name,
                'model': 'Gold-Hoyle', 'b0': b0, 'q': q_val, 'alpha': None,
                'k': None, 'g': None, 'r0': None, 'b': None,
                'MSE': mse, 'MAE': mae, 'NRMSE': nrmse, 'normalized_residual': nres, 'AIC': aic, 'Rrope': rope_radius,
                'n_points_inside': n_inside
            })
        else:
            gold_hoyle_q[region_name].append(None)
            gold_hoyle_nrmse[region_name].append(None)
            gold_hoyle_aic[region_name].append(None)

        result = fit_lundquist(r_norm, bz_norm, bphi_norm, sign_phi)
        if result[0] is not None:
            b0, alpha, mse, mae, nrmse, nres, aic, bz_fit, bphi_fit = result
            lundquist_alpha[region_name].append(alpha)
            lundquist_nrmse[region_name].append(nrmse)
            lundquist_aic[region_name].append(aic)
            print(f"  {region_name} Lundquist: alpha={alpha:.3f}, NRMSE={nrmse:.4f}, AIC={aic:.1f}")
            plot_fit(r_norm, bz_norm, bphi_norm, bz_fit, bphi_fit, "Lundquist", region_name, timestep, mse, mae, nrmse,
                     nres, aic)
            fit_results.append({
                'timestep': timestep, 'time': physical_time, 'region': region_name,
                'model': 'Lundquist', 'b0': b0, 'q': None, 'alpha': alpha,
                'k': None, 'g': None, 'r0': None, 'b': None,
                'MSE': mse, 'MAE': mae, 'NRMSE': nrmse, 'normalized_residual': nres, 'AIC': aic, 'Rrope': rope_radius,
                'n_points_inside': n_inside
            })
        else:
            lundquist_alpha[region_name].append(None)
            lundquist_nrmse[region_name].append(None)
            lundquist_aic[region_name].append(None)

        result = fit_exponential(r_norm, bz_norm, bphi_norm, sign_phi)
        if result[0] is not None:
            b0, k_val, g_val, mse, mae, nrmse, nres, aic, bz_fit, bphi_fit = result
            exponential_k[region_name].append(k_val)
            exponential_g[region_name].append(g_val)
            exponential_nrmse[region_name].append(nrmse)
            exponential_aic[region_name].append(aic)
            print(f"  {region_name} Exponential: k={k_val:.3f}, g={g_val:.3f}, NRMSE={nrmse:.4f}, AIC={aic:.1f}")
            plot_fit(r_norm, bz_norm, bphi_norm, bz_fit, bphi_fit, "Exponential", region_name, timestep, mse, mae,
                     nrmse, nres, aic)
            fit_results.append({
                'timestep': timestep, 'time': physical_time, 'region': region_name,
                'model': 'Exponential', 'b0': b0, 'q': None, 'alpha': None,
                'k': k_val, 'g': g_val, 'r0': None, 'b': None,
                'MSE': mse, 'MAE': mae, 'NRMSE': nrmse, 'normalized_residual': nres, 'AIC': aic, 'Rrope': rope_radius,
                'n_points_inside': n_inside
            })
        else:
            exponential_k[region_name].append(None)
            exponential_g[region_name].append(None)
            exponential_nrmse[region_name].append(None)
            exponential_aic[region_name].append(None)

        result = fit_step_like(r_norm, bz_norm, bphi_norm, sign_phi)
        if result[0] is not None:
            b0, g_val, r0, b_val, mse, mae, nrmse, nres, aic, bz_fit, bphi_fit = result
            step_like_r0[region_name].append(r0)
            step_like_nrmse[region_name].append(nrmse)
            step_like_aic[region_name].append(aic)
            print(f"  {region_name} Step-like: r0={r0:.3f}, NRMSE={nrmse:.4f}, AIC={aic:.1f}")
            plot_fit(r_norm, bz_norm, bphi_norm, bz_fit, bphi_fit, "Step-like", region_name, timestep, mse, mae, nrmse,
                     nres, aic)
            fit_results.append({
                'timestep': timestep, 'time': physical_time, 'region': region_name,
                'model': 'Step-like', 'b0': b0, 'q': None, 'alpha': None,
                'k': None, 'g': g_val, 'r0': r0, 'b': b_val,
                'MSE': mse, 'MAE': mae, 'NRMSE': nrmse, 'normalized_residual': nres, 'AIC': aic, 'Rrope': rope_radius,
                'n_points_inside': n_inside
            })
        else:
            step_like_r0[region_name].append(None)
            step_like_nrmse[region_name].append(None)
            step_like_aic[region_name].append(None)

        region_fits = [r for r in fit_results if r['timestep'] == timestep and r['region'] == region_name]
        if len(region_fits) > 0:
            best = min(region_fits, key=lambda x: x['AIC'])
            best_model_per_timestep[region_name].append(best['model'])
            print(
                f"  {region_name} BEST MODEL by AIC: {best['model']} (AIC={best['AIC']:.1f}, NRMSE={best['NRMSE']:.4f})")
        else:
            best_model_per_timestep[region_name].append(None)

df_results = pd.DataFrame(fit_results)
df_results.to_csv(OUTPUT_DIR / "all_fit_results.csv", index=False)
print(f"\nSaved all fit results to: {OUTPUT_DIR / 'all_fit_results.csv'}")

time_float_valid = []
gold_hoyle_q_foot_valid = []
gold_hoyle_q_apex_valid = []
gold_hoyle_nrmse_foot_valid = []
gold_hoyle_nrmse_apex_valid = []
rope_radius_foot_valid = []
rope_radius_apex_valid = []
n_points_foot_valid = []
n_points_apex_valid = []

for i, t in enumerate(time_values):
    gh_foot = gold_hoyle_q['footpoint'][i]
    gh_apex = gold_hoyle_q['apex'][i]
    nr_foot = gold_hoyle_nrmse['footpoint'][i]
    nr_apex = gold_hoyle_nrmse['apex'][i]
    rr_foot = rope_radius_detected['footpoint'][i]
    rr_apex = rope_radius_detected['apex'][i]
    np_foot = n_points_foot[i] if i < len(n_points_foot) else None
    np_apex = n_points_apex[i] if i < len(n_points_apex) else None
    if gh_foot is not None and gh_apex is not None:
        time_float_valid.append(t)
        gold_hoyle_q_foot_valid.append(gh_foot)
        gold_hoyle_q_apex_valid.append(gh_apex)
        gold_hoyle_nrmse_foot_valid.append(nr_foot)
        gold_hoyle_nrmse_apex_valid.append(nr_apex)
        rope_radius_foot_valid.append(rr_foot)
        rope_radius_apex_valid.append(rr_apex)
        n_points_foot_valid.append(np_foot)
        n_points_apex_valid.append(np_apex)

plot_evolution(
    time_float_valid,
    {'footpoint': gold_hoyle_q_foot_valid, 'apex': gold_hoyle_q_apex_valid},
    r'$q$ [Mm$^{-1}$]',
    'Gold-Hoyle Twist Parameter Evolution',
    'gold_hoyle_q_evolution.png'
)

plot_evolution(
    time_float_valid,
    {'footpoint': gold_hoyle_nrmse_foot_valid, 'apex': gold_hoyle_nrmse_apex_valid},
    'NRMSE',
    'Gold-Hoyle Fit Quality Evolution (NRMSE)',
    'gold_hoyle_nrmse_evolution.png'
)

plot_evolution(
    time_float_valid,
    {'footpoint': rope_radius_foot_valid, 'apex': rope_radius_apex_valid},
    'Rope Radius [Mm]',
    'Rope Radius Evolution (Current Inversion Boundary)',
    'rope_radius_evolution.png'
)

plot_evolution(
    time_float_valid,
    {'footpoint': n_points_foot_valid, 'apex': n_points_apex_valid},
    'Number of Points Inside Rope',
    'Points Inside Rope Evolution',
    'n_points_inside_rope.png'
)

plot_evolution_ratio(
    time_float_valid,
    gold_hoyle_q_foot_valid,
    gold_hoyle_q_apex_valid,
    'q_footpoint / q_apex',
    'Gold-Hoyle Twist Parameter Ratio',
    'gold_hoyle_q_ratio.png'
)

print(f"\nBest model per timestep (AIC):")
for i, t in enumerate(time_values):
    bm_foot = best_model_per_timestep['footpoint'][i] if i < len(best_model_per_timestep['footpoint']) else None
    bm_apex = best_model_per_timestep['apex'][i] if i < len(best_model_per_timestep['apex']) else None
    print(f"  t={t:.2f}: footpoint={bm_foot}, apex={bm_apex}")

print(f"\nAll fitting plots saved to: {OUTPUT_DIR}")
