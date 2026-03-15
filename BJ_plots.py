import os
import pandas as pd
import matplotlib.pyplot as plt


input_base = 'C:/' # insert path for the desired folder here, the folder _must_ contain previously collected data, look up VJB_analysis.py for reference 
output_plots = os.path.join(input_base, 'plots')
os.makedirs(output_plots, exist_ok=True)

time_steps = None


def get_available_time_steps(base_path):
    b_files = os.listdir(os.path.join(base_path, 'B_maps'))
    time_steps_ = set()
    for f in b_files:
        if f.startswith('B_vr_t') and f.endswith('.txt'):
            t = int(f.split('_t')[1].split('.')[0])
            time_steps_.add(t)
    return sorted(time_steps_)


def create_plots_for_timestep(t_idx_, base_path, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.08, top=0.92,
        wspace=0.3, hspace=0.3
    )

    b_file = os.path.join(base_path, 'B_maps', f'B_vr_t{t_idx_:03d}.txt')
    if not os.path.exists(b_file):
        return False

    for row, (field, prefix, titles, units) in enumerate([
        ('B', 'B', ['Br', 'Bφ', 'Bz'], ['G', 'G', 'G']),
        ('J', 'J', ['Jr', 'Jφ', 'Jz'], ['A/km²', 'A/km²', 'A/km²'])
    ]):
        base_path_field = os.path.join(base_path, f'{field}_maps')

        for col, (comp, title, unit) in enumerate(zip(['vr', 'vphi', 'vz'], titles, units)):
            file_path = f'{base_path_field}/{prefix}_{comp}_t{t_idx_:03d}.txt'

            if os.path.exists(file_path):
                data = pd.read_csv(file_path)

                if len(data) > 0:
                    ax = axes[row, col]

                    scatter = ax.scatter(
                        data['x_prime'], data['y_prime'],
                        c=data[comp], cmap='coolwarm',
                        s=1, alpha=0.5, rasterized=True
                    )

                    ax.set_xlabel("X'", fontsize=12)
                    ax.set_ylabel("Y'", fontsize=12)
                    ax.set_title(f'{title} [{unit}]', fontsize=14, pad=15)
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=10)

                    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=10)

                    stats_text = f'min={data[comp].min():.2f}\nmax={data[comp].max():.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                            fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    time_val = t_idx_ * 0.119
    plt.suptitle(f'Cylindrical components at t = {time_val:.3f} (step {t_idx_})',
                 fontsize=16, y=0.98)

    output_file = os.path.join(output_dir, f'components_t{t_idx_:03d}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return True


if time_steps is None:
    time_steps = get_available_time_steps(input_base)

successful = 0
failed = 0

for t_idx in time_steps:
    create_plots_for_timestep(t_idx, input_base, output_plots)
