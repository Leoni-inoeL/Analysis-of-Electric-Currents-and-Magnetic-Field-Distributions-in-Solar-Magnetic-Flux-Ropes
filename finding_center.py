import pandas as pd
from datetime import datetime
from data_loader import load_and_prepare_data
from center_of_mass import center_of_mass
from moments_analysis import moments_analysis
from gradient_symmetry import gradient_symmetry
from circle_bubbling_method import circle_bubbling_method
from metrics import calculate_metrics, calculate_average_center


def run_comparison():
    target_date_ = datetime(2025, 11, 11, 2, 0, 0)
    sample_map, data_clean, reference_center = load_and_prepare_data(target_date_)

    methods = [
        center_of_mass,
        moments_analysis,
        gradient_symmetry,
        circle_bubbling_method
    ]

    results = []
    uncertainties_ = {}

    for method in methods:
        result = method(data_clean)
        if result[0] is not None:
            center, method_name_, uncertainty = result
            error, dx, dy = calculate_metrics(center, reference_center)

            results.append({
                'Method': method_name_,
                'Center_X': center[0],
                'Center_Y': center[1],
                'Error_pixels': error,
                'Delta_X': dx,
                'Delta_Y': dy,
            })

            uncertainties_[method_name_] = uncertainty

            print(f"\n{method_name_}:")
            print(f"  Center: ({center[0]:.2f}, {center[1]:.2f})")
            print(f"  Error: {error:.2f} pixels")
            print(f"  Delta: (Δx={dx:+.2f}, Δy={dy:+.2f})")

            if uncertainty:
                std_x, std_y = uncertainty['std_pixels']
                print(f"  Uncertainty (1σ): ±({std_x:.2f}, {std_y:.2f}) pixels")

                if 'correlation' in uncertainty:
                    print(f"  Correlation: {uncertainty['correlation']:.3f}")

                if 'final_diameter' in uncertainty:
                    print(f"  Final diameter: {uncertainty['final_diameter']:.2f} pixels")

                if 'edge_pixels_used' in uncertainty:
                    print(f"  Edge pixels used: {uncertainty['edge_pixels_used']} "
                          f"(out of {uncertainty['total_edge_pixels']})")

    ref_info = f"({reference_center[0]:.2f}, {reference_center[1]:.2f})"
    print(f"\n{'Reference (given center coordinates)':<25}: {ref_info}")

    df_results = pd.DataFrame(results)

    averaging_results_ = calculate_average_center(df_results, reference_center)

    if averaging_results_:
        simple_avg_ = averaging_results_['simple_average']
        weighted_avg = averaging_results_['weighted_average']

        print(f"\nAverage of {averaging_results_['methods_used']} methods:")
        print("Simple Average:")
        print(f"  Center: ({simple_avg_['center'][0]:.2f}, {simple_avg_['center'][1]:.2f})")
        print(f"  Error: {simple_avg_['error']:.2f} pixels")

        print("\nWeighted Average:")
        print(f"  Center: ({weighted_avg['center'][0]:.2f}, {weighted_avg['center'][1]:.2f})")
        print(f"  Error: {weighted_avg['error']:.2f} pixels")

    return df_results, averaging_results_, uncertainties_


results_df, averaging_results, uncertainties = run_comparison()

if not results_df.empty:
    best_method = results_df.loc[results_df['Error_pixels'].idxmin()]
    print(f"\nMost Accurate: {best_method['Method']} "
          f"(Error: {best_method['Error_pixels']:.2f} px)")

    if averaging_results:
        simple_avg = averaging_results['simple_average']
        improvement = best_method['Error_pixels'] - simple_avg['error']
        print(f"Average Improvement: {improvement:+.2f} px")

    print("\nError/Uncertainty ratio:")
    for _, row in results_df.iterrows():
        method_name = row['Method']
        if (uncertainties and method_name in uncertainties and
                uncertainties[method_name]):
            unc = uncertainties[method_name]
            mean_std = np.mean(unc['std_pixels'])
            reliability = row['Error_pixels'] / mean_std if mean_std > 0 else 0
            print(f"  {method_name}: {reliability:.3f}")
