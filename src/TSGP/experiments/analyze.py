"""Analyze and visualize results from completed experimental runs."""

import pandas as pd
import numpy as np
import glob
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages

# Set plotting style
rcParams['pdf.fonttype'] = 42
#rcParams['font.family'] = "Times New Roman"
rcParams['font.size'] = 14

# Color and style settings
colors = plt.cm.tab10.colors
line_styles = ['-', '--', '-.', ':']
shapes = ["o", "s", '^', 'd', '*', 'v', '<', '>', 'p', 'H']
grid_background = "#ECECEC"

# All best-fitness variants, in the order they should appear in plots
BEST_FITNESS_METRICS = [
    ("best_fitness", "Best Training Fitness"),
    ("best_fitness_test", "Best Test Fitness"),
]

def load_experiment_metadata(folder_path):
    """Load metadata from experiment folder."""
    metadata_yaml = folder_path / 'experiment_metadata.yaml'
    metadata_pkl = folder_path / 'experiment_metadata.pkl'
    
    if metadata_yaml.exists():
        with open(metadata_yaml, 'r') as f:
            return yaml.safe_load(f)
    elif metadata_pkl.exists():
        import pickle
        with open(metadata_pkl, 'rb') as f:
            return pickle.load(f)
    return None

def load_monitor_files(folder_path):
    """Load all monitor CSV files from a folder and aggregate."""
    pattern = str(folder_path / "**" / "monitor_run_*.csv")
    monitor_files = glob.glob(pattern, recursive=True)
    
    if not monitor_files:
        print(f"No monitor files found in {folder_path}")
        return None
    
    # Load all monitor files
    dfs = []
    for file in monitor_files:
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"  Skipping empty monitor file: {file}")
            continue
        if df.empty:
            print(f"  Skipping empty monitor file: {file}")
            continue
        dfs.append(df)

    if not dfs:
        print(f"No non-empty monitor files found in {folder_path}")
        return None

    # Concatenate all runs
    combined = pd.concat(dfs, ignore_index=True)

    return combined

def aggregate_data(df):
    """Aggregate monitor data across runs."""
    # Define aggregation functions
    dict_to_group = {
        'best_fitness': ['median', ('q1', lambda x: x.quantile(0.25)), 
                        ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'best_fitness_test': ['median', ('q1', lambda x: x.quantile(0.25)),
                             ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'fitness_avg': ['median', ('q1', lambda x: x.quantile(0.25)),
                        ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'fitness_test_avg': ['median', ('q1', lambda x: x.quantile(0.25)), 
                             ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'fitness_25percentile': ['median', ('q1', lambda x: x.quantile(0.25)), 
                                 ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'fitness_median': ['median', ('q1', lambda x: x.quantile(0.25)), 
                           ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'fitness_75percentile': ['median', ('q1', lambda x: x.quantile(0.25)), 
                                 ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'size_best': ['median', ('q1', lambda x: x.quantile(0.25)), 
                     ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'size_avg': ['median', ('q1', lambda x: x.quantile(0.25)), 
                    ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'time_used': ['median', ('q1', lambda x: x.quantile(0.25)), 
                    ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'unique_rate': ['median', ('q1', lambda x: x.quantile(0.25)), 
                       ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'semantic_distance_25percentile': ['median', ('q1', lambda x: x.quantile(0.25)), 
                                          ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'semantic_distance_median': ['median', ('q1', lambda x: x.quantile(0.25)), 
                                  ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        'semantic_distance_75percentile': ['median', ('q1', lambda x: x.quantile(0.25)), 
                                          ('q2', lambda x: x.quantile(0.75)), 'mean', 'std'],
        
    }
    
    # Only include columns that exist in the data
    dict_to_group = {k: v for k, v in dict_to_group.items() if k in df.columns}
    
    # Aggregate by generation
    aggr = df.groupby('generation').agg(dict_to_group)
    
    # Flatten column names
    aggr.columns = ['{}_{}'.format(col[0], col[1]) for col in aggr.columns]
    aggr = aggr.reset_index()
    
    return aggr

def final_generation_values(df):
    """Return one row per run, taken at that run's last generation."""
    run_key = 'seed' if 'seed' in df.columns else None
    if run_key is None:
        # Fall back to the last generation present across the whole frame
        return df[df['generation'] == df['generation'].max()]
    last_idx = df.groupby(run_key)['generation'].idxmax()
    return df.loc[last_idx]

def plot_boxplots(final_list, names, title):
    """Boxplot of the final-generation best fitness of every run, one panel per metric."""
    metrics = [(m, label) for m, label in BEST_FITNESS_METRICS
               if any(m in final for final in final_list)]
    if not metrics:
        return None

    fig, axs = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6), squeeze=False)
    axs = axs[0]
    fig.suptitle(f"{title} - Final Generation Best Fitness", fontsize=20, fontweight='bold')

    for ax, (metric, label) in zip(axs, metrics):
        data, tick_labels, box_colors = [], [], []
        for i, (final, name) in enumerate(zip(final_list, names)):
            if metric not in final:
                continue
            values = final[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            data.append(values.values)
            tick_labels.append(name)
            box_colors.append(colors[i % len(colors)])

        if not data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)

        ax.grid(color=grid_background, linestyle='--', linewidth=0.5)
        ax.set_title(label)
        ax.set_ylabel("Fitness (RMSE)")
        ax.tick_params(axis='x', rotation=20)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

def plot_inter_run_variability(ax, aggr, metric, aggregate, title, y_label, color, name, line_style, shape, step_size=5, ylim=None):
    """Plot the variability *between* runs (inter-run variability)."""
    thinned = list(range(0, len(aggr), step_size))
    
    # Plot main line
    ax.plot(aggr["generation"], aggr[f"{metric}_{aggregate}"], 
            color=color, label=name, linestyle=line_style)
    
    # Add markers at thinned points
    ax.scatter(aggr["generation"].iloc[thinned], aggr[f"{metric}_{aggregate}"].iloc[thinned], 
               marker=shape, color=color)
    
    # Add confidence ribbon
    if aggregate == "median":
        ax.fill_between(aggr["generation"],
                        aggr[f"{metric}_q1"],
                        aggr[f"{metric}_q2"],
                        color=color, alpha=0.1)
    else:
        ax.fill_between(aggr["generation"],
                        aggr[f"{metric}_mean"] - aggr[f"{metric}_std"],
                        aggr[f"{metric}_mean"] + aggr[f"{metric}_std"],
                        color=color, alpha=0.1)
    

    if ylim is not None:
        # set y-limit 
        ax.set_ylim(top=ylim)
        ax.set_ylim(bottom=0)
    
    # Style
    ax.grid(color=grid_background, linestyle='--', linewidth=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel(y_label)
    ax.set_title(title)

def plot_intra_run_variability(ax, aggr, metric, aggregate, title, y_label, color, name, line_style, shape, step_size=5, ylim=None, ribbon_lower=None, ribbon_upper=None):
    """
    Plot the intra-run variability (*within* individual runs) aggregated over multiple runs.
    
    Args:
        ribbon_lower: Column name for lower ribbon bound (without aggregate suffix).
                     Defaults to '{metric_base}_25percentile'.
        ribbon_upper: Column name for upper ribbon bound (without aggregate suffix).
                     Defaults to '{metric_base}_75percentile'.
    """
    thinned = list(range(0, len(aggr), step_size))
    if aggregate != "median":
        raise ValueError("Intra-run variability plotting currently only supports 'median' aggregate.")
    
    metric_base = metric.rsplit('_median', 1)[0]
    
    # Default ribbon columns
    if ribbon_lower is None:
        ribbon_lower = f"{metric_base}_25percentile"
    if ribbon_upper is None:
        ribbon_upper = f"{metric_base}_75percentile"

    # Plot main line
    ax.plot(aggr["generation"], aggr[f"{metric}_{aggregate}"], 
            color=color, label=name, linestyle=line_style)
    
    # Add markers at thinned points
    ax.scatter(aggr["generation"].iloc[thinned], aggr[f"{metric}_{aggregate}"].iloc[thinned], 
               marker=shape, color=color)
    
    # Add confidence ribbon
    if aggregate == "median":
        ax.fill_between(aggr["generation"],
                        aggr[f"{ribbon_lower}_{aggregate}"],
                        aggr[f"{ribbon_upper}_{aggregate}"],
                        color=color, alpha=0.1)
    else:
        raise NotImplementedError("Intra-run variability with mean/std not implemented.")
    
    if ylim is not None:
        # set y-limit 
        ax.set_ylim(top=ylim)
        ax.set_ylim(bottom=0)
    
    # Style
    ax.grid(color=grid_background, linestyle='--', linewidth=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel(y_label)
    ax.set_title(title)

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize experimental results')
    parser.add_argument('-f', '--folders', nargs='+',
                        help='Experiment folders to analyze and compare',
                        required=True)
    parser.add_argument('-n', '--names', nargs='+',
                        help='Names for each experiment (optional)',
                        default=None)
    parser.add_argument('-o', '--output',
                        help='Output PDF file path',
                        required=True)
    parser.add_argument('--title',
                        help='Main title for the plots',
                        default='Experiments')
    parser.add_argument('--boxplots', action=argparse.BooleanOptionalAction,
                        help='Add a page with boxplots of the final-generation best fitness of every run',
                        default=True)
    args = parser.parse_args()
    
    # Convert folder paths to Path objects
    folders = [Path(f) for f in args.folders]
    
    # Generate names if not provided
    if args.names:
        names = args.names
    else:
        names = [f.name for f in folders]
    
    if len(names) != len(folders):
        print(f"Error: Number of names ({len(names)}) must match number of folders ({len(folders)})")
        return
    
    # Load and aggregate data from all folders
    aggr_list = []
    final_list = []
    metadata_list = []
    plot_names = []
    plot_folders = []

    print(f"\nLoading data from {len(folders)} experiment folders...")
    for i, folder in enumerate(folders):
        print(f"\nProcessing: {folder}")

        # Load monitor files
        df = load_monitor_files(folder)
        if df is None:
            print(f"Skipping {folder} - no data found")
            continue

        # Convert time to hours
        df["time_used"] = df["time_used"] / 3600

        # Aggregate
        aggr = aggregate_data(df)
        aggr_list.append(aggr)
        final_list.append(final_generation_values(df))
        metadata_list.append(load_experiment_metadata(folder))
        plot_names.append(names[i])
        plot_folders.append(folder)

        print(f"  Loaded {len(df)} total records from {folder}")
        print(f"  Generations: {aggr['generation'].min()} to {aggr['generation'].max()}")

    if not aggr_list:
        print("No data to plot!")
        return

    names = plot_names

    # Create figure with subplots
    num_plots = 8
    num_cols = 2
    num_rows = num_plots // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 6.8 * num_rows))
    axs = axs.flatten()
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.text(0.5, 0.98, args.title, ha='center', va='top', fontsize=24, fontweight='bold')
    
    # Add metadata information for first experiment
    if metadata_list[0]:
        metadata = metadata_list[0]
        info_text = f"Dataset: {metadata['problem']['dataset']}\n" \
                   f"Population Size: {metadata['popsize']}\n" \
                   f"Functions: {metadata['problem']['functions']}\n" \
                   f"Runs: {len(glob.glob(str(plot_folders[0] / 'monitor_run_*.csv')))}"
        
        fig.text(0.02, 0.96, info_text, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle="round", alpha=0.1, edgecolor='none', facecolor='lightgrey'))
    
    # Generate plots for each experiment
    for i, (aggr, name) in enumerate(zip(aggr_list, names)):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        shape = shapes[i % len(shapes)]
        
        # Get final generation values
        final_gen = aggr.iloc[-1]
        
        print(f"\n{'='*80}")
        print(f"Final Generation Stats for: {name}")
        print(f"{'='*80}")
        print(f"Generation: {final_gen['generation']}")
        print(f"Best Training Fitness (median): {final_gen['best_fitness_median']:.6f}")
        print(f"Best Test Fitness (median): {final_gen['best_fitness_test_median']:.6f}")
        print(f"Best Solution Size (median): {final_gen['size_best_median']:.2f} nodes")
        print(f"Avg Program Size (median): {final_gen['size_avg_median']:.2f} nodes")
        print(f"Cumulative Time (mean): {final_gen['time_used_mean']:.4f} hours")
        print(f"Unique Rate (median): {final_gen['unique_rate_median']:.4f}")
        print(f"Semantic Distance (median): {final_gen['semantic_distance_median_median']:.4f}")
        
        # Best Training fitness
        plot_inter_run_variability(axs[0], aggr, "best_fitness", "median",
                     "Best Training Fitness Over Generations",
                     "Fitness (RMSE)", color, name, line_style, shape)
        
        # BestTest fitness
        plot_inter_run_variability(axs[1], aggr, "best_fitness_test", "median",
                     "Best Test Fitness Over Generations",
                     "Fitness (RMSE)", color, name, line_style, shape)
        
        # Best solution size
        plot_inter_run_variability(axs[2], aggr, "size_best", "median",
                     "Best Solution Size Over Generations",
                     "Number of Nodes", color, name, line_style, shape)
        
        # Average size
        plot_inter_run_variability(axs[3], aggr, "size_avg", "median",
                     "Average Program Size Over Generations",
                     "Number of Nodes", color, name, line_style, shape)
        
        # Unique rate
        plot_inter_run_variability(axs[4], aggr, "unique_rate", "median",
                     "Population Diversity (Unique Rate) Over Generations",
                     "Unique Rate", color, name, line_style, shape)
        
        # Runtime
        plot_inter_run_variability(axs[5], aggr, "time_used", "mean",
                     "Cumulative Runtime Over Generations",
                     "Time (hours)", color, name, line_style, shape)
        
        # Semantic distance
        plot_intra_run_variability(axs[6], aggr, "semantic_distance_median", "median",
                     "Semantic Distance (Parent-Child) Over Generations (Intra-Run Variability)",
                     "Euclidean Distance", color, name, line_style, shape, ylim = 10)
        
        # Median training fitness
        plot_intra_run_variability(axs[7], aggr, "fitness_median", "median",
                     "Median Training Fitness Over Generations (Intra-Run Variability)",
                     "Fitness (RMSE)", color, name, line_style, shape, ylim = 2)

    # Hide unused subplots
    for idx in range(num_plots, num_rows * num_cols):
        axs[idx].set_visible(False)
    
    # Create custom legend handles with markers
    handles = []
    for i, name in enumerate(names):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        shape = shapes[i % len(shapes)]
        line = plt.Line2D([0], [0], color=color, linestyle=line_style, marker=shape, 
                         markersize=10, label=name, linewidth=2)
        handles.append(line)
    
    # Add shared legend to figure
    fig.legend(handles=handles, loc='lower center', ncol=len(names), 
              bbox_to_anchor=(0.5, -0.01), fontsize=12)
    
    # Save to PDF
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    box_fig = plot_boxplots(final_list, names, args.title) if args.boxplots else None

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, orientation="landscape", bbox_inches='tight')
        if box_fig is not None:
            pdf.savefig(box_fig, orientation="landscape", bbox_inches='tight')

    print(f"\n{'='*80}")
    print(f"Plot saved to: {output_path}")
    print(f"{'='*80}")
    plt.close(fig)
    if box_fig is not None:
        plt.close(box_fig)

if __name__ == "__main__":
    main()
