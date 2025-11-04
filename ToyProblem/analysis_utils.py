import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


def analyze_goal_encoding(speaker_network, ddcl_channel, config):
    """Optional: Analyze how different goals are encoded"""
    if not config.get("enable_goal_encoding_analysis", False):
        return None
        
    device = torch.device(config["device"])
    grid_size = config["grid_size"]
    
    goal_analysis = []
    total_freq = sum(config["goal_frequencies"].values())
    
    speaker_network.eval()
    with torch.no_grad():
        for goal, freq in config["goal_frequencies"].items():
            goal_normalized = torch.FloatTensor([goal[0], goal[1]]).to(device)
            z = speaker_network(goal_normalized.unsqueeze(0))
            bits = ddcl_channel.calculate_total_bits_from_z(z).item()
            
            prob = freq / total_freq
            theoretical_bits = -np.log2(prob)
            efficiency = (theoretical_bits / bits) * 100
            
            goal_analysis.append({
                'goal': goal,
                'frequency': freq,
                'bits': bits,
                'theoretical_bits': theoretical_bits,
                'efficiency_percent': efficiency
            })
    
    return pd.DataFrame(goal_analysis)


def calculate_theoretical_bounds(config):
    """Calculate theoretical Shannon bounds"""
    goal_frequencies = config["goal_frequencies"]
    total_freq = sum(goal_frequencies.values())
    
    entropy = 0.0
    for freq in goal_frequencies.values():
        prob = freq / total_freq
        entropy -= prob * np.log2(prob)
    
    return entropy


def analyze_and_visualize(speaker_network, ddcl_channel, env_config, wandb_run):
    print("\n--- Starting Post-Training Analysis ---")

    # Set up matplotlib for high-quality plots suitable for papers
    plt.rcParams.update({
        'font.size': 16,  # Base font size
        'axes.titlesize': 20,  # Title font size
        'axes.labelsize': 18,  # Axis label font size
        'xtick.labelsize': 14,  # X-axis tick label size
        'ytick.labelsize': 14,  # Y-axis tick label size
        'legend.fontsize': 14,  # Legend font size
        'figure.titlesize': 20,  # Figure title size
        'lines.linewidth': 2,  # Line width
        'axes.linewidth': 1.5,  # Axes border width
        'font.family': 'sans-serif',  # More universally available
        'text.usetex': False,  # Set to True if you have LaTeX installed
    })

    sns.set_theme(style="whitegrid", font_scale=1.0)
    grid_size = env_config["grid_size"]
    device = env_config["device"]

    # Generate all possible goal locations in proper order
    # Create goals in (x,y) format matching environment expectations
    all_goals = []
    for y in range(grid_size):
        for x in range(grid_size):
            all_goals.append([x, y])
    all_goals = np.array(all_goals)
    all_goals_tensor = torch.FloatTensor(all_goals).to(device)

    with torch.no_grad():
        # Get deterministic speaker output for analysis
        z_vectors = speaker_network(all_goals_tensor)
        comm_bits = ddcl_channel.calculate_total_bits_from_z(z_vectors).cpu().numpy()

        # Create heatmap of communication costs
        # Reshape to match grid: rows are y-coordinates (top to bottom), cols are x-coordinates
        cost_grid = comm_bits.reshape(grid_size, grid_size)

        # Create masks for goal vs non-goal positions
        goal_mask = np.zeros((grid_size, grid_size), dtype=bool)
        goal_freq_grid = np.zeros((grid_size, grid_size))
        goal_prob_grid = np.zeros((grid_size, grid_size))

        if "goal_frequencies" in env_config and env_config["goal_frequencies"]:
            goal_freq_map = env_config["goal_frequencies"]
            total_frequency = sum(goal_freq_map.values())

            for (x, y), freq in goal_freq_map.items():
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    goal_mask[y, x] = True  # Note: y first for grid indexing
                    goal_freq_grid[y, x] = freq
                    goal_prob_grid[y, x] = freq / total_frequency

        # Create the heatmap with proper orientation
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

        # Create a modified colormap that fades non-goal positions
        import matplotlib.patches as patches

        # Base heatmap with all positions
        heatmap = sns.heatmap(
            cost_grid,
            annot=False,  # We'll add custom annotations
            cmap="viridis_r",
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': 'Communication Bits'},
            square=True,
            xticklabels=range(grid_size),
            yticklabels=range(grid_size),
            alpha=0.3  # Make everything semi-transparent initially
        )

        # Overlay highlighted goal positions
        for y in range(grid_size):
            for x in range(grid_size):
                if goal_mask[y, x]:
                    # Highlight goal positions with full opacity
                    rect = patches.Rectangle((x, y), 1, 1,
                                             linewidth=3, edgecolor='red',
                                             facecolor=plt.cm.viridis_r(cost_grid[y, x] / cost_grid.max()),
                                             alpha=1.0)
                    ax.add_patch(rect)

                    # Add custom annotation with bits and frequency info
                    bits_text = f'{cost_grid[y, x]:.2f}'
                    freq_text = f'f={int(goal_freq_grid[y, x])}'
                    prob_text = f'p={goal_prob_grid[y, x]:.3f}'

                    # Multi-line annotation
                    annotation = f'{bits_text}\n{freq_text}\n{prob_text}'

                    ax.text(x + 0.5, y + 0.5, annotation,
                            ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white', alpha=0.8, edgecolor='red'))
                else:
                    # Add faded annotation for non-goal positions
                    ax.text(x + 0.5, y + 0.5, f'{cost_grid[y, x]:.2f}',
                            ha='center', va='center',
                            fontsize=8, alpha=0.6, color='gray')

        ax.set_title("Communication Bits: Goal Positions vs Non-Goal Positions", fontsize=18, pad=15)
        ax.set_xlabel("X Coordinate", fontsize=16)
        ax.set_ylabel("Y Coordinate", fontsize=16)

        # Add legend explaining the annotations
        legend_text = ("Red boxes: Goal positions\n"
                       "Top: Communication bits\n"
                       "Middle: Frequency (f)\n"
                       "Bottom: Probability (p)\n"
                       "Faded: Non-goal positions")
        ax.text(1.02, 0.5, legend_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Invert y-axis to match typical coordinate convention (origin at bottom-left)
        ax.invert_yaxis()

        plt.tight_layout()

        # Save plot offline
        plt.savefig('communication_bits_heatmap.pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig('communication_bits_heatmap.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')

        # Log to wandb
        wandb_run.log({"analysis/communication_bits_heatmap": wandb.Image(fig)})
        plt.close()
        print("Saved and logged communication bits heatmap (PDF and PNG).")

        # Create a separate, cleaner visualization focusing just on goal positions
        if "goal_frequencies" in env_config and env_config["goal_frequencies"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)

            # Left plot: Communication bits for all positions with goal highlights
            im1 = ax1.imshow(cost_grid, cmap='viridis_r', alpha=0.3)
            ax1.set_title("Communication Bits (All Positions)", fontsize=16)

            # Right plot: Goal frequencies
            freq_grid = np.full((grid_size, grid_size), np.nan)
            goal_freq_map = env_config["goal_frequencies"]
            total_frequency = sum(goal_freq_map.values())

            for (x, y), freq in goal_freq_map.items():
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    freq_grid[y, x] = freq / total_frequency  # Convert to probability

            # Mask for showing only goal positions
            masked_freq = np.ma.masked_where(np.isnan(freq_grid), freq_grid)
            im2 = ax2.imshow(masked_freq, cmap='Reds', vmin=0, vmax=masked_freq.max())
            ax2.set_title("Goal Sampling Probabilities", fontsize=16)

            # Add annotations and highlights for both plots
            for (x, y), freq in goal_freq_map.items():
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    prob = freq / total_frequency

                    # Left plot: highlight goal with communication cost
                    rect1 = patches.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8,
                                              linewidth=3, edgecolor='red',
                                              facecolor='none')
                    ax1.add_patch(rect1)
                    ax1.text(x, y, f'{cost_grid[y, x]:.2f}\n({freq})',
                             ha='center', va='center', fontweight='bold', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                    # Right plot: show probability
                    ax2.text(x, y, f'{prob:.3f}\n(f={freq})',
                             ha='center', va='center', fontweight='bold', fontsize=11,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            # Format both plots
            for ax in [ax1, ax2]:
                ax.set_xlim(-0.5, grid_size - 0.5)
                ax.set_ylim(-0.5, grid_size - 0.5)
                ax.set_xticks(range(grid_size))
                ax.set_yticks(range(grid_size))
                ax.set_xlabel("X Coordinate", fontsize=14)
                ax.set_ylabel("Y Coordinate", fontsize=14)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3)

            # Add colorbars
            plt.colorbar(im1, ax=ax1, label='Communication Bits', shrink=0.8)
            plt.colorbar(im2, ax=ax2, label='Sampling Probability', shrink=0.8)

            plt.suptitle("Goal Position Analysis", fontsize=18, y=1.02)
            plt.tight_layout()

            # Save goal-focused visualization
            plt.savefig('goal_analysis_comparison.pdf', dpi=300, bbox_inches='tight')
            plt.savefig('goal_analysis_comparison.png', dpi=300, bbox_inches='tight')
            wandb_run.log({"analysis/goal_analysis_comparison": wandb.Image(fig)})
            plt.close()
            print("Saved and logged goal analysis comparison.")

        # Handle goal frequencies analysis - only for goals that have defined frequencies
        if "goal_frequencies" in env_config and env_config["goal_frequencies"]:
            goal_freq_map = env_config["goal_frequencies"]
            analysis_data = []

            # Only analyze goals that have defined frequencies
            for goal_tuple, frequency in goal_freq_map.items():
                # Find the index of this goal in our all_goals array
                goal_idx = None
                for i, goal in enumerate(all_goals):
                    if goal[0] == goal_tuple[0] and goal[1] == goal_tuple[1]:
                        goal_idx = i
                        break

                if goal_idx is not None:
                    analysis_data.append({
                        "goal_x": int(goal_tuple[0]),
                        "goal_y": int(goal_tuple[1]),
                        "goal_label": f"({int(goal_tuple[0])}, {int(goal_tuple[1])})",
                        "frequency": frequency,
                        "cost": comm_bits[goal_idx]
                    })

            if len(analysis_data) >= 3:  # Need at least 3 points for meaningful analysis
                df = pd.DataFrame(analysis_data).sort_values("frequency", ascending=False)

                # Create frequency categories using actual data distribution
                if len(df) >= 3:
                    # Use quantiles if we have enough data points
                    try:
                        freq_bins = pd.qcut(df['frequency'], q=min(3, len(df)),
                                            labels=['Low', 'Medium', 'High'][:min(3, len(df))],
                                            duplicates='drop')
                    except ValueError:
                        # If qcut fails due to duplicate edges, use simple thresholds
                        freq_median = df['frequency'].median()
                        freq_75 = df['frequency'].quantile(0.75)
                        freq_bins = pd.cut(df['frequency'],
                                           bins=[0, freq_median, freq_75, float('inf')],
                                           labels=['Low', 'Medium', 'High'],
                                           include_lowest=True)
                else:
                    freq_bins = ['Medium'] * len(df)  # Default category for small datasets

                df['freq_category'] = freq_bins

                # Bar chart
                fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

                # Create color palette
                colors = {'Low': '#3498db', 'Medium': '#f39c12', 'High': '#e74c3c'}
                bar_colors = [colors.get(cat, '#95a5a6') for cat in df['freq_category']]

                bars = ax.bar(range(len(df)), df['cost'], color=bar_colors, alpha=0.8, edgecolor='black')

                ax.set_title("Communication Bits per Goal Location\n(Sorted by Goal Frequency)",
                             fontsize=18, pad=15)
                ax.set_xlabel("Goal Location (Sorted by Frequency)", fontsize=16)
                ax.set_ylabel("Communication Bits", fontsize=16)

                # Set x-axis labels
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['goal_label'], rotation=45, ha='right')

                # Create legend
                unique_categories = df['freq_category'].unique()
                legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors.get(cat, '#95a5a6'),
                                                 label=f'{cat} Frequency') for cat in unique_categories]
                ax.legend(handles=legend_elements, loc='upper right')

                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom')

                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)

                plt.tight_layout()

                # Save plot offline
                plt.savefig('bits_vs_frequency_barchart.pdf', dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                plt.savefig('bits_vs_frequency_barchart.png', dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')

                # Log to wandb
                wandb_run.log({"analysis/bits_vs_frequency_barchart": wandb.Image(fig)})
                plt.close()
                print("Saved and logged sorted bits vs. frequency bar chart (PDF and PNG).")

                # Calculate correlation
                correlation, p_value = pearsonr(df["frequency"], df["cost"])
                wandb_run.summary["analysis/correlation_freq_vs_cost"] = correlation
                wandb_run.summary["analysis/correlation_p_value"] = p_value
                print(f"Pearson Correlation between frequency and cost: {correlation:.4f} (p={p_value:.4f})")

                # Scatter plot for correlation visualization
                fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

                # Create scatter plot with category colors
                for category in unique_categories:
                    mask = df['freq_category'] == category
                    if mask.any():
                        ax.scatter(df[mask]["frequency"], df[mask]["cost"],
                                   c=colors.get(category, '#95a5a6'),
                                   label=f'{category} Frequency',
                                   s=100, alpha=0.7, edgecolors='black', linewidth=1)

                # Add trend line if we have enough points
                if len(df) > 2:
                    z = np.polyfit(df["frequency"], df["cost"], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(df["frequency"].min(), df["frequency"].max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')

                ax.set_title(f"Communication Bits vs Goal Frequency\n(Correlation: r = {correlation:.3f})",
                             fontsize=18, pad=15)
                ax.set_xlabel("Goal Frequency", fontsize=16)
                ax.set_ylabel("Communication Bits", fontsize=16)

                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                ax.set_axisbelow(True)

                plt.tight_layout()

                # Save scatter plot offline
                plt.savefig('frequency_cost_correlation.pdf', dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                plt.savefig('frequency_cost_correlation.png', dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')

                # Log to wandb
                wandb_run.log({"analysis/frequency_cost_correlation": wandb.Image(fig)})
                plt.close()
                print("Saved and logged frequency vs cost correlation plot (PDF and PNG).")

            else:
                print(
                    f"Insufficient data for frequency analysis: only {len(analysis_data)} goals with defined frequencies")
        else:
            print("No goal frequencies defined - skipping frequency analysis")

        # Additional analysis: Show distribution of communication costs
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.hist(comm_bits, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title("Distribution of Communication Bits Across All Goals", fontsize=18)
        ax.set_xlabel("Communication Bits", fontsize=16)
        ax.set_ylabel("Number of Goals", fontsize=16)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"Mean: {comm_bits.mean():.2f}\nStd: {comm_bits.std():.2f}\nMin: {comm_bits.min():.2f}\nMax: {comm_bits.max():.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('communication_bits_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('communication_bits_distribution.png', dpi=300, bbox_inches='tight')
        wandb_run.log({"analysis/communication_bits_distribution": wandb.Image(fig)})
        plt.close()
        print("Saved and logged communication bits distribution.")

    # Reset matplotlib parameters to default after plotting
    plt.rcParams.update(plt.rcParamsDefault)

    print("--- Analysis complete. All plots saved offline and logged to wandb ---")
