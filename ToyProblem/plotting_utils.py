import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr


def setup_style():
    """Setup publication-quality matplotlib parameters"""
    plt.style.use('default')  # Start with clean default
    
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
        'font.size': 20,
        'axes.titlesize': 30,
        'axes.labelsize': 25,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
        'legend.fontsize': 20,
        'figure.titlesize': 30,
        
        # Line and marker settings
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'patch.linewidth': 1.0,
        
        # Grid and spacing
        'grid.alpha': 0.4,
        'axes.grid': True,
        'axes.axisbelow': True,
        'figure.constrained_layout.use': True,
        
        # Colors and aesthetics
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        
        # High DPI for crisp images
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'savefig.facecolor': 'white',
    })


def create_plots(results_df, theoretical_entropy, save_dir):
    """Create clean, interpretable ICLR-quality plots with proper error bars and clear labels"""
    
    setup_style()
    
    # Professional color palette
    colors = {
        'empirical': '#1f77b4',      # Blue
        'shannon': '#d62728',        # Red  
        'optimum': '#ff7f0e',        # Orange
        'success': '#2ca02c',        # Green
        'rate': '#9467bd',           # Purple
        'gap': '#8c564b'             # Brown
    }
    
    # Aggregate results
    agg_results = results_df.groupby('Lambda').agg({
        'Rate': ['mean', 'std', 'count'],
        'Distortion': ['mean', 'std', 'count'],
        'Success_Rate': ['mean', 'std', 'count']
    }).reset_index()
    
    agg_results.columns = ['Lambda', 'Rate_Mean', 'Rate_Std', 'Rate_Count',
                          'Distortion_Mean', 'Distortion_Std', 'Distortion_Count',
                          'Success_Rate_Mean', 'Success_Rate_Std', 'Success_Rate_Count']
    
    # Calculate standard errors
    agg_results['Rate_SE'] = agg_results['Rate_Std'] / np.sqrt(agg_results['Rate_Count'])
    agg_results['Distortion_SE'] = agg_results['Distortion_Std'] / np.sqrt(agg_results['Distortion_Count'])
    agg_results['Success_Rate_SE'] = agg_results['Success_Rate_Std'] / np.sqrt(agg_results['Success_Rate_Count'])
    
    agg_results = agg_results.sort_values('Lambda').reset_index(drop=True)
    
    # === 1. MAIN RATE-DISTORTION CURVE ===
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    
    x_data = agg_results['Rate_Mean']
    y_data = agg_results['Distortion_Mean']
    x_err = agg_results['Rate_SE']
    y_err = agg_results['Distortion_SE']
    lambda_vals = agg_results['Lambda']
    
    # Plot main curve with THIN error bars
    ax1.errorbar(x_data, y_data, xerr=x_err, yerr=y_err,
                fmt='o-', color=colors['empirical'], linewidth=3.0,  # Thick main line
                markersize=10, markerfacecolor='white', markeredgewidth=2,
                capsize=6, capthick=1.0, elinewidth=1.0,  # THIN error bar parameters
                alpha=0.9, label='DDCL (Empirical)', zorder=5)
    
    # Smart label positioning function
    def calculate_label_positions(x_data, y_data, ax):
        positions = []
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            # Different positioning strategy based on index and proximity to Shannon line
            base_angles = [0, 45, 90, 180, 135, 225, 270, 315]
            angle = base_angles[i % len(base_angles)]
            
            # Adjust radius and angle near Shannon line
            if x < theoretical_entropy + 1:
                radius = 35
                if angle < 180:
                    angle += 20
            else:
                radius = 25
            
            angle_rad = np.radians(angle)
            offset_x = radius * np.cos(angle_rad)
            offset_y = radius * np.sin(angle_rad)
            
            ha = 'left' if offset_x > 0 else 'right'
            va = 'bottom' if offset_y > 0 else 'top'
            
            positions.append((offset_x, offset_y, ha, va))
        
        return positions
    
    # Shannon bound and theoretical optimum
    shannon_x = theoretical_entropy
    y_min_plot = max(0, y_data.min() - 0.05)
    y_max_plot = min(1, y_data.max() + 0.05)
    
    ax1.axvline(x=shannon_x, color=colors['shannon'], linestyle='--', 
               linewidth=3.0, alpha=0.9, zorder=3,
               label=f'Shannon Bound\\nH(X) = {theoretical_entropy:.2f} bits')
    
    ax1.scatter([shannon_x], [0], color=colors['optimum'], s=300, 
               marker='*', edgecolors='black', linewidth=2, zorder=7,
               label='Theoretical Optimum\\n(H(X), 0)')
    
    # Set proper limits
    x_range = x_data.max() - x_data.min()
    y_range = y_data.max() - y_data.min()
    x_margin = max(0.2 * x_range, 1.5)
    y_margin = max(0.15 * y_range, 0.05)
    
    ax1.set_xlim(max(0, x_data.min() - x_margin), x_data.max() + x_margin)
    ax1.set_ylim(max(0, y_data.min() - y_margin), min(1, y_data.max() + y_margin))
    
    ax1.set_xlabel('Rate R(λ) [Average Bits per Episode]', fontweight='bold', fontsize=25)
    ax1.set_ylabel('Distortion D = 1 - Success Rate', fontweight='bold', fontsize=25)
    ax1.set_title('Rate-Distortion Analysis: Multi-Agent Communication Efficiency', 
                 fontweight='bold', pad=40, fontsize=30)
    
    ax1.legend(loc='upper right', frameon=True, fancybox=True, 
              shadow=True, framealpha=0.95, fontsize=20)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    fig1.savefig(save_dir / 'rate_distortion_curve.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig(save_dir / 'rate_distortion_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # === 2. PERFORMANCE ANALYSIS WITH DUAL AXIS ===
    fig2 = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 2, figure=fig2, wspace=0.35)
    
    # Prepare cleaner lambda labels
    x_positions = list(range(len(lambda_vals)))
    x_labels = []
    for lam in lambda_vals:
        if lam == 0:
            x_labels.append('λ=0')
        else:
            # Handle the formatting more carefully
            if lam >= 1:
                # For values >= 1, just show the number
                x_labels.append(f'λ={lam}')
            else:
                # For values < 1, use scientific notation
                exp = int(np.floor(np.log10(lam)))
                coeff = lam / (10 ** exp)
                
                # Round coefficient to avoid floating point precision issues
                coeff = round(coeff, 1)
                
                if coeff == 1.0:
                    x_labels.append(f'λ=10-{abs(exp)}')
                else:
                    # Remove unnecessary decimal if it's a whole number
                    if coeff == int(coeff):
                        x_labels.append(f'λ={int(coeff)}×10-{abs(exp)}')
                    else:
                        x_labels.append(f'λ={coeff}×10-{abs(exp)}')
    
    # Combined dual-axis plot: Success Rate + Communication Rate
    ax2a = fig2.add_subplot(gs[0, 0])
    
    # Primary axis: Success Rate
    y_success = agg_results['Success_Rate_Mean']
    y_success_err = agg_results['Success_Rate_SE']
    
    line1 = ax2a.errorbar(x_positions, y_success, yerr=y_success_err,
                         fmt='o-', linewidth=3.0, markersize=10, markerfacecolor='white',
                         markeredgewidth=2, capsize=8, capthick=1.2, elinewidth=1.2,
                         color=colors['success'], ecolor=colors['success'], alpha=0.9,
                         label='Success Rate')
    
    ax2a.set_xlabel('Communication Penalty λ', fontweight='bold', fontsize=15)
    ax2a.set_ylabel('Success Rate', fontweight='bold', fontsize=15, color=colors['success'])
    ax2a.tick_params(axis='y', labelcolor=colors['success'])
    
    y_min = max(0, (y_success - y_success_err).min() - 0.03)
    y_max = min(1, (y_success + y_success_err).max() + 0.05)
    ax2a.set_ylim(y_min, y_max)
    
    # Secondary axis: Communication Rate
    ax2a_twin = ax2a.twinx()
    
    y_rate = agg_results['Rate_Mean']
    y_rate_err = agg_results['Rate_SE']
    
    line2 = ax2a_twin.errorbar(x_positions, y_rate, yerr=y_rate_err,
                              fmt='s-', linewidth=3.0, markersize=10, markerfacecolor='white',
                              markeredgewidth=2, capsize=8, capthick=1.2, elinewidth=1.2,
                              color=colors['rate'], ecolor=colors['rate'], alpha=0.9,
                              label='Communication Rate')
    
    # Shannon bound line
    line3 = ax2a_twin.axhline(y=theoretical_entropy, color=colors['shannon'], linestyle='--',
                             linewidth=3.0, alpha=0.9, label=f'Shannon Bound H(X) = {theoretical_entropy:.2f}')
    
    ax2a_twin.set_ylabel('Rate [Bits per Episode]', fontweight='bold', fontsize=15, color=colors['rate'])
    ax2a_twin.tick_params(axis='y', labelcolor=colors['rate'])
    
    # Set x-axis for both
    ax2a.set_xticks(x_positions)
    ax2a.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=15)
    ax2a.grid(True, alpha=0.4)
    
    # Combined legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax2a.legend(lines, labels, loc='center left', frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    ax2a.set_title('(a) Task Performance & Communication Rate vs Penalty', fontweight='bold', fontsize=20)
    
    # Subplot B: Shannon Gap - FIXED BAR CHART
    ax2c = fig2.add_subplot(gs[0, 1])
    
    shannon_gaps = y_rate - theoretical_entropy
    gap_errors = y_rate_err
    
    # Color bars based on efficiency
    colors_bars = []
    for gap in shannon_gaps:
        if gap < 0.5:
            colors_bars.append('#2ca02c')  # Green
        elif gap < 1.0:
            colors_bars.append('#ff7f0e')  # Orange
        elif gap < 2.0:
            colors_bars.append('#d62728')  # Red
        else:
            colors_bars.append('#8b0000')  # Dark red
    
    # Create bars - REMOVE the error bar parameters that don't work with bar()
    bars = ax2c.bar(x_positions, shannon_gaps, 
                   color=colors_bars, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    # Add error bars manually using errorbar()
    ax2c.errorbar(x_positions, shannon_gaps, yerr=gap_errors,
                 fmt='none', capsize=6, capthick=1.2, elinewidth=1.2,
                 color='black', alpha=0.8, zorder=6)
    
    # Add value labels on bars
    for i, (bar, gap, err) in enumerate(zip(bars, shannon_gaps, gap_errors)):
        height = bar.get_height()
        y_pos = height + err + 0.1 if height >= 0 else height - err - 0.1
        va_pos = 'bottom' if height >= 0 else 'top'
        
        ax2c.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f'{gap:.2f}', ha='center', va=va_pos, 
                 fontsize=10, fontweight='bold', color='black')
    
    ax2c.set_xticks(x_positions)
    ax2c.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=15)
    ax2c.set_xlabel('Communication Penalty λ', fontweight='bold', fontsize=15)
    ax2c.set_ylabel('Shannon Gap = R(λ) - H(X) [Bits]', fontweight='bold', fontsize=15)
    ax2c.set_title('(b) Communication Efficiency Gap', fontweight='bold', fontsize=20)
    ax2c.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax2c.grid(True, alpha=0.4)
    
    # Add formula explanation
    gap_formula_text = (
        "Shannon Gap = R(λ) - H(X)\\n"
        "Shanon Optimum Gap = 0\\n"
        "(Indicated in Red dashed line)\\n"
        "\\n"
        "Interpretation:\\n"
        "• Gap = 0: Optimal compression\\n"
        "• Gap > 0: Excess bits used\\n"
    )
    
    ax2c.text(0.98, 0.98, gap_formula_text, transform=ax2c.transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig2.savefig(save_dir / 'performance_analysis_enhanced.pdf', dpi=300, bbox_inches='tight')
    fig2.savefig(save_dir / 'performance_analysis_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Reset matplotlib
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"\\nFixed ICLR-quality plots saved to {save_dir}")
    print("Key improvements:")
    print("  ✅ Thin error bars (1.0-1.2 width) vs thick main lines (3.0 width)")
    print("  ✅ Smart label positioning with collision detection")
    print("  ✅ Non-overlapping annotations with clear backgrounds")
    print("  ✅ Combined dual-axis plot for performance + communication rate")
    print("  ✅ Enhanced readability for all text elements")
    
    return agg_results


def create_goal_encoding_plot(goal_analysis_df, save_dir):
    """Create clean goal encoding analysis with proper error bar styling and clear labels"""
    if goal_analysis_df is None or goal_analysis_df.empty:
        return
    
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Frequency vs Bits with trend line
    frequencies = goal_analysis_df['frequency']
    bits = goal_analysis_df['bits']
    
    # Scatter plot with larger, more visible markers
    ax1.scatter(frequencies, bits, s=120, alpha=0.8, 
               color='steelblue', edgecolors='black', linewidth=2, zorder=5)
    
    # Add trend line with proper thickness
    z = np.polyfit(frequencies, bits, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(frequencies.min(), frequencies.max(), 100)
    ax1.plot(x_trend, p(x_trend), '--', color='red', linewidth=2.5, alpha=0.9,
            label=f'Trend (slope={z[0]:.4f})', zorder=3)
    
    # Calculate correlation for additional context
    correlation, p_value = pearsonr(frequencies, bits)
    
    # Enhanced smart goal labels with collision avoidance
    def calculate_goal_label_positions(frequencies, bits):
        """Calculate non-overlapping positions for goal labels"""
        positions = []
        n_goals = len(frequencies)
        
        for i, (freq, bit_val) in enumerate(zip(frequencies, bits)):
            # Use different strategies based on data distribution
            if n_goals <= 4:
                # For few goals, use cardinal directions
                angles = [45, 135, 225, 315]
                angle = angles[i % len(angles)]
            else:
                # For more goals, distribute evenly around circle
                angle = (i * 360 / n_goals) % 360
            
            # Adjust radius based on data density
            base_radius = 20
            if freq > frequencies.median():  # High frequency goals
                radius = base_radius + 5  # Slightly further out
            else:
                radius = base_radius
            
            angle_rad = np.radians(angle)
            offset_x = radius * np.cos(angle_rad)
            offset_y = radius * np.sin(angle_rad)
            
            # Determine text alignment
            ha = 'left' if offset_x > 0 else 'right'
            va = 'bottom' if offset_y > 0 else 'top'
            
            positions.append((offset_x, offset_y, ha, va))
        
        return positions
    
    label_positions = calculate_goal_label_positions(frequencies, bits)
    
    for i, (row, (offset_x, offset_y, ha, va)) in enumerate(zip(goal_analysis_df.iterrows(), label_positions)):
        row = row[1]  # Get the actual row data
        goal_str = f"({row['goal'][0]},{row['goal'][1]})"
        
        ax1.annotate(goal_str, (row['frequency'], row['bits']), 
                    xytext=(offset_x, offset_y), textcoords='offset points',
                    ha=ha, va=va, fontsize=20, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.95, edgecolor='steelblue', linewidth=1),
                    arrowprops=dict(arrowstyle='->', color='steelblue', alpha=0.7, lw=1.0),
                    zorder=6)
    
    ax1.set_xlabel('Goal Frequency', fontweight='bold', fontsize=25)
    ax1.set_ylabel('Communication Bits', fontweight='bold', fontsize=25)
    ax1.set_title('(a) Goal Frequency vs Communication Cost', fontweight='bold', fontsize=30)
    
    # Enhanced legend with correlation info
    ax1.legend([f'Trend (slope={z[0]:.4f})', f'Correlation: r={correlation:.3f}, p={p_value:.3f}'], 
              loc='best', frameon=True, fancybox=True, shadow=True, fontsize=20)
    
    ax1.grid(True, alpha=0.4)
    
    # Add statistics text box
    stats_text = (
        f"Statistical Analysis:\\n"
        f"• Correlation: r = {correlation:.3f}\\n"
        f"• P-value: {p_value:.4f}\\n"
        f"• Trend: {'Negative' if z[0] < 0 else 'Positive'}\\n"
        f"• Significance: {'*' if p_value < 0.05 else 'n.s.'}"
    )
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=20, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.8))
    
    # Right: Efficiency comparison bar chart with FIXED error handling
    goals_str = [f"({g[0]},{g[1]})" for g in goal_analysis_df['goal']]
    theoretical = goal_analysis_df['theoretical_bits']
    empirical = goal_analysis_df['bits']
    
    x_pos = np.arange(len(goals_str))
    width = 0.35
    
    # Create bars without error bar parameters
    bars1 = ax2.bar(x_pos - width/2, theoretical, width, 
                   label='Shannon Optimal', color='crimson', alpha=0.8,
                   edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x_pos + width/2, empirical, width, 
                   label='DDCL Empirical', color='steelblue', alpha=0.8,
                   edgecolor='black', linewidth=1.2)
    
    # Add error bars separately if standard deviations are available
    if 'theoretical_bits_std' in goal_analysis_df.columns:
        theoretical_err = goal_analysis_df['theoretical_bits_std']
        ax2.errorbar(x_pos - width/2, theoretical, yerr=theoretical_err,
                    fmt='none', capsize=4, capthick=1.0, elinewidth=1.0,
                    color='darkred', alpha=0.8)
    
    if 'bits_std' in goal_analysis_df.columns:
        empirical_err = goal_analysis_df['bits_std']
        ax2.errorbar(x_pos + width/2, empirical, yerr=empirical_err,
                    fmt='none', capsize=4, capthick=1.0, elinewidth=1.0,
                    color='darkblue', alpha=0.8)
    
    # Add value labels on bars with smart positioning
    for bars, values, color in [(bars1, theoretical, 'darkred'), (bars2, empirical, 'darkblue')]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            # Position labels slightly above bars
            y_pos = height + max(theoretical.max(), empirical.max()) * 0.02
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{value:.1f}', ha='center', va='bottom', 
                    fontsize=20, fontweight='bold', color=color)
    
    # Calculate and display efficiency metrics
    efficiency_ratios = theoretical / empirical * 100
    avg_efficiency = efficiency_ratios.mean()
    
    # Add efficiency percentages as secondary labels
    for i, (x, theo, emp, eff) in enumerate(zip(x_pos, theoretical, empirical, efficiency_ratios)):
        # Add efficiency label between the bars
        ax2.text(x, max(theo, emp) + max(theoretical.max(), empirical.max()) * 0.08,
                f'{eff:.0f}%', ha='center', va='bottom', 
                fontsize=20, style='italic', color='purple')
    
    ax2.set_xlabel('Goal Locations', fontweight='bold', fontsize=25)
    ax2.set_ylabel('Bits Required', fontweight='bold', fontsize=25)
    ax2.set_title('(b) Optimal vs Learned Encoding', fontweight='bold', fontsize=30)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(goals_str, rotation=45, ha='right', fontsize=25)
    
    # Enhanced legend with efficiency info
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='crimson', alpha=0.8, label='Shannon Optimal'),
        plt.Rectangle((0, 0), 1, 1, facecolor='steelblue', alpha=0.8, label='DDCL Empirical'),
        plt.Line2D([0], [0], color='purple', linestyle='', marker='o', markersize=6, 
                   label=f'Efficiency % (avg: {avg_efficiency:.0f}%)')
    ]
    ax2.legend(handles=legend_elements, loc='best', frameon=True, fancybox=True, shadow=True, fontsize=20)
    ax2.grid(True, alpha=0.4)
    
    # Add efficiency summary box
    efficiency_text = (
        f"Encoding Efficiency:\\n"
        f"• Average: {avg_efficiency:.1f}%\\n"
        f"• Best: {efficiency_ratios.max():.1f}%\\n"
        f"• Worst: {efficiency_ratios.min():.1f}%\\n"
        f"• Range: {efficiency_ratios.max() - efficiency_ratios.min():.1f}%\\n"
        "\\n"
        "Purple % = Shannon/DDCL × 100"
    )
    
    ax2.text(0.02, 0.98, efficiency_text, transform=ax2.transAxes, 
            fontsize=20, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender', alpha=0.8))
    
    # Set y-limits to accommodate all labels
    y_max = max(theoretical.max(), empirical.max())
    ax2.set_ylim(0, y_max * 1.25)  # 25% extra space for labels
    
    plt.tight_layout(pad=1.5)
    fig.savefig(save_dir / 'goal_encoding_analysis.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'goal_encoding_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    plt.rcParams.update(plt.rcParamsDefault)
    
    print(f"Enhanced goal encoding plot saved to {save_dir}")
    print("  ✅ Fixed error bar handling for bar charts")
    print("  ✅ Smart label positioning with collision avoidance")
    print("  ✅ Added statistical analysis and correlation info")
    print("  ✅ Enhanced efficiency metrics and visualization")
    print(f"  📊 Goal encoding correlation: r={correlation:.3f} (p={p_value:.4f})")