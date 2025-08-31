import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns

# Set modern style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_modern_plot(x_data, y_data, title="Data Visualization", 
                      x_label="X Values", y_label="Y Values", 
                      plot_type="line", color_scheme="modern"):
    """
    Create a modern matplotlib visualization
    
    Parameters:
    - x_data, y_data: Data arrays
    - title: Plot title
    - x_label, y_label: Axis labels
    - plot_type: 'line', 'scatter', 'bar', 'area'
    - color_scheme: 'modern', 'gradient', 'minimal'
    """
    
    # Create figure with modern proportions
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # Color schemes
    colors = {
        'modern': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
        'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c'],
        'minimal': ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6']
    }
    
    primary_color = colors[color_scheme][0]
    
    # Plot based on type
    if plot_type == "line":
        ax.plot(x_data, y_data, linewidth=3, color=primary_color, 
                marker='o', markersize=6, markerfacecolor='white', 
                markeredgecolor=primary_color, markeredgewidth=2,
                alpha=0.8)
    
    elif plot_type == "scatter":
        scatter = ax.scatter(x_data, y_data, s=80, c=y_data, 
                           cmap='viridis', alpha=0.7, edgecolors='white', 
                           linewidth=1)
        plt.colorbar(scatter, ax=ax, shrink=0.8)
    
    elif plot_type == "bar":
        bars = ax.bar(x_data, y_data, color=primary_color, 
                     alpha=0.8, edgecolor='white', linewidth=1)
        # Add gradient effect
        for bar in bars:
            bar.set_alpha(0.7)
    
    elif plot_type == "area":
        ax.fill_between(x_data, y_data, alpha=0.3, color=primary_color)
        ax.plot(x_data, y_data, linewidth=2, color=primary_color)
    
    # Modern styling
    ax.set_title(title, fontsize=20, fontweight='bold', pad=30, 
                color='#2c3e50')
    ax.set_xlabel(x_label, fontsize=14, fontweight='600', color='#34495e')
    ax.set_ylabel(y_label, fontsize=14, fontweight='600', color='#34495e')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=11, 
                  colors='#34495e', length=0)
    
    # Background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    # Tight layout with padding
    plt.tight_layout(pad=2.0)
    
    return fig, ax

# Example usage with sample data
def demo_plots():
    """Demonstrate different plot types with sample data"""
    
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = np.sin(x) * np.exp(-x/5) + np.random.normal(0, 0.1, 50)
    
    # Create subplots for different styles
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Modern Matplotlib Templates', fontsize=24, fontweight='bold', y=0.95)
    
    # Line plot
    axes[0,0].plot(x, y, linewidth=3, color='#2E86AB', marker='o', 
                   markersize=4, markerfacecolor='white', 
                   markeredgecolor='#2E86AB', alpha=0.8)
    axes[0,0].set_title('Modern Line Plot', fontsize=16, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_facecolor('#fafafa')
    
    # Scatter plot
    scatter = axes[0,1].scatter(x, y, s=60, c=y, cmap='viridis', 
                               alpha=0.7, edgecolors='white', linewidth=1)
    axes[0,1].set_title('Modern Scatter Plot', fontsize=16, fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_facecolor('#fafafa')
    
    # Bar plot
    x_bar = np.arange(1, 11)
    y_bar = np.random.randint(10, 100, 10)
    bars = axes[1,0].bar(x_bar, y_bar, color='#A23B72', alpha=0.8, 
                        edgecolor='white', linewidth=1)
    axes[1,0].set_title('Modern Bar Plot', fontsize=16, fontweight='bold')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].set_facecolor('#fafafa')
    
    # Area plot
    axes[1,1].fill_between(x, y, alpha=0.3, color='#F18F01')
    axes[1,1].plot(x, y, linewidth=2, color='#F18F01')
    axes[1,1].set_title('Modern Area Plot', fontsize=16, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_facecolor('#fafafa')
    
    # Style all subplots
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', labelsize=10, colors='#34495e', length=0)
    
    plt.tight_layout()
    return fig

# Time series specific template
def create_time_series_plot(dates, values, title="Time Series Data"):
    """Create a modern time series plot"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot with modern styling
    ax.plot(dates, values, linewidth=3, color='#2E86AB', 
            marker='o', markersize=5, markerfacecolor='white',
            markeredgecolor='#2E86AB', markeredgewidth=2, alpha=0.8)
    
    # Fill area under curve
    ax.fill_between(dates, values, alpha=0.2, color='#2E86AB')
    
    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Modern styling
    ax.set_title(title, fontsize=20, fontweight='bold', pad=30)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#fafafa')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(axis='both', labelsize=11, colors='#34495e', length=0)
    
    plt.tight_layout()
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Demo with sample data
    x_sample = np.linspace(0, 10, 30)
    y_sample = 2 * x_sample + np.random.normal(0, 2, 30)
    
    # Create modern plot
    fig, ax = create_modern_plot(x_sample, y_sample, 
                               title="Modern Data Visualization",
                               x_label="X Values", 
                               y_label="Y Values",
                               plot_type="scatter",
                               color_scheme="modern")
    
    # Show demo plots
    demo_fig = demo_plots()
    
    # Time series example
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    values = np.cumsum(np.random.randn(365)) + 100
    
    ts_fig, ts_ax = create_time_series_plot(dates, values, 
                                          "Time Series Visualization")
    
    plt.show()