import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For better aesthetics

plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style

def plot_histograms_styled(image1, image2, title1="Image 1", title2="Image 2"):
    """Generates side-by-side histograms for two images with improved styling."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram for Image 1
    axes[0].hist(image1.ravel(), bins=256, range=[0,256], color='dodgerblue', alpha=0.7, density=True)
    axes[0].set_title(title1, fontsize=14)
    axes[0].set_xlabel("Pixel Intensity", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Histogram for Image 2
    axes[1].hist(image2.ravel(), bins=256, range=[0,256], color='tomato', alpha=0.7, density=True)
    axes[1].set_title(title2, fontsize=14)
    axes[1].set_xlabel("Pixel Intensity", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle("Image Histograms", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    return fig

def plot_difference_styled(image1, image2, cmap='viridis'):
    """Creates a heatmap of the absolute difference between two images with improved styling."""
    diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(diff, cmap=cmap)
    ax.set_title("Absolute Difference Map", fontsize=16, fontweight='bold')
    ax.set_xlabel("Width", fontsize=12)
    ax.set_ylabel("Height", fontsize=12)
    
    # Add a colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Magnitude', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_comparative_bar(data_dict, title="Comparative Bar Chart", ylabel="Value"):
    """
    Generates a comparative bar chart from a dictionary of data.
    Args:
        data_dict (dict): Dictionary where keys are labels and values are numerical.
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    methods = list(data_dict.keys())
    values = list(data_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(methods)) # Get a nice color palette
    
    bars = ax.bar(methods, values, color=colors, width=0.6)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10) # Rotate labels for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values), f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
        
    plt.ylim(0, max(values) * 1.15) # Adjust y-limit for space for labels
    plt.tight_layout()
    return fig

# Example Usage (can be run standalone for testing)
if __name__ == '__main__':
    # Create dummy images
    img_a = np.random.randint(0, 150, size=(100, 100), dtype=np.uint8)
    img_b = img_a.copy()
    # Introduce some changes in img_b
    img_b[20:50, 20:50] = np.random.randint(100, 255, size=(30,30), dtype=np.uint8)
    img_b[0:10, 0:10] = 0


    # Test plot_histograms_styled
    fig_hist = plot_histograms_styled(img_a, img_b, "Original Dummy Image", "Modified Dummy Image")
    plt.show()

    # Test plot_difference_styled
    fig_diff = plot_difference_styled(img_a, img_b)
    plt.show()

    # Test plot_comparative_bar
    comp_data = {
        "Method A": 75.5,
        "Method B": 88.2,
        "Method C": 60.0,
        "Method D": 92.1
    }
    fig_bar = plot_comparative_bar(comp_data, title="PSNR Comparison", ylabel="PSNR (dB)")
    plt.show()
