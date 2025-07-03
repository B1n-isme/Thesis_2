import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec

def create_grid_plot():
    horizons = [7, 14, 30, 60]
    image_paths = []

    # Add ETS images
    for h in horizons:
        image_paths.append(f'results/results_{h}d/cv/plot/ETS_cv_{h}d.png')

    # # Add TFT 90d image
    # image_paths.append('results/results_90d/cv/plot/TFT_cv_90d.png')

    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig) # 3 rows, 2 columns

    # Define the axes based on the GridSpec
    axes = [
        fig.add_subplot(gs[0, 0]), # Row 0, Col 0
        fig.add_subplot(gs[0, 1]), # Row 0, Col 1
        fig.add_subplot(gs[1, 0]), # Row 1, Col 0
        fig.add_subplot(gs[1, 1]), # Row 1, Col 1
        # fig.add_subplot(gs[2, :])  # Row 2, spans both columns (centered)
    ]

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            axes[i].imshow(img)
            # Extract information from the filename to create a more readable title
            filename = img_path.split('/')[-1] # e.g., ETS_cv_7d.png
            parts = filename.replace('.png', '').split('_') # e.g., ['ETS', 'cv', '7d']
            model = parts[0].upper()
            horizon = parts[-1].replace('d', '') # Remove 'd' for cleaner display
            if model == 'TFT':
                title = f'{model} on {horizon}-days horizon'
            else:
                title = f'{model} on {horizon}-days horizon'
            axes[i].set_title(title) # Set title to filename
            axes[i].axis('off') # Hide axes
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            axes[i].set_title(f"Image not found: {img_path.split('/')[-1]}")
            axes[i].axis('off')
        except Exception as e:
            print(f"An error occurred while loading {img_path}: {e}")
            axes[i].set_title(f"Error loading: {img_path.split('/')[-1]}")
            axes[i].axis('off')

    # Adjust layout for better spacing
    # plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    plt.show()
    fig.savefig('results/insights/plot_cv_grid.png')

if __name__ == '__main__':
    create_grid_plot()
