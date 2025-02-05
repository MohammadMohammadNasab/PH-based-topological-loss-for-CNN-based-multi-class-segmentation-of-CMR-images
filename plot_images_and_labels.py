import os
import numpy as np
import matplotlib.pyplot as plt

def plot_image_and_label(image, label, image_path, output_dir):
    plt.figure(figsize=(12, 6))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image: {image_path}")
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    # Plot segmentation label
    plt.subplot(1, 2, 2)
    plt.title(f"Segmentation Mask")
    plt.imshow(label, cmap='viridis')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    
    # Generate output filename from full path
    rel_path = os.path.relpath(image_path)
    path_without_ext = os.path.splitext(rel_path)[0]
    safe_path = path_without_ext.replace('/', '_').replace('\\', '_')
    output_path = os.path.join(output_dir, f"{safe_path}_plot.png")
    
    # Save and close figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_images_and_labels_in_folder(folder_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('_image.npy'):
                try:
                    image_path = os.path.join(root, file)
                    label_path = image_path.replace('_image.npy', '_label.npy')
                    
                    if os.path.exists(label_path):
                        image = np.load(image_path)
                        label = np.load(label_path)
                        plot_image_and_label(image, label, image_path, output_dir)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    test_folder = "data/preprocessed/test"
    output_dir = "test_plots"
    plot_images_and_labels_in_folder(test_folder, output_dir)
