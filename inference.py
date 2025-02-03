import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from CCA import connected_component_analysis
from unet import UNet
from topo import multi_class_topological_post_processing

CLASS_LABELS = {0: 'Background', 1: 'RV', 2: 'MY', 3: 'LV'}

# **Define Priors**
SINGLE_CLASS_PRIOR = {
    (1,): (1, 0),
    (2,): (1, 1),
    (3,): (1, 0),
}

MULTI_CLASS_PRIOR = {
    (1,): (1, 0),
    (2,): (1, 1),
    (3,): (1, 0),
    (1, 2): (1, 1),
    (1, 3): (2, 0),
    (2, 3): (1, 0)
}

# **Argument Parser**
def parse_args():
    parser = argparse.ArgumentParser(description="Inference with different post-processing methods")
    parser.add_argument('--description', type=str, required=True, help="Description of the test run")
    parser.add_argument('--image_path', type=str, required=True, help="Path to a single test image")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model")
    parser.add_argument('--output_path', type=str, default="output.png", help="Path to save the output visualization")
    return parser.parse_args()

# **Load and Prepare Model**
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = UNet(in_channels=1, n_classes=4, depth=5, wf=48, padding=True, batch_norm=True, up_mode='upsample').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# **Perform Inference**
def run_inference(model, image, device):
    image = image.to(device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    print('model output shape:', outputs.shape)
    pred_probs = torch.softmax(outputs, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()
    return pred_labels

# **Apply Post-Processing Methods**
def apply_cca(segmentation_map):
    processed_map = np.zeros_like(segmentation_map)
    for class_idx in range(1, 4):
        class_mask = (segmentation_map == class_idx)
        if class_mask.any():
            processed_mask = connected_component_analysis(class_mask)
            processed_map[processed_mask == 1] = class_idx
    return processed_map

def apply_topo(image, model, device, prior):
    input_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
    model_topo = multi_class_topological_post_processing(
        input_tensor, model, prior, lr=1e-3, num_its=100, mse_lambda=1000, parallel=False
    )
    refined_output = model_topo(input_tensor)
    pred_probs = torch.softmax(refined_output, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1).squeeze(0).detach().cpu().numpy()
    print(f'pred labels shape: {pred_labels.shape}')
    #printing unique values in pred_labels
    print(f'unique values in pred_labels: {np.unique(pred_labels)}')
    return pred_labels

# **Plot the Results**
def plot_results(image, segmentations, output_path):
    titles = ["True label","1 No Post-Processing", "2️ CCA Applied", "3️ Topo (Single Class)", "4️ Topo (Multi-Class)"]
    
    fig, axes = plt.subplots(1, len(segmentations) , figsize=(20, 5))
    
    for a in axes:
        a.imshow(image.squeeze(), cmap='gray')
        a.set_xticks([])
        a.set_yticks([])
        plt.setp(a.spines.values(), color=None)
    # Segmentation Maps
    for i in range(len(segmentations)):
        axes[i].imshow(segmentations[i], alpha =0.5)
        axes[i].set_title(titles[i])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# **Main Function**
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    model = load_model(args.model_path, device)

    # Load Image
    # Ensure consistent size through center cropping
    def center_crop(img, target_size):
        h, w = img.shape
        th, tw = target_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img[i:i+th, j:j+tw]

    # Center crop both image and label
    image = torch.load(args.image_path)
    
    image = center_crop(image, (224, 224))
    image = image.unsqueeze(0).to(device)
    label = torch.load(args.image_path.replace("image", "label"))
    label = center_crop(label, (224, 224))
    label = label.numpy()
    # Perform Inference in 4 Ways
    pred_no_processing = run_inference(model, image, device)
    pred_with_cca = apply_cca(pred_no_processing)
    pred_topo_single = apply_topo(image, model, device, SINGLE_CLASS_PRIOR)
    pred_topo_multi = apply_topo(image, model, device, MULTI_CLASS_PRIOR)
    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, f'predictions_{args.description}.png')
    # Plot and Save Results
    plot_results(image.squeeze(0).cpu().numpy(), 
                 [label, pred_no_processing, pred_with_cca, pred_topo_single, pred_topo_multi],
                 out_path)

if __name__ == "__main__":
    main()
