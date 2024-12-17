import os
import numpy as np
import torch
import matplotlib.pyplot as plt  # For plotting and saving images
import data_analysis.constants as const
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
from tqdm import tqdm

# Import the DeepGaze IIE model
import deepgaze_pytorch

# Reference: https://github.com/matthias-k/deepgaze
# PLEASE NOTE: REMEMBER TO ACTIVATE THE gen_attention_reasoning ENVIRONMENT BEFORE RUNNING THIS SCRIPT !!!

# Set the device (CPU or GPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pre-trained DeepGaze IIE model
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)
model.eval()

# Use a uniform center bias (log density of 1)
# centerbias_template = np.zeros((1024, 1024))
centerbias_template = np.zeros((const.SCREEN_RESOLUTION_WIDTH_PX, const.SCREEN_RESOLUTION_HEIGHT_PX))

# Directory containing your images
image_dir = '/home/baiy4/reading-model/baseline_models/ReaderAgent_scanpath/stimuli'
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# Output directory for saliency maps
output_dir = '/home/baiy4/reading-model/baseline_models/ReaderAgent_scanpath/attention_reasoning'
os.makedirs(output_dir, exist_ok=True)

# Output directory for saliency images
output_image_dir = '/home/baiy4/reading-model/baseline_models/ReaderAgent_scanpath/attention_reasoning_images'  # Replace with your desired directory
os.makedirs(output_image_dir, exist_ok=True)

# Process each image in the directory
for filename in tqdm(os.listdir(image_dir)):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(image_dir, filename)
        try:
            # Load the image
            image = Image.open(image_path).convert('RGB')
            # Resize the image to match the center bias size
            # image = image.resize((1024, 1024), Image.BILINEAR)
            image_np = np.array(image)

            # Resize center bias to match image size (should already be 1024x1024)
            centerbias = centerbias_template.copy()
            # Renormalize log density
            centerbias -= logsumexp(centerbias)

            # Prepare tensors
            image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).float().to(DEVICE)  # Shape: [1, 3, H, W]
            centerbias_tensor = torch.tensor([centerbias]).float().to(DEVICE)  # Shape: [1, H, W]

            # Generate the log density prediction
            with torch.no_grad():
                log_density_prediction = model(image_tensor, centerbias_tensor)

            # Convert log density to probability density
            density_prediction = torch.exp(log_density_prediction)
            density_prediction = density_prediction.squeeze().cpu().numpy()

            # Normalize the saliency map to [0, 1]
            density_prediction -= density_prediction.min()
            density_prediction /= density_prediction.max()

            # Save the saliency map as a .npy file
            idx = int(os.path.splitext(filename)[0])
            output_filename = f"qid_{idx:03d}.npy"
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, density_prediction)

            # Save the saliency map as a PNG image
            output_image_filename = f"qid_{idx:03d}.png"
            output_image_path = os.path.join(output_image_dir, output_image_filename)

            # TODO: Save the saliency map as an image
            print(f"the density_prediction shape is {density_prediction.shape}")

            # Option 1: Save the saliency map alone
            plt.figure(figsize=(5, 5), dpi=200)
            plt.axis('off')
            plt.imshow(density_prediction, cmap='hot')
            plt.tight_layout(pad=0)
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
            plt.close()


        except Exception as e:
            print(f"Error processing {filename}: {e}")
