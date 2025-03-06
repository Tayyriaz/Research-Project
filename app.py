from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, normalize, resize
import os
from werkzeug.utils import secure_filename
import yaml
from argparse import ArgumentParser, Namespace
import time
from utils import get_model
from bilateral_solver import bilateral_solver_output

# Initialize Flask App
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload and results directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
try:
    state_dict = torch.hub.load_state_dict_from_url(
        "https://www.robots.ox.ac.uk/~vgg/research/selfmask/shared_files/selfmask_nq20.pt",
        map_location=device
    )

    # Parse Arguments
    parser = ArgumentParser("SelfMask demo")
    parser.add_argument("--config", type=str, default="duts-dino-k234-nq20-224-swav-mocov2-dino-p16-sr10100.yaml")
    args = parser.parse_args()

    base_args = yaml.safe_load(open(args.config, 'r'))
    base_args.pop("dataset_name")
    args = vars(args)
    args.update(base_args)
    args = Namespace(**args)

    # Load Model
    model = get_model(arch="maskformer", configs=args).to(device)
    model.load_state_dict(state_dict)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Image Processing Parameters
size = 384
max_size = 512
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@torch.no_grad()
def process_image(image_path):
    try:
        # Open and convert image
        image = Image.open(image_path).convert("RGB")
        pil_image = resize(image, size=size, max_size=max_size)
        image_tensor = normalize(to_tensor(pil_image), mean=list(mean), std=list(std))

        # Run Model
        dict_outputs = model(image_tensor[None].to(device))
        batch_pred_masks = dict_outputs["mask_pred"]

        if len(batch_pred_masks.shape) == 5:
            batch_pred_masks = batch_pred_masks[:, -1, ...]

        H, W = image_tensor.shape[-2:]
        batch_pred_masks = F.interpolate(batch_pred_masks, scale_factor=4, mode="bilinear", align_corners=False)[..., :H, :W]
        pred_mask = batch_pred_masks[0, 0]

        pred_mask = (pred_mask > 0.5).cpu().numpy().astype(np.uint8) * 255
        pred_mask_bi, _ = bilateral_solver_output(img=pil_image, target=pred_mask)
        pred_mask_bi = np.clip(pred_mask_bi, 0, 255).astype(np.uint8)

        attn_map = cv2.applyColorMap(pred_mask_bi, cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.addWeighted(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), 0.5, attn_map, 0.5, 0)

        return pred_mask_bi, heatmap
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF)'}), 400

    try:
        # Save original image
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        base_filename = f"{timestamp}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], base_filename)
        file.save(file_path)

        # Process image
        mask, heatmap = process_image(file_path)
        
        if mask is None or heatmap is None:
            return jsonify({'error': 'Error processing image'}), 500

        # Save results
        mask_filename = f"mask_{base_filename}"
        heatmap_filename = f"heatmap_{base_filename}"
        
        mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
        heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], heatmap_filename)
        
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(heatmap_path, heatmap)

        # Return results
        return jsonify({
            'success': True,
            'image_path': f"/static/uploads/{base_filename}",
            'mask_path': f"/static/results/{mask_filename}",
            'heatmap_path': f"/static/results/{heatmap_filename}",
            'is_cached': False
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/logout')
def logout():
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    if model is None:
        print("Warning: Model failed to load. Application may not work correctly.")
    app.run(debug=True, port=8010)

