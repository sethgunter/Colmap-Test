from flask import Flask, request, send_file, Response, session
import subprocess
import os
import shutil
import logging
import json
import zipfile
import io
import time
import uuid
import psutil
import resource
import GPUtil
import glob
import plyfile
import pycolmap
import numpy as np
import cv2
import torch
import sqlite3
from pathlib import Path

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fixed-secret-key-for-testing')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.after_request
def add_security_headers(response: Response):
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
        "connect-src 'self' file:; "
        "worker-src 'self' blob:;"
    )
    return response

@app.route('/')
def index():
    logger.debug("Serving index.html")
    return app.send_static_file('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    logger.debug(f"Serving static file: {path}")
    file_path = os.path.join(app.static_folder, path)
    if not os.path.exists(file_path):
        logger.error(f"Static file not found: {file_path}")
        return {"status": "error", "message": f"Static file not found: {path}"}, 404
    return app.send_static_file(path)

@app.route('/output/<path:path>')
def serve_output(path):
    logger.debug(f"Serving output file: {path}")
    try:
        file_path = os.path.join('/app/colmap_project', path)
        if not os.path.exists(file_path):
            logger.error(f"Output file not found: {file_path}")
            return {"status": "error", "message": f"File not found: {path}"}, 404
        return send_file(file_path)
    except Exception as e:
        logger.error(f"Error serving output file {path}: {str(e)}")
        return {"status": "error", "message": f"Error serving file: {str(e)}"}, 500

def terminate_child_processes():
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                logger.debug(f"Terminating child process {child.pid} ({child.name()})")
                child.terminate()
                child.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired) as e:
                logger.warning(f"Failed to terminate child process {child.pid}: {e}")
    except Exception as e:
        logger.error(f"Error while terminating child processes: {e}")

def debug_file_locks(directory):
    try:
        result = subprocess.run(['lsof', directory], capture_output=True, text=True)
        logger.debug(f"lsof output for {directory}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to run lsof on {directory}: {e.stderr}")
    except FileNotFoundError:
        logger.warning("lsof not installed, cannot debug file locks")

def cleanup_old_requests(current_request_id):
    project_dir = '/app/colmap_project'
    try:
        if not os.path.exists(project_dir):
            logger.debug(f"No project directory found at {project_dir}")
            return True
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and item != current_request_id:
                for attempt in range(5):
                    try:
                        terminate_child_processes()
                        debug_file_locks(item_path)
                        shutil.rmtree(item_path)
                        logger.debug(f"Successfully removed old directory: {item_path}")
                        break
                    except OSError as e:
                        logger.warning(f"Attempt {attempt+1} to remove {item_path} failed: {e}")
                        time.sleep(3)
                else:
                    logger.error(f"Failed to remove old directory {item_path} after retries")
                    return False
        return True
    except Exception as e:
        logger.error(f"Cleanup of old requests failed: {e}")
        return False

def check_resources(current_request_id):
    if not cleanup_old_requests(current_request_id):
        logger.error("Failed to clean up old request directories")
        return False, "Failed to clean up old request directories"
    try:
        disk = shutil.disk_usage('/app')
        free_gb = disk.free / (1024**3)
        logger.debug(f"Disk space free: {free_gb:.2f} GB")
        if free_gb < 5:
            logger.error(f"Low disk space: {free_gb:.2f} GB available")
            return False, f"Low disk space: {free_gb:.2f} GB available"
        gpus = GPUtil.getGPUs()
        if not gpus:
            logger.error("No GPU available")
            return False, "No GPU available"
        gpu = gpus[0]
        free_memory_mb = gpu.memoryFree
        logger.debug(f"GPU memory free: {free_memory_mb} MB")
        if free_memory_mb < 10000:
            logger.error(f"Insufficient GPU memory: {free_memory_mb} MB available")
            return False, f"Insufficient GPU memory: {free_memory_mb} MB available"
        available_ram = psutil.virtual_memory().available / (1024 ** 2)
        logger.debug(f"Available RAM: {available_ram} MB")
        if available_ram < 20000:
            logger.error(f"Insufficient RAM: {available_ram} MB available")
            return False, f"Insufficient RAM: {available_ram} MB available"
        return True, ""
    except Exception as e:
        logger.error(f"Resource check failed: {e}")
        return False, f"Resource check failed: {e}"

def check_ram_for_fusion():
    available_ram = psutil.virtual_memory().available / (1024 ** 2)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        free_memory_mb = gpu.memoryFree
    else:
        logger.warning("No GPU detected before stereo fusion")
    if available_ram < 8000:
        logger.error(f"Insufficient RAM for stereo fusion: {available_ram} MB available")
        return False, f"Insufficient RAM for fusion: {available_ram} MB available"
    return True, available_ram

def merge_ply_files(ply_files, output_path):
    all_vertices = []
    all_colors = []
    for ply_path in ply_files:
        ply_data = plyfile.PlyData.read(ply_path)
        vertices = ply_data['vertex']
        coords = np.array([(v['x'], v['y'], v['z']) for v in vertices], dtype=np.float32)
        colors = np.array([(v['red'], v['green'], v['blue']) for v in vertices], dtype=np.uint8)
        all_vertices.append(coords)
        all_colors.append(colors)
    merged_vertices = np.concatenate(all_vertices)
    merged_colors = np.concatenate(all_colors)
    vertex_data = np.array(
        [(v[0], v[1], v[2], c[0], c[1], c[2]) for v, c in zip(merged_vertices, merged_colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    vertex_element = plyfile.PlyElement.describe(vertex_data, 'vertex')
    plyfile.PlyData([vertex_element]).write(output_path)
    logger.debug(f"Merged {len(ply_files)} PLY files into {output_path}")

def export_sparse_ply_and_poses(sparse_model_dir, output_sparse_ply, poses_dir, poses_json_path):
    try:
        logger.debug("Exporting sparse point cloud")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'model_converter',
            '--input_path', sparse_model_dir,
            '--output_path', output_sparse_ply,
            '--output_type', 'PLY'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Sparse point cloud export failed: {stderr}")
            return False, f"Sparse point cloud export failed: {stderr}"
        sparse_size = os.path.getsize(output_sparse_ply) / (1024 ** 2)
        logger.debug(f"Sparse point cloud size: {sparse_size:.2f} MB")
    except subprocess.TimeoutExpired:
        logger.error("Sparse point cloud export timed out")
        return False, "Sparse point cloud export timed out"
    try:
        logger.debug("Exporting camera poses")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'model_converter',
            '--input_path', sparse_model_dir,
            '--output_path', poses_dir,
            '--output_type', 'TXT'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Model converter failed: {stderr}")
            return False, f"Model converter failed: {stderr}"
    except subprocess.TimeoutExpired:
        logger.error("Model conversion timed out")
        return False, "Model conversion timed out"
    try:
        with open(os.path.join(poses_dir, 'images.txt')) as f:
            lines = f.readlines()[4::2]
            poses = {}
            for line in lines:
                parts = line.strip().split()
                img_name = parts[-1]
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                poses[img_name] = {'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz, 'tx': tx, 'ty': ty, 'tz': tz}
        with open(poses_json_path, 'w') as f:
            json.dump(poses, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to parse camera poses: {str(e)}")
        return False, f"Failed to parse camera poses: {str(e)}"
    return True, ""

# Note: Only the run_superpoint_superglue function is shown.
# Replace this function in your existing app.py, keeping all other code unchanged.

def run_superpoint_superglue(images_dir, database_path, vocab_tree_path, masks_dir=None):
    """Run SuperPoint for feature detection and SuperGlue for feature matching with sequential matching."""
    try:
        from superpoint_superglue.models.superpoint import SuperPoint
        from superpoint_superglue.models.superglue import SuperGlue
        import pycolmap
        import sqlite3
    except ImportError as e:
        logger.error(f"Failed to import SuperPoint/SuperGlue or pycolmap: {str(e)}")
        return False, f"Import failed: {str(e)}"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.debug(f"Using device: {device}")

    # Initialize SuperPoint
    superpoint_config = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1000,  # Reduced to avoid CUDA memory issues
        'weight_path': '/app/superpoint_superglue/models/weights/superpoint_v1.pth'
    }
    superpoint_model = SuperPoint(superpoint_config).eval().to(device)
    logger.debug(f"SuperPoint config: {superpoint_config}")

    # Initialize SuperGlue
    superglue_config = {
        'weights_path': '/app/superpoint_superglue/models/weights/superglue_indoor.pth',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2
    }
    superglue_model = SuperGlue(superglue_config).eval().to(device)
    logger.debug(f"SuperGlue config: {superglue_config}")

    # Load images and optional masks
    image_files = sorted(glob.glob(os.path.join(images_dir, '*')))
    if not image_files:
        logger.error("No images found in images_dir")
        return False, "No images found"
    logger.debug(f"Found {len(image_files)} images: {image_files[:5]}...")

    mask_files = sorted(glob.glob(os.path.join(masks_dir, '*'))) if masks_dir and os.path.exists(masks_dir) else []
    use_masks = len(mask_files) > 0
    if use_masks and len(mask_files) != len(image_files):
        logger.warning(f"Mismatch: {len(mask_files)} masks vs {len(image_files)} images")
        use_masks = False
    logger.debug(f"Found {len(mask_files)} masks, use_masks: {use_masks}")

    # Create feature directory
    feature_dir = os.path.join(os.path.dirname(database_path), 'features')
    os.makedirs(feature_dir, exist_ok=True)
    logger.debug(f"Created feature directory: {feature_dir}")

    # Process images with SuperPoint
    keypoints_dict = {}
    descriptors_dict = {}
    image_sizes = {}
    images_data = {}
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            continue
        img_height, img_width = img.shape
        image_sizes[img_name] = (img_width, img_height)
        images_data[img_name] = img
        logger.debug(f"Loaded image {img_name} with shape: ({img_height}, {img_width})")

        # Apply mask if available
        mask = None
        if use_masks:
            mask_path = os.path.join(masks_dir, img_name.replace('.jpg', '.png'))
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask.shape != img.shape:
                    logger.warning(f"Mask size mismatch for {img_name}, ignoring mask")
                    mask = None
                else:
                    mask = (mask > 0).astype(np.uint8)
            logger.debug(f"Mask for {img_name}: {'applied' if mask is not None else 'not applied'}")

        # Prepare image for SuperPoint
        img_tensor = torch.from_numpy(img / 255.0).float()[None, None].to(device)
        logger.debug(f"Image tensor shape for {img_name}: {list(img_tensor.shape)}")
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()[None, None].to(device)
            logger.debug(f"Mask tensor shape for {img_name}: {list(mask_tensor.shape)}")
        else:
            mask_tensor = None

        # Run SuperPoint
        with torch.no_grad():
            pred = superpoint_model({'image': img_tensor})
            keypoints = pred['keypoints'][0].cpu().numpy()
            scores = pred['scores'][0].cpu().numpy()
            descriptors = pred['descriptors'][0].cpu().numpy()  # Shape: [256, num_keypoints]
            logger.debug(f"SuperPoint output for {img_name}: keypoints shape {keypoints.shape}, scores shape {scores.shape}, descriptors shape {descriptors.shape}")

        if mask is not None:
            # Filter keypoints by mask
            mask_np = mask_tensor[0, 0].cpu().numpy()
            valid = []
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < mask_np.shape[1] and 0 <= y < mask_np.shape[0] and mask_np[y, x] > 0:
                    valid.append(True)
                else:
                    valid.append(False)
            valid = np.array(valid)
            keypoints = keypoints[valid]
            scores = scores[valid]
            descriptors = descriptors[:, valid]  # Shape: [256, num_valid_keypoints]
            logger.debug(f"After mask filtering for {img_name}: keypoints shape {keypoints.shape}, scores shape {scores.shape}, descriptors shape {descriptors.shape}")

        # Store results
        keypoints_dict[img_name] = np.hstack([keypoints, scores[:, None]])
        descriptors_dict[img_name] = descriptors
        logger.debug(f"Stored {len(keypoints)} keypoints for {img_name}")

    # Initialize database using COLMAP CLI
    try:
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'database_creator',
            '--database_path', database_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode != 0:
            logger.error(f"Database creation failed: {stderr}")
            return False, f"Database creation failed: {stderr}"
        logger.debug(f"Database creation output: {stdout}")
    except subprocess.TimeoutExpired:
        logger.error("Database creation timed out")
        return False, "Database creation timed out"

    # Add data to database using SQLite and pycolmap
    try:
        # Add camera and images via SQLite
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Add camera (SPHERE model)
        camera_model = 10  # SPHERE model ID
        width = 1920
        height = 960
        params = b''  # SPHERE model has no parameters
        prior_focal_length = 1920.0  # Default focal length (image width)
        cursor.execute(
            "INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?)",
            (camera_model, width, height, params, prior_focal_length)
        )
        camera_id = cursor.lastrowid
        logger.debug(f"Added camera ID: {camera_id}")

        # Add images
        image_id_map = {}
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            cursor.execute(
                "INSERT INTO images (name, camera_id) VALUES (?, ?)",
                (img_name, camera_id)
            )
            image_id = cursor.lastrowid
            image_id_map[img_name] = image_id
            logger.debug(f"Added image {img_name} with ID {image_id}")

        conn.commit()
        conn.close()
        logger.debug("SQLite database committed")

        # Use pycolmap for keypoints, descriptors, matches
        db = pycolmap.Database(database_path)
        logger.debug(f"Opened pycolmap database: {database_path}")

        # Perform SuperGlue matching (sequential pairs)
        matches_dict = {}
        overlap = 2
        sequential_pairs = []
        for i in range(len(image_files) - 1):
            img1_name = os.path.basename(image_files[i])
            for j in range(i + 1, min(i + overlap + 1, len(image_files))):
                img2_name = os.path.basename(image_files[j])
                sequential_pairs.append((img1_name, img2_name))
        logger.debug(f"Generated {len(sequential_pairs)} sequential pairs: {sequential_pairs[:5]}...")

        for img1_name, img2_name in sequential_pairs:
            if img1_name not in keypoints_dict or img2_name not in keypoints_dict:
                logger.warning(f"Skipping pair {img1_name}, {img2_name}: missing keypoints")
                continue

            try:
                kp1 = torch.from_numpy(keypoints_dict[img1_name][:, :2]).float().to(device)[None]
                scores1 = torch.from_numpy(keypoints_dict[img1_name][:, 2]).float().to(device)[None]
                desc1 = torch.from_numpy(descriptors_dict[img1_name]).float().to(device).unsqueeze(0)  # Shape: [1, 256, num_keypoints]
                kp2 = torch.from_numpy(keypoints_dict[img2_name][:, :2]).float().to(device)[None]
                scores2 = torch.from_numpy(keypoints_dict[img2_name][:, 2]).float().to(device)[None]
                desc2 = torch.from_numpy(descriptors_dict[img2_name]).float().to(device).unsqueeze(0)  # Shape: [1, 256, num_keypoints]
                logger.debug(f"SuperGlue input for pair {img1_name}, {img2_name}:")
                logger.debug(f"  keypoints0 shape: {list(kp1.shape)}, scores0 shape: {list(scores1.shape)}, descriptors0 shape: {list(desc1.shape)}")
                logger.debug(f"  keypoints1 shape: {list(kp2.shape)}, scores1 shape: {list(scores2.shape)}, descriptors1 shape: {list(desc2.shape)}")

                # Load images for SuperGlue
                img1 = images_data[img1_name]
                img2 = images_data[img2_name]
                img1_tensor = torch.from_numpy(img1 / 255.0).float()[None, None].to(device)
                img2_tensor = torch.from_numpy(img2 / 255.0).float()[None, None].to(device)
                logger.debug(f"  image0 shape: {list(img1_tensor.shape)}, image1 shape: {list(img2_tensor.shape)}")

                data = {
                    'keypoints0': kp1,
                    'scores0': scores1,
                    'descriptors0': desc1,
                    'keypoints1': kp2,
                    'scores1': scores2,
                    'descriptors1': desc2,
                    'image0_size': torch.tensor([image_sizes[img1_name]], dtype=torch.float).to(device),
                    'image1_size': torch.tensor([image_sizes[img2_name]], dtype=torch.float).to(device),
                    'image0': img1_tensor,
                    'image1': img2_tensor
                }
                logger.debug(f"  image0_size shape: {list(data['image0_size'].shape)}, image1_size shape: {list(data['image1_size'].shape)}")

                # Run SuperGlue
                with torch.no_grad():
                    pred = superglue_model(data)
                    matches = pred['matches0'][0].cpu().numpy()
                    valid = matches > -1
                    matches0 = np.where(valid)[0]
                    matches1 = matches[valid]

                matches_dict[(img1_name, img2_name)] = np.vstack([matches0, matches1]).T
                logger.debug(f"Found {len(matches0)} matches between {img1_name} and {img2_name}")
            except Exception as e:
                logger.error(f"SuperGlue failed for pair {img1_name}, {img2_name}: {str(e)}")
                db.close()
                return False, f"SuperGlue failed for pair {img1_name}, {img2_name}: {str(e)}"

        # Add keypoints
        for img_name, keypoints in keypoints_dict.items():
            if img_name not in image_id_map:
                logger.warning(f"Image {img_name} not in database")
                continue
            image_id = image_id_map[img_name]
            num_keypoints = len(keypoints)
            if num_keypoints == 0:
                logger.warning(f"No keypoints for {img_name}")
                continue
            keypoints_data = keypoints[:, :2].astype(np.float32)
            descriptors_data = descriptors_dict[img_name].astype(np.float32)
            db.add_keypoints(image_id, keypoints_data)
            db.add_descriptors(image_id, descriptors_data)
            logger.debug(f"Added {num_keypoints} keypoints for {img_name} to database")

        # Add matches and two-view geometry
        for (img1_name, img2_name), matches in matches_dict.items():
            if img1_name not in image_id_map or img2_name not in image_id_map:
                logger.warning(f"Image pair {img1_name}, {img2_name} not in database")
                continue
            image_id1 = image_id_map[img1_name]
            image_id2 = image_id_map[img2_name]
            if len(matches) == 0:
                logger.warning(f"No matches for pair {img1_name}, {img2_name}")
                continue
            db.add_matches(image_id1, image_id2, matches)
            db.add_two_view_geometry(image_id1, image_id2, matches)
            logger.debug(f"Added matches for pair {image_id1}, {image_id2} to database")

        db.close()
        logger.debug("Successfully populated COLMAP database")
        return True, ""
    except Exception as e:
        logger.error(f"Failed to populate database: {str(e)}")
        if 'db' in locals():
            db.close()
        return False, f"Database population failed: {str(e)}"

@app.route('/process-video', methods=['POST'])
def process_video():
    logger.debug("Received POST request to /process-video")
    logger.debug(f"Request files: {list(request.files.keys())}")
    logger.debug(f"Incoming form data: {request.form}")

    session_id = request.form.get('session_id')
    if not session_id or session_id.strip() == '':
        session_id = str(uuid.uuid4())
        logger.debug(f"Generated new session_id: {session_id}")
    else:
        logger.debug(f"Using provided session_id: {session_id}")
    request_id = session.get(f'request_id_{session_id}', str(uuid.uuid4()))
    session[f'request_id_{session_id}'] = request_id
    logger.debug(f"Session state: request_id_{session_id} = {request_id}")

    base_dir = os.path.join('/app/colmap_project', request_id)
    video_dir = os.path.join(base_dir, 'video')
    images_dir = os.path.join(base_dir, 'images')
    masks_dir = os.path.join(base_dir, 'masks')
    database_path = os.path.join(base_dir, 'database.db')
    sparse_dir = os.path.join(base_dir, 'sparse')
    poses_dir = os.path.join(base_dir, 'poses')
    sparse_cubic_dir = os.path.join(base_dir, 'sparse-cubic')

    if not cleanup_old_requests(request_id):
        logger.error("Failed to clean up old request directories")
        response = {"status": "error", "message": "Failed to clean up old request directories", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    try:
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(poses_dir, exist_ok=True)
        os.makedirs(sparse_cubic_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        response = {"status": "error", "message": f"Failed to create directories: {e}", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    is_video = 'video' in request.files and request.files['video'].filename != ''
    image_files = request.files.getlist('images')
    mask_files = request.files.getlist('masks')

    if not is_video and not image_files:
        logger.error("No video or images provided in request")
        response = {"status": "error", "message": "No video or images provided", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 400
    if is_video and (image_files or mask_files):
        logger.error("Video provided with images or masks; please provide only a video or images/masks")
        response = {"status": "error", "message": "Please provide either a video or images/masks, not both", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 400
    if mask_files and not image_files:
        logger.error("Masks provided without images")
        response = {"status": "error", "message": "Masks provided without corresponding images", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 400

    input_save_time = time.time()
    session[f'input_save_time_{session_id}'] = input_save_time
    logger.debug(f"Stored input_save_time for session_id {session_id}: {input_save_time}")
    if is_video:
        video = request.files['video']
        video_path = os.path.join(video_dir, video.filename)
        logger.debug(f"Saving video: {video_path}")
        try:
            with open(video_path, 'wb') as f:
                f.write(video.read())
            logger.debug(f"Video saved: {video_path}, size: {os.path.getsize(video_path)} bytes")
            process = subprocess.Popen(['ffprobe', video_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            _, ffprobe_stderr = process.communicate(timeout=30)
            if process.returncode != 0:
                logger.error(f"Invalid video file: {ffprobe_stderr}")
                response = {"status": "error", "message": f"Invalid video file: {ffprobe_stderr}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 400
        except Exception as e:
            logger.error(f"Failed to save video: {str(e)}")
            response = {"status": "error", "message": f"Failed to save video: {str(e)}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500

        try:
            logger.debug("Extracting frames")
            process = subprocess.Popen([
                'ffmpeg', '-i', video_path, '-r', '2', '-vf', 'scale=1920:960', '-y',
                os.path.join(images_dir, 'frame_%04d.jpg')
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=300)
            if process.returncode != 0:
                logger.error(f"Frame extraction failed: {stderr}")
                response = {"status": "error", "message": f"Frame extraction failed: {stderr}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
            logger.debug(f"Frame extraction output: {stdout}")
            frame_count = len(glob.glob(os.path.join(images_dir, 'frame_*.jpg')))
            logger.debug(f"Extracted {frame_count} frames")
            if frame_count == 0:
                logger.error("No frames extracted")
                response = {"status": "error", "message": "No frames extracted from video", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
        except subprocess.TimeoutExpired:
            logger.error("Frame extraction timed out")
            terminate_child_processes()
            response = {"status": "error", "message": "Frame extraction timed out", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
    else:
        for image in image_files:
            if image.filename == '':
                continue
            ext = os.path.splitext(image.filename)[1]
            current_count = len(glob.glob(os.path.join(images_dir, '*')))
            image_path = os.path.join(images_dir, f"frame_{current_count:04d}{ext}")
            try:
                image.save(image_path)
                logger.debug(f"Saved image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to save image {image.filename}: {str(e)}")
                response = {"status": "error", "message": f"Failed to save image: {str(e)}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
        for mask in mask_files:
            if mask.filename == '':
                continue
            ext = os.path.splitext(mask.filename)[1]
            current_count = len(glob.glob(os.path.join(masks_dir, '*')))
            mask_path = os.path.join(masks_dir, f"frame_{current_count:04d}{ext}")
            try:
                mask.save(mask_path)
                logger.debug(f"Saved mask: {mask_path}")
            except Exception as e:
                logger.error(f"Failed to save mask {mask.filename}: {str(e)}")
                response = {"status": "error", "message": f"Failed to save mask: {str(e)}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
        logger.debug(f"Saved {len(image_files)} images and {len(mask_files)} masks for session {session_id}")
        if request.form.get('complete') != 'true':
            response = {
                'status': 'partial',
                'message': 'Images and masks received, send next chunk or mark complete',
                'session_id': session_id
            }
            logger.debug(f"Sending response: {response}")
            return response, 200

    resource_ok, resource_message = check_resources(request_id)
    if not resource_ok:
        logger.error(f"Resource check failed: {resource_message}")
        response = {"status": "error", "message": resource_message, "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    try:
        logger.debug("Creating database")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'database_creator',
            '--database_path', database_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode != 0:
            logger.error(f"Database creation failed: {stderr}")
            response = {"status": "error", "message": f"Database creation failed: {stderr}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
        logger.debug(f"Database creation output: {stdout}")
        image_files = glob.glob(os.path.join(images_dir, '*'))
        logger.debug(f"Images in {images_dir} ({len(image_files)}): {[os.path.basename(f) for f in image_files]}")
    except subprocess.TimeoutExpired:
        logger.error("Database creation timed out")
        response = {"status": "error", "message": "Database creation timed out", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    mask_files = glob.glob(os.path.join(masks_dir, '*'))
    use_masks = len(mask_files) > 0
    if use_masks:
        logger.debug(f"Found {len(mask_files)} masks in {masks_dir}")

    # Run SuperPoint and SuperGlue with sequential matching
    try:
        logger.debug("Running SuperPoint and SuperGlue")
        success, error_message = run_superpoint_superglue(
            images_dir,
            database_path,
            '/app/vocab_tree.bin',
            masks_dir if use_masks else None
        )
        if not success:
            logger.error(f"SuperPoint/SuperGlue processing failed: {error_message}")
            response = {"status": "error", "message": f"SuperPoint/SuperGlue processing failed: {error_message}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
    except Exception as e:
        logger.error(f"SuperPoint/SuperGlue processing failed: {str(e)}")
        response = {"status": "error", "message": f"SuperPoint/SuperGlue processing failed: {str(e)}", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    try:
        logger.debug("Running sparse reconstruction")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'mapper',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--output_path', sparse_dir,
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.sphere_camera', '1'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Sparse reconstruction failed: {stderr}")
            response = {"status": "error", "message": f"Sparse reconstruction failed: {stderr} {stdout}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
    except subprocess.TimeoutExpired:
        logger.error("Sparse reconstruction timed out")
        response = {"status": "error", "message": "Sparse reconstruction timed out", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    sparse_model_dir = os.path.join(sparse_dir, '0')
    if not os.path.exists(sparse_model_dir):
        logger.error("Sparse model not found")
        response = {"status": "error", "message": "Sparse reconstruction failed: no model generated", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    try:
        logger.debug(f"Sparse finished with : {stdout}")
        logger.debug("Running cubic reprojection")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'sphere_cubic_reprojecer',
            '--image_path', images_dir,
            '--input_path', sparse_model_dir,
            '--output_path', sparse_cubic_dir
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Cubic reprojection failed: {stderr}")
            response = {"status": "error", "message": f"Cubic reprojection failed: {stderr} {stdout}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
    except subprocess.TimeoutExpired:
        logger.error("Cubic reprojection timed out")
        response = {"status": "error", "message": "Cubic reprojection timed out", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    output_sparse_ply = os.path.join(base_dir, 'sparse.ply')
    poses_json_path = os.path.join(base_dir, 'camera_poses.json')
    success, error_message = export_sparse_ply_and_poses(sparse_model_dir, output_sparse_ply, poses_dir, poses_json_path)
    if not success:
        response = {"status": "error", "message": error_message, "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    response = {
        'status': 'sparse_success',
        'message': 'Sparse reconstruction complete, ready for dense processing',
        'sparse_ply_path': f'/output/{request_id}/sparse.ply',
        'poses_path': f'/output/{request_id}/camera_poses.json',
        'session_id': session_id,
        'input_save_time': input_save_time
    }
    logger.debug(f"Sending response: {response}")
    return response, 200

@app.route('/process-dense', methods=['POST'])
def process_dense():
    logger.debug("Received POST request to /process-dense")
    logger.debug(f"Incoming form data: {request.form}")
    session_id = request.form.get('session_id')
    logger.debug(f"Incoming session_id: {session_id}")
    if not session_id or session_id.strip() == '':
        logger.error("No valid session ID provided")
        response = {"status": "error", "message": "No valid session ID provided", "session_id": session_id or ""}
        logger.debug(f"Sending response: {response}")
        return response, 400

    request_id = session.get(f'request_id_{session_id}')
    if not request_id:
        logger.error("Invalid session ID: no associated request ID found")
        response = {"status": "error", "message": "Invalid session ID", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 400

    base_dir = os.path.join('/app/colmap_project', request_id)
    images_dir = os.path.join(base_dir, 'images')
    sparse_model_dir = os.path.join(base_dir, 'sparse', '0')
    sparse_cubic_dir = os.path.join(base_dir, 'sparse-cubic')
    dense_base_dir = os.path.join(base_dir, 'dense_chunks')
    poses_dir = os.path.join(base_dir, 'poses')

    if not all(os.path.exists(d) for d in [base_dir, images_dir, sparse_model_dir, sparse_cubic_dir]):
        logger.error("Required project directories missing")
        response = {"status": "error", "message": "Project directories missing", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    try:
        os.makedirs(dense_base_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create dense directory: {e}")
        response = {"status": "error", "message": f"Failed to create dense directory: {e}", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    resource_ok, resource_message = check_resources(request_id)
    if not resource_ok:
        logger.error(f"Resource check failed: {resource_message}")
        response = {"status": "error", "message": resource_message, "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    cubic_image_files = glob.glob(os.path.join(sparse_cubic_dir, '*.jpg'))
    chunk_size = 100
    overlap = 20
    step = chunk_size - overlap
    image_list = sorted(cubic_image_files)
    chunks = [image_list[i:i + chunk_size] for i in range(0, len(image_list), step) if image_list[i:i + chunk_size]]
    logger.debug(f"Split {len(image_list)} images into {len(chunks)} chunks")

    partial_ply_files = []
    for idx, chunk in enumerate(chunks):
        chunk_dir = os.path.join(dense_base_dir, f'chunk_{idx}')
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_image_dir = os.path.join(chunk_dir, 'images')
        os.makedirs(chunk_image_dir, exist_ok=True)
        chunk_sparse_dir = os.path.join(chunk_dir, 'sparse')
        os.makedirs(chunk_sparse_dir, exist_ok=True)

        for img_path in chunk:
            shutil.copy(img_path, chunk_image_dir)
        chunk_image_names = [os.path.basename(img) for img in chunk]
        logger.debug(f"Chunk {idx}: {len(chunk_image_names)}")

        for img_name in chunk_image_names:
            if not os.path.exists(os.path.join(chunk_image_dir, img_name)):
                logger.error(f"Image missing in chunk {idx}: {img_name}")
                response = {"status": "error", "message": f"Image missing in chunk {idx}: {img_name}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500

        try:
            shutil.copytree(os.path.join(sparse_cubic_dir, 'sparse'), chunk_sparse_dir, dirs_exist_ok=True)
            reconstruction = pycolmap.Reconstruction(chunk_sparse_dir)
            chunk_image_names_set = set(chunk_image_names)
            images_to_remove = []
            for img_id, img in reconstruction.images.items():
                img_name = os.path.basename(img.name)
                if img_name not in chunk_image_names_set:
                    if reconstruction.exists_image(img_id):
                        images_to_remove.append((img_id, img_name))

            for img_id, img_name in images_to_remove:
                try:
                    reconstruction.deregister_image(img_id)
                except Exception as e:
                    logger.warning(f"Chunk {idx} failed to deregister image {img_name} (ID: {img_id}): {str(e)}")

            reconstruction.write(chunk_sparse_dir)
            reconstruction = pycolmap.Reconstruction(chunk_sparse_dir)
            filtered_image_names = [(img_id, os.path.basename(img.name)) 
                                for img_id, img in reconstruction.images.items()]
            logger.debug(f"Chunk {idx} sparse model filtered to {len(reconstruction.images)} images")

            if len(reconstruction.images) != len(chunk_image_names):
                logger.warning(f"Chunk {idx} deregister_image failed, falling back to new reconstruction")
                new_reconstruction = pycolmap.Reconstruction()
                for cam_id, cam in reconstruction.cameras.items():
                    new_reconstruction.add_camera(cam)

                valid_image_ids = []
                for img_id, img in reconstruction.images.items():
                    img_name = os.path.basename(img.name)
                    if img_name in chunk_image_names_set and reconstruction.is_image_registered(img_id):
                        new_reconstruction.add_image(img)
                        valid_image_ids.append(img_id)

                for point3d_id, point3d in reconstruction.points3D.items():
                    track = point3d.track
                    has_valid_ref = any(elem.image_id in valid_image_ids for elem in track.elements)
                    if has_valid_ref:
                        new_reconstruction.add_point3D(point3d.xyz, point3d.track, point3d.color)

                new_reconstruction.write(chunk_sparse_dir)
                reconstruction = pycolmap.Reconstruction(chunk_sparse_dir)
                filtered_image_names = [(img_id, os.path.basename(img.name)) 
                                    for img_id, img in reconstruction.images.items()]

                if len(reconstruction.images) != len(chunk_image_names):
                    logger.error(f"Chunk {idx} sparse model has {len(reconstruction.images)} images, expected {len(chunk_image_names)}")
                    response = {"status": "error", "message": f"Chunk {idx} sparse model filtering failed: expected {len(chunk_image_names)} images, got {len(reconstruction.images)}", "session_id": session_id}
                    logger.debug(f"Sending response: {response}")
                    return response, 500
        except Exception as e:
            logger.error(f"Failed to filter sparse model for chunk {idx}: {str(e)}")
            response = {"status": "error", "message": f"Failed to filter sparse model for chunk {idx}: {str(e)}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500

        try:
            logger.debug(f"Undistorting images for chunk {idx}")
            process = subprocess.Popen([
                'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
                'colmap', 'image_undistorter',
                '--image_path', chunk_image_dir,
                '--input_path', chunk_sparse_dir,
                '--output_path', chunk_dir,
                '--output_type', 'COLMAP',
                '--max_image_size', '600'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Undistortion failed for chunk {idx}: {stderr}")
                response = {"status": "error", "message": f"Undistortion failed for chunk {idx}: {stderr}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
        except subprocess.TimeoutExpired:
            logger.error(f"Undistortion timed out for chunk {idx}")
            response = {"status": "error", "message": f"Undistortion timed out for chunk {idx}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500

        try:
            logger.debug(f"Running patch match stereo for chunk {idx}")
            process = subprocess.Popen([
                'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
                'colmap', 'patch_match_stereo',
                '--workspace_path', chunk_dir,
                '--workspace_format', 'COLMAP',
                '--PatchMatchStereo.gpu_index', '0',
                '--PatchMatchStereo.max_image_size', '400',
                '--PatchMatchStereo.window_radius', '3',
                '--PatchMatchStereo.num_samples', '3',
                '--PatchMatchStereo.num_iterations', '3',
                '--PatchMatchStereo.cache_size', '4'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Patch match failed for chunk {idx}: {stderr}")
                response = {"status": "error", "message": f"Patch match failed for chunk {idx}: {stderr}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
        except subprocess.TimeoutExpired:
            logger.error(f"Patch match timed out for chunk {idx}")
            response = {"status": "error", "message": f"Patch match timed out for chunk {idx}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500

        depth_maps_dir = os.path.join(chunk_dir, 'stereo', 'depth_maps')
        if not os.path.exists(depth_maps_dir) or not os.listdir(depth_maps_dir):
            logger.error(f"No depth maps for chunk {idx}")
            response = {"status": "error", "message": f"No depth maps for chunk {idx}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500

        ram_ok, available_ram = check_ram_for_fusion()
        if not ram_ok:
            logger.error(f"Insufficient RAM for chunk {idx}: {available_ram}")
            response = {"status": "error", "message": available_ram, "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
        gpus = GPUtil.getGPUs()
        free_memory_mb = gpus[0].memoryFree if gpus else 0
        if free_memory_mb < 3000:
            logger.error(f"Insufficient GPU memory for chunk {idx}: {free_memory_mb} MB")
            response = {"status": "error", "message": f"Insufficient GPU memory for chunk {idx}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
        disk = shutil.disk_usage('/app')
        free_gb = disk.free / (1024**3)
        if free_gb < 1:
            logger.error(f"Insufficient disk space for chunk {idx}: {free_gb:.2f} GB")
            response = {"status": "error", "message": f"Insufficient disk space for chunk {idx}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
        logger.debug(f"Resources for chunk {idx}: RAM={available_ram} MB, GPU={free_memory_mb} MB, Disk={free_gb} GB")
        cache_size = min(4, max(1, int((available_ram / 1024) * 0.5)))
        partial_ply = os.path.join(chunk_dir, f'dense_chunk_{idx}.ply')
        try:
            logger.debug(f"Running stereo fusion for chunk {idx}")
            process = subprocess.Popen([
                'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
                'colmap', 'stereo_fusion',
                '--workspace_path', chunk_dir,
                '--workspace_format', 'COLMAP',
                '--input_type', 'photometric',
                '--output_path', partial_ply,
                '--StereoFusion.min_num_pixels', '2',
                '--StereoFusion.check_num_images', '2',
                '--StereoFusion.max_reproj_error', '2',
                '--StereoFusion.max_depth_error', '0.3',
                '--StereoFusion.cache_size', str(cache_size)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Stereo fusion failed for chunk {idx}: {stderr} {stdout}")
                response = {"status": "error", "message": f"Stereo fusion failed for chunk {idx}: {stderr}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
        except subprocess.TimeoutExpired:
            logger.error(f"Stereo fusion timed out for chunk {idx}")
            response = {"status": "error", "message": f"Stereo fusion timed out for chunk {idx}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
        except Exception as e:
            logger.error(f"Unexpected error during stereo fusion for chunk {idx}: {str(e)}")
            response = {"status": "error", "message": f"Unexpected error during stereo fusion for chunk {idx}: {str(e)}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500

        if not os.path.exists(partial_ply):
            logger.error(f"No dense point cloud for chunk {idx}")
            response = {"status": "error", "message": f"No dense point cloud for chunk {idx}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
        partial_ply_files.append(partial_ply)

    output_dense_ply = os.path.join(base_dir, 'dense.ply')
    try:
        logger.debug("Merging partial dense point clouds")
        merge_ply_files(partial_ply_files, output_dense_ply)
        dense_size = os.path.getsize(output_dense_ply) / (1024 ** 2)
        logger.debug(f"Merged dense point cloud size: {dense_size:.2f} MB")
        os.chmod(output_dense_ply, 0o644)
    except Exception as e:
        logger.error(f"Failed to merge point clouds: {str(e)}")
        response = {"status": "error", "message": f"Failed to merge point clouds: {str(e)}", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if os.path.exists(os.path.join(base_dir, 'sparse.ply')):
            zip_file.write(os.path.join(base_dir, 'sparse.ply'), 'sparse.ply')
        if os.path.exists(output_dense_ply):
            zip_file.write(output_dense_ply, 'dense.ply')
        if os.path.exists(os.path.join(base_dir, 'camera_poses.json')):
            zip_file.write(os.path.join(base_dir, 'camera_poses.json'), 'camera_poses.json')

    zip_buffer.seek(0)
    zip_temp_path = os.path.join(base_dir, 'reconstruction_bundle.zip')
    with open(zip_temp_path, 'wb') as f:
        f.write(zip_buffer.getvalue())

    response = {
        'status': 'success',
        'message': 'Dense processing complete',
        'sparse_ply_path': f'/output/{request_id}/sparse.ply',
        'dense_ply_path': f'/output/{request_id}/dense.ply',
        'poses_path': f'/output/{request_id}/camera_poses.json',
        'zip_path': f'/output/{request_id}/reconstruction_bundle.zip',
        'session_id': session_id,
        'input_save_time': session.get(f'input_save_time_{session_id}')
    }
    logger.debug(f"Sending response: {response}")
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)