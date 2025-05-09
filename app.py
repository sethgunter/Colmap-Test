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
from scipy.spatial import KDTree  # Added for loop closure detection
import cv2  # Added for feature matching
import sqlite3  # Added for database updates

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024
# Ensure consistent secret_key for session persistence
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fixed-secret-key-for-testing')

# Set up logging
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
    """Terminate any child processes that might be holding files open."""
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
    """Log files that might be locked in the directory using lsof."""
    try:
        result = subprocess.run(['lsof', directory], capture_output=True, text=True)
        logger.debug(f"lsof output for {directory}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to run lsof on {directory}: {e.stderr}")
    except FileNotFoundError:
        logger.warning("lsof not installed, cannot debug file locks")

def cleanup_old_requests(current_request_id):
    """Clean up all directories in /app/colmap_project except the current request's directory."""
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
    """Check available disk space, GPU memory, and RAM after cleaning up old requests."""
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
    """Check available RAM and log GPU memory before stereo fusion."""
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
    """Merge multiple PLY files into a single PLY file, handling variable vertex attributes."""
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
    """Export sparse PLY and camera poses."""
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

@app.route('/process-video', methods=['POST'])
def process_video():
    logger.debug("Received POST request to /process-video")
    logger.debug(f"Request files: {list(request.files.keys())}")
    logger.debug(f"Incoming form data: {request.form}")

    # Initialize session and request ID
    session_id = request.form.get('session_id')
    if not session_id or session_id.strip() == '':
        session_id = str(uuid.uuid4())
        logger.debug(f"Generated new session_id: {session_id}")
    else:
        logger.debug(f"Using provided session_id: {session_id}")
    request_id = session.get(f'request_id_{session_id}', str(uuid.uuid4()))
    session[f'request_id_{session_id}'] = request_id
    logger.debug(f"Session state: request_id_{session_id} = {request_id}")

    # Define directories
    base_dir = os.path.join('/app/colmap_project', request_id)
    video_dir = os.path.join(base_dir, 'video')
    images_dir = os.path.join(base_dir, 'images')
    database_path = os.path.join(base_dir, 'database.db')
    sparse_dir = os.path.join(base_dir, 'sparse')
    poses_dir = os.path.join(base_dir, 'poses')
    sparse_cubic_dir = os.path.join(base_dir, 'sparse-cubic')

    # Clean up old requests
    if not cleanup_old_requests(request_id):
        logger.error("Failed to clean up old request directories")
        response = {"status": "error", "message": "Failed to clean up old request directories", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    # Create directories
    try:
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(poses_dir, exist_ok=True)
        os.makedirs(sparse_cubic_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        response = {"status": "error", "message": f"Failed to create directories: {e}", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    # Check for video or images
    is_video = 'video' in request.files and request.files['video'].filename != ''
    image_files = request.files.getlist('images')

    # Validate inputs
    if not is_video and not image_files:
        logger.error("No video or images provided in request")
        response = {"status": "error", "message": "No video or images provided", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 400
    if is_video and image_files:
        logger.error("Both video and images provided; please provide only one")
        response = {"status": "error", "message": "Please provide either a video or images, not both", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 400

    input_save_time = time.time()
    session[f'input_save_time_{session_id}'] = input_save_time
    logger.debug(f"Stored input_save_time for session_id {session_id}: {input_save_time}")
    if is_video:
        # Handle video upload
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

        # Extract frames
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
        # Handle chunked image uploads
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
        logger.debug(f"Saved {len(image_files)} images for session {session_id}")

        if request.form.get('complete') != 'true':
            response = {
                'status': 'partial',
                'message': 'Images received, send next chunk or mark complete',
                'session_id': session_id
            }
            logger.debug(f"Sending response: {response}")
            return response, 200

    # Check resources
    resource_ok, resource_message = check_resources(request_id)
    if not resource_ok:
        logger.error(f"Resource check failed: {resource_message}")
        response = {"status": "error", "message": resource_message, "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    # Create database
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

    try:
        logger.debug("Running feature extraction")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'feature_extractor',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--ImageReader.camera_model', 'SPHERE',
            '--ImageReader.camera_params', '1,960,480',
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '1',
            '--SiftExtraction.gpu_index', '0',
            '--SiftExtraction.peak_threshold', '0.0001',
            '--SiftExtraction.max_num_features', '13000',
            '--SiftExtraction.estimate_affine_shape', '1',
            '--SiftExtraction.max_num_orientations', '3'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Feature extraction failed: {stderr}")
            response = {"status": "error", "message": f"Feature extraction failed: {stderr} {stdout}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
    except subprocess.TimeoutExpired:
        logger.error("Feature extraction timed out")
        response = {"status": "error", "message": "Feature extraction timed out", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    try:
        logger.debug("Running feature matching")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'sequential_matcher',
            '--database_path', database_path,
            '--SequentialMatching.overlap', '5',
            '--SequentialMatching.quadratic_overlap', '0',
            '--SequentialMatching.loop_detection', '0',  # Disable vocab tree for raw poses
            '--SiftMatching.use_gpu', '1',
            '--SiftMatching.gpu_index', '0',
            '--SiftMatching.min_num_inliers', '15'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Feature matching failed: {stderr}")
            response = {"status": "error", "message": f"Feature matching failed: {stderr} {stdout}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
    except subprocess.TimeoutExpired:
        logger.error("Feature matching timed out")
        response = {"status": "error", "message": "Feature matching timed out", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    try:
        logger.debug("Running initial sparse reconstruction")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'mapper',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--output_path', sparse_dir,
            '--Mapper.min_num_matches', '10',
            '--Mapper.init_min_num_inliers', '30',
            '--Mapper.ba_global_max_num_iterations', '50',
            '--Mapper.multiple_models', '0',
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.sphere_camera', '1',
            '--Mapper.ba_local_max_num_iterations', '100'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Initial sparse reconstruction failed: {stderr}")
            response = {"status": "error", "message": f"Initial sparse reconstruction failed: {stderr} {stdout}", "session_id": session_id}
            logger.debug(f"Sending response: {response}")
            return response, 500
    except subprocess.TimeoutExpired:
        logger.error("Initial sparse reconstruction timed out")
        response = {"status": "error", "message": "Initial sparse reconstruction timed out", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    sparse_model_dir = os.path.join(sparse_dir, '0')
    if not os.path.exists(sparse_model_dir):
        logger.error("Sparse model not found")
        response = {"status": "error", "message": "Initial sparse reconstruction failed: no model generated", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

        # --- Start of Loop Closure Detection (Modified for SphereSfM) ---
        # --- Start of Loop Closure Detection (Using pycolmap) ---
    try:
        logger.debug("Running distance-based loop closure detection")
        # Load reconstruction
        reconstruction = pycolmap.Reconstruction(sparse_model_dir)

        # Extract poses
        poses = []
        image_ids = []
        image_names = []
        translations = []
        for image_id, image in reconstruction.images.items():
            if not image.has_pose:
                logger.warning(f"Image {image.name} has no pose, skipping")
                continue
            # Use projection_center for translation vector
            t = image.projection_center()
            # Use cam_from_world for pose
            pose = image.cam_from_world
            R = pose.rotation.matrix()  # Convert Rotation3d to 3x3 matrix
            poses.append((t, R))
            translations.append(t)
            image_ids.append(image_id)
            image_names.append(image.name)

        translations = np.array(translations)
        if len(translations) == 0:
            logger.warning("No valid poses found, skipping loop closure detection")
            raise Exception("No valid poses found")

        # Build KD-tree for proximity search
        kdtree = KDTree(translations)
        distance_threshold = 0.5  # meters, adjust based on scene scale
        temporal_threshold = 10  # Minimum frame difference for loop closures

        # Find candidate pairs
        candidate_pairs = []
        for i, t in enumerate(translations):
            indices = kdtree.query_ball_point(t, distance_threshold)
            for j in indices:
                if i < j and abs(image_ids[i] - image_ids[j]) >= temporal_threshold:  # Enforce 10-frame difference
                    candidate_pairs.append((image_ids[i], image_ids[j], image_names[i], image_names[j]))

        # Geometric verification
        loop_closures = []
        for img_id1, img_id2, img_name1, img_name2 in candidate_pairs:
            # Load images
            img1_path = os.path.join(images_dir, img_name1)
            img2_path = os.path.join(images_dir, img_name2)
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                logger.warning(f"Failed to load images: {img_name1}, {img_name2}")
                continue

            # Extract and match features
            sift = cv2.SIFT_create(nfeatures=13000, contrastThreshold=0.0001)
            kp1, desc1 = sift.detectAndCompute(img1, None)
            kp2, desc2 = sift.detectAndCompute(img2, None)
            if desc1 is None or desc2 is None:
                logger.warning(f"No features detected for pair: {img_name1}, {img_name2}")
                continue

            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Estimate fundamental matrix with RANSAC
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.5)
            inlier_count = np.sum(inliers) if inliers is not None else 0
            if inlier_count < 50:  # Require at least 50 inliers
                logger.debug(f"Insufficient inliers ({inlier_count}) for pair: {img_name1}, {img_name2}")
                continue

            # Triangulate points and check reprojection error
            P1 = np.hstack((poses[i][1], poses[i][0].reshape(-1, 1)))  # R | t
            P2 = np.hstack((poses[j][1], poses[j][0].reshape(-1, 1)))  # R | t
            points4d = cv2.triangulatePoints(P1, P2, pts1[inliers.ravel() > 0].T, pts2[inliers.ravel() > 0].T)
            points3d = points4d[:3] / points4d[3]

            # Reprojection error for spherical model (simplified)
            reproj_error = 0
            for k in range(points3d.shape[1]):
                pt3d = points3d[:, k]
                proj1 = P1 @ np.append(pt3d, 1)
                proj2 = P2 @ np.append(pt3d, 1)
                reproj_error += np.linalg.norm(proj1[:2] / proj1[2] - pts1[inliers.ravel() > 0][k]) ** 2
                reproj_error += np.linalg.norm(proj2[:2] / proj2[2] - pts2[inliers.ravel() > 0][k]) ** 2
            reproj_error = np.sqrt(reproj_error / (2 * points3d.shape[1]))

            if reproj_error > 1.5:  # Stricter threshold for 1920x960 images
                logger.debug(f"High reprojection error ({reproj_error}) for pair: {img_name1}, {img_name2}")
                continue

            loop_closures.append((img_id1, img_id2, matches, inliers))
            logger.debug(f"Valid loop closure found: {img_name1}, {img_name2} with {inlier_count} inliers, reproj error {reproj_error}")

        # Update database with loop closures
        if loop_closures:
            logger.debug("Updating database with loop closures")
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            for img_id1, img_id2, matches, inliers in loop_closures:
                cursor.execute("SELECT image_id, data FROM keypoints WHERE image_id IN (?, ?)", (img_id1, img_id2))
                keypoints = {row[0]: np.frombuffer(row[1], dtype=np.float32).reshape(-1, 6) for row in cursor.fetchall()}
                match_data = []
                for i, m in enumerate(matches):
                    if inliers[i]:
                        match_data.append((m.queryIdx, m.trainIdx))
                match_data = np.array(match_data, dtype=np.uint32)
                cursor.execute("INSERT OR REPLACE INTO matches (pair_id, data) VALUES (?, ?)",
                              ((min(img_id1, img_id2) << 32) | max(img_id1, img_id2), match_data.tobytes()))
            conn.commit()
            conn.close()

            # Rerun mapper with loop closures
            logger.debug("Running sparse reconstruction with loop closures")
            process = subprocess.Popen([
                'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
                'colmap', 'mapper',
                '--database_path', database_path,
                '--image_path', images_dir,
                '--output_path', sparse_dir,
                '--Mapper.min_num_matches', '10',
                '--Mapper.init_min_num_inliers', '30',
                '--Mapper.ba_global_max_num_iterations', '50',
                '--Mapper.multiple_models', '0',
                '--Mapper.ba_refine_focal_length', '0',
                '--Mapper.ba_refine_principal_point', '0',
                '--Mapper.ba_refine_extra_params', '0',
                '--Mapper.sphere_camera', '1',
                '--Mapper.ba_local_max_num_iterations', '100'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Final sparse reconstruction failed: {stderr}")
                response = {"status": "error", "message": f"Final sparse reconstruction failed: {stderr} {stdout}", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500

            sparse_model_dir = os.path.join(sparse_dir, '0')
            if not os.path.exists(sparse_model_dir):
                logger.error("Final sparse model not found")
                response = {"status": "error", "message": "Final sparse reconstruction failed: no model generated", "session_id": session_id}
                logger.debug(f"Sending response: {response}")
                return response, 500
        else:
            logger.warning("No loop closures found, proceeding with initial reconstruction")
    except Exception as e:
        logger.error(f"Loop closure detection failed: {str(e)}")
        response = {"status": "error", "message": f"Loop closure detection failed: {str(e)}", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500
    # --- End of Loop Closure Detection ---

    try:
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

    # Export sparse PLY and camera poses
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

    # Define directories
    base_dir = os.path.join('/app/colmap_project', request_id)
    images_dir = os.path.join(base_dir, 'images')
    sparse_model_dir = os.path.join(base_dir, 'sparse', '0')
    sparse_cubic_dir = os.path.join(base_dir, 'sparse-cubic')
    dense_base_dir = os.path.join(base_dir, 'dense_chunks')
    poses_dir = os.path.join(base_dir, 'poses')

    # Verify required directories exist
    if not all(os.path.exists(d) for d in [base_dir, images_dir, sparse_model_dir, sparse_cubic_dir]):
        logger.error("Required project directories missing")
        response = {"status": "error", "message": "Project directories missing", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    # Create dense directory
    try:
        os.makedirs(dense_base_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create dense directory: {e}")
        response = {"status": "error", "message": f"Failed to create dense directory: {e}", "session_id": session_id}
        logger.debug(f"Sending response: {response}")
        return response, 500

    # Check resources
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
                logger.debug(f"Raw stderr content: {repr(stderr)}")
                if not stderr:
                    logger.error(f"No stderr output from stereo fusion for chunk {idx}")
                logger.debug(f"stdout content: {repr(stdout)}")
                for handler in logger.handlers:
                    handler.flush()
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