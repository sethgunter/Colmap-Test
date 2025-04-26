from flask import Flask, request, send_file, Response
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
import plyfile  # For merging PLY files
import pycolmap

app = Flask(__name__, static_folder='static', static_url_path='/static')

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
    logger.debug(f"Attempting to serve static file: {path}")
    return app.send_static_file(path)

@app.route('/output/<path:path>')
def serve_output(path):
    logger.debug(f"Attempting to serve output file: {path}")
    try:
        file_path = os.path.join('/app/colmap_project', path)
        if not os.path.exists(file_path):
            logger.error(f"Output file not found: {path}")
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
        if free_memory_mb < 3000:
            logger.error(f"Insufficient GPU memory: {free_memory_mb} MB available")
            return False, f"Insufficient GPU memory: {free_memory_mb} MB available"

        available_ram = psutil.virtual_memory().available / (1024 ** 2)
        logger.debug(f"Available RAM: {available_ram} MB")
        if available_ram < 8192:
            logger.error(f"Insufficient RAM: {available_ram} MB available")
            return False, f"Insufficient RAM: {available_ram} MB available"

        return True, ""
    except Exception as e:
        logger.error(f"Resource check failed: {e}")
        return False, f"Resource check failed: {e}"

def check_ram_for_fusion():
    """Check available RAM and log GPU memory before stereo fusion."""
    available_ram = psutil.virtual_memory().available / (1024 ** 2)
    logger.debug(f"Available RAM before stereo fusion: {available_ram} MB")
    
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        free_memory_mb = gpu.memoryFree
        logger.debug(f"GPU memory free before stereo fusion: {free_memory_mb} MB")
    else:
        logger.warning("No GPU detected before stereo fusion")

    if available_ram < 8192:
        logger.error(f"Insufficient RAM for stereo fusion: {available_ram} MB available")
        return False, f"Insufficient RAM for fusion: {available_ram} MB available"
    return True, available_ram

def merge_ply_files(ply_files, output_path):
    """Merge multiple PLY files into a single PLY file."""
    all_vertices = []
    all_colors = []
    for ply_path in ply_files:
        ply_data = plyfile.PlyData.read(ply_path)
        vertices = ply_data['vertex']
        all_vertices.append(vertices[['x', 'y', 'z']])
        all_colors.append(vertices[['red', 'green', 'blue']])
    
    # Concatenate all vertices and colors
    merged_vertices = np.concatenate([v.data for v in all_vertices])
    merged_colors = np.concatenate([c.data for c in all_colors])
    
    # Create new vertex element
    vertex_data = np.array(
        [(v['x'], v['y'], v['z'], c['red'], c['green'], c['blue']) 
         for v, c in zip(merged_vertices, merged_colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    
    # Create PLY element and write to file
    vertex_element = plyfile.PlyElement.describe(vertex_data, 'vertex')
    plyfile.PlyData([vertex_element]).write(output_path)
    logger.debug(f"Merged {len(ply_files)} PLY files into {output_path}")

@app.route('/process-video', methods=['POST'])
def process_video():
    logger.debug("Received POST request to /process-video")
    if 'video' not in request.files:
        logger.error("No video provided in request")
        return {"status": "error", "message": "No video provided"}, 400

    request_id = str(uuid.uuid4())
    base_dir = os.path.join('/app/colmap_project', request_id)
    video_dir = os.path.join(base_dir, 'video')
    images_dir = os.path.join(base_dir, 'images')
    database_path = os.path.join(base_dir, 'database.db')
    sparse_dir = os.path.join(base_dir, 'sparse')
    dense_base_dir = os.path.join(base_dir, 'dense_chunks')  # Directory for chunked dense workspaces
    poses_dir = os.path.join(base_dir, 'poses')
    sparse_cubic_dir = os.path.join(base_dir, 'sparse-cubic')

    if os.path.exists(base_dir):
        for attempt in range(5):
            try:
                terminate_child_processes()
                debug_file_locks(base_dir)
                shutil.rmtree(base_dir)
                logger.debug(f"Successfully removed {base_dir}")
                break
            except OSError as e:
                logger.warning(f"Attempt {attempt+1} to remove {base_dir} failed: {e}")
                time.sleep(3)
        else:
            logger.error(f"Failed to remove {base_dir} after retries")
            return {"status": "error", "message": f"Cannot clean up previous run: {base_dir} is busy"}, 500

    try:
        os.makedirs(video_dir)
        os.makedirs(images_dir)
        os.makedirs(sparse_dir)
        os.makedirs(dense_base_dir)
        os.makedirs(poses_dir, exist_ok=True)
        os.makedirs(sparse_cubic_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        return {"status": "error", "message": f"Failed to create directories: {e}"}, 500

    video = request.files['video']
    video_path = os.path.join(video_dir, video.filename)
    logger.debug(f"Saving video: {video_path}")
    try:
        video.save(video_path)
        video_save_time = time.time()
        logger.debug(f"Video saved at timestamp: {video_save_time}")
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        return {"status": "error", "message": f"Failed to save video: {str(e)}"}, 500

    resource_ok, resource_message = check_resources(request_id)
    if not resource_ok:
        return {"status": "error", "message": resource_message}, 500

    # Extract frames
    try:
        logger.debug("Extracting frames")
        process = subprocess.Popen([
            'ffmpeg', '-i', video_path, '-r', '2', '-vf', 'scale=1920:960',
            os.path.join(images_dir, 'frame_%04d.jpg')
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=300)
        if process.returncode != 0:
            logger.error(f"Frame extraction failed: {stderr}")
            return {"status": "error", "message": f"Frame extraction failed: {stderr}"}, 500
        logger.debug(f"Frame extraction output: {stdout}")
    except subprocess.TimeoutExpired:
        logger.error("Frame extraction timed out")
        terminate_child_processes()
        return {"status": "error", "message": "Frame extraction timed out"}, 500

    # Database creation
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
            return {"status": "error", "message": f"Database creation failed: {stderr}"}, 500
    except subprocess.TimeoutExpired:
        logger.error("Database creation timed out")
        return {"status": "error", "message": "Database creation timed out"}, 500

    # Feature extraction
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
            '--SiftExtraction.max_num_features', '11000',
            '--SiftExtraction.estimate_affine_shape', '1',
            '--SiftExtraction.max_num_orientations', '3'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=600)
        if process.returncode != 0:
            logger.error(f"Feature extraction failed: {stderr}")
            return {"status": "error", "message": f"Feature extraction failed: {stderr}"}, 500
    except subprocess.TimeoutExpired:
        logger.error("Feature extraction timed out")
        return {"status": "error", "message": "Feature extraction timed out"}, 500

    # Feature matching
    try:
        logger.debug("Running feature matching")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'sequential_matcher',
            '--database_path', database_path,
            '--SequentialMatching.overlap', '5',
            '--SequentialMatching.quadratic_overlap', '0',
            '--SequentialMatching.loop_detection', '1',
            '--SequentialMatching.vocab_tree_path', '/app/vocab_tree.bin',
            '--SequentialMatching.loop_detection_period', '20',
            '--SequentialMatching.loop_detection_num_images', '50',
            '--SequentialMatching.loop_detection_num_nearest_neighbors', '1',
            '--SequentialMatching.loop_detection_num_checks', '256',
            '--SequentialMatching.loop_detection_num_images_after_verification', '0',
            '--SequentialMatching.loop_detection_max_num_features', '-1',
            '--SiftMatching.use_gpu', '1',
            '--SiftMatching.gpu_index', '0',
            '--SiftMatching.min_num_inliers', '30'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=600)
        if process.returncode != 0:
            logger.error(f"Feature matching failed: {stderr}")
            return {"status": "error", "message": f"Feature matching failed: {stderr}"}, 500
    except subprocess.TimeoutExpired:
        logger.error("Feature matching timed out")
        return {"status": "error", "message": "Feature matching timed out"}, 500

    # Sparse reconstruction
    try:
        logger.debug("Running sparse reconstruction")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'mapper',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--output_path', sparse_dir,
            '--Mapper.min_num_matches', '10',
            '--Mapper.init_min_num_inliers', '30',
            '--Mapper.ba_global_max_num_iterations', '50',
            '--Mapper.multiple_models', '1',
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.sphere_camera', '1'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=600)
        if process.returncode != 0:
            logger.error(f"Sparse reconstruction failed: {stderr}")
            return {"status": "error", "message": f"Sparse reconstruction failed: {stderr}"}, 500
    except subprocess.TimeoutExpired:
        logger.error("Sparse reconstruction timed out")
        return {"status": "error", "message": "Sparse reconstruction timed out"}, 500

    sparse_model_dir = os.path.join(sparse_dir, '0')
    if not os.path.exists(sparse_model_dir):
        logger.error("Sparse model not found")
        return {"status": "error", "message": "Sparse reconstruction failed: no model generated"}, 500

    # Cubic reprojection
    try:
        logger.debug("Running cubic reprojection")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'sphere_cubic_reprojecer',
            '--image_path', images_dir,
            '--input_path', sparse_model_dir,
            '--output_path', sparse_cubic_dir
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=600)
        if process.returncode != 0:
            logger.error(f"Cubic reprojection failed: {stderr}")
            return {"status": "error", "message": f"Cubic reprojection failed: {stderr}"}, 500
    except subprocess.TimeoutExpired:
        logger.error("Cubic reprojection timed out")
        return {"status": "error", "message": "Cubic reprojection timed out"}, 500
    
    # Log contents of sparse_cubic_dir to confirm images are there
    cubic_image_files = glob.glob(os.path.join(sparse_cubic_dir, '*.jpg'))
    logger.debug(f"Found {len(cubic_image_files)} perspective images in {sparse_cubic_dir}: {cubic_image_files[:5]}...")
    if not cubic_image_files:
        logger.error(f"No perspective images found in {sparse_cubic_dir}")
        return {"status": "error", "message": f"No perspective images found in {sparse_cubic_dir}"}, 500

    # Chunk the perspective images
    chunk_size = 50  # Adjust based on your memory limits (50 works for most setups)
    overlap = 20     # Overlap ensures consistency across chunks
    step = chunk_size - overlap
    image_list = sorted(cubic_image_files)
    chunks = [image_list[i:i + chunk_size] for i in range(0, len(image_list), step) if image_list[i:i + chunk_size]]
    logger.debug(f"Split {len(image_list)} images into {len(chunks)} chunks")

    # Process each chunk for dense reconstruction
    partial_ply_files = []
    for idx, chunk in enumerate(chunks):
        # Set up chunk workspace
        chunk_dir = os.path.join(dense_base_dir, f'chunk_{idx}')
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_image_dir = os.path.join(chunk_dir, 'images')
        os.makedirs(chunk_image_dir, exist_ok=True)
        chunk_sparse_dir = os.path.join(chunk_dir, 'sparse')
        os.makedirs(chunk_sparse_dir, exist_ok=True)

        # Copy images to chunk workspace
        for img_path in chunk:
            shutil.copy(img_path, chunk_image_dir)
        chunk_image_names = [os.path.basename(img) for img in chunk]
        logger.debug(f"Chunk {idx}: {len(chunk_image_names)} images copied: {chunk_image_names[:5]}...")

        # Verify images exist
        for img_name in chunk_image_names:
            if not os.path.exists(os.path.join(chunk_image_dir, img_name)):
                logger.error(f"Image missing in chunk {idx}: {img_name}")
                return {"status": "error", "message": f"Image missing in chunk {idx}: {img_name}"}, 500

        # Filter sparse model for this chunk
        try:
            shutil.copytree(os.path.join(sparse_cubic_dir, 'sparse'), chunk_sparse_dir, dirs_exist_ok=True)
            reconstruction = pycolmap.Reconstruction(chunk_sparse_dir)
            logger.debug(f"Chunk {idx} sparse model originally has {len(reconstruction.images)} images")

            # Keep only images in this chunk
            chunk_image_names_set = set(chunk_image_names)
            images_to_keep = {img_id: img for img_id, img in reconstruction.images.items()
                            if img.name in chunk_image_names_set}
            for img_id in list(reconstruction.images.keys()):
                if img_id not in images_to_keep:
                    reconstruction.deregister_image(img_id)
            reconstruction.write(chunk_sparse_dir)
            logger.debug(f"Chunk {idx} sparse model filtered to {len(reconstruction.images)} images")
        except Exception as e:
            logger.error(f"Failed to filter sparse model for chunk {idx}: {str(e)}")
            return {"status": "error", "message": f"Sparse model filtering failed for chunk {idx}: {str(e)}"}, 500

        # Run image_undistorter
        try:
            logger.debug(f"Undistorting images for chunk {idx}")
            process = subprocess.Popen([
                'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
                'colmap', 'image_undistorter',
                '--image_path', chunk_image_dir,
                '--input_path', chunk_sparse_dir,
                '--output_path', chunk_dir,
                '--output_type', 'COLMAP',
                '--max_image_size', '400'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=1200)
            if process.returncode != 0:
                logger.error(f"Undistortion failed for chunk {idx}: {stderr}")
                return {"status": "error", "message": f"Undistortion failed for chunk {idx}: {stderr}"}, 500
        except subprocess.TimeoutExpired:
            logger.error(f"Undistortion timed out for chunk {idx}")
            return {"status": "error", "message": f"Undistortion timed out for chunk {idx}"}, 500

        # Run patch_match_stereo
        try:
            logger.debug(f"Running patch match stereo for chunk {idx}")
            process = subprocess.Popen([
                'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
                'colmap', 'patch_match_stereo',
                '--workspace_path', chunk_dir,
                '--workspace_format', 'COLMAP',
                '--PatchMatchStereo.gpu_index', '0',
                '--PatchMatchStereo.max_image_size', '400',
                '--PatchMatchStereo.window_radius', '4',
                '--PatchMatchStereo.num_samples', '3',
                '--PatchMatchStereo.num_iterations', '2',
                '--PatchMatchStereo.filter', '1',
                '--PatchMatchStereo.filter_min_ncc', '0.5',
                '--PatchMatchStereo.filter_min_triangulation_angle', '5.0',
                '--PatchMatchStereo.filter_min_num_consistent', '2',
                '--PatchMatchStereo.cache_size', '4'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=5400)
            if process.returncode != 0:
                logger.error(f"Patch match failed for chunk {idx}: {stderr}")
                return {"status": "error", "message": f"Patch match failed for chunk {idx}: {stderr}"}, 500
        except subprocess.TimeoutExpired:
            logger.error(f"Patch match timed out for chunk {idx}")
            return {"status": "error", "message": f"Patch match timed out for chunk {idx}"}, 500

        # Verify depth maps
        depth_maps_dir = os.path.join(chunk_dir, 'stereo', 'depth_maps')
        if not os.path.exists(depth_maps_dir) or not os.listdir(depth_maps_dir):
            logger.error(f"No depth maps for chunk {idx}")
            return {"status": "error", "message": f"No depth maps for chunk {idx}"}, 500

        # Run stereo_fusion
        ram_ok, available_ram = check_ram_for_fusion()
        if not ram_ok:
            return {"status": "error", "message": available_ram}, 500
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
                '--StereoFusion.min_num_pixels', '6',
                '--StereoFusion.check_num_images', '3',
                '--StereoFusion.max_reproj_error', '1.5',
                '--StereoFusion.max_depth_error', '0.2',
                '--StereoFusion.cache_size', str(cache_size)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=5400)
            if process.returncode != 0:
                logger.error(f"Stereo fusion failed for chunk {idx}: {stderr}")
                return {"status": "error", "message": f"Stereo fusion failed for chunk {idx}: {stderr}"}, 500
        except subprocess.TimeoutExpired:
            logger.error(f"Stereo fusion timed out for chunk {idx}")
            return {"status": "error", "message": f"Stereo fusion timed out for chunk {idx}"}, 500

        if not os.path.exists(partial_ply):
            logger.error(f"No dense point cloud for chunk {idx}")
            return {"status": "error", "message": f"No dense point cloud for chunk {idx}"}, 500
        partial_ply_files.append(partial_ply)
    # Merge partial dense point clouds
    output_dense_ply = os.path.join(base_dir, 'dense.ply')
    try:
        logger.debug("Merging partial dense point clouds")
        merge_ply_files(partial_ply_files, output_dense_ply)
        dense_size = os.path.getsize(output_dense_ply) / (1024 ** 2)
        logger.debug(f"Merged dense point cloud size: {dense_size:.2f} MB")
        os.chmod(output_dense_ply, 0o644)
    except Exception as e:
        logger.error(f"Failed to merge point clouds: {str(e)}")
        return {"status": "error", "message": f"Failed to merge point clouds: {str(e)}"}, 500

    # Export camera poses
    try:
        logger.debug("Exporting camera poses")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'model_converter',
            '--input_path', sparse_model_dir,
            '--output_path', poses_dir,
            '--output_type', 'TXT'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode != 0:
            logger.error(f"Model converter failed: {stderr}")
            return {"status": "error", "message": f"Model converter failed: {stderr}"}, 500
    except subprocess.TimeoutExpired:
        logger.error("Model conversion timed out")
        return {"status": "error", "message": "Model conversion timed out"}, 500

    poses_json_path = os.path.join(base_dir, 'camera_poses.json')
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
        return {"status": "error", "message": f"Failed to parse camera poses: {str(e)}"}, 500

    # Export sparse point cloud
    output_sparse_ply = os.path.join(base_dir, 'sparse.ply')
    try:
        logger.debug("Exporting sparse point cloud")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'model_converter',
            '--input_path', sparse_model_dir,
            '--output_path', output_sparse_ply,
            '--output_type', 'PLY'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode != 0:
            logger.error(f"Sparse point cloud export failed: {stderr}")
            return {"status": "error", "message": f"Sparse point cloud export failed: {stderr}"}, 500
        sparse_size = os.path.getsize(output_sparse_ply) / (1024 ** 2)
        logger.debug(f"Sparse point cloud size: {sparse_size:.2f} MB")
    except subprocess.TimeoutExpired:
        logger.error("Sparse point cloud export timed out")
        return {"status": "error", "message": "Sparse point cloud export timed out"}, 500

    # Create zip archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if os.path.exists(output_sparse_ply):
            zip_file.write(output_sparse_ply, 'sparse.ply')
        if os.path.exists(output_dense_ply):
            zip_file.write(output_dense_ply, 'dense.ply')
        if os.path.exists(poses_json_path):
            zip_file.write(poses_json_path, 'camera_poses.json')

    zip_buffer.seek(0)
    zip_temp_path = os.path.join(base_dir, 'reconstruction_bundle.zip')
    with open(zip_temp_path, 'wb') as f:
        f.write(zip_buffer.getvalue())

    response = {
        'status': 'success',
        'message': 'Processing complete',
        'sparse_ply_path': f'/output/{request_id}/sparse.ply',
        'dense_ply_path': f'/output/{request_id}/dense.ply',
        'poses_path': f'/output/{request_id}/camera_poses.json',
        'zip_path': f'/output/{request_id}/reconstruction_bundle.zip',
        'video_save_time': video_save_time
    }, 200

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)