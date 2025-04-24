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
    response = app.send_static_file('index.html')
    return response

@app.route('/static/<path:path>')
def serve_static(path):
    logger.debug(f"Attempting to serve static file: {path}")
    response = app.send_static_file(path)
    return response

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
    dense_dir = os.path.join(base_dir, 'dense')
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
        os.makedirs(dense_dir)
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
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        return {"status": "error", "message": f"Failed to save video: {str(e)}"}, 500

    resource_ok, resource_message = check_resources(request_id)
    if not resource_ok:
        return {"status": "error", "message": resource_message}, 500

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
    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}")
        return {"status": "error", "message": f"Frame extraction failed: {str(e)}"}, 500
    finally:
        terminate_child_processes()

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
        logger.debug(f"Database creation output: {stdout}")
        terminate_child_processes()

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
        logger.debug(f"Feature extraction output: {stdout}")
        terminate_child_processes()

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
        logger.debug(f"Feature matching output: {stdout}")
        terminate_child_processes()

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
        logger.debug(f"Sparse reconstruction output: {stdout}")
        terminate_child_processes()

        sparse_model_dir = os.path.join(sparse_dir, '0')
        if not os.path.exists(sparse_model_dir):
            logger.error("Sparse model not found")
            return {"status": "error", "message": "Sparse reconstruction failed: no model generated"}, 500

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
        logger.debug(f"Cubic reprojection output: {stdout}")
        terminate_child_processes()

        resource_ok, resource_message = check_resources(request_id)
        if not resource_ok:
            return {"status": "error", "message": resource_message}, 500

        logger.debug("Running dense reconstruction")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'image_undistorter',
            '--image_path', sparse_cubic_dir,
            '--input_path', os.path.join(sparse_cubic_dir, 'sparse'),
            '--output_path', dense_dir,
            '--output_type', 'COLMAP',
            '--max_image_size', '1000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=600)
        if process.returncode != 0:
            logger.error(f"Image undistortion failed: {stderr}")
            return {"status": "error", "message": f"Image undistortion failed: {stderr}"}, 500
        logger.debug(f"Image undistortion output: {stdout}")
        terminate_child_processes()

        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'patch_match_stereo',
            '--workspace_path', dense_dir,
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.gpu_index', '0',
            '--PatchMatchStereo.max_image_size', '1000',
            '--PatchMatchStereo.window_radius', '5',
            '--PatchMatchStereo.num_samples', '10',
            '--PatchMatchStereo.num_iterations', '5',
            '--PatchMatchStereo.filter', '0',
            '--PatchMatchStereo.cache_size', '8'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=2400)
        if process.returncode != 0:
            logger.error(f"Patch match stereo failed: {stderr}")
            return {"status": "error", "message": f"Patch match stereo failed: {stderr}"}, 500
        logger.debug(f"Patch match stereo output: {stdout}")
        terminate_child_processes()

        depth_maps_dir = os.path.join(dense_dir, 'stereo', 'depth_maps')
        if not os.path.exists(depth_maps_dir) or not os.listdir(depth_maps_dir):
            logger.error("No depth maps found after patch_match_stereo")
            return {"status": "error", "message": "No depth maps generated by patch_match_stereo"}, 500
        depth_map_files = os.listdir(depth_maps_dir)
        logger.debug(f"Found {len(depth_map_files)} depth maps in {depth_maps_dir}")
        for f in depth_map_files:
            file_path = os.path.join(depth_maps_dir, f)
            file_size = os.path.getsize(file_path) / (1024 ** 2)
            logger.debug(f"Depth map {f}: {file_size:.2f} MB")

        ram_ok, available_ram = check_ram_for_fusion()
        if not ram_ok:
            return {"status": "error", "message": available_ram}, 500

        available_ram_gb = available_ram / 1024
        cache_size = max(1, int(available_ram_gb * 0.5))

        logger.debug(f"Verifying workspace at {dense_dir}")
        required_dirs = ['images', 'sparse', 'stereo']
        for subdir in required_dirs:
            subdir_path = os.path.join(dense_dir, subdir)
            if not os.path.exists(subdir_path):
                logger.error(f"Workspace directory missing: {subdir_path}")
                return {"status": "error", "message": f"Workspace directory missing: {subdir_path}"}, 500
            if not os.access(subdir_path, os.R_OK | os.W_OK):
                logger.error(f"No read/write permissions for {subdir_path}")
                return {"status": "error", "message": f"No read/write permissions for {subdir_path}"}, 500
        logger.debug(f"Workspace verification passed: {dense_dir}")

        try:
            process = subprocess.Popen(['colmap', 'stereo_fusion', '--help'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=10)
            if process.returncode != 0:
                logger.error(f"Failed to check stereo_fusion options: {stderr}")
                return {"status": "error", "message": f"Failed to check stereo_fusion options: {stderr}"}, 500
            logger.debug(f"stereo_fusion help output: {stdout}")
            use_min_tri_angle = 'min_tri_angle' in stdout
            if not use_min_tri_angle:
                logger.warning("min_tri_angle option not supported; omitting")
        except Exception as e:
            logger.error(f"Failed to check stereo_fusion options: {str(e)}")
            return {"status": "error", "message": f"Failed to check stereo_fusion options: {str(e)}"}, 500

        output_dense_ply = os.path.join(base_dir, 'dense.ply')
        cmd = [
            'colmap', 'stereo_fusion',
            '--workspace_path', dense_dir,
            '--workspace_format', 'COLMAP',
            '--input_type', 'photometric',
            '--output_path', output_dense_ply,
            '--StereoFusion.min_num_pixels', '5',
            '--StereoFusion.max_reproj_error', '2',
            '--StereoFusion.max_depth_error', '0.25',
            '--StereoFusion.cache_size', str(cache_size)
        ]
        
        logger.debug(f"Executing command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=2400)
        except subprocess.TimeoutExpired:
            logger.error("Stereo fusion timed out")
            process.terminate()
            return {"status": "error", "message": "Stereo fusion timed out"}, 500

        logger.debug(f"Stereo fusion stdout: {stdout}")
        if stderr:
            logger.debug(f"Stereo fusion stderr: {stderr}")
        if process.returncode != 0:
            logger.error(f"Stereo fusion failed: {stderr}")
            return {"status": "error", "message": f"Stereo fusion failed: {stderr}"}, 500
        logger.debug(f"Stereo fusion completed successfully")
        terminate_child_processes()

        if not os.path.exists(output_dense_ply):
            logger.error("Dense point cloud file not found")
            return {"status": "error", "message": "Dense reconstruction failed: no point cloud generated"}, 500
        # Log dense point cloud size and check permissions
        dense_size = os.path.getsize(output_dense_ply) / (1024 ** 2)
        logger.debug(f"Dense point cloud size: {dense_size:.2f} MB")
        if not os.access(output_dense_ply, os.R_OK):
            logger.error(f"No read permissions for {output_dense_ply}")
            return {"status": "error", "message": f"No read permissions for dense.ply"}, 500
        # Set read permissions
        try:
            os.chmod(output_dense_ply, 0o644)
            logger.debug(f"Set read permissions for {output_dense_ply}")
        except Exception as e:
            logger.warning(f"Failed to set permissions for {output_dense_ply}: {str(e)}")

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
        logger.debug(f"Model converter output: {stdout}")
        terminate_child_processes()

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

        logger.debug("Exporting sparse point cloud")
        output_sparse_ply = os.path.join(base_dir, 'sparse.ply')
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
        logger.debug(f"Point cloud export output: {stdout}")
        if os.path.exists(output_sparse_ply):
            sparse_size = os.path.getsize(output_sparse_ply) / (1024 ** 2)
            logger.debug(f"Sparse point cloud size: {sparse_size:.2f} MB")
        else:
            logger.error("Sparse point cloud file not found")
        terminate_child_processes()

        logger.debug("Creating zip archive")
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
            'zip_path': f'/output/{request_id}/reconstruction_bundle.zip'
        }, 200

        def cleanup():
            time.sleep(600)
            for attempt in range(5):
                try:
                    terminate_child_processes()
                    debug_file_locks(base_dir)
                    shutil.rmtree(base_dir)
                    logger.debug(f"Post-response cleanup: Successfully removed {base_dir}")
                    break
                except OSError as e:
                    logger.warning(f"Post-response cleanup attempt {attempt+1} failed: {e}")
                    time.sleep(3)
            else:
                logger.error(f"Post-response cleanup failed for {base_dir}")

        from threading import Thread
        cleanup_thread = Thread(target=cleanup)
        cleanup_thread.start()

        return response

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}, 500
    finally:
        terminate_child_processes()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)