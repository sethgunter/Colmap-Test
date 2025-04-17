# app.py
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
    return send_file(os.path.join('/app/colmap_project', path))

def terminate_child_processes():
    """Terminate any child processes that might be holding files open."""
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                logger.debug(f"Terminating child process {child.pid}")
                child.terminate()
                child.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired) as e:
                logger.warning(f"Failed to terminate child process {child.pid}: {e}")
    except Exception as e:
        logger.error(f"Error while terminating child processes: {e}")

@app.route('/process-video', methods=['POST'])
def process_video():
    logger.debug("Received POST request to /process-video")
    if 'video' not in request.files:
        logger.error("No video provided in request")
        return "No video provided", 400

    # Generate unique directory for this request
    request_id = str(uuid.uuid4())
    base_dir = os.path.join('/app/colmap_project', request_id)
    video_dir = os.path.join(base_dir, 'video')
    images_dir = os.path.join(base_dir, 'images')
    database_path = os.path.join(base_dir, 'database.db')
    sparse_dir = os.path.join(base_dir, 'sparse')
    dense_dir = os.path.join(base_dir, 'dense')
    poses_dir = os.path.join(base_dir, 'poses')

    # Clean up specific request directory if it exists (shouldn't normally exist)
    if os.path.exists(base_dir):
        for attempt in range(3):
            try:
                terminate_child_processes()  # Terminate any lingering processes
                shutil.rmtree(base_dir)
                logger.debug(f"Successfully removed {base_dir}")
                break
            except OSError as e:
                logger.warning(f"Attempt {attempt+1} to remove {base_dir} failed: {e}")
                time.sleep(1)
        else:
            logger.error(f"Failed to remove {base_dir} after retries")
            return f"Cannot clean up previous run: {base_dir} is busy", 500

    # Create directories
    try:
        os.makedirs(video_dir)
        os.makedirs(images_dir)
        os.makedirs(sparse_dir)
        os.makedirs(dense_dir)
        os.makedirs(poses_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        return f"Failed to create directories: {e}", 500

    # Save uploaded video
    video = request.files['video']
    video_path = os.path.join(video_dir, video.filename)
    logger.debug(f"Saving video: {video_path}")
    try:
        video.save(video_path)
    except Exception as e:
        logger.error(f"Failed to save video: {str(e)}")
        return f"Failed to save video: {str(e)}", 500

    # Extract frames using FFmpeg
    try:
        logger.debug("Extracting frames")
        process = subprocess.Popen([
            'ffmpeg', '-i', video_path, '-r', '2', '-vf', 'scale=1280:720',
            os.path.join(images_dir, 'frame_%04d.jpg')
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Frame extraction failed: {stderr}")
            return f"Frame extraction failed: {stderr}", 500
        logger.debug(f"Frame extraction output: {stdout}")
    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}")
        return f"Frame extraction failed: {str(e)}", 500

    # COLMAP processing pipeline
    try:
        # 1. Create database
        logger.debug("Creating database")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'database_creator',
            '--database_path', database_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Database creation failed: {stderr}")
            return f"Database creation failed: {stderr}", 500
        logger.debug(f"Database creation output: {stdout}")

        # 2. Feature extraction
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
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Feature extraction failed: {stderr}")
            return f"Feature extraction failed: {stderr}", 500
        logger.debug(f"Feature extraction output: {stdout}")

        # 3. Feature matching
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
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Feature matching failed: {stderr}")
            return f"Feature matching failed: {stderr}", 500
        logger.debug(f"Feature matching output: {stdout}")

        # 4. Sparse reconstruction
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
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Sparse reconstruction failed: {stderr}")
            return f"Sparse reconstruction failed: {stderr}", 500
        logger.debug(f"Sparse reconstruction output: {stdout}")

        # Verify sparse model
        sparse_model_dir = os.path.join(sparse_dir, '0')
        if not os.path.exists(sparse_model_dir):
            logger.error("Sparse model not found")
            return "Sparse reconstruction failed", 500

        # 5. Dense reconstruction
        logger.debug("Running dense reconstruction")
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'image_undistorter',
            '--image_path', images_dir,
            '--input_path', sparse_model_dir,
            '--output_path', dense_dir,
            '--output_type', 'COLMAP',
            '--max_image_size', '2000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Image undistortion failed: {stderr}")
            return f"Image undistortion failed: {stderr}", 500
        logger.debug(f"Image undistortion output: {stdout}")

        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'patch_match_stereo',
            '--workspace_path', dense_dir,
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.gpu_index', '0',
            '--PatchMatchStereo.max_image_size', '2000',
            '--PatchMatchStereo.window_radius', '5',
            '--PatchMatchStereo.num_samples', '15',
            '--PatchMatchStereo.num_iterations', '5'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Patch match stereo failed: {stderr}")
            return f"Patch match stereo failed: {stderr}", 500
        logger.debug(f"Patch match stereo output: {stdout}")

        output_dense_ply = os.path.join(dense_dir, 'fused.ply')
        process = subprocess.Popen([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'stereo_fusion',
            '--workspace_path', dense_dir,
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', output_dense_ply,
            '--StereoFusion.min_num_pixels', '5'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Stereo fusion failed: {stderr}")
            return f"Stereo fusion failed: {stderr}", 500
        logger.debug(f"Stereo fusion output: {stdout}")

        if not os.path.exists(output_dense_ply):
            logger.error("Dense point cloud file not found")
            return "Dense reconstruction failed", 500

        # 6. Export camera poses
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
            return f"Model converter failed: {stderr}", 500
        logger.debug(f"Model converter output: {stdout}")

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
            return f"Failed to parse camera poses: {str(e)}", 500

        # 7. Export sparse point cloud
        logger.debug("Exporting sparse point cloud")
        output_sparse_ply = os.path.join(base_dir, 'sparse.ply')
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
            return f"Sparse point cloud export failed: {stderr}", 500
        logger.debug(f"Point cloud export output: {stdout}")

        # Create zip archive
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

        # Return JSON response
        response = {
            'status': 'success',
            'message': 'Processing complete',
            'sparse_ply_path': f'/output/{request_id}/sparse.ply',
            'dense_ply_path': f'/output/{request_id}/dense.ply',
            'poses_path': f'/output/{request_id}/camera_poses.json',
            'zip_path': f'/output/{request_id}/reconstruction_bundle.zip'
        }, 200

        # Clean up after response
        def cleanup():
            for attempt in range(3):
                try:
                    terminate_child_processes()
                    shutil.rmtree(base_dir)
                    logger.debug(f"Post-response cleanup: Successfully removed {base_dir}")
                    break
                except OSError as e:
                    logger.warning(f"Post-response cleanup attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            else:
                logger.error(f"Post-response cleanup failed for {base_dir}")

        # Schedule cleanup after response is sent
        from threading import Thread
        cleanup_thread = Thread(target=cleanup)
        cleanup_thread.start()

        return response

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}", 500
    finally:
        terminate_child_processes()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)