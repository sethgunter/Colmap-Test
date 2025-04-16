from flask import Flask, request, send_file, Response
import subprocess
import os
import shutil
import logging
import json
import zipfile
import io

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.after_request
def add_security_headers(response: Response):
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
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
    return send_file(os.path.join('/app/colmap_project', path))

@app.route('/process-video', methods=['POST'])
def process_video():
    logger.debug("Received POST request to /process-video")
    if 'frames' not in request.files:
        logger.error("No frames provided in request")
        return "No frames provided", 400

    # Set up working directories
    base_dir = '/app/colmap_project'
    images_dir = os.path.join(base_dir, 'images')
    database_path = os.path.join(base_dir, 'database.db')
    sparse_dir = os.path.join(base_dir, 'sparse')
    poses_dir = os.path.join(base_dir, 'poses')

    # Clean up previous runs
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(images_dir)
    os.makedirs(sparse_dir)
    os.makedirs(poses_dir, exist_ok=True)

    # Save uploaded frames
    frames = request.files.getlist('frames')
    logger.debug(f"Received {len(frames)} frames")
    if not frames:
        logger.error("No frames found in request.files.getlist('frames')")
        return "No frames uploaded", 400

    for frame in frames:
        frame_path = os.path.join(images_dir, frame.filename)
        logger.debug(f"Saving frame: {frame_path}")
        try:
            frame.save(frame_path)
        except Exception as e:
            logger.error(f"Failed to save frame {frame.filename}: {str(e)}")
            return f"Failed to save frame {frame.filename}: {str(e)}", 500

    # COLMAP processing pipeline with SphereSfM parameters
    try:
        # 1. Create database
        logger.debug("Creating database")
        result = subprocess.run([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'database_creator',
            '--database_path', database_path
        ], check=True, capture_output=True, text=True)
        logger.debug(f"Database creation output: {result.stdout}")

        # 2. Feature extraction
        logger.debug("Running feature extraction")
        result = subprocess.run([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'feature_extractor',
            '--database_path', database_path,
            '--image_path', images_dir,
            '--ImageReader.camera_model', 'SPHERE',
            '--ImageReader.camera_params', '1,960,480',  # focal_length=1, cx=960, cy=480 for 1920x960
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '0',
            '--SiftExtraction.peak_threshold', '0.0001',
            '--SiftExtraction.max_num_features', '11000',
            '--SiftExtraction.estimate_affine_shape', '1',
            '--SiftExtraction.max_num_orientations', '3'
        ], check=True, capture_output=True, text=True)
        logger.debug(f"Feature extraction output: {result.stdout}")

        # 3. Feature matching
        logger.debug("Running feature matching")
        result = subprocess.run([
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
            '--SiftMatching.use_gpu', '0',
            '--SiftMatching.min_num_inliers', '30'
        ], check=True, capture_output=True, text=True)
        logger.debug(f"Feature matching output: {result.stdout}")

        # 4. Sparse reconstruction
        logger.debug("Running sparse reconstruction")
        result = subprocess.run([
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
        ], check=True, capture_output=True, text=True)
        logger.debug(f"Sparse reconstruction output: {result.stdout}")

        # Verify sparse model exists
        sparse_model_dir = os.path.join(sparse_dir, '0')
        if not os.path.exists(sparse_model_dir):
            logger.error("Sparse model not found")
            return "Sparse reconstruction failed, no model generated", 500

        # 5. Export camera poses
        logger.debug("Exporting camera poses")
        result = subprocess.run([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'model_converter',
            '--input_path', sparse_model_dir,
            '--output_path', poses_dir,
            '--output_type', 'TXT'
        ], check=True, capture_output=True, text=True)
        logger.debug(f"Model converter output: {result.stdout}")

        # Parse images.txt to create camera_poses.json
        poses_json_path = os.path.join(base_dir, 'camera_poses.json')
        try:
            with open(os.path.join(poses_dir, 'images.txt')) as f:
                lines = f.readlines()[4::2]  # Skip header, take image lines
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

        # 6. Export sparse point cloud as PLY
        logger.debug("Exporting sparse point cloud")
        output_ply = os.path.join(base_dir, 'sparse.ply')
        result = subprocess.run([
            'xvfb-run', '--auto-servernum', '--server-args', '-screen 0 1024x768x24',
            'colmap', 'model_converter',
            '--input_path', sparse_model_dir,
            '--output_path', output_ply,
            '--output_type', 'PLY'
        ], check=True, capture_output=True, text=True)
        logger.debug(f"Point cloud export output: {result.stdout}")

        if not os.path.exists(output_ply):
            logger.error("Point cloud file not found")
            return "Point cloud export failed", 500

        # Create a proper zip archive
        logger.debug("Creating zip archive")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if os.path.exists(output_ply):
                zip_file.write(output_ply, 'sparse.ply')
            else:
                logger.warning("sparse.ply not found, skipping")
            if os.path.exists(poses_json_path):
                zip_file.write(poses_json_path, 'camera_poses.json')
            else:
                logger.warning("camera_poses.json not found, skipping")

        zip_buffer.seek(0)
        # Save zip_buffer to a temporary file for response
        zip_temp_path = os.path.join(base_dir, 'reconstruction_bundle.zip')
        with open(zip_temp_path, 'wb') as f:
            f.write(zip_buffer.getvalue())

        # Return JSON response with paths for webview
        return {
            'status': 'success',
            'message': 'Processing complete',
            'ply_path': '/output/sparse.ply',
            'poses_path': '/output/camera_poses.json',
            'zip_path': '/output/reconstruction_bundle.zip'
        }, 200

    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP command failed: stdout={e.stdout}, stderr={e.stderr}")
        return f"COLMAP processing failed: {e.stderr}", 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)