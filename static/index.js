const { createFFmpeg, fetchFile } = FFmpeg;

const videoInput = document.getElementById('videoInput');
const processButton = document.getElementById('processButton');
const message = document.getElementById('message');
const showModelButton = document.getElementById('showModelButton');
const downloadButton = document.getElementById('downloadButton');
const container = document.getElementById('container');

let ffmpeg;
let ffmpegReady = false;
let pointCloudCenter;
let scalingFactor;

async function loadFFmpeg() {
    // Check for SharedArrayBuffer support
    if (typeof SharedArrayBuffer === 'undefined') {
        message.textContent = 'SharedArrayBuffer is not available. Ensure the page is served with cross-origin isolation headers (COOP: same-origin, COEP: require-corp).';
        console.error('SharedArrayBuffer is not supported. Check server headers.');
        return;
    }

    ffmpeg = createFFmpeg({
        log: true,
        corePath: 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.11.0/dist/ffmpeg-core.js'
    });
    message.textContent = 'Loading FFmpeg...';
    try {
        await ffmpeg.load();
        ffmpegReady = true;
        processButton.disabled = false;
        message.textContent = 'FFmpeg loaded. Select a video.';
    } catch (error) {
        console.error('FFmpeg load failed:', error);
        message.textContent = `FFmpeg error: ${error.message}. Check console for details.`;
    }
}

function initThreeJS(plyPath, posesPath) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    let sceneCenter = new THREE.Vector3(0, 0, 0);
    let sceneExtent = 1;

    const group = new THREE.Group();
    scene.add(group);

    const loader = new THREE.PLYLoader();
    loader.load(plyPath, (geometry) => {
        geometry.computeVertexNormals();
        const material = new THREE.PointsMaterial({ size: 0.02, vertexColors: true });
        const points = new THREE.Points(geometry, material);

        const positions = geometry.attributes.position.array;
        const coords = { x: [], y: [], z: [] };
        for (let i = 0; i < positions.length; i += 3) {
            coords.x.push(positions[i]);
            coords.y.push(positions[i + 1]);
            coords.z.push(positions[i + 2]);
        }
        const median = (arr) => {
            const sorted = [...arr].sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        };
        sceneCenter.set(median(coords.x), median(coords.y), median(coords.z));
        points.position.sub(sceneCenter);

        group.add(points);

        const iqr = (arr) => {
            const sorted = [...arr].sort((a, b) => a - b);
            const q1 = sorted[Math.floor(sorted.length * 0.25)];
            const q3 = sorted[Math.floor(sorted.length * 0.75)];
            return q3 - q1;
        };
        sceneExtent = Math.max(iqr(coords.x), iqr(coords.y), iqr(coords.z)) * 1.5;

        console.log('Scene center:', sceneCenter);
        console.log('Scene extent:', sceneExtent);

        fetch(posesPath)
            .then(response => response.json())
            .then(poses => {
                const coneGeometry = new THREE.ConeGeometry(0.05, 0.2, 8);
                const coneMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                const sphereGeometry = new THREE.SphereGeometry(sceneExtent * 0.01, 8, 8);
                const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });

                for (const [_, pose] of Object.entries(poses)) {
                    const q = new THREE.Quaternion(pose.qx, pose.qy, pose.qz, pose.qw);
                    const qInv = q.clone().conjugate();
                    const t = new THREE.Vector3(pose.tx, pose.ty, pose.tz);
                    const tInv = t.clone().applyQuaternion(qInv).negate();
                    tInv.sub(sceneCenter);

                    const cone = new THREE.Mesh(coneGeometry, coneMaterial);
                    cone.position.copy(tInv);
                    cone.setRotationFromQuaternion(qInv);
                    cone.rotateX(Math.PI / 2);
                    group.add(cone);

                    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
                    sphere.position.copy(tInv);
                    group.add(sphere);
                }

                group.rotation.x = Math.PI;

                camera.position.set(0, 0, sceneExtent * 2);
                camera.up.set(0, 1, 0);
                camera.lookAt(0, 0, 0);
            })
            .catch(error => console.error('Poses load error:', error));
    }, undefined, (error) => console.error('PLY load error:', error));

    scene.background = new THREE.Color(0x000000);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 1);
    scene.add(directionalLight);

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

async function processVideo() {
    if (!ffmpegReady) {
        message.textContent = 'FFmpeg not ready. Please wait or refresh.';
        return;
    }
    const videoFile = videoInput.files[0];
    if (!videoFile) {
        message.textContent = 'Please select a video file.';
        return;
    }

    processButton.disabled = true;
    message.textContent = 'Extracting frames...';
    showModelButton.style.display = 'none';
    downloadButton.style.display = 'none';
    container.style.display = 'none';

    try {
        ffmpeg.FS('writeFile', 'input.mp4', await fetchFile(videoFile));
        await ffmpeg.run('-i', 'input.mp4', '-r', '2', '-vf', 'scale=1920:960', 'frame_%04d.jpg');

        const frames = [];
        const files = ffmpeg.FS('readdir', '/');
        for (const file of files) {
            if (file.startsWith('frame_')) {
                const data = ffmpeg.FS('readFile', file);
                frames.push(new File([data.buffer], file, { type: 'image/jpeg' }));
            }
        }

        if (frames.length === 0) {
            throw new Error('No frames extracted.');
        }

        message.textContent = 'Uploading frames...';
        const formData = new FormData();
        frames.forEach(frame => formData.append('frames', frame));

        const response = await fetch('/process-video', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${await response.text()}`);
        }

        const result = await response.json();
        message.textContent = result.message;
        showModelButton.style.display = 'block';
        downloadButton.style.display = 'block';

        showModelButton.onclick = () => {
            container.style.display = 'block';
            initThreeJS(result.dense_ply_path, result.poses_path); // Use dense.ply
            showModelButton.style.display = 'none';
            message.textContent = 'Model displayed. Rotate, zoom, pan with mouse.';
        };

        downloadButton.onclick = () => {
            const zipLink = document.createElement('a');
            zipLink.href = result.zip_path;
            zipLink.download = 'reconstruction_bundle.zip';
            zipLink.click();
            message.textContent = 'Download started.';
        };
    } catch (error) {
        console.error('Processing error:', error);
        message.textContent = `Error: ${error.message}`;
    } finally {
        processButton.disabled = false;
    }
}

processButton.disabled = true;
loadFFmpeg();
processButton.addEventListener('click', processVideo);