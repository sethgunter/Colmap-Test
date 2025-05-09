const videoInput = document.getElementById('videoInput');
const imagesInput = document.getElementById('imagesInput');
const processButton = document.getElementById('processButton');
const confirmButton = document.getElementById('confirmButton');
const message = document.getElementById('message');
const viewDenseButton = document.getElementById('viewDenseButton');
const viewSparseButton = document.getElementById('viewSparseButton');
const downloadButton = document.getElementById('downloadButton');
const container = document.getElementById('container');
const progressBar = document.getElementById('progressBar');
const progressContainer = document.getElementById('progressContainer');
const timerDisplay = document.getElementById('timer');

async function processInput() {
    const videoFile = videoInput.files[0];
    const imageFiles = imagesInput.files;

    // Validate inputs
    if (!videoFile && (!imageFiles || imageFiles.length === 0)) {
        message.textContent = 'Please select a video or a folder of images.';
        return;
    }
    if (videoFile && imageFiles.length > 0) {
        message.textContent = 'Please select either a video or images, not both.';
        return;
    }

    if (!timerDisplay) {
        console.error('Timer display element not found');
        message.textContent = 'Error: Timer element missing';
        return;
    }

    processButton.disabled = true;
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    message.textContent = videoFile ? 'Uploading video...' : 'Uploading images...';
    timerDisplay.textContent = 'Processing time: 00:00';

    let timerInterval = null;
    let sessionId = null;

    try {
        let result = null;
        if (videoFile) {
            const formData = new FormData();
            formData.append('video', videoFile);
            formData.append('complete', 'true');
            result = await uploadData(formData, 1, 1);
            sessionId = result.session_id;
        } else {
            const sortedImageFiles = Array.from(imageFiles).sort((a, b) =>
                a.name.localeCompare(b.name, undefined, { numeric: true })
            );
            const chunkSize = 100;
            for (let i = 0; i < sortedImageFiles.length; i += chunkSize) {
                const chunk = sortedImageFiles.slice(i, i + chunkSize);
                const formData = new FormData();
                formData.append('session_id', sessionId || '');
                chunk.forEach((file, index) => {
                    formData.append('images', file, `frame_${(i + index).toString().padStart(4, '0')}.jpg`);
                });
                if (i + chunkSize >= sortedImageFiles.length) {
                    formData.append('complete', 'true');
                }
                message.textContent = `Uploading images ${i + 1} to ${Math.min(i + chunkSize, sortedImageFiles.length)}...`;
                const chunkResult = await uploadData(formData, sortedImageFiles.length, i + chunkSize);
                sessionId = chunkResult.session_id || sessionId;
                result = chunkResult.status === 'sparse_success' ? chunkResult : result;
            }
        }

        async function uploadData(formData, totalFiles, uploadedFiles) {
            const xhr = new XMLHttpRequest();
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = (e.loaded / e.total) * 100 * (uploadedFiles / totalFiles);
                    progressBar.style.width = `${percent}%`;
                    message.textContent = `Uploading: ${Math.round(percent)}%`;
                }
            });

            xhr.open('POST', '/process-video');
            xhr.responseType = 'json';
            return new Promise((resolve, reject) => {
                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        console.log('Raw response:', xhr.response);
                        resolve(xhr.response);
                    } else {
                        reject(new Error(`Server responded with status ${xhr.status}: ${xhr.response?.message || xhr.statusText}`));
                    }
                };
                xhr.onerror = () => reject(new Error('Network error during upload'));
                xhr.send(formData);
            });
        }

        if (!result || result.status !== 'sparse_success') {
            throw new Error(result?.message || 'Sparse reconstruction failed');
        }

        console.log('Sparse reconstruction response:', result);
        const inputSaveTime = result.input_save_time * 1000;
        if (!inputSaveTime) {
            console.error('Input save time not provided by server');
            throw new Error('Server did not provide input save time');
        }

        timerInterval = setInterval(() => {
            const elapsedMs = Date.now() - inputSaveTime;
            const elapsed = Math.floor(elapsedMs / 1000);
            const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `Processing time: ${minutes}:${seconds}`;
        }, 1000);

        message.textContent = 'Sparse reconstruction complete. Click Confirm to proceed with dense processing.';
        progressBar.style.width = '50%';
        viewSparseButton.style.display = 'block';
        confirmButton.style.display = 'block';

        viewSparseButton.onclick = () => {
            container.style.display = 'block';
            initThreeJS(result.sparse_ply_path, result.poses_path);
            message.textContent = 'Sparse model displayed. Click Confirm to proceed with dense processing.';
        };

        confirmButton.onclick = async () => {
            confirmButton.disabled = true;
            message.textContent = 'Processing dense reconstruction...';
            progressBar.style.width = '75%';

            try {
                const formData = new FormData();
                formData.append('session_id', sessionId);
                const denseResult = await fetch('/process-dense', {
                    method: 'POST',
                    body: formData
                }).then(response => {
                    if (!response.ok) {
                        throw new Error(`Dense processing failed: ${response.statusText}`);
                    }
                    return response.json();
                });

                console.log('Dense reconstruction response:', denseResult);
                if (denseResult.status !== 'success') {
                    throw new Error(denseResult.message || 'Dense reconstruction failed');
                }

                message.textContent = 'Dense reconstruction complete';
                progressBar.style.width = '100%';
                viewDenseButton.style.display = 'block';
                downloadButton.style.display = 'block';

                viewDenseButton.onclick = () => {
                    container.style.display = 'block';
                    initThreeJS(denseResult.dense_ply_path, denseResult.poses_path);
                    message.textContent = 'Dense model displayed';
                };

                downloadButton.onclick = () => {
                    const zipLink = document.createElement('a');
                    zipLink.href = denseResult.zip_path;
                    zipLink.download = 'reconstruction_bundle.zip';
                    zipLink.click();
                    message.textContent = 'Download started';
                };

                if (timerInterval) {
                    clearInterval(timerInterval);
                    const elapsedMs = Date.now() - inputSaveTime;
                    const elapsed = Math.floor(elapsedMs / 1000);
                    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
                    const seconds = (elapsed % 60).toString().padStart(2, '0');
                    timerDisplay.textContent = `Processing time: ${minutes}:${seconds}`;
                }
            } catch (error) {
                console.error('Dense processing error:', error);
                message.textContent = `Error: ${error.message}`;
                if (timerInterval) {
                    clearInterval(timerInterval);
                    timerDisplay.textContent = 'Processing time: stopped due to error';
                }
            } finally {
                confirmButton.disabled = false;
            }
        };
    } catch (error) {
        console.error('Sparse processing error:', error);
        message.textContent = `Error: ${error.message}`;
        if (timerInterval) {
            clearInterval(timerInterval);
            timerDisplay.textContent = 'Processing time: stopped due to error';
        }
    } finally {
        processButton.disabled = false;
        progressContainer.style.display = 'none';
    }
}

function initThreeJS(plyPath, posesPath) {
    console.log(`Initializing Three.js with PLY: ${plyPath}, Poses: ${posesPath}`);
    try {
        container.innerHTML = ''; // Clear previous scene
        container.style.display = 'block'; // Ensure container is visible

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
            console.log(`Loaded PLY with ${geometry.attributes.position.count} points`);
            if (geometry.attributes.position.count === 0) {
                console.warn('Point cloud is empty');
                message.textContent = 'Warning: Point cloud is empty';
                return;
            }

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

            fetch(posesPath)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Failed to fetch poses: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(poses => {
                    console.log(`Loaded ${Object.keys(poses).length} camera poses`);
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
                .catch(error => {
                    console.error(`Error loading poses: ${error.message}`);
                    message.textContent = `Error loading camera poses: ${error.message}`;
                });
        }, (progress) => {
            console.log(`Loading PLY: ${(progress.loaded / progress.total * 100).toFixed(2)}%`);
        }, (error) => {
            console.error(`Error loading PLY: ${error.message}`);
            message.textContent = `Error loading point cloud: ${error.message}`;
        });

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
    } catch (error) {
        console.error(`Error initializing Three.js: ${error.message}`);
        message.textContent = `Error initializing visualization: ${error.message}`;
    }
}

processButton.addEventListener('click', processInput);