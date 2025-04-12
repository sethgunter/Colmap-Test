const { createFFmpeg, fetchFile } = FFmpeg;

const videoInput = document.getElementById('videoInput');
const processButton = document.getElementById('processButton');
const message = document.getElementById('message');

let ffmpeg;
let ffmpegReady = false;

async function loadFFmpeg() {
    ffmpeg = createFFmpeg({ log: true });
    message.textContent = 'Loading FFmpeg...';
    try {
        await ffmpeg.load();
        ffmpegReady = true;
        processButton.disabled = false;
        message.textContent = 'FFmpeg loaded. Select a video to begin.';
    } catch (error) {
        message.textContent = `Failed to load FFmpeg: ${error.message}`;
    }
}

async function processVideo() {
    if (!ffmpegReady) {
        message.textContent = 'FFmpeg is not ready yet. Please wait.';
        return;
    }

    const videoFile = videoInput.files[0];
    if (!videoFile) {
        message.textContent = 'Please select a video file first.';
        return;
    }

    processButton.disabled = true;
    message.textContent = 'Extracting frames...';

    try {
        ffmpeg.FS('writeFile', 'input.mp4', await fetchFile(videoFile));
        await ffmpeg.run('-i', 'input.mp4', '-r', '1.5', '-vf', 'scale=1920:960', 'frame_%04d.jpg');

        const frames = [];
        const files = ffmpeg.FS('readdir', '/');
        for (const file of files) {
            if (file.startsWith('frame_')) {
                const data = ffmpeg.FS('readFile', file);
                frames.push(new File([data.buffer], file, { type: 'image/jpeg' }));
            }
        }

        message.textContent = 'Uploading frames...';

        const formData = new FormData();
        frames.forEach((frame) => {
            formData.append('frames', frame);
        });

        const response = await fetch('/process-video', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'reconstruction_bundle.zip';
            a.click();
            window.URL.revokeObjectURL(url);
            message.textContent = 'Processing complete! File downloaded.';
        } else {
            const errorText = await response.text();
            message.textContent = `Upload failed: ${errorText}`;
        }
    } catch (error) {
        message.textContent = `Error: ${error.message}`;
    } finally {
        processButton.disabled = false;
    }
}

processButton.disabled = true;
loadFFmpeg();
processButton.addEventListener('click', processVideo);