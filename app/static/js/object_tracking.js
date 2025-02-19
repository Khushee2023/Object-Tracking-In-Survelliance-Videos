document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        form: document.getElementById('uploadForm'),
        videoInput: document.getElementById('videoFile'),
        progressBar: document.getElementById('progressBar'),
        progressText: document.getElementById('progressText'),
        downloadLink: document.getElementById('downloadLink'),
        logOutput: document.getElementById('logOutput'),
        clearButton: document.getElementById('clearButton'),
        confidenceThreshold: document.getElementById('confidenceThreshold'),
        confidenceValue: document.getElementById('confidenceValue'),
        nmsThreshold: document.getElementById('nmsThreshold'),
        nmsValue: document.getElementById('nmsValue')
    };

    // Update threshold values display
    elements.confidenceThreshold.addEventListener('input', (e) => {
        elements.confidenceValue.textContent = (e.target.value / 100).toFixed(2);
    });

    elements.nmsThreshold.addEventListener('input', (e) => {
        elements.nmsValue.textContent = (e.target.value / 100).toFixed(2);
    });

    elements.form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData();
        const videoFile = elements.videoInput.files[0];

        if (!videoFile) {
            showMessage('Please select a video file', 'error');
            return;
        }

        formData.append('videoFile', videoFile);
        formData.append('confidenceThreshold', elements.confidenceThreshold.value / 100);
        formData.append('nmsThreshold', elements.nmsThreshold.value / 100);

        try {
            showMessage('Uploading video...', 'info');
            elements.progressBar.style.width = '0%';

            const response = await fetch('/model3/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');
            
            const data = await response.json();
            if (!data.success) throw new Error(data.message);

            await monitorProgress(data.taskId);
        } catch (error) {
            showMessage(`Error: ${error.message}`, 'error');
        }
    });

    async function monitorProgress(taskId) {
        const checkStatus = setInterval(async () => {
            try {
                const response = await fetch(`/model3/status/${taskId}`);
                const status = await response.json();

                if (status.error) {
                    clearInterval(checkStatus);
                    showMessage(`Error: ${status.error}`, 'error');
                    return;
                }

                elements.progressBar.style.width = `${status.progress}%`;
                elements.progressText.textContent = `Processing: ${status.progress}%`;

                if (status.complete) {
                    clearInterval(checkStatus);
                    showMessage('Processing complete!', 'success');
                    elements.downloadLink.href = `/model3/output/${status.outputFile}`;
                    elements.downloadLink.style.display = 'block';
                }
            } catch (error) {
                clearInterval(checkStatus);
                showMessage(`Error checking status: ${error.message}`, 'error');
            }
        }, 1000);
    }

    function showMessage(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const formattedMessage = `[${timestamp}] ${message}`;
        elements.progressText.textContent = message;
        elements.logOutput.textContent += `\n${formattedMessage}`;
        elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
    }

    elements.clearButton.addEventListener('click', () => {
        elements.form.reset();
        elements.progressText.textContent = 'Waiting for upload...';
        elements.progressBar.style.width = '0%';
        elements.downloadLink.style.display = 'none';
        elements.logOutput.textContent = '';
        elements.confidenceValue.textContent = '0.50';
        elements.nmsValue.textContent = '0.40';
    });

    // File drag and drop handling
    const uploadArea = document.querySelector('.upload-area');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('highlight');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('highlight');
        });
    });

    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        elements.videoInput.files = dt.files;
    });
});