document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        form: document.getElementById('uploadForm'),
        videoInput: document.getElementById('videoFile'),
        progressBar: document.getElementById('progressBar'),
        progressText: document.getElementById('progressText'),
        downloadLink: document.getElementById('downloadLink'),
        logOutput: document.getElementById('logOutput'),
        clearButton: document.getElementById('clearButton')
    };

    elements.form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData();
        const videoFile = elements.videoInput.files[0];

        if (!videoFile) {
            showMessage('Please select a video file', 'error');
            return;
        }

        formData.append('videoFile', videoFile);

        try {
            showMessage('Uploading video...', 'info');
            elements.progressBar.style.width = '0%';

            const response = await fetch('/model5/upload', {
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
                const response = await fetch(`/model5/status/${taskId}`);
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
                    elements.downloadLink.href = `/model5/output/${status.outputFile}`;
                    elements.downloadLink.style.display = 'block';
                }
            } catch (error) {
                clearInterval(checkStatus);
                showMessage(`Error checking status: ${error.message}`, 'error');
            }
        }, 1000);
    }

    function showMessage(message, type = 'info') {
        elements.progressText.textContent = message;
        elements.logOutput.textContent += `\n${message}`;
        elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
    }

    elements.clearButton.addEventListener('click', () => {
        elements.form.reset();
        elements.progressText.textContent = 'Waiting for upload...';
        elements.progressBar.style.width = '0%';
        elements.downloadLink.style.display = 'none';
        elements.logOutput.textContent = '';
    });
});
