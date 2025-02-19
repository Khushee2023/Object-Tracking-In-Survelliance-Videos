// DOM Elements
const uploadInput = document.getElementById('upload-input');
const previewImage = document.getElementById('preview-image');
const processedImage = document.getElementById('processed-image');

// Buttons
const viewHistogramBtn = document.getElementById('view-histogram');
const viewRGBChannelsBtn = document.getElementById('rgb-channels');
const applyBlurBtn = document.getElementById('apply-blur');
const clearFormBtn = document.getElementById('clear-form');

// Image Preview Function
uploadInput.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// Function to Send Image for Processing
function processImage(action) {
    const file = uploadInput.files[0];
    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('action', action);

    fetch('/model1/process_image', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        processedImage.src = url;
        processedImage.style.display = 'block';
    })
    .catch(err => console.error('Error:', err));
}

// Event Listeners
viewHistogramBtn.addEventListener('click', () => processImage('histogram'));
viewRGBChannelsBtn.addEventListener('click', () => processImage('rgb_channels'));
applyBlurBtn.addEventListener('click', () => processImage('blur'));

// Clear Form
clearFormBtn.addEventListener('click', () => {
    uploadInput.value = '';
    previewImage.style.display = 'none';
    processedImage.style.display = 'none';
});
