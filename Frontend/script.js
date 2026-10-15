const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadPrompt = document.getElementById('uploadPrompt');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const loader = document.getElementById('loader');
const resultCard = document.getElementById('resultCard');
const predictionClass = document.getElementById('predictionClass');
const predictionConfidence = document.getElementById('predictionConfidence');

let selectedFile = null;

// Trigger file browser on click
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadPrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        predictBtn.disabled = false;
        resultCard.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.classList.add('hidden');
    uploadPrompt.classList.remove('hidden');
    predictBtn.disabled = true;
    resultCard.classList.add('hidden');
});

// Handle Prediction Request to Backend
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);

    // Update UI states
    predictBtn.classList.add('hidden');
    loader.classList.remove('hidden');
    resultCard.classList.add('hidden');

    try {
        // Change URL to match your backend API endpoint later
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to get prediction from server.');
        }

        const data = await response.json();

        // Display results
        predictionClass.textContent = data.class_name;
        predictionConfidence.textContent = (data.confidence * 100).toFixed(2) + '%';
        
        loader.classList.add('hidden');
        resultCard.classList.remove('hidden');
        predictBtn.classList.remove('hidden');

    } catch (error) {
        console.error(error);
        alert('Error connecting to backend server. Make sure Flask is running.');
        loader.classList.add('hidden');
        predictBtn.classList.remove('hidden');
    }
});