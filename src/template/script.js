let historyStack = [];
let currentResults = [];
function displayResults(imagePaths) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';
    if (imagePaths && imagePaths.length > 0) {
        imagePaths.forEach(path => {
            const container = document.createElement('div');
            container.classList.add('image-container');
            const img = document.createElement('img');
            img.src = path;
            img.onclick = () => handleImageClick(path);
            const info = document.createElement('p');
            info.classList.add('image-info');
            info.textContent = path.split('/').slice(-2).join('/');
            container.appendChild(img);
            container.appendChild(info);
            resultsDiv.appendChild(container);
        });
    } else {
        resultsDiv.innerHTML = '<p>No images found.</p>';
    }
}
async function getImageIndex(selectedImagePath) {
    try {
        const response = await fetch('./data/dicts/keyframes_id_search.json');
        const imagePaths = await response.json();
        selectedImagePath = selectedImagePath.split('/').slice(-4).join('/');
        const index = imagePaths.indexOf(selectedImagePath);
        return index;
    } catch (error) {
        console.error('Error fetching or processing JSON file:', error);
        return -1;
    }
}
async function handleImageClick(path) {
    try {
        const index = await getImageIndex(path);
        const k = document.getElementById('k').value;
        if (index !== -1) {
            const response = await fetch('/image_click/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index, k })
            });
            const data = await response.json();
            if (data.image_paths) {
                historyStack.push([...currentResults]);
                currentResults = data.image_paths;
                displayResults(data.image_paths);
                document.getElementById('back-button').style.display = 'block';
            } else {
                alert('No images found.');
            }
        } else {
            alert('Image not found in the list.');
        }
    } catch (error) {
        alert('Failed to handle image click: ' + error.message);
    }
}

document.getElementById('load-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const selectedFiles = Array.from(document.querySelectorAll('input[name="bin_files"]:checked')).map(cb => cb.value);
    const rerankFile = document.querySelector('input[name="rerank_file"]:checked')?.value || null;
    try {
        const response = await fetch('/load_index/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bin_files: selectedFiles, rerank_file: rerankFile })
        });
        const data = await response.json();
        alert(data.status);
    } catch (error) {
        alert("Failed to load index: " + error.message);
    }
});

document.getElementById('search-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = document.getElementById('text').value;
    const k = document.getElementById('k').value;
    try {
        const response = await fetch('/search/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, k })
        });
        const data = await response.json();
        if (currentResults.length > 0) {
            historyStack.push([...currentResults]);
        }
        currentResults = data.image_paths;
        displayResults(data.image_paths);
        document.getElementById('back-button').style.display = 'block';
    } catch (error) {
        alert("Search failed: " + error.message);
    }
});
document.getElementById('back-button').addEventListener('click', () => {
    if (historyStack.length > 0) {
        currentResults = historyStack.pop();
        displayResults(currentResults);
    }
    if (historyStack.length === 0) {
        document.getElementById('back-button').style.display = 'none';
    }
});