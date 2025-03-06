async function processImage() {
    let imageInput = document.getElementById("imageInput").files[0];
    let formData = new FormData();
    formData.append("data", imageInput);

    let response = await fetch("/gradio/api/predict/", {
        method: "POST",
        body: formData
    });

    let result = await response.json();
    document.getElementById("maskImage").src = result["data"][0];  
    document.getElementById("heatmapImage").src = result["data"][1];  
}
