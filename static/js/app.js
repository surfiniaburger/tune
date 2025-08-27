document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("image-input");
    const imagePreview = document.getElementById("image-preview");
    const runButton = document.getElementById("run-button");
    const statusDiv = document.getElementById("status");
    const resultsDiv = document.getElementById("results");
    const diagnosisText = document.getElementById("diagnosis-text");
    const audioPlayer = document.getElementById("audio-player");

    let selectedFile = null;
    let ws = null;

    imageInput.addEventListener("change", () => {
        selectedFile = imageInput.files[0];
        if (selectedFile) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = "block";
            };
            reader.readAsDataURL(selectedFile);
            runButton.disabled = false;
            statusDiv.textContent = "Image selected. Ready to diagnose.";
        }
    });

    runButton.addEventListener("click", () => {
        if (!selectedFile) {
            statusDiv.textContent = "Please select an image first.";
            return;
        }

        runButton.disabled = true;
        statusDiv.textContent = "Connecting to server...";
        resultsDiv.style.display = "none";

        // Start WebSocket connection
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            statusDiv.textContent = "Connection open. Reading image file...";
            const reader = new FileReader();
            reader.onload = (e) => {
                // e.target.result is a data URL like "data:image/jpeg;base64,..."
                // We need to extract just the base64 part.
                const base64Data = e.target.result.split(",")[1];
                const message = {
                    mime_type: selectedFile.type,
                    data: base64Data,
                };
                ws.send(JSON.stringify(message));
                statusDiv.textContent = "Image sent to server. Awaiting diagnosis...";
            };
            reader.readAsDataURL(selectedFile);
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                console.log("Received message:", message);

                if (message.status) {
                    statusDiv.textContent = message.message;
                } else if (message.error) {
                    statusDiv.textContent = `Error: ${message.error}`;
                    runButton.disabled = false;
                } else if (message.diagnosis && message.audio_url) {
                    // This is the final result
                    diagnosisText.textContent = message.diagnosis;
                    audioPlayer.src = message.audio_url;
                    resultsDiv.style.display = "block";
                    statusDiv.textContent = "Diagnosis complete!";
                    runButton.disabled = false;
                    ws.close();
                }
            } catch (error) {
                console.error("Error parsing message from server:", error);
                statusDiv.textContent = "Received an invalid message from the server.";
            }
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed.");
            statusDiv.textContent = "Connection closed. Ready for next diagnosis.";
            runButton.disabled = false;
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            statusDiv.textContent = "Connection error. Please try again.";
            runButton.disabled = false;
        };
    });
});
