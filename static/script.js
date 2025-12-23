document.getElementById("predictForm").addEventListener("submit", function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById("result");

        if (data.prediction === 1) {
            resultDiv.innerHTML = "⚠️ Disease Detected";
            resultDiv.style.color = "red";
        } else {
            resultDiv.innerHTML = "✅ No Disease Detected";
            resultDiv.style.color = "green";
        }
    })
    .catch(error => {
        console.error(error);
        document.getElementById("result").innerHTML = "❌ Error predicting result";
    });
});
