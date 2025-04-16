// Handle form submission and display predicted installation time
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("multiStepForm");
  if (!form) return;

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    // Example prediction: you can replace this with real model inference result
    const predicted = 10;

    const result = document.getElementById("predictionResult");
    result.innerText = `This Solar Panel Installation will take ${predicted} to ${predicted + 10} Hours to Complete`;
    result.style.display = "block";
  });
});
