const dropZone = document.getElementById("drop_zone");
const trainBtn = document.getElementById("train_btn");
const resultBtn = document.getElementById("result_btn");
const statusDiv = document.getElementById("status");

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#3b82f6";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "#aaa";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#aaa";

  const file = e.dataTransfer.files[0];
  if (!file.name.endsWith(".xlsx")) {
    alert("Please upload an Excel (.xlsx) file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  statusDiv.innerText = "Uploading file...";

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((res) => res.text())
    .then((msg) => {
      statusDiv.innerText = msg;
      trainBtn.disabled = false;
    })
    .catch((err) => {
      statusDiv.innerText = "Upload failed: " + err;
    });
});

trainBtn.addEventListener("click", () => {
  statusDiv.innerText = "Training in progress...";
  fetch("/train")
    .then((res) => {
    if (!res.ok) throw new Error("HTTP " + res.status);
    return res.text();
  })
    .then((msg) => {
    statusDiv.innerText = msg;
    resultBtn.disabled = false;
  })
    .catch((err) => {
    statusDiv.innerText = "Training failed: " + err;
  });
});
