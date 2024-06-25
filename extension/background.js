chrome.runtime.onInstalled.addListener(() => {
	chrome.contextMenus.create({
		id: "predict",
		title: "Predict news type",
		contexts: ["selection"]
	});
});


chrome.contextMenus.onClicked.addListener((info, tab) => {
	if (info.menuItemId === "predict" && info.selectionText) {
		chrome.scripting.executeScript({
			target: { tabId: tab.id },
			function: showPrediction,
			args: [info.selectionText]
		});
	}
});


function showPrediction(selectedText) {
	fetch('http://localhost:5000/predict', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ text: selectedText })
	})
	.then(response => response.json())
	.then(data => {
		const predictionBox = document.createElement('div');
		predictionBox.id = 'prediction-box';

		predictionBox.textContent = ` Selected news type is ${data.prediction} `;

		let borderColor = '#ff0000';
		if (data.prediction === 'reliable') {
			borderColor = '#33cc33';
		} else if (data.prediction === 'unknown') {
			borderColor = '#666699';
		}

		predictionBox.style.width = 'fit-content';
		predictionBox.style.height = 'fit-content';
        predictionBox.style.position = 'absolute';
        predictionBox.style.zIndex = '9999';
        predictionBox.style.backgroundColor = '#ffffff';
		predictionBox.style.fontSize = '20px';
		predictionBox.style.color = '#000000';
		predictionBox.style.border = '3px solid ' + borderColor;
        predictionBox.style.padding = '10px';
        predictionBox.style.borderRadius = '10px';
        predictionBox.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';

        const range = window.getSelection().getRangeAt(0);
        const rect = range.getBoundingClientRect();
        predictionBox.style.top = `${rect.bottom + window.scrollY}px`;
        predictionBox.style.left = `${rect.left + window.scrollX}px`;

        document.body.appendChild(predictionBox);

		document.addEventListener('selectionchange', () => {
            if (document.getElementById('prediction-box')) {
                document.getElementById('prediction-box').remove();
            }
        });
	})
	.catch(error => {
		console.error('Error:', error);
	});
}

