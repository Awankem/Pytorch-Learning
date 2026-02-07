document.addEventListener('DOMContentLoaded', () => {
    const xInput = document.getElementById('x-input');
    const predictBtn = document.getElementById('predict-btn');
    const yResult = document.getElementById('y-result');
    const modelWeight = document.getElementById('model-weight');
    const modelBias = document.getElementById('model-bias');
    const formulaWeight = document.getElementById('formula-weight');
    const formulaBias = document.getElementById('formula-bias');

    let chart;

    // Initialize Chart
    const initChart = () => {
        const ctx = document.getElementById('predictionChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Model Predictions',
                    data: [],
                    backgroundColor: 'rgba(99, 102, 241, 0.5)',
                    borderColor: '#6366f1',
                    showLine: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: { display: true, text: 'Input (X)', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        title: { display: true, text: 'Output (Y)', color: '#94a3b8' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });
    };

    // Fetch Model Information
    const fetchModelInfo = async () => {
        try {
            const response = await fetch('/model-info');
            const data = await response.json();

            modelWeight.textContent = data.weights.toFixed(4);
            modelBias.textContent = data.bias.toFixed(4);
            formulaWeight.textContent = data.weights.toFixed(4);
            formulaBias.textContent = data.bias.toFixed(4);

            // Generate some line data for the chart
            updateChartLine(data.weights, data.bias);
        } catch (error) {
            console.error('Error fetching model info:', error);
        }
    };

    const updateChartLine = (w, b) => {
        const lineData = [];
        for (let x = 0; x <= 1; x += 0.1) {
            lineData.push({ x: x, y: w * x + b });
        }
        chart.data.datasets[0].data = lineData;
        chart.update();
    };

    // Handle Prediction
    const handlePredict = async () => {
        const xVal = parseFloat(xInput.value);
        if (isNaN(xVal)) return;

        predictBtn.textContent = 'Predicting...';
        predictBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: xVal })
            });
            const data = await response.json();

            if (data.status === 'success') {
                yResult.textContent = data.prediction.toFixed(4);

                // Add dot to chart
                chart.data.datasets.push({
                    label: 'New Prediction',
                    data: [{ x: data.x, y: data.prediction }],
                    backgroundColor: '#ec4899',
                    pointRadius: 8,
                    pointHoverRadius: 10
                });
                chart.update();
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Failed to get prediction from server.');
        } finally {
            predictBtn.textContent = 'Predict Result';
            predictBtn.disabled = false;
        }
    };

    // Event Listeners
    predictBtn.addEventListener('click', handlePredict);

    // Init
    initChart();
    fetchModelInfo();
});
