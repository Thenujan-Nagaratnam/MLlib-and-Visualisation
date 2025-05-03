document.addEventListener('DOMContentLoaded', function() {
    const lyricsForm = document.getElementById('lyrics-form');
    const resultsSection = document.getElementById('results-section');
    const loadingIndicator = document.getElementById('loading');
    const newAnalysisBtn = document.getElementById('new-analysis');
    const predictedGenreElement = document.getElementById('predicted-genre');
    const confidenceElement = document.getElementById('confidence');
    let genreChart = null;

    lyricsForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        
        // Prepare form data
        const formData = new FormData(lyricsForm);
        
        try {
            // Send POST request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }
            
            // Parse JSON response
            const result = await response.json();
            
            // Update UI with results
            updateResults(result);
            
            // Hide loading indicator and form, show results
            loadingIndicator.style.display = 'none';
            lyricsForm.closest('.input-section').style.display = 'none';
            resultsSection.style.display = 'block';
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the lyrics. Please try again.');
            loadingIndicator.style.display = 'none';
        }
    });

    newAnalysisBtn.addEventListener('click', function() {
        // Reset form
        lyricsForm.reset();
        
        // Hide results, show form
        resultsSection.style.display = 'none';
        lyricsForm.closest('.input-section').style.display = 'block';
        
        // Destroy existing chart to prevent memory leaks
        if (genreChart) {
            genreChart.destroy();
        }
    });

    // Function to update results UI
    function updateResults(data) {
        // Update predicted genre and confidence
        predictedGenreElement.textContent = data.predicted_genre;
        confidenceElement.textContent = (data.probabilities[data.predicted_genre] * 100).toFixed(2) + '%';
        
        // Create chart for visualization
        createChart(data.probabilities);
    }

    // Function to create chart visualization
    function createChart(probabilities) {
        const ctx = document.getElementById('genre-chart').getContext('2d');
        
        // Convert probabilities object to arrays for Chart.js
        const labels = Object.keys(probabilities);
        const data = Object.values(probabilities).map(val => (val * 100).toFixed(2));
        
        // Generate background colors (gradient of purples)
        const backgroundColors = labels.map((_, index) => {
            const opacity = 0.7 + (index / (labels.length * 2));
            return `rgba(98, 0, 234, ${opacity})`;
        });
        
        // Destroy previous chart instance if it exists
        if (genreChart) {
            genreChart.destroy();
        }
        
        // Create new chart
        genreChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Genre Probability (%)',
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: 'rgba(98, 0, 234, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Probability (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Genre'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Probability: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    }
});