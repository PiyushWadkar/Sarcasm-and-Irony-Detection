/**
 * Chart.js Visualizations for Prediction Distribution and Model Metrics.
 */

let distributionChartInstance = null;

// 1. Prediction Distribution Doughnut Chart
function initDistributionChart(canvasId, sarcasticCount = 0, ironicalCount = 0, literalCount = 0) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const data = [sarcasticCount, ironicalCount, literalCount];
    const total = data.reduce((a, b) => a + b, 0);

    // If completely empty, show placeholder segment
    const displayData = total === 0 ? [1, 1, 1] : data;
    const backgroundColors = total === 0 
        ? ["rgba(244, 63, 94, 0.3)", "rgba(245, 158, 11, 0.3)", "rgba(16, 185, 129, 0.3)"]
        : ["#f43f5e", "#f59e0b", "#10b981"];

    distributionChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Sarcastic", "Ironical", "Literal"],
            datasets: [{
                data: displayData,
                backgroundColor: backgroundColors,
                borderColor: "#121829",
                borderWidth: 3,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "70%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#94a3b8",
                        font: { family: "Inter", size: 12 },
                        padding: 16,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (total === 0) return `${context.label}: 0 (No data)`;
                            const count = data[context.dataIndex];
                            const pct = ((count / total) * 100).toFixed(1);
                            return `${context.label}: ${count} (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Update Distribution Chart Live
window.updateDistributionChart = function(sarcastic, ironical, literal) {
    if (!distributionChartInstance) return;

    const total = sarcastic + ironical + literal;
    if (total === 0) {
        distributionChartInstance.data.datasets[0].data = [1, 1, 1];
        distributionChartInstance.data.datasets[0].backgroundColor = [
            "rgba(244, 63, 94, 0.3)", "rgba(245, 158, 11, 0.3)", "rgba(16, 185, 129, 0.3)"
        ];
    } else {
        distributionChartInstance.data.datasets[0].data = [sarcastic, ironical, literal];
        distributionChartInstance.data.datasets[0].backgroundColor = ["#f43f5e", "#f59e0b", "#10b981"];
    }
    distributionChartInstance.update();
};

// 2. Model Baseline Comparison Bar Chart
function initModelComparisonChart(canvasId, comparisonData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !comparisonData) return;

    const lr = comparisonData.logistic_regression.metrics;
    const svm = comparisonData.linear_svm.metrics;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Accuracy (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1 (%)"],
            datasets: [
                {
                    label: "Logistic Regression",
                    data: [lr.accuracy, lr.macro_precision, lr.macro_recall, lr.macro_f1],
                    backgroundColor: "rgba(139, 92, 246, 0.8)",
                    borderColor: "#8b5cf6",
                    borderWidth: 1,
                    borderRadius: 6
                },
                {
                    label: "Linear SVM (Calibrated)",
                    data: [svm.accuracy, svm.macro_precision, svm.macro_recall, svm.macro_f1],
                    backgroundColor: "rgba(99, 102, 241, 0.8)",
                    borderColor: "#6366f1",
                    borderWidth: 1,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.08)" },
                    ticks: { color: "#94a3b8" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: { color: "#f8fafc", font: { family: "Inter", size: 12 } }
                }
            }
        }
    });
}

// 3. Per-Class F1 Score Breakdown Chart
function initPerClassChart(canvasId, perClassData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !perClassData) return;

    const classes = Object.keys(perClassData);
    const f1Scores = classes.map(c => perClassData[c].f1_score);
    const precisions = classes.map(c => perClassData[c].precision);
    const recalls = classes.map(c => perClassData[c].recall);

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: classes,
            datasets: [
                {
                    label: "Precision (%)",
                    data: precisions,
                    backgroundColor: "rgba(6, 182, 212, 0.75)",
                    borderRadius: 4
                },
                {
                    label: "Recall (%)",
                    data: recalls,
                    backgroundColor: "rgba(168, 85, 247, 0.75)",
                    borderRadius: 4
                },
                {
                    label: "F1-Score (%)",
                    data: f1Scores,
                    backgroundColor: "rgba(244, 63, 94, 0.75)",
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "rgba(255, 255, 255, 0.08)" },
                    ticks: { color: "#94a3b8" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: { color: "#f8fafc", font: { family: "Inter", size: 12 } }
                }
            }
        }
    });
}
