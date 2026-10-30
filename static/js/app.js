/**
 * Main Interactive Application Logic for Sarcasm & Irony Detection.
 */

document.addEventListener("DOMContentLoaded", () => {
    initCharacterCounter();
    initExampleChips();
    initSinglePredictionForm();
    loadDashboardStats();
    loadRecentHistory();
});

// Toast Helper
function showToast(message, type = "success") {
    const toastEl = document.getElementById("appToast");
    if (!toastEl) return;

    const toastIcon = document.getElementById("toastIcon");
    const toastText = document.getElementById("toastText");

    toastText.textContent = message;

    if (type === "success") {
        toastIcon.className = "bi bi-check-circle-fill text-success";
    } else if (type === "warning") {
        toastIcon.className = "bi bi-exclamation-triangle-fill text-warning";
    } else {
        toastIcon.className = "bi bi-x-circle-fill text-danger";
    }

    const toast = new bootstrap.Toast(toastEl, { delay: 3500 });
    toast.show();
}

// 1. Character Counter
function initCharacterCounter() {
    const textarea = document.getElementById("textInput");
    const counter = document.getElementById("charCount");
    if (!textarea || !counter) return;

    textarea.addEventListener("input", () => {
        const len = textarea.value.length;
        counter.textContent = `${len} / 2000`;
        if (len > 2000) {
            counter.classList.add("text-danger");
        } else {
            counter.classList.remove("text-danger");
        }
    });
}

// 2. Preset Example Sentence Chips
function initExampleChips() {
    const chips = document.querySelectorAll(".example-chip");
    const textarea = document.getElementById("textInput");
    const counter = document.getElementById("charCount");

    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            const sentence = chip.getAttribute("data-sentence");
            if (textarea && sentence) {
                textarea.value = sentence;
                textarea.dispatchEvent(new Event("input"));
                textarea.focus();
                showToast("Example loaded. Click 'Analyze Text' to run the model.", "success");
            }
        });
    });
}

// 3. Single Prediction Form
function initSinglePredictionForm() {
    const form = document.getElementById("analyzeForm");
    const textarea = document.getElementById("textInput");
    const btnSubmit = document.getElementById("btnAnalyze");
    const btnClear = document.getElementById("btnClear");
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultPanel = document.getElementById("resultPanel");

    if (btnClear && textarea) {
        btnClear.addEventListener("click", () => {
            textarea.value = "";
            textarea.dispatchEvent(new Event("input"));
            if (resultPanel) resultPanel.classList.add("d-none");
            if (resultPlaceholder) resultPlaceholder.classList.remove("d-none");
            textarea.focus();
        });
    }

    if (!form || !btnSubmit) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = textarea.value.trim();

        if (!text) {
            showToast("Please enter a sentence to analyze.", "warning");
            textarea.focus();
            return;
        }

        if (text.length > 2000) {
            showToast("Text exceeds maximum 2000 characters.", "danger");
            return;
        }

        // Set Loading State
        btnSubmit.disabled = true;
        btnSubmit.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span> Analyzing...`;

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            const data = await resp.json();

            if (!resp.ok || data.status !== "success") {
                throw new Error(data.message || "Failed to process prediction.");
            }

            renderPredictionResult(data.data);
            showToast(`Predicted as ${data.data.predicted_class} (${data.data.confidence_percentage}%)`, "success");

            // Refresh stats and history
            loadDashboardStats();
            loadRecentHistory();

        } catch (err) {
            showToast(err.message, "danger");
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = `<i class="bi bi-magic me-1"></i> Analyze Text`;
        }
    });
}

// 4. Render Prediction Result Card
function renderPredictionResult(res) {
    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultPanel = document.getElementById("resultPanel");
    if (!resultPanel) return;

    if (resultPlaceholder) resultPlaceholder.classList.add("d-none");
    resultPanel.classList.remove("d-none");

    // Class Badge
    const badgeEl = document.getElementById("resClassBadge");
    const classIcon = document.getElementById("resClassIcon");
    const className = document.getElementById("resClassName");

    let badgeClass = "badge-literal";
    let iconClass = "bi-check-circle";

    if (res.predicted_class === "Sarcastic") {
        badgeClass = "badge-sarcastic";
        iconClass = "bi-fire";
    } else if (res.predicted_class === "Ironical") {
        badgeClass = "badge-ironical";
        iconClass = "bi-lightning-charge";
    }

    badgeEl.className = `badge ${badgeClass} px-3 py-2 fs-6 rounded-pill d-inline-flex align-items-center gap-1`;
    classIcon.className = `bi ${iconClass}`;
    className.textContent = res.predicted_class;

    // Confidence
    const confVal = document.getElementById("resConfidenceVal");
    const confBar = document.getElementById("resConfidenceBar");
    confVal.textContent = `${res.confidence_percentage}%`;
    confBar.style.width = `${res.confidence_percentage}%`;

    if (res.predicted_class === "Sarcastic") {
        confBar.style.background = "linear-gradient(90deg, #f43f5e, #fb7185)";
    } else if (res.predicted_class === "Ironical") {
        confBar.style.background = "linear-gradient(90deg, #f59e0b, #fbbf24)";
    } else {
        confBar.style.background = "linear-gradient(90deg, #10b981, #34d399)";
    }

    // Probabilities Breakdown
    const probs = res.probabilities || {};
    const pSarcastic = Math.round((probs["Sarcastic"] || 0) * 100);
    const pIronical = Math.round((probs["Ironical"] || 0) * 100);
    const pLiteral = Math.round((probs["Non-Sarcastic / Literal"] || probs["Literal"] || 0) * 100);

    const barS = document.getElementById("probBarSarcastic");
    const barI = document.getElementById("probBarIronical");
    const barL = document.getElementById("probBarLiteral");

    if (barS) { barS.style.width = `${pSarcastic}%`; barS.textContent = `${pSarcastic}%`; }
    if (barI) { barI.style.width = `${pIronical}%`; barI.textContent = `${pIronical}%`; }
    if (barL) { barL.style.width = `${pLiteral}%`; barL.textContent = `${pLiteral}%`; }

    // Explanation
    const explanationEl = document.getElementById("resExplanation");
    if (explanationEl) {
        explanationEl.textContent = res.explanation;
    }

    // Top Contributing Features
    const featureContainer = document.getElementById("resTopFeatures");
    if (featureContainer) {
        featureContainer.innerHTML = "";
        if (res.top_features && res.top_features.length > 0) {
            res.top_features.forEach(f => {
                const span = document.createElement("span");
                span.className = "feature-token feature-token-positive";
                span.innerHTML = `<i class="bi bi-tag-fill me-1 small text-purple-light"></i>${escapeHtml(f.feature)} <small class="opacity-75">(${f.weight > 0 ? '+' : ''}${f.weight})</small>`;
                featureContainer.appendChild(span);
            });
        } else {
            featureContainer.innerHTML = `<span class="text-secondary small fst-italic">No high-weight n-gram triggers identified.</span>`;
        }
    }
}

// 5. Dashboard Statistics Loader
async function loadDashboardStats() {
    try {
        const resp = await fetch("/api/stats");
        if (!resp.ok) return;
        const json = await resp.json();
        if (json.status !== "success") return;

        const s = json.data;
        const totalEl = document.getElementById("statTotal");
        const sarcEl = document.getElementById("statSarcastic");
        const ironEl = document.getElementById("statIronical");
        const litEl = document.getElementById("statLiteral");

        if (totalEl) totalEl.textContent = s.total_predictions.toLocaleString();
        if (sarcEl) sarcEl.textContent = `${s.sarcastic_count.toLocaleString()} (${s.sarcastic_pct}%)`;
        if (ironEl) ironEl.textContent = `${s.ironical_count.toLocaleString()} (${s.ironical_pct}%)`;
        if (litEl) litEl.textContent = `${s.literal_count.toLocaleString()} (${s.literal_pct}%)`;

        // Update Global Chart
        if (window.updateDistributionChart) {
            window.updateDistributionChart(s.sarcastic_count, s.ironical_count, s.literal_count);
        }
    } catch (e) {
        console.warn("Could not refresh dashboard stats:", e);
    }
}

// 6. Recent History Table
async function loadRecentHistory() {
    const tableBody = document.getElementById("recentHistoryBody");
    if (!tableBody) return;

    try {
        const resp = await fetch("/api/history?limit=6");
        if (!resp.ok) return;
        const json = await resp.json();
        if (json.status !== "success" || !json.data) return;

        if (json.data.length === 0) {
            tableBody.innerHTML = `<tr><td colspan="4" class="text-center text-secondary py-3">No predictions recorded yet.</td></tr>`;
            return;
        }

        tableBody.innerHTML = "";
        json.data.forEach(item => {
            const tr = document.createElement("tr");

            let badgeHtml = `<span class="badge bg-success-subtle text-success border border-success-subtle">Literal</span>`;
            if (item.predicted_class === "Sarcastic") {
                badgeHtml = `<span class="badge bg-danger-subtle text-danger border border-danger-subtle">Sarcastic</span>`;
            } else if (item.predicted_class === "Ironical") {
                badgeHtml = `<span class="badge bg-warning-subtle text-warning border border-warning-subtle">Ironical</span>`;
            }

            const truncatedText = item.input_text.length > 55 ? item.input_text.substring(0, 55) + "..." : item.input_text;
            const timeAgo = formatTime(item.created_at);

            tr.innerHTML = `
                <td class="text-truncate" style="max-width: 280px;" title="${escapeHtml(item.input_text)}">
                    ${escapeHtml(truncatedText)}
                </td>
                <td>${badgeHtml}</td>
                <td><span class="fw-semibold">${Math.round(item.confidence * 100)}%</span></td>
                <td class="text-secondary small text-nowrap">${timeAgo}</td>
            `;
            tableBody.appendChild(tr);
        });
    } catch (e) {
        console.warn("Could not load recent history:", e);
    }
}

// Helper: Format relative timestamp or time
function formatTime(isoString) {
    if (!isoString) return "";
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
        return isoString;
    }
}

// Helper: Escape HTML
function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
