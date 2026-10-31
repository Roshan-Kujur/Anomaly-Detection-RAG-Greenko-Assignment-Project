
/* =====================================================
   GREENKO AI COPILOT
   Frontend Application Logic

   Architecture:
   Frontend -> FastAPI -> ML Inference
                      -> Conditional RAG
                      -> LLM Generation
                      -> Frontend
===================================================== */


/* ================= API CONFIGURATION ================= */

// Change this if your FastAPI server uses another address.
const API_BASE_URL =
    window.GREENKO_API_BASE_URL || "http://127.0.0.1:8000";

const API_ENDPOINT = `${API_BASE_URL}/api/run`;;


/* ================= DOM REFERENCES ================= */

const assetTypeInput = document.getElementById("assetType");
const assetIdInput = document.getElementById("assetId");

const telemetryForm = document.getElementById("telemetryForm");
const analyzeButton = document.getElementById("analyzeButton");

const solarFields = document.querySelectorAll(".solar-field");
const windFields = document.querySelectorAll(".wind-field");

const timestampInput = document.getElementById("timestampId");
const temperatureInput = document.getElementById("temperature");
const vibrationInput = document.getElementById("vibration");
const powerInput = document.getElementById("power");
const irradianceInput = document.getElementById("irradiance");
const windSpeedInput = document.getElementById("windSpeed");

const apiStatus = document.getElementById("apiStatus");

const errorMessage = document.getElementById("errorMessage");
const analysisProgress = document.getElementById("analysisProgress");
const progressText = document.getElementById("progressText");

const emptyState = document.getElementById("emptyState");
const diagnosticResults = document.getElementById("diagnosticResults");


/* ================= KPI REFERENCES ================= */

const kpiAsset = document.getElementById("kpiAsset");
const kpiAssetType = document.getElementById("kpiAssetType");
const kpiPower = document.getElementById("kpiPower");
const kpiTemperature = document.getElementById("kpiTemperature");
const kpiHealth = document.getElementById("kpiHealth");
const kpiHealthMeta = document.getElementById("kpiHealthMeta");


/* ================= ML RESULT REFERENCES ================= */

const predictionValue = document.getElementById("predictionValue");
const predictionMeta = document.getElementById("predictionMeta");
const anomalyCount = document.getElementById("anomalyCount");

const anomalyFeaturePanel =
    document.getElementById("anomalyFeaturePanel");

const anomalyFeatures =
    document.getElementById("anomalyFeatures");

const normalResult =
    document.getElementById("normalResult");


/* ================= RAG RESULT REFERENCES ================= */

const ragResults = document.getElementById("ragResults");
const ragStatusBadge = document.getElementById("ragStatusBadge");

const ragFailureMessage =
    document.getElementById("ragFailureMessage");

const anomalySummary = document.getElementById("anomalySummary");

const overallAssessment =
    document.getElementById("overallAssessment");

const potentialCauses =
    document.getElementById("potentialCauses");

const troubleshootingChecks =
    document.getElementById("troubleshootingChecks");

const preventiveRecommendations =
    document.getElementById("preventiveRecommendations");

const evidenceLimitations =
    document.getElementById("evidenceLimitations");

const retrievedDocuments =
    document.getElementById("retrievedDocuments");


/* ================= ANALYSIS HISTORY ================= */

const historyBody = document.getElementById("historyBody");


/* =====================================================
   1. GENERAL DOM HELPERS
===================================================== */

function createElement(tag, className, text) {

    const element = document.createElement(tag);

    if (className) {
        element.className = className;
    }

    if (text !== undefined && text !== null) {
        element.textContent = String(text);
    }

    return element;
}


function setText(element, value, fallback = "—") {

    if (!element) {
        return;
    }

    element.textContent =
        value === null || value === undefined || value === ""
            ? fallback
            : String(value);

}


function formatNumber(value, decimals = 3) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return "—";
    }

    return number.toFixed(decimals);
}


function clearElement(element) {

    if (element) {
        element.replaceChildren();
    }

}


/* =====================================================
   2. API STATUS
===================================================== */

function updateApiStatus(status, label) {

    const dot = apiStatus.querySelector(".status-dot");

    apiStatus.classList.remove("online", "offline");

    if (status === "online") {

        apiStatus.classList.add("online");

    } else if (status === "offline") {

        apiStatus.classList.add("offline");

    }

    apiStatus.replaceChildren(
        dot,
        document.createTextNode(` API: ${label}`)
    );

}


/* =====================================================
   3. ASSET TYPE HANDLING
===================================================== */

function updateAssetFields() {

    const assetType = assetTypeInput.value;

    const showSolar = assetType === "solar";

    solarFields.forEach(field => {

        const input = field.querySelector("input");

        field.classList.toggle("hidden", !showSolar);

        if (input) {

            input.disabled = !showSolar;
            input.required = showSolar;

            if (!showSolar) {
                input.value = "";
            }

        }

    });


    windFields.forEach(field => {

        const input = field.querySelector("input");

        field.classList.toggle("hidden", showSolar);

        if (input) {

            input.disabled = showSolar;
            input.required = !showSolar;

            if (showSolar) {
                input.value = "";
            }

        }

    });

    updateKPIInputs();

}


/* =====================================================
   4. INPUT VALUE HANDLING
===================================================== */

function getNumericValue(input) {

    if (!input || input.value.trim() === "") {
        return null;
    }

    const value = Number(input.value);

    return Number.isFinite(value) ? value : null;

}


/* =====================================================
   5. COLLECT RAW TELEMETRY
===================================================== */

function collectTelemetryData() {

    const assetType = assetTypeInput.value;

    const telemetry = {
    timestamp_id: getNumericValue(timestampInput),
    asset_id: assetIdInput.value.trim(),
    asset_type: assetType,
    hour: new Date().getHours(),
    temperature_c: getNumericValue(temperatureInput),
    vibration_mm_s: getNumericValue(vibrationInput),
    power_mw: getNumericValue(powerInput)
};

    // Only send the sensor relevant to the selected asset.

// Initialize both sensor fields.
// The inactive sensor is represented as zero.
telemetry.irradiance_wm2 = 0;
telemetry.wind_speed_ms = 0;

// Assign the actual value for the selected asset type.
if (assetType === "solar") {
    telemetry.irradiance_wm2 = getNumericValue(irradianceInput);
} else if (assetType === "wind") {
    telemetry.wind_speed_ms = getNumericValue(windSpeedInput);
}

    return telemetry;

}


/* =====================================================
   6. INPUT VALIDATION
===================================================== */

function validateTelemetry(data) {

    if (!data.asset_id) {
        return "Please enter an Asset ID.";
    }

    if (!["solar", "wind"].includes(data.asset_type)) {
        return "Please select a valid asset type.";
    }

    if (data.timestamp_id === null) {
        return "Please enter a valid Timestamp ID.";
    }

    if (!Number.isInteger(data.timestamp_id)) {
        return "Timestamp ID must be a whole number.";
    }

    const commonFields = {

        temperature_c: "Temperature",

        vibration_mm_s: "Vibration",

        power_mw: "Power Output"

    };

    for (const [field, label] of Object.entries(commonFields)) {

        if (data[field] === null) {
            return `Please enter a valid ${label} reading.`;
        }

    }


    if (
        data.asset_type === "solar" &&
        data.irradiance_wm2 === null
    ) {

        return "Please enter a valid Solar Irradiance reading.";

    }


    if (
        data.asset_type === "wind" &&
        data.wind_speed_ms === null
    ) {

        return "Please enter a valid Wind Speed reading.";

    }

    return null;

}


/* =====================================================
   7. KPI CARD UPDATES
===================================================== */

function updateKPIInputs() {

    const assetType = assetTypeInput.value;

    setText(
        kpiAsset,
        assetIdInput.value.trim(),
        "—"
    );

    setText(
        kpiAssetType,
        assetType === "solar"
            ? "Solar Inverter"
            : "Wind Turbine Generator",
        "Awaiting asset selection"
    );

    const power = getNumericValue(powerInput);

    const temperature = getNumericValue(temperatureInput);

    setText(
        kpiPower,
        power === null ? "—" : `${formatNumber(power)} MW`
    );

    setText(
        kpiTemperature,
        temperature === null
            ? "—"
            : `${formatNumber(temperature)} °C`
    );

}


/* =====================================================
   8. ERROR HANDLING
===================================================== */

function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove("hidden");

}


function clearError() {

    errorMessage.textContent = "";

    errorMessage.classList.add("hidden");

}


function showProgress(message) {

    progressText.textContent = message;

    analysisProgress.classList.remove("hidden");

}


function hideProgress() {

    analysisProgress.classList.add("hidden");

}


/* =====================================================
   9. BACKEND API COMMUNICATION
===================================================== */

async function analyzeTelemetry(data) {

    const response = await fetch(API_ENDPOINT, {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(data)

    });


    if (!response.ok) {

        const errorBody = await response.text();

        let errorDetail = errorBody;

        try {

            const parsedError = JSON.parse(errorBody);

            errorDetail =
                parsedError.detail ||
                parsedError.message ||
                errorBody;

        } catch {
            // Retain the original response text.
        }

        throw new Error(
            `Backend error (${response.status}): ${errorDetail}`
        );

    }


    try {

        return await response.json();

    } catch {

        throw new Error(
            "Backend returned an invalid JSON response."
        );

    }

}


/* =====================================================
   10. NORMALIZE BACKEND RESPONSE
===================================================== */

/*
Expected response contract:

{
    "asset": {
        "asset_id": "...",
        "asset_type": "solar"
    },

    "inference": {
        "is_anomaly": true,
        "prediction": "Anomaly",
        "anomaly_count": 1,
        "anomaly_features": []
    },

    "rag": {
        "status": "completed",

        "diagnostic": {
            "asset_id": "...",
            "asset_type": "Solar",
            "anomaly_summary": "...",
            "potential_causes": [],
            "troubleshooting_checks": [],
            "preventive_recommendations": [],
            "evidence_limitations": [],
            "overall_assessment": "..."
        },

        "sources": []
    }
}
*/

function normalizeBackendResponse(response) {

    if (!response || typeof response !== "object") {
        throw new Error("Invalid response received from backend.");
    }

    // 1. Handle normal telemetry
    if (response.status === "normal") {

        const normalRecord = response.normal_records?.[0] || {};

        return {
            asset: {
                asset_id: normalRecord.asset_id || "",
                asset_type: normalRecord.asset_type || ""
            },

            inference: {
                is_anomaly: false,
                prediction: "Normal",
                anomaly_count: 0,
                anomaly_features: []
            },

            rag: {
                status: "skipped",
                diagnostic: null,
                sources: []
            }
        };
    }

    // 2. Handle anomalous telemetry
    if (response.status === "anomaly_detected") {

        const anomalyRecord = response.anomaly_records?.[0] || {};

        const anomalyDetails = anomalyRecord.anomaly_details || {};

        const diagnostics = response.diagnostics || [];

        const successfulDiagnostic = diagnostics.find(
            item => item.status === "success"
        );

        const failedDiagnostic = diagnostics.some(
            item => item.status === "failed"
        );

        // Extract anomaly features
        let anomalyFeatures = anomalyDetails.anomaly_features || [];

        if (
            anomalyFeatures.length === 0 &&
            anomalyDetails.feature_deviations
        ) {
            anomalyFeatures = Object.keys(
                anomalyDetails.feature_deviations
            );
        }

        return {
            asset: {
                asset_id: anomalyRecord.asset_id || "",
                asset_type: anomalyRecord.asset_type || ""
            },

            inference: {
                is_anomaly: true,
                prediction: "Anomaly",
                anomaly_count:
                    anomalyDetails.anomaly_count ??
                    anomalyFeatures.length,
                anomaly_features: anomalyFeatures
            },

            rag: {
                status: successfulDiagnostic
                    ? "completed"
                    : failedDiagnostic
                        ? "failed"
                        : "unavailable",

                diagnostic: successfulDiagnostic
                    ? successfulDiagnostic.diagnostic
                    : null,

                sources: successfulDiagnostic?.diagnostic?.sources || []
            }
        };
    }

    throw new Error(
        `Unexpected backend response status: ${response.status}`
    );
}

/* =====================================================
   11. DISPLAY ML PREDICTION
===================================================== */

function displayPrediction(inference) {

    const isAnomaly = inference.is_anomaly;

    predictionValue.classList.remove(
        "prediction-normal",
        "prediction-anomaly",
        "prediction-warning"
    );


    if (isAnomaly) {

        predictionValue.textContent =
            inference.prediction || "Anomaly Detected";

        predictionValue.classList.add("prediction-anomaly");

        kpiHealth.textContent = "Anomaly";

        kpiHealth.className =
            "kpi-value health-anomaly";

        kpiHealthMeta.textContent =
            "Anomaly detected by ML inference";


    } else {

        predictionValue.textContent =
            inference.prediction || "Normal";

        predictionValue.classList.add("prediction-normal");

        kpiHealth.textContent = "Normal";

        kpiHealth.className =
            "kpi-value health-normal";

        kpiHealthMeta.textContent =
            "No anomaly detected by ML inference";

    }


    const count = inference.anomaly_count;

    setText(
        anomalyCount,
        count === undefined ? "—" : count
    );


    const features = Array.isArray(inference.anomaly_features)
        ? inference.anomaly_features
        : [];

    anomalyFeatures.replaceChildren();


    if (isAnomaly && features.length > 0) {

        anomalyFeaturePanel.classList.remove("hidden");

        features.forEach(feature => {

            const badge = createElement(
                "span",
                "feature-badge",
                feature
            );

            anomalyFeatures.appendChild(badge);

        });

    } else {

        anomalyFeaturePanel.classList.add("hidden");

    }


    // Display the model score only if the backend provides one.
    if (
        inference.score !== undefined &&
        inference.score !== null
    ) {

        predictionMeta.textContent =
            `Model score: ${inference.score}`;

    } else {

        predictionMeta.textContent =
            "Prediction returned by ML inference";

    }

}


/* =====================================================
   12. DISPLAY NORMAL RESULT
===================================================== */

function displayNormalResult() {

    normalResult.classList.remove("hidden");

    ragResults.classList.add("hidden");

}


/* =====================================================
   13. RAG STATUS BADGE
===================================================== */

function updateRAGStatus(status) {

    const labels = {

        completed: "RAG Completed",

        skipped: "RAG Skipped",

        failed: "RAG Failed",

        unavailable: "RAG Unavailable",

        pending: "Processing"

    };


    const badgeClasses = {

        completed: "badge badge-success",

        skipped: "badge badge-neutral",

        failed: "badge badge-danger",

        unavailable: "badge badge-danger",

        pending: "badge badge-pending"

    };


    ragStatusBadge.className =
        badgeClasses[status] || badgeClasses.pending;

    ragStatusBadge.textContent =
        labels[status] || "RAG Status Unknown";

}


/* =====================================================
   14. RENDER GENERIC TEXT LIST
===================================================== */

function renderTextList(element, items, emptyMessage) {

    element.replaceChildren();


    if (!Array.isArray(items) || items.length === 0) {

        element.appendChild(
            createElement(
                "p",
                "muted-text",
                emptyMessage
            )
        );

        return;

    }


    const list = document.createElement("ul");


    items.forEach(item => {

        const li = document.createElement("li");

        li.textContent = String(item);

        list.appendChild(li);

    });


    element.appendChild(list);

}


/* =====================================================
   15. DISPLAY POTENTIAL CAUSES
===================================================== */

function displayPotentialCauses(causes) {

    potentialCauses.replaceChildren();


    if (!Array.isArray(causes) || causes.length === 0) {

        potentialCauses.appendChild(
            createElement(
                "p",
                "muted-text",
                "No potential causes were provided."
            )
        );

        return;

    }


    causes.forEach((cause, index) => {

        const card = createElement(
            "div",
            "content-item"
        );


        const heading = createElement(
            "h4",
            "",
            `${index + 1}. ${cause.cause || "Possible cause"}`
        );


        card.appendChild(heading);


        if (cause.explanation) {

            const explanation = createElement(
                "p",
                "",
                cause.explanation
            );

            card.appendChild(explanation);

        }


        // Supporting observations

        if (
            Array.isArray(cause.supporting_observations) &&
            cause.supporting_observations.length > 0
        ) {

            card.appendChild(
                createElement(
                    "p",
                    "",
                    "Supporting observations"
                )
            );

            const observations = document.createElement("ul");

            cause.supporting_observations.forEach(item => {

                const li = document.createElement("li");

                li.textContent = String(item);

                observations.appendChild(li);

            });

            card.appendChild(observations);

        }


        // Evidence references

        if (
            Array.isArray(cause.evidence_sources) &&
            cause.evidence_sources.length > 0
        ) {

            const evidenceContainer = createElement(
                "div",
                "cause-evidence"
            );

            cause.evidence_sources.forEach(source => {

                const reference = createElement(
                    "span",
                    "evidence-reference",
                    source
                );

                evidenceContainer.appendChild(reference);

            });

            card.appendChild(evidenceContainer);

        }


        potentialCauses.appendChild(card);

    });

}


/* =====================================================
   16. DISPLAY TROUBLESHOOTING CHECKS
===================================================== */

function displayTroubleshootingChecks(checks) {

    troubleshootingChecks.replaceChildren();


    if (!Array.isArray(checks) || checks.length === 0) {

        troubleshootingChecks.appendChild(
            createElement(
                "p",
                "muted-text",
                "No troubleshooting checks were provided."
            )
        );

        return;

    }


    checks.forEach((item, index) => {

        const card = createElement(
            "div",
            "content-item"
        );

        card.appendChild(
            createElement(
                "h4",
                "",
                `${index + 1}. ${item.check || "Inspection check"}`
            )
        );


        if (item.reason) {

            card.appendChild(
                createElement(
                    "p",
                    "",
                    item.reason
                )
            );

        }

        troubleshootingChecks.appendChild(card);

    });

}


/* =====================================================
   17. DISPLAY PREVENTIVE RECOMMENDATIONS
===================================================== */

function displayPreventiveRecommendations(items) {

    renderTextList(
        preventiveRecommendations,
        items,
        "No preventive recommendations were provided."
    );

}


/* =====================================================
   18. DISPLAY EVIDENCE LIMITATIONS
===================================================== */

function displayEvidenceLimitations(items) {

    renderTextList(
        evidenceLimitations,
        items,
        "No evidence limitations were reported."
    );

}


/* =====================================================
   19. DISPLAY RETRIEVED DOCUMENT SOURCES
===================================================== */

function displayRetrievedDocuments(sources) {

    retrievedDocuments.replaceChildren();


    if (!Array.isArray(sources) || sources.length === 0) {

        retrievedDocuments.appendChild(
            createElement(
                "p",
                "muted-text",
                "No source references were returned by the backend."
            )
        );

        return;

    }


    sources.forEach(source => {

        const card = createElement(
            "article",
            "document-item"
        );


        const heading = createElement(
            "div",
            "document-heading"
        );


        const sourceName =
            source.source_file ||
            source.document_name ||
            source.document_id ||
            "Unknown source";


        heading.appendChild(
            createElement("h4", "", sourceName)
        );


        const evidenceId =
            source.evidence_id || "Source";


        heading.appendChild(
            createElement(
                "span",
                "document-id",
                evidenceId
            )
        );


        card.appendChild(heading);


        const passage =
            source.text ||
            source.excerpt ||
            source.page_content ||
            "";


        if (passage) {

            card.appendChild(
                createElement(
                    "p",
                    "",
                    passage
                )
            );

        }


        // Document metadata

        const metadata = createElement(
            "div",
            "document-meta"
        );


        const metadataItems = [

            source.page !== undefined
                ? `Page: ${source.page}`
                : null,

            source.document_id
                ? `Document ID: ${source.document_id}`
                : null,

            source.chunk_id
                ? `Chunk ID: ${source.chunk_id}`
                : null,

            source.relevance_score !== undefined &&
            source.relevance_score !== null
                ? `Relevance: ${source.relevance_score}`
                : null

        ].filter(Boolean);


        metadataItems.forEach(item => {

            metadata.appendChild(
                createElement("span", "", item)
            );

        });


        card.appendChild(metadata);

        retrievedDocuments.appendChild(card);

    });

}


/* =====================================================
   20. DISPLAY COMPLETE RAG DIAGNOSTIC
===================================================== */

function displayRAGDiagnostic(
    diagnostic,
    sources
) {

    if (!diagnostic) {

        throw new Error(
            "RAG completed without returning a diagnostic object."
        );

    }


    setText(
        anomalySummary,
        diagnostic.anomaly_summary,
        "No anomaly summary provided."
    );


    setText(
        overallAssessment,
        diagnostic.overall_assessment,
        "No overall assessment provided."
    );


    displayPotentialCauses(
        diagnostic.potential_causes
    );


    displayTroubleshootingChecks(
        diagnostic.troubleshooting_checks
    );


    displayPreventiveRecommendations(
        diagnostic.preventive_recommendations
    );


    displayEvidenceLimitations(
        diagnostic.evidence_limitations
    );


    displayRetrievedDocuments(sources);

}


/* =====================================================
   21. DISPLAY RAG RESULTS
===================================================== */

function displayRAGResults(result) {

    normalResult.classList.add("hidden");

    ragResults.classList.remove("hidden");

    ragFailureMessage.classList.add("hidden");

    ragFailureMessage.textContent = "";


    const rag = result.rag;

    const diagnostic = result.diagnostic;

    const sources = result.sources;


    // Missing RAG response

    if (!rag) {

        updateRAGStatus("unavailable");

        ragFailureMessage.textContent =
            "An anomaly was detected by ML inference, but the backend did not return a RAG result.";

        ragFailureMessage.classList.remove("hidden");

        displayRetrievedDocuments([]);

        return;

    }


    const status = String(
        rag.status || "unavailable"
    ).toLowerCase();


    updateRAGStatus(status);


    // RAG processing failure

    if (status === "failed" || status === "unavailable") {

        const errorText =
            rag.error_message ||
            rag.message ||
            "RAG diagnostic generation was unsuccessful.";

        ragFailureMessage.textContent =
            errorText;

        ragFailureMessage.classList.remove("hidden");


        // A partial diagnostic may still be displayed.
        if (diagnostic) {

            displayRAGDiagnostic(
                diagnostic,
                sources
            );

        } else {

            displayRetrievedDocuments([]);

        }

        return;

    }


    // RAG skipped should not occur for a true anomaly,
    // but handle it safely.

    if (status === "skipped") {

        updateRAGStatus("skipped");

        ragFailureMessage.textContent =
            "RAG processing was skipped by the backend.";

        ragFailureMessage.classList.remove("hidden");

        return;

    }


    // Completed successfully

    try {

        displayRAGDiagnostic(
            diagnostic,
            sources
        );

    } catch (error) {

        updateRAGStatus("failed");

        ragFailureMessage.textContent =
            "The RAG response could not be displayed: " +
            error.message;

        ragFailureMessage.classList.remove("hidden");

    }

}


/* =====================================================
   22. DISPLAY COMPLETE BACKEND RESPONSE
===================================================== */

function displayResults(rawResponse) {

    const result = normalizeBackendResponse(rawResponse);

    const inference = result.inference;

    const isAnomaly = inference.is_anomaly;


    diagnosticResults.classList.remove("hidden");

    emptyState.classList.add("hidden");


    // Asset details returned by backend, when available.

    const returnedAssetId =
        result.asset.asset_id ||
        assetIdInput.value.trim();

    const returnedAssetType =
        result.asset.asset_type ||
        assetTypeInput.value;


    setText(kpiAsset, returnedAssetId);

    setText(
        kpiAssetType,
        returnedAssetType
    );


    // Display ML prediction first.

    displayPrediction(inference);


    // Normal assets do not invoke RAG.

    if (!isAnomaly) {

        displayNormalResult();

        ragResults.classList.add("hidden");

        return {
            isAnomaly: false,
            ragStatus: "skipped"
        };

    }


    // Only anomalous observations should reach this point.

    displayRAGResults(result);


    const ragStatus =
        result.rag?.status || "unavailable";


    return {
        isAnomaly: true,
        ragStatus
    };

}


/* =====================================================
   23. ANALYSIS HISTORY
===================================================== */

function addHistoryRow(data, result) {

    const emptyRow =
        document.getElementById("historyEmptyRow");


    if (emptyRow) {
        emptyRow.remove();
    }


    const row = document.createElement("tr");


    const cells = [

        new Date().toLocaleTimeString(),

        data.asset_id,

        data.asset_type === "solar"
            ? "Solar"
            : "Wind",

        result.isAnomaly
            ? "Anomaly"
            : "Normal",

        result.ragStatus

    ];


    cells.forEach((value, index) => {

        const cell = document.createElement("td");

        if (index === 3) {

            const badge = createElement(
                "span",
                result.isAnomaly
                    ? "history-badge history-anomaly"
                    : "history-badge history-normal",
                value
            );

            cell.appendChild(badge);

        } else if (index === 4) {

            const badge = createElement(
                "span",
                "history-badge history-neutral",
                value
            );

            cell.appendChild(badge);

        } else {

            cell.textContent = value;

        }

        row.appendChild(cell);

    });


    // Show most recent analysis first.

    historyBody.prepend(row);

}


/* =====================================================
   24. FORM SUBMISSION
===================================================== */

telemetryForm.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        clearError();


        const telemetryData =
            collectTelemetryData();


        // Frontend validation

        const validationError =
            validateTelemetry(telemetryData);


        if (validationError) {

            showError(validationError);

            return;

        }


        // Begin analysis

        analyzeButton.disabled = true;

        analyzeButton.innerHTML =
            '<span>Analyzing...</span><span class="loading-indicator"></span>';

        telemetryForm.classList.add("loading");

        showProgress(
            "Submitting telemetry to the backend for ML inference..."
        );


        try {

            updateApiStatus("neutral", "Request in progress");


            // One request to FastAPI.
            // Backend handles ML and conditional RAG.

            const response = await analyzeTelemetry(
                telemetryData
            );


            showProgress(
                "Backend analysis response received. Updating diagnostics..."
            );


            // Process combined backend response.

            const result = displayResults(response);


            // Update API status.

            updateApiStatus(
                "online",
                "Response received"
            );


            // Record session history.

            addHistoryRow(
                telemetryData,
                result
            );


            progressText.textContent =
                result.isAnomaly
                    ? `Anomaly detected. RAG status: ${result.ragStatus}.`
                    : "Analysis complete. No anomaly detected; RAG was skipped.";

            // Keep the final message visible briefly.
            // It will be hidden when the next analysis begins.

        } catch (error) {

            console.error(
                "Greenko Copilot analysis error:",
                error
            );


            updateApiStatus(
                "offline",
                "Request failed"
            );


            showError(
                error.message ||
                "Unable to complete telemetry analysis."
            );


            showProgress(
                "Analysis request was unsuccessful."
            );


        } finally {

            analyzeButton.disabled = false;

            analyzeButton.innerHTML =
                '<span>Analyze Telemetry</span><span class="button-arrow">→</span>';

            telemetryForm.classList.remove("loading");

        }

    }
);


/* =====================================================
   25. EVENT LISTENERS
===================================================== */

// Asset type selection.

assetTypeInput.addEventListener(
    "change",
    updateAssetFields
);


// Update KPI preview when asset changes.

assetIdInput.addEventListener(
    "input",
    updateKPIInputs
);


// Update KPI values when telemetry is entered.

[
    powerInput,
    temperatureInput
].forEach(input => {

    input.addEventListener(
        "input",
        updateKPIInputs
    );

});


/* =====================================================
   26. INITIALIZE APPLICATION
===================================================== */

function initializeCopilot() {

    updateAssetFields();

    updateKPIInputs();

    updateApiStatus(
        "neutral",
        "Not connected"
    );

}


// Script uses defer, so HTML elements are already parsed.

initializeCopilot();
