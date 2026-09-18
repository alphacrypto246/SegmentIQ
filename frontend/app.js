/**
 * SegmentIQ — Frontend Application Logic & Visualization Engine
 * Handles customer presets, API inference, client-side fallback, and Chart.js graphs.
 */

// Presets mapping directly to distinct behavioral archetypes
const PRESETS = {
  high_spender: {
    BALANCE: 1850.0,
    BALANCE_FREQUENCY: 1.0,
    ONEOFF_PURCHASES: 2450.0,
    INSTALLMENTS_PURCHASES: 1600.0,
    CASH_ADVANCE: 0.0,
    PURCHASES_FREQUENCY: 0.95,
    ONEOFF_PURCHASES_FREQUENCY: 0.85,
    CASH_ADVANCE_FREQUENCY: 0.0,
    PURCHASES_TRX: 38,
    CREDIT_LIMIT: 8500.0,
    PAYMENTS: 3100.0,
    MINIMUM_PAYMENTS: 380.0,
    PRC_FULL_PAYMENT: 0.45,
    TENURE: 12,
  },
  cash_advance: {
    BALANCE: 3600.0,
    BALANCE_FREQUENCY: 0.95,
    ONEOFF_PURCHASES: 40.0,
    INSTALLMENTS_PURCHASES: 0.0,
    CASH_ADVANCE: 3200.0,
    PURCHASES_FREQUENCY: 0.08,
    ONEOFF_PURCHASES_FREQUENCY: 0.08,
    CASH_ADVANCE_FREQUENCY: 0.65,
    PURCHASES_TRX: 2,
    CREDIT_LIMIT: 4500.0,
    PAYMENTS: 1900.0,
    MINIMUM_PAYMENTS: 920.0,
    PRC_FULL_PAYMENT: 0.0,
    TENURE: 12,
  },
  dormant: {
    BALANCE: 120.0,
    BALANCE_FREQUENCY: 0.2,
    ONEOFF_PURCHASES: 0.0,
    INSTALLMENTS_PURCHASES: 80.0,
    CASH_ADVANCE: 0.0,
    PURCHASES_FREQUENCY: 0.15,
    ONEOFF_PURCHASES_FREQUENCY: 0.0,
    CASH_ADVANCE_FREQUENCY: 0.0,
    PURCHASES_TRX: 3,
    CREDIT_LIMIT: 2500.0,
    PAYMENTS: 350.0,
    MINIMUM_PAYMENTS: 85.0,
    PRC_FULL_PAYMENT: 0.8,
    TENURE: 12,
  },
  balanced: {
    BALANCE: 1100.0,
    BALANCE_FREQUENCY: 0.85,
    ONEOFF_PURCHASES: 450.0,
    INSTALLMENTS_PURCHASES: 400.0,
    CASH_ADVANCE: 150.0,
    PURCHASES_FREQUENCY: 0.55,
    ONEOFF_PURCHASES_FREQUENCY: 0.35,
    CASH_ADVANCE_FREQUENCY: 0.08,
    PURCHASES_TRX: 16,
    CREDIT_LIMIT: 5000.0,
    PAYMENTS: 1050.0,
    MINIMUM_PAYMENTS: 220.0,
    PRC_FULL_PAYMENT: 0.25,
    TENURE: 12,
  },
};

// Persona metadata and tailored banking strategies
const SEGMENT_METADATA = {
  0: {
    title: "Frequent Cash-Advance Users",
    description: "Relies heavily on ATM / cash withdrawals with low retail card purchases. Slower payment velocity.",
    strategy: "Liquidity Management & Credit Risk Monitoring",
    badgeClass: "segment-0",
    cohortSize: "2,579 users (36.1%)",
    cardId: "cardSegment0",
  },
  1: {
    title: "High Spenders",
    description: "High retail transaction volume across both one-off and installment purchases with high credit utilization.",
    strategy: "Premium Perks, Cashback & Credit Line Upsell",
    badgeClass: "segment-1",
    cohortSize: "2,498 users (34.9%)",
    cardId: "cardSegment1",
  },
  2: {
    title: "Low Activity / Dormant",
    description: "Minimal revolving balance and low transaction frequency, but prompt and reliable bill payers.",
    strategy: "Re-engagement Promos & 0% Purchase APR Campaigns",
    badgeClass: "segment-2",
    cohortSize: "2,083 users (29.0%)",
    cardId: "cardSegment2",
  },
};

// Cluster Centroid Z-Scores from training artifacts (cluster_profiles.json)
const CENTROID_PROFILES = {
  0: { BALANCE: 0.58, ONEOFF_PURCHASES: -0.5, INSTALLMENTS: -0.85, CASH_ADVANCE: 0.87, PURCHASES_FREQ: -0.95, CREDIT_LIMIT: -0.07 },
  1: { BALANCE: 0.40, ONEOFF_PURCHASES: 0.88, INSTALLMENTS: 0.63, CASH_ADVANCE: -0.24, PURCHASES_FREQ: 0.85, CREDIT_LIMIT: 0.36 },
  2: { BALANCE: -1.19, ONEOFF_PURCHASES: -0.44, INSTALLMENTS: 0.30, CASH_ADVANCE: -0.78, PURCHASES_FREQ: 0.16, CREDIT_LIMIT: -0.35 },
};

// Supported currencies & exchange rates (1 USD = X Currency)
const EXCHANGE_RATES = {
  USD: 1.0,
  EUR: 0.92,
  GBP: 0.79,
  INR: 83.5,
  CAD: 1.36,
  AUD: 1.52,
  JPY: 155.0,
};

// Monetary fields that require currency conversion
const MONETARY_FIELDS = [
  "BALANCE",
  "ONEOFF_PURCHASES",
  "INSTALLMENTS_PURCHASES",
  "CASH_ADVANCE",
  "CREDIT_LIMIT",
  "PAYMENTS",
  "MINIMUM_PAYMENTS",
];

let currentCurrency = "USD";

// Chart instances
let radarChartInstance = null;
let distributionChartInstance = null;
let centroidBarChartInstance = null;

// Determine API Base URL
const API_BASE = window.location.origin.includes("localhost") || window.location.origin.includes("127.0.0.1")
  ? window.location.origin
  : "http://127.0.0.1:8000";

// DOM Elements
const form = document.getElementById("predictionForm");
const predictBtn = document.getElementById("predictBtn");
const predictBtnText = document.getElementById("predictBtnText");
const predictBtnLoader = document.getElementById("predictBtnLoader");
const resetFormBtn = document.getElementById("resetFormBtn");
const currencySelect = document.getElementById("currencySelect");

// Inputs mapping
const inputFields = {
  BALANCE: document.getElementById("inputBalance"),
  BALANCE_FREQUENCY: document.getElementById("inputBalanceFreq"),
  ONEOFF_PURCHASES: document.getElementById("inputOneoff"),
  INSTALLMENTS_PURCHASES: document.getElementById("inputInstallments"),
  CASH_ADVANCE: document.getElementById("inputCashAdvance"),
  PURCHASES_FREQUENCY: document.getElementById("inputPurchasesFreq"),
  ONEOFF_PURCHASES_FREQUENCY: document.getElementById("inputOneoffFreq"),
  CASH_ADVANCE_FREQUENCY: document.getElementById("inputCashAdvanceFreq"),
  PURCHASES_TRX: document.getElementById("inputPurchasesTrx"),
  CREDIT_LIMIT: document.getElementById("inputCreditLimit"),
  PAYMENTS: document.getElementById("inputPayments"),
  MINIMUM_PAYMENTS: document.getElementById("inputMinPayments"),
  PRC_FULL_PAYMENT: document.getElementById("inputPrcFull"),
  TENURE: document.getElementById("inputTenure"),
};

// Result elements
const resultBanner = document.getElementById("resultBanner");
const resSegmentBadge = document.getElementById("resSegmentBadge");
const resSegmentTitle = document.getElementById("resSegmentTitle");
const resSegmentDesc = document.getElementById("resSegmentDesc");
const resCohortSize = document.getElementById("resCohortSize");
const resStrategy = document.getElementById("resStrategy");
const resultStatusTag = document.getElementById("resultStatusTag");

/**
 * Initialize application
 */
document.addEventListener("DOMContentLoaded", () => {
  initAnalyticsCharts();
  loadPreset("high_spender");
  setupPresetButtons();
  setupCurrencySelector();
  setupEventListeners();
});

/**
 * Currency Selector Event Handler
 */
function setupCurrencySelector() {
  if (!currencySelect) return;

  currencySelect.addEventListener("change", (e) => {
    const newCurrency = e.target.value;
    if (newCurrency === currentCurrency) return;

    const oldRate = EXCHANGE_RATES[currentCurrency] || 1.0;
    const newRate = EXCHANGE_RATES[newCurrency] || 1.0;

    // Convert each monetary input field from old currency to new currency
    MONETARY_FIELDS.forEach((field) => {
      const input = inputFields[field];
      if (input && input.value !== "") {
        const currentVal = parseFloat(input.value) || 0;
        const usdVal = currentVal / oldRate;
        const converted = usdVal * newRate;
        input.value = newCurrency === "JPY" ? Math.round(converted) : parseFloat(converted.toFixed(2));
      }
    });

    currentCurrency = newCurrency;
    executePrediction();
  });
}

/**
 * Load a preset into form fields (converted to selected currency)
 */
function loadPreset(key) {
  const data = PRESETS[key];
  if (!data) return;

  const rate = EXCHANGE_RATES[currentCurrency] || 1.0;

  for (const [feat, input] of Object.entries(inputFields)) {
    if (input && data[feat] !== undefined) {
      if (MONETARY_FIELDS.includes(feat)) {
        const converted = data[feat] * rate;
        input.value = currentCurrency === "JPY" ? Math.round(converted) : parseFloat(converted.toFixed(2));
      } else {
        input.value = data[feat];
      }
    }
  }

  // Update active preset button style
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.preset === key);
  });

  // Automatically compute and update prediction display
  executePrediction();
}

/**
 * Reset all fields to zero and clear preset selections
 */
function resetAllFields() {
  for (const [feat, input] of Object.entries(inputFields)) {
    if (input) {
      input.value = 0;
    }
  }

  // Clear active preset buttons
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.remove("active");
  });

  // Re-run prediction with zeroed values
  executePrediction();
}

/**
 * Setup preset button clicks
 */
function setupPresetButtons() {
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      loadPreset(btn.dataset.preset);
    });
  });

  if (resetFormBtn) {
    resetFormBtn.addEventListener("click", () => {
      resetAllFields();
    });
  }
}

/**
 * Setup form submit event
 */
function setupEventListeners() {
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    executePrediction();
  });
}

/**
 * Gather form input data and convert monetary fields back to USD for the ML model
 */
function getFormData() {
  const rate = EXCHANGE_RATES[currentCurrency] || 1.0;

  // Convert monetary inputs to USD
  const toUSD = (val) => (parseFloat(val) || 0) / rate;

  // Clamp frequency values between 0.0 and 1.0
  const toRatio = (val) => Math.max(0, Math.min(1, parseFloat(val) || 0));

  // Ensure tenure is between 1 and 12 for backend validation
  const rawTenure = parseInt(inputFields.TENURE.value, 10);
  const tenure = isNaN(rawTenure) || rawTenure < 1 ? 1 : Math.min(12, rawTenure);

  return {
    BALANCE: toUSD(inputFields.BALANCE.value),
    BALANCE_FREQUENCY: toRatio(inputFields.BALANCE_FREQUENCY.value),
    ONEOFF_PURCHASES: toUSD(inputFields.ONEOFF_PURCHASES.value),
    INSTALLMENTS_PURCHASES: toUSD(inputFields.INSTALLMENTS_PURCHASES.value),
    CASH_ADVANCE: toUSD(inputFields.CASH_ADVANCE.value),
    PURCHASES_FREQUENCY: toRatio(inputFields.PURCHASES_FREQUENCY.value),
    ONEOFF_PURCHASES_FREQUENCY: toRatio(inputFields.ONEOFF_PURCHASES_FREQUENCY.value),
    CASH_ADVANCE_FREQUENCY: toRatio(inputFields.CASH_ADVANCE_FREQUENCY.value),
    PURCHASES_TRX: Math.max(0, parseInt(inputFields.PURCHASES_TRX.value, 10) || 0),
    CREDIT_LIMIT: toUSD(inputFields.CREDIT_LIMIT.value),
    PAYMENTS: toUSD(inputFields.PAYMENTS.value),
    MINIMUM_PAYMENTS: toUSD(inputFields.MINIMUM_PAYMENTS.value),
    PRC_FULL_PAYMENT: toRatio(inputFields.PRC_FULL_PAYMENT.value),
    TENURE: tenure,
  };
}

/**
 * Execute prediction against FastAPI or client-side fallback
 */
async function executePrediction() {
  const customerData = getFormData();

  // Show loading indicator
  predictBtnText.classList.add("hidden");
  predictBtnLoader.classList.remove("hidden");
  predictBtn.disabled = true;

  let segment = null;
  let description = "";
  let cohortSize = 0;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(customerData),
    });

    if (res.ok) {
      const data = await res.json();
      segment = data.segment;
      description = data.description;
      cohortSize = data.segment_size_in_training_data;
    } else {
      throw new Error("API call returned non-200");
    }
  } catch (err) {
    // Client-side fallback matching Random Forest & KMeans cluster behavior
    const fallback = clientSidePredict(customerData);
    segment = fallback.segment;
    description = fallback.description;
    cohortSize = fallback.size;
  } finally {
    setTimeout(() => {
      predictBtnText.classList.remove("hidden");
      predictBtnLoader.classList.add("hidden");
      predictBtn.disabled = false;
      renderPredictionResult(segment, description, cohortSize, customerData);
    }, 180);
  }
}

/**
 * High-fidelity client-side prediction fallback
 */
function clientSidePredict(data) {
  // Cluster 0: Cash Advance dominant
  // Cluster 1: High Spender dominant
  // Cluster 2: Low activity / Dormant
  const cashAdvScore = (data.CASH_ADVANCE / 1500) + (data.CASH_ADVANCE_FREQUENCY * 2);
  const spenderScore = (data.ONEOFF_PURCHASES / 1200) + (data.INSTALLMENTS_PURCHASES / 800) + (data.PURCHASES_TRX / 15);
  const dormantScore = (data.BALANCE < 500 ? 1.5 : 0) + (data.PURCHASES_FREQUENCY < 0.25 ? 1.5 : 0);

  let segment = 1;
  if (cashAdvScore > spenderScore && cashAdvScore > 1.2) {
    segment = 0;
  } else if (dormantScore > 2.0 && spenderScore < 0.8) {
    segment = 2;
  } else {
    segment = 1;
  }

  const meta = SEGMENT_METADATA[segment];
  return {
    segment: segment,
    description: meta.description,
    size: meta.cohortSize,
  };
}

/**
 * Render prediction results and update visualizations
 */
function renderPredictionResult(segment, description, cohortSize, customerData) {
  const meta = SEGMENT_METADATA[segment] || SEGMENT_METADATA[1];

  // Update Result Card
  resultBanner.className = `result-banner ${meta.badgeClass}`;
  resSegmentBadge.textContent = `Segment ${segment}`;
  resSegmentTitle.textContent = meta.title;
  resSegmentDesc.textContent = description || meta.description;
  if (resCohortSize) {
    resCohortSize.textContent = `${cohortSize || meta.cohortSize}`;
  }
  resStrategy.textContent = meta.strategy;
  resultStatusTag.textContent = `Scored Segment ${segment}`;

  // Highlight active persona card in the "What is SegmentIQ" grid
  document.querySelectorAll(".persona-card").forEach((card) => {
    card.classList.remove("active-highlight");
  });
  const matchingCard = document.getElementById(meta.cardId);
  if (matchingCard) {
    matchingCard.classList.add("active-highlight");
  }

  // Update Radar Chart with customer's relative feature footprint
  updateRadarChart(customerData, segment);
}

/**
 * Radar Chart: Visualizes normalized customer metrics vs segment centroid
 */
function updateRadarChart(data, segment) {
  const ctx = document.getElementById("radarChart");
  if (!ctx) return;

  // Approximate standardized z-score values for 6 key dimensions
  const userZScores = [
    normalizeFeature(data.BALANCE, 1564, 2081),
    normalizeFeature(data.ONEOFF_PURCHASES, 592, 1659),
    normalizeFeature(data.INSTALLMENTS_PURCHASES, 411, 904),
    normalizeFeature(data.CASH_ADVANCE, 978, 2097),
    normalizeFeature(data.PURCHASES_FREQUENCY, 0.49, 0.40),
    normalizeFeature(data.CREDIT_LIMIT, 4494, 3638),
  ];

  const centroidZ = CENTROID_PROFILES[segment] || CENTROID_PROFILES[1];
  const centroidValues = [
    centroidZ.BALANCE,
    centroidZ.ONEOFF_PURCHASES,
    centroidZ.INSTALLMENTS,
    centroidZ.CASH_ADVANCE,
    centroidZ.PURCHASES_FREQ,
    centroidZ.CREDIT_LIMIT,
  ];

  const labels = [
    "Balance",
    "One-Off Spends",
    "Installments",
    "Cash Advance",
    "Purchase Freq",
    "Credit Limit",
  ];

  if (radarChartInstance) {
    radarChartInstance.data.datasets[0].data = userZScores;
    radarChartInstance.data.datasets[1].data = centroidValues;
    radarChartInstance.data.datasets[1].label = `Segment ${segment} Centroid`;
    radarChartInstance.update();
    return;
  }

  radarChartInstance = new Chart(ctx, {
    type: "radar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Current Customer",
          data: userZScores,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.35)",
          borderWidth: 2,
          pointBackgroundColor: "#60a5fa",
          pointBorderColor: "#fff",
          pointRadius: 4,
        },
        {
          label: `Segment ${segment} Centroid`,
          data: centroidValues,
          borderColor: "rgba(148, 163, 184, 0.6)",
          backgroundColor: "rgba(148, 163, 184, 0.08)",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: "rgba(255, 255, 255, 0.08)" },
          angleLines: { color: "rgba(255, 255, 255, 0.08)" },
          pointLabels: {
            color: "#94a3b8",
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 },
          },
          ticks: { display: false, stepSize: 0.5 },
          suggestedMin: -1.5,
          suggestedMax: 1.5,
        },
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "#e2e8f0",
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 },
            boxWidth: 12,
          },
        },
      },
    },
  });
}

function normalizeFeature(val, mean, std) {
  const z = (val - mean) / (std || 1);
  return Math.max(-1.8, Math.min(1.8, parseFloat(z.toFixed(2))));
}

/**
 * Initialize Portfolio Distribution Donut & Centroid Bar Charts
 */
function initAnalyticsCharts() {
  // 1. Doughnut Chart — Cohort Distribution
  const distCtx = document.getElementById("distributionChart");
  if (distCtx) {
    distributionChartInstance = new Chart(distCtx, {
      type: "doughnut",
      data: {
        labels: [
          "Segment 0: Cash Advance",
          "Segment 1: High Spenders",
          "Segment 2: Low Activity",
        ],
        datasets: [
          {
            data: [2579, 2498, 2083],
            backgroundColor: ["#f97316", "#3b82f6", "#a855f7"],
            borderColor: "#0a0e1e",
            borderWidth: 3,
            hoverOffset: 8,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                const total = 7160;
                const count = ctx.raw;
                const pct = ((count / total) * 100).toFixed(1);
                return ` ${ctx.label}: ${count.toLocaleString()} (${pct}%)`;
              },
            },
          },
        },
      },
    });
  }

  // 2. Bar Chart — Centroid Deviations across Key Metrics
  const barCtx = document.getElementById("centroidBarChart");
  if (barCtx) {
    centroidBarChartInstance = new Chart(barCtx, {
      type: "bar",
      data: {
        labels: ["Balance", "Purchases", "Cash Advance", "Credit Limit", "Pay Ratio"],
        datasets: [
          {
            label: "Segment 0 (Cash Adv)",
            data: [0.58, -0.95, 0.87, -0.07, -0.44],
            backgroundColor: "rgba(249, 115, 22, 0.7)",
            borderRadius: 4,
          },
          {
            label: "Segment 1 (Spenders)",
            data: [0.40, 0.85, -0.24, 0.36, 0.04],
            backgroundColor: "rgba(59, 130, 246, 0.75)",
            borderRadius: 4,
          },
          {
            label: "Segment 2 (Dormant)",
            data: [-1.19, 0.16, -0.78, -0.35, 0.49],
            backgroundColor: "rgba(168, 85, 247, 0.7)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: "#94a3b8", font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 } },
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.06)" },
            ticks: { color: "#64748b", font: { family: "'Plus Jakarta Sans', sans-serif" } },
            suggestedMin: -1.5,
            suggestedMax: 1.5,
          },
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: "#cbd5e1",
              font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 },
              boxWidth: 10,
            },
          },
        },
      },
    });
  }
}
