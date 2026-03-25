"""
Generate a self-contained static HTML dashboard from comparison results.
Embeds all CSV data and PNG images directly into a single index.html file
with interactive Plotly charts.

Run: python generate_static_site.py
Output: static_site/index.html
"""

import base64
import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "ml_model" / "comparison_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "static_site"


def read_csv(filename):
    with open(RESULTS_DIR / filename, newline="") as f:
        return list(csv.DictReader(f))


def img_to_data_uri(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    clf_data = read_csv("classification_metrics.csv")
    reg_data = read_csv("regression_metrics.csv")
    summary_data = read_csv("model_summary.csv")

    fi_files = {
        "Xgboost": "feature_importance_xgboost.csv",
        "Random Forest": "feature_importance_random_forest.csv",
        "Gradient Boosting": "feature_importance_gradient_boosting.csv",
    }
    fi_data = {}
    for name, fname in fi_files.items():
        path = RESULTS_DIR / fname
        if path.exists():
            fi_data[name] = read_csv(fname)

    images = {
        "confusion_matrices": img_to_data_uri("confusion_matrices.png"),
        "roc_curves": img_to_data_uri("roc_curves.png"),
        "regressor_comparison": img_to_data_uri("regressor_comparison.png"),
        "feature_importance": img_to_data_uri("feature_importance.png"),
        "classifier_comparison": img_to_data_uri("classifier_comparison.png"),
        "training_time": img_to_data_uri("training_time.png"),
    }

    best_clf = max(clf_data, key=lambda r: float(r["F1 Score"]))
    fastest_clf = min(clf_data, key=lambda r: float(r["Training Time (s)"]))

    clf_json = json.dumps(clf_data)
    reg_json = json.dumps(reg_data)
    summary_json = json.dumps(summary_data)
    fi_json = json.dumps(fi_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Model Comparison Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #ffffff; color: #1a1a1a; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px 24px; }}
h1 {{ font-size: 2rem; margin-bottom: 4px; }}
.caption {{ color: #6b7280; font-size: 0.95rem; margin-bottom: 20px; }}
.banner {{ background: #f0fdf4; border: 1px solid #86efac; border-radius: 12px; padding: 16px 24px; margin-bottom: 20px; }}
.banner h3 {{ color: #166534; margin: 0 0 8px; }}
.banner p {{ color: #166534; margin: 0; }}
.banner b {{ font-weight: 700; }}
.metrics-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }}
.metric-card {{ background: #e6f7f1; border: 1px solid #10a37f; border-radius: 12px; padding: 16px; text-align: center; }}
.metric-card .label {{ font-size: 0.85rem; color: #1a1a1a; margin-bottom: 4px; }}
.metric-card .value {{ font-size: 1.8rem; font-weight: 700; color: #0d8c6d; }}
.divider {{ border: none; border-top: 1px solid #e5e5e5; margin: 24px 0; }}
.tabs {{ display: flex; gap: 0; border-bottom: 2px solid #e5e5e5; margin-bottom: 24px; }}
.tab-btn {{ padding: 10px 24px; border: none; background: none; cursor: pointer; font-size: 1rem; color: #6b7280; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s; }}
.tab-btn.active {{ color: #10a37f; border-bottom-color: #10a37f; font-weight: 600; }}
.tab-btn:hover {{ color: #0d8c6d; }}
.tab-content {{ display: none; }}
.tab-content.active {{ display: block; }}
h2 {{ font-size: 1.4rem; margin-bottom: 16px; color: #1a1a1a; }}
h3 {{ font-size: 1.15rem; margin: 20px 0 12px; color: #1a1a1a; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 0.9rem; }}
th {{ background: #f9fafb; padding: 10px 12px; text-align: left; border-bottom: 2px solid #e5e5e5; font-weight: 600; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; }}
tr:hover td {{ background: #f9fafb; }}
.chart-img {{ width: 100%; border-radius: 8px; margin: 16px 0; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
select {{ padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 0.95rem; margin-bottom: 16px; }}
.download-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 12px; }}
.download-btn {{ display: inline-block; padding: 10px 20px; background: #10a37f; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 0.9rem; text-align: center; text-decoration: none; }}
.download-btn:hover {{ background: #0d8c6d; }}
@media (max-width: 768px) {{
    .metrics-row {{ grid-template-columns: repeat(2, 1fr); }}
    .two-col {{ grid-template-columns: 1fr; }}
    .download-row {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="container">
    <h1>ML Model Comparison Dashboard</h1>
    <p class="caption">7 Models Compared on 500K Health Insurance Claims (4 Traditional + 3 Neural Networks)</p>

    <div class="banner">
        <h3>Best Model: {best_clf['Model']}</h3>
        <p>
            Classification F1: <b>{float(best_clf['F1 Score']):.4f}</b> &nbsp;|&nbsp;
            AUC-ROC: <b>{float(best_clf['AUC-ROC']):.4f}</b> &nbsp;|&nbsp;
            Training: <b>{float(best_clf['Training Time (s)']):.1f}s</b>
            (fastest: {fastest_clf['Model']} at {fastest_clf['Training Time (s)']}s)
        </p>
    </div>

    <div class="metrics-row">
        <div class="metric-card"><div class="label">Models Compared</div><div class="value">7</div></div>
        <div class="metric-card"><div class="label">Training Samples</div><div class="value">500,000</div></div>
        <div class="metric-card"><div class="label">Test Samples</div><div class="value">100,000</div></div>
        <div class="metric-card"><div class="label">Features</div><div class="value">11</div></div>
    </div>

    <hr class="divider">

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('classification')">Classification</button>
        <button class="tab-btn" onclick="switchTab('regression')">Regression</button>
        <button class="tab-btn" onclick="switchTab('training')">Training Time</button>
        <button class="tab-btn" onclick="switchTab('features')">Feature Importance</button>
    </div>

    <!-- Classification Tab -->
    <div id="tab-classification" class="tab-content active">
        <h2>Classification: Approved vs Denied</h2>
        <div id="clf-chart"></div>

        <div class="two-col">
            <div>
                <h3>Confusion Matrix Summary</h3>
                <table id="cm-table"></table>
            </div>
            <div>
                <h3>Raw Metrics Table</h3>
                <table id="clf-table"></table>
            </div>
        </div>

        {"<h3>Confusion Matrices</h3><img class='chart-img' src='" + images['confusion_matrices'] + "'/>" if images['confusion_matrices'] else ""}
        {"<h3>ROC Curves</h3><img class='chart-img' src='" + images['roc_curves'] + "'/>" if images['roc_curves'] else ""}
    </div>

    <!-- Regression Tab -->
    <div id="tab-regression" class="tab-content">
        <h2>Regression: Processing Days Prediction</h2>
        <div id="reg-chart"></div>
        <h3>Regression Metrics Table</h3>
        <table id="reg-table"></table>
        {"<img class='chart-img' src='" + images['regressor_comparison'] + "'/>" if images['regressor_comparison'] else ""}
    </div>

    <!-- Training Time Tab -->
    <div id="tab-training" class="tab-content">
        <h2>Training Time Comparison</h2>
        <div id="training-chart"></div>
        <h3>Speed Ranking</h3>
        <table id="speed-table"></table>
    </div>

    <!-- Feature Importance Tab -->
    <div id="tab-features" class="tab-content">
        <h2>Feature Importance (Tree-Based Models)</h2>
        <select id="fi-model-select" onchange="renderFeatureImportance()"></select>
        <div id="fi-chart"></div>
        <table id="fi-table"></table>
        {"<h3>All Models Feature Importance</h3><img class='chart-img' src='" + images['feature_importance'] + "'/>" if images['feature_importance'] else ""}
    </div>

    <hr class="divider">
    <h2>Complete Model Summary</h2>
    <table id="summary-table"></table>

    <hr class="divider">
    <h2>Export Data</h2>
    <p style="color:#6b7280; margin-bottom:12px;">Download CSVs to import into any BI tool:</p>
    <div class="download-row">
        <button class="download-btn" onclick="downloadCSV('classification')">Classification Metrics CSV</button>
        <button class="download-btn" onclick="downloadCSV('regression')">Regression Metrics CSV</button>
        <button class="download-btn" onclick="downloadCSV('summary')">Model Summary CSV</button>
    </div>
</div>

<script>
const COLORS = ["#10a37f","#3b82f6","#f59e0b","#ef4444","#8b5cf6","#ec4899","#06b6d4"];
const clfData = {clf_json};
const regData = {reg_json};
const summaryData = {summary_json};
const fiData = {fi_json};

// Tab switching
function switchTab(name) {{
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    event.target.classList.add('active');
    // Trigger resize for Plotly charts
    window.dispatchEvent(new Event('resize'));
}}

// Build HTML table
function buildTable(tableId, data, columns) {{
    const table = document.getElementById(tableId);
    let html = '<thead><tr>' + columns.map(c => '<th>' + c + '</th>').join('') + '</tr></thead><tbody>';
    data.forEach(row => {{
        html += '<tr>' + columns.map(c => {{
            let v = row[c];
            if (v !== undefined && !isNaN(v) && v !== '' && c !== 'TN' && c !== 'FP' && c !== 'FN' && c !== 'TP') {{
                let num = parseFloat(v);
                if (Math.abs(num) < 0.01 && num !== 0) v = num.toExponential(2);
                else if (c.includes('Time')) v = num.toFixed(1);
                else if (num !== Math.floor(num)) v = num.toFixed(4);
            }}
            return '<td>' + v + '</td>';
        }}).join('') + '</tr>';
    }});
    html += '</tbody>';
    table.innerHTML = html;
}}

// Classification chart
function renderClfChart() {{
    const metrics = ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"];
    const traces = clfData.map((row, i) => ({{
        name: row.Model,
        x: metrics,
        y: metrics.map(m => parseFloat(row[m])),
        type: 'bar',
        marker: {{ color: COLORS[i % COLORS.length] }},
        text: metrics.map(m => parseFloat(row[m]).toFixed(4)),
        textposition: 'outside',
    }}));
    const minVal = Math.min(...clfData.flatMap(r => metrics.map(m => parseFloat(r[m]))));
    Plotly.newPlot('clf-chart', traces, {{
        barmode: 'group', height: 500,
        title: 'Classification Metrics Comparison',
        paper_bgcolor:'#fff', plot_bgcolor:'#fff',
        font: {{ color:'#4a4a4a', size:13 }},
        yaxis: {{ range: [Math.max(0, minVal - 0.02), 1.005], title:'Score', gridcolor:'#f0f0f0' }},
        xaxis: {{ gridcolor:'#f0f0f0' }},
    }}, {{ responsive: true }});
}}

// Regression chart
function renderRegChart() {{
    const metrics = ["MAE (days)","RMSE (days)","R2 Score"];
    const traces = metrics.map((metric, ci) => ({{
        x: regData.map(r => r.Model),
        y: regData.map(r => parseFloat(r[metric])),
        type: 'bar',
        marker: {{ color: regData.map((_, i) => COLORS[i % COLORS.length]) }},
        text: regData.map(r => parseFloat(r[metric]).toFixed(4)),
        textposition: 'outside',
        showlegend: false,
        xaxis: ci === 0 ? 'x' : 'x' + (ci+1),
        yaxis: ci === 0 ? 'y' : 'y' + (ci+1),
    }}));
    Plotly.newPlot('reg-chart', traces, {{
        grid: {{ rows:1, columns:3, pattern:'independent' }},
        annotations: metrics.map((m, i) => ({{
            text: m, font: {{ size: 14 }}, showarrow: false,
            x: (i + 0.5) / 3, y: 1.08, xref:'paper', yref:'paper',
        }})),
        height: 420, title: 'Regression Metrics Comparison',
        paper_bgcolor:'#fff', plot_bgcolor:'#fff',
        font: {{ color:'#4a4a4a', size:13 }},
    }}, {{ responsive: true }});
}}

// Training time chart
function renderTrainingChart() {{
    const trace1 = {{
        y: clfData.map(r => r.Model), x: clfData.map(r => parseFloat(r['Training Time (s)'])),
        type: 'bar', orientation: 'h',
        marker: {{ color: clfData.map((_, i) => COLORS[i % COLORS.length]) }},
        text: clfData.map(r => parseFloat(r['Training Time (s)']).toFixed(1) + 's'),
        textposition: 'outside', showlegend: false,
        xaxis: 'x', yaxis: 'y',
    }};
    const trace2 = {{
        y: regData.map(r => r.Model), x: regData.map(r => parseFloat(r['Training Time (s)'])),
        type: 'bar', orientation: 'h',
        marker: {{ color: regData.map((_, i) => COLORS[i % COLORS.length]) }},
        text: regData.map(r => parseFloat(r['Training Time (s)']).toFixed(1) + 's'),
        textposition: 'outside', showlegend: false,
        xaxis: 'x2', yaxis: 'y2',
    }};
    Plotly.newPlot('training-chart', [trace1, trace2], {{
        grid: {{ rows:1, columns:2, pattern:'independent' }},
        annotations: [
            {{ text:'Classification', font:{{size:14}}, showarrow:false, x:0.22, y:1.08, xref:'paper', yref:'paper' }},
            {{ text:'Regression', font:{{size:14}}, showarrow:false, x:0.78, y:1.08, xref:'paper', yref:'paper' }},
        ],
        height: 420, title: 'Training Time (500K samples)',
        paper_bgcolor:'#fff', plot_bgcolor:'#fff',
        font: {{ color:'#4a4a4a', size:13 }},
    }}, {{ responsive: true }});
}}

// Feature importance
function renderFeatureImportance() {{
    const select = document.getElementById('fi-model-select');
    const modelName = select.value;
    const data = fiData[modelName];
    if (!data) return;

    const top = data.slice(0, 11);
    Plotly.newPlot('fi-chart', [{{
        y: top.map(r => r.Feature),
        x: top.map(r => parseFloat(r.Importance)),
        type: 'bar', orientation: 'h',
        marker: {{ color: top.map(r => parseFloat(r.Importance)), colorscale: 'Greens' }},
    }}], {{
        title: modelName + ' - Top Features',
        yaxis: {{ categoryorder: 'total ascending', gridcolor: '#f0f0f0' }},
        xaxis: {{ gridcolor: '#f0f0f0' }},
        paper_bgcolor:'#fff', plot_bgcolor:'#fff',
        font: {{ color:'#4a4a4a', size:13 }},
        height: 400,
    }}, {{ responsive: true }});

    buildTable('fi-table', data, ['Feature','Importance','Model']);
}}

// CSV download
function downloadCSV(type) {{
    let data, filename;
    if (type === 'classification') {{ data = clfData; filename = 'classification_metrics.csv'; }}
    else if (type === 'regression') {{ data = regData; filename = 'regression_metrics.csv'; }}
    else {{ data = summaryData; filename = 'model_summary.csv'; }}

    const cols = Object.keys(data[0]);
    const csv = [cols.join(','), ...data.map(r => cols.map(c => r[c]).join(','))].join('\\n');

    const blob = new Blob([csv], {{ type: 'text/csv' }});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
}}

// Init
renderClfChart();

buildTable('cm-table', clfData.map(r => ({{
    Model: r.Model, TN: r.TN, FP: r.FP, FN: r.FN, TP: r.TP,
    'Total Errors': parseInt(r.FP) + parseInt(r.FN),
}})), ['Model','TN','FP','FN','TP','Total Errors']);

buildTable('clf-table', clfData, ['Model','Accuracy','Precision','Recall','F1 Score','AUC-ROC','Training Time (s)']);
buildTable('reg-table', regData, ['Model','MAE (days)','RMSE (days)','R2 Score','Training Time (s)']);
buildTable('summary-table', summaryData, Object.keys(summaryData[0]));

// Speed ranking
const speedData = [
    ...clfData.map(r => ({{Model:r.Model, 'Training Time (s)':r['Training Time (s)'], Task:'Classification'}})),
    ...regData.map(r => ({{Model:r.Model, 'Training Time (s)':r['Training Time (s)'], Task:'Regression'}})),
].sort((a, b) => parseFloat(a['Training Time (s)']) - parseFloat(b['Training Time (s)']));
buildTable('speed-table', speedData, ['Model','Training Time (s)','Task']);

// Feature importance setup
const fiSelect = document.getElementById('fi-model-select');
Object.keys(fiData).forEach(name => {{
    const opt = document.createElement('option');
    opt.value = name; opt.textContent = name;
    fiSelect.appendChild(opt);
}});
if (Object.keys(fiData).length > 0) renderFeatureImportance();

// Lazy render charts on tab switch
document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        setTimeout(() => {{
            renderClfChart();
            renderRegChart();
            renderTrainingChart();
        }}, 50);
    }});
}});
</script>
</body>
</html>"""

    output_path = OUTPUT_DIR / "index.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Static site generated: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
