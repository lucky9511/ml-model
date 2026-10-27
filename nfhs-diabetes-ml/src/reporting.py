"""
Clinical reporting and visualization helper module for nfhs-diabetes-ml.
Includes Plotly interactive gauges, counterfactual comparison visuals,
batch triage distribution charts, and hospital-grade printable HTML clinical reports.
"""

from datetime import datetime
from typing import Any, Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def create_risk_gauge_fig(
    predicted_prob: float,
    optimal_threshold: float = 0.7225,
    baseline_prev: float = 0.0224,
) -> go.Figure:
    """
    Construct an interactive Plotly semicircular gauge for patient risk score.
    """
    risk_pct = round(predicted_prob * 100, 1)
    threshold_pct = round(optimal_threshold * 100, 1)

    # Color for bar based on tier
    if predicted_prob < 0.30:
        bar_color = "#059669"  # Emerald
    elif predicted_prob < optimal_threshold:
        bar_color = "#d97706"  # Amber
    else:
        bar_color = "#dc2626"  # Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 42, "weight": "bold", "color": bar_color}},
            delta={
                "reference": round(baseline_prev * 100, 1),
                "position": "bottom",
                "increasing": {"color": "#dc2626"},
                "decreasing": {"color": "#059669"},
                "suffix": "% vs Baseline",
            },
            title={
                "text": "<b>Predicted Diabetes Risk Probability</b>",
                "font": {"size": 16, "color": "#1e293b"},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": bar_color, "thickness": 0.35},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, 30], "color": "#ecfdf5"},
                    {"range": [30, threshold_pct], "color": "#fffbeb"},
                    {"range": [threshold_pct, 100], "color": "#fef2f2"},
                ],
                "threshold": {
                    "line": {"color": "#991b1b", "width": 4},
                    "thickness": 0.85,
                    "value": threshold_pct,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=25, r=25, t=40, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def create_counterfactual_comparison_fig(
    current_prob: float,
    simulated_prob: float,
) -> go.Figure:
    """
    Construct side-by-side bar chart showing Current vs Simulated 'What-If' Risk.
    """
    categories = ["Current Assessment", "What-If Simulated"]
    values = [round(current_prob * 100, 1), round(simulated_prob * 100, 1)]
    colors = ["#ef4444" if current_prob >= 0.72 else "#f59e0b", "#10b981"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=categories,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                marker_color=colors,
                width=0.45,
            )
        ]
    )

    fig.update_layout(
        title="<b>Current vs. Simulated 'What-If' Diabetes Risk Score</b>",
        yaxis=dict(title="Probability (%)", range=[0, max(values) * 1.3 + 10]),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def create_batch_distribution_fig(results_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Generate Donut chart of Risk Tiers and Histogram of Predicted Probabilities.
    """
    tier_counts = results_df["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["risk_tier", "count"]

    color_map = {
        "High Risk / Screening Indicated": "#ef4444",
        "Moderate Risk": "#f59e0b",
        "Low Risk": "#10b981",
    }

    # Donut Chart
    fig_donut = px.pie(
        tier_counts,
        values="count",
        names="risk_tier",
        hole=0.55,
        color="risk_tier",
        color_discrete_map=color_map,
        title="<b>Cohort Risk Tier Stratification</b>",
    )
    fig_donut.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Histogram
    fig_hist = px.histogram(
        results_df,
        x="predicted_risk_pct",
        nbins=20,
        color="risk_tier",
        color_discrete_map=color_map,
        title="<b>Distribution of Predicted Risk Scores (%)</b>",
        labels={"predicted_risk_pct": "Predicted Diabetes Risk (%)"},
    )
    fig_hist.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig_donut, fig_hist


def generate_clinical_html_report(
    patient_data: Dict[str, Any],
    predicted_prob: float,
    optimal_threshold: float,
    risk_tier: str,
    top_risk_factors: List[Tuple[str, float]],
    top_protective_factors: List[Tuple[str, float]],
    clinical_advice: List[str],
) -> str:
    """
    Generate an exhaustive, printable hospital-branded HTML Clinical Report.
    """
    now_str = datetime.now().strftime("%B %d, %Y at %H:%M")
    risk_pct = f"{predicted_prob * 100:.1f}%"
    thresh_pct = f"{optimal_threshold * 100:.1f}%"

    if predicted_prob >= optimal_threshold:
        tier_badge = f'<span style="background:#fee2e2; color:#991b1b; padding:6px 14px; border-radius:20px; font-weight:bold;">HIGH RISK — ACTIONABLE SCREENING INDICATED</span>'
    elif predicted_prob >= 0.30:
        tier_badge = f'<span style="background:#fef3c7; color:#92400e; padding:6px 14px; border-radius:20px; font-weight:bold;">MODERATE RISK TIER</span>'
    else:
        tier_badge = f'<span style="background:#d1fae5; color:#065f46; padding:6px 14px; border-radius:20px; font-weight:bold;">LOW CLINICAL RISK</span>'

    risk_rows = "".join(
        [f"<tr><td><b>{feat}</b></td><td style='color:#b91c1c; font-weight:bold;'>+{val:.3f} log-odds</td></tr>" for feat, val in top_risk_factors]
    ) or "<tr><td colspan='2'>No significant risk elevating features noted.</td></tr>"

    protective_rows = "".join(
        [f"<tr><td><b>{feat}</b></td><td style='color:#059669; font-weight:bold;'>{val:.3f} log-odds</td></tr>" for feat, val in top_protective_factors]
    ) or "<tr><td colspan='2'>No significant protective features noted.</td></tr>"

    advice_list = "".join([f"<li style='margin-bottom:8px;'>{item}</li>" for item in clinical_advice])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Clinical Diabetes Risk Assessment Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            body {{
                font-family: 'Inter', sans-serif;
                color: #1e293b;
                line-height: 1.5;
                padding: 40px;
                max-width: 850px;
                margin: 0 auto;
                background-color: #ffffff;
            }}
            .header-banner {{
                border-bottom: 3px solid #0d9488;
                padding-bottom: 20px;
                margin-bottom: 25px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .report-title {{
                font-size: 24px;
                font-weight: 800;
                color: #0f172a;
                margin: 0;
            }}
            .report-sub {{
                font-size: 13px;
                color: #64748b;
                margin-top: 4px;
            }}
            .summary-box {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 25px;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                font-size: 13px;
            }}
            th, td {{
                padding: 10px 12px;
                text-align: left;
                border-bottom: 1px solid #e2e8f0;
            }}
            th {{
                background-color: #f1f5f9;
                font-weight: 700;
                color: #334155;
            }}
            .advice-section {{
                background: #eff6ff;
                border-left: 4px solid #3b82f6;
                padding: 18px;
                border-radius: 6px;
                margin-top: 25px;
            }}
            .footer {{
                margin-top: 40px;
                border-top: 1px solid #cbd5e1;
                padding-top: 15px;
                font-size: 11px;
                color: #94a3b8;
                text-align: center;
            }}
            @media print {{
                body {{ padding: 0; }}
                .no-print {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="header-banner">
            <div>
                <h1 class="report-title">🩺 Clinical Diabetes Risk Assessment</h1>
                <div class="report-sub">NFHS-5 Tabular Diagnostic Decision Support System | Generated {now_str}</div>
            </div>
            <div>
                {tier_badge}
            </div>
        </div>

        <div class="summary-box">
            <h3 style="margin-top:0; color:#0f172a; font-size:16px;">Assessment Summary</h3>
            <div class="grid-2">
                <div>
                    <div><strong>Patient Profile:</strong> {patient_data.get('age', 'N/A')} yrs, {patient_data.get('residence', 'N/A')}, {patient_data.get('region', 'N/A')} Region</div>
                    <div><strong>Anthropometry:</strong> BMI {patient_data.get('bmi', 'N/A')} kg/m² ({patient_data.get('bmi_category', 'N/A')})</div>
                    <div><strong>Vitals:</strong> SBP {patient_data.get('systolic_bp', 'N/A')} mmHg / DBP {patient_data.get('diastolic_bp', 'N/A')} mmHg</div>
                </div>
                <div>
                    <div><strong>Estimated Risk Score:</strong> <span style="font-size:18px; font-weight:800; color:#b91c1c;">{risk_pct}</span></div>
                    <div><strong>Triage Decision Cutoff:</strong> {thresh_pct}</div>
                    <div><strong>National Baseline Risk:</strong> 2.24%</div>
                </div>
            </div>
        </div>

        <div class="grid-2">
            <div>
                <h4 style="margin-bottom:5px; color:#991b1b;">⬆️ Key Risk Contributors (SHAP)</h4>
                <table>
                    <thead><tr><th>Factor</th><th>Impact</th></tr></thead>
                    <tbody>{risk_rows}</tbody>
                </table>
            </div>
            <div>
                <h4 style="margin-bottom:5px; color:#065f46;">⬇️ Protective Attributes (SHAP)</h4>
                <table>
                    <thead><tr><th>Factor</th><th>Impact</th></tr></thead>
                    <tbody>{protective_rows}</tbody>
                </table>
            </div>
        </div>

        <div class="advice-section">
            <h4 style="margin-top:0; color:#1e40af;">📋 Clinical Action & Lifestyle Protocols</h4>
            <ul>{advice_list}</ul>
        </div>

        <div class="footer">
            Confidential Medical Diagnostic Aid. Based on NFHS-5 Women's Recode (India 2019–21). 
            This assessment provides statistical risk stratification and should be corroborated with formal clinical HbA1c or FPG biological testing.
        </div>
    </body>
    </html>
    """
    return html
