"""
Unit tests for clinical reporting, Plotly gauge generators, and HTML export.
"""

import pandas as pd
import pytest
import plotly.graph_objects as go

from src.reporting import (
    create_risk_gauge_fig,
    create_counterfactual_comparison_fig,
    create_batch_distribution_fig,
    generate_clinical_html_report,
)


def test_create_risk_gauge_fig():
    """Verify Plotly semicircular risk gauge generation."""
    fig = create_risk_gauge_fig(0.45, optimal_threshold=0.72)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert fig.data[0].type == "indicator"


def test_create_counterfactual_comparison_fig():
    """Verify counterfactual side-by-side bar chart generation."""
    fig = create_counterfactual_comparison_fig(current_prob=0.82, simulated_prob=0.35)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].x) == ["Current Assessment", "What-If Simulated"]


def test_create_batch_distribution_fig():
    """Verify batch triage Donut and Histogram construction."""
    mock_df = pd.DataFrame({
        "predicted_risk_pct": [12.0, 45.0, 78.0, 85.0],
        "risk_tier": ["Low Risk", "Moderate Risk", "High Risk / Screening Indicated", "High Risk / Screening Indicated"],
    })

    fig_donut, fig_hist = create_batch_distribution_fig(mock_df)
    assert isinstance(fig_donut, go.Figure)
    assert isinstance(fig_hist, go.Figure)


def test_generate_clinical_html_report():
    """Verify hospital-grade printable HTML report generation."""
    patient_data = {
        "age": 45,
        "residence": "Urban",
        "region": "South",
        "bmi": 28.5,
        "bmi_category": "Overweight (25.0-29.9)",
        "systolic_bp": 138,
        "diastolic_bp": 88,
    }
    top_risks = [("Age (Years)", 0.52), ("BMI: Overweight (25.0-29.9)", 0.28)]
    top_protective = [("Diet: Pulses/Beans Frequency", -0.12)]
    advice = ["Targeted weight reduction of 5–10% recommended."]

    html = generate_clinical_html_report(
        patient_data=patient_data,
        predicted_prob=0.78,
        optimal_threshold=0.7225,
        risk_tier="High / Actionable Screening Indicated",
        top_risk_factors=top_risks,
        top_protective_factors=top_protective,
        clinical_advice=advice,
    )

    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "Clinical Diabetes Risk Assessment" in html
    assert "78.0%" in html
    assert "Age (Years)" in html
    assert "Targeted weight reduction" in html
