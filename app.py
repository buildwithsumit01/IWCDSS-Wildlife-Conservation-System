"""
IWCDSS - Intelligent Wildlife Conservation Decision Support System
Tiger Presence/Absence Prediction & Conservation Management System
Integrated Platform covering Tasks 1 through 9
"""

import os
import io
import time
import json
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Import existing core pipeline
import pipeline

# -----------------------------------------------------------------------------
# Configuration & Theme Styling (Forest Green & White)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="IWCDSS - Wildlife Conservation Decision Support",
    page_icon="🐅",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Primary Colors & Typography */
    :root {
        --forest-dark: #1b4332;
        --forest-mid: #2d6a4f;
        --forest-light: #52b788;
        --forest-pale: #e8f5e9;
        --text-dark: #1a2e26;
        --border-color: #d1e7dd;
    }
    
    /* Global Styles */
    .stApp {
        background-color: #fcfdfc;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(27, 67, 50, 0.15);
    }
    .header-box h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 6px !important;
    }
    .header-box p {
        color: #e8f5e9 !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }
    
    /* Card Containers */
    .iwcdss-card {
        background-color: #ffffff;
        border: 1px solid #d8e2dc;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .iwcdss-card-header {
        color: #1b4332;
        font-weight: 600;
        font-size: 1.25rem;
        border-bottom: 2px solid #52b788;
        padding-bottom: 8px;
        margin-bottom: 14px;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 4px;
    }
    .badge-high { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
    .badge-mod { background-color: #fff3cd; color: #664d03; border: 1px solid #ffecb5; }
    .badge-low { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
    .badge-full { background-color: #cfe2ff; color: #084298; border: 1px solid #b6d4fe; }

    /* Distinction Callout Box */
    .distinction-box {
        background-color: #f0f7f4;
        border-left: 5px solid #2d6a4f;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
ASSETS_DIR = os.path.join(APP_DIR, "assets")

# -----------------------------------------------------------------------------
# Artifact Loader (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models & preprocessing artifacts...")
def get_artifacts():
    return pipeline.load_artifacts(MODELS_DIR)

try:
    art = get_artifacts()
    artifacts_ready = True
except Exception as e:
    art = {"error": str(e)}
    artifacts_ready = False

# -----------------------------------------------------------------------------
# Header Display
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1>🐅 IWCDSS — Intelligent Wildlife Conservation Decision Support System</h1>
    <p>Integrated Framework: Knowledge Representation • Rule-Based Expert Systems • State Space Search • ML & DNN Modeling • Explainable AI (XAI)</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Navigation Tabs (5 Core Sections as Approved)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Overview & Foundation",
    "🎯 Tiger Presence Predictor",
    "📈 Model Benchmarks & Evaluation",
    "🔍 Explainable AI (SHAP & LIME)",
    "🗺️ Patrol Route Optimizer"
])

tab_overview, tab_predict, tab_benchmark, tab_xai, tab_patrol = tabs

# =============================================================================
# SECTION 1: OVERVIEW DASHBOARD & FOUNDATIONAL TASKS (Tasks 1, 2, 4)
# =============================================================================
with tab_overview:
    st.markdown("## 🌿 Executive System Architecture & Foundations")
    
    # Task 4 Dataset Distinction Callout (Requirement 4)
    st.markdown("""
    <div class="distinction-box">
        <h4 style="color: #1b4332; margin-top: 0;">📌 Crucial Project Dataset Distinction (Task 4 vs. Tasks 5–9)</h4>
        <p style="margin-bottom: 6px;">
            <strong>1. Task 4 Data Preparation & EDA Dataset (7,989 Records):</strong> In Task 4, WEKA data preprocessing performed missing-value imputation, duplicate removal (pruning 96 duplicate records from 8,085 down to 7,989 instances), and interquartile-range (IQR) outlier/extreme-value classification (flagging 5,130 extreme spatial coordinates).
        </p>
        <p style="margin-bottom: 0;">
            <strong>2. Tasks 5–9 Feature Matrix & Modeling Dataset (8,085 Records):</strong> Tasks 5 through 9 utilized the baseline field survey dataset (<code>Data.csv</code>, 8,085 rows $\\times$ 22 columns) to generate the full 40-column feature matrix (<code>feature_matrix.csv</code>). This ensured no rare observation points or delicate prey signs were lost before frequency and dummy encoding.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        st.markdown("""
        <div class="iwcdss-card">
            <div class="iwcdss-card-header">Task 1: Knowledge Representation & Domain Ontology</div>
            <p>Formal semantic modeling of the wildlife reserve ecosystem defining <strong>12 primary conservation entities</strong> and <strong>17 semantic object property relationships</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        t1_img = os.path.join(ASSETS_DIR, "task1_ontology_er.png")
        if os.path.exists(t1_img):
            st.image(t1_img, caption="Task 1: IWCDSS Conceptual Entity-Relationship & Domain Architecture", use_container_width=True)
        
        with st.expander("View 12 Core Entities & Semantic Relationships"):
            st.markdown("""
            **Core Entities:**
            1. `Wildlife Species` (SpeciesID, CommonName, ConservationStatus, PopulationCount)
            2. `Habitat` (HabitatID, HabitatType, VegetationType, Area, WaterAvailability)
            3. `Protected Area` (ProtectedAreaID, AreaName, Location, Zone)
            4. `Camera Trap` (CameraTrapID, GPSLocation, InstallationDate, Status)
            5. `Animal Observation` (ObservationID, ObservationDate, ObservationTime, ImageID)
            6. `Environmental Sensor` (SensorID, SensorType, SensorLocation, Value)
            7. `Weather` (WeatherID, Temperature, Humidity, Rainfall, WindSpeed)
            8. `Poaching Incident` (IncidentID, IncidentDate, Severity)
            9. `Ranger` (RangerID, RangerName, Experience)
            10. `Patrol Route` (RouteID, Distance)
            11. `Risk Level` (RiskCategory, RiskScore)
            12. `Conservation Recommendation` (RecommendationID, Priority, RecommendedAction)

            **Key Relationships:**
            - `Wildlife Species` — *livesIn* ➔ `Habitat`
            - `Camera Trap` — *installedIn* ➔ `Protected Area`
            - `Camera Trap` — *captures* ➔ `Animal Observation`
            - `Animal Observation` — *identifies* ➔ `Wildlife Species`
            - `Ranger` — *patrols* ➔ `Patrol Route`
            - `Protected Area` — *hasRiskLevel* ➔ `Risk Level`
            - `Risk Level` — *triggers* ➔ `Conservation Recommendation`
            """)

    with col_t2:
        st.markdown("""
        <div class="iwcdss-card">
            <div class="iwcdss-card-header">Task 2: Rule-Based Threat Monitoring Expert System</div>
            <p>Inference engine executing 10 formal production rules (R1–R10) for threat detection, habitat stress alerts, and surveillance scheduling via forward and backward chaining.</p>
        </div>
        """, unsafe_allow_html=True)

        t2_img = os.path.join(ASSETS_DIR, "task2_expert_system.jpeg")
        if os.path.exists(t2_img):
            st.image(t2_img, caption="Task 2: Rule-Based Expert System Architecture & Chaining Engine", use_container_width=True)

        with st.expander("View Task 2 Formal Rules (R1 to R10) & Test Scenarios"):
            st.markdown("""
            | Rule | IF (Condition) | THEN (Action / Recommendation) |
            | :--- | :--- | :--- |
            | **R1** | Conservation Status = "Endangered" AND Previous Poaching = "Yes" | Risk = "High" ➔ Recommend: Increase Ranger Patrols |
            | **R2** | Camera Trap Status = "Inactive" | Recommend: Repair or Install Camera Trap |
            | **R3** | Patrol Frequency = "Low" | Recommend: Increase Patrol Frequency |
            | **R4** | Temperature > 40°C | Recommend: Habitat Heat Warning |
            | **R5** | Humidity < 30% | Recommend: Habitat Dryness Alert |
            | **R6** | Risk = "High" (R1 fired) | Recommend: Deploy Drone Surveillance |
            | **R7** | Animal Observation = "Not Detected" | Recommend: Conduct Wildlife Survey |
            | **R8** | Poaching Incidents Last Year ≥ 4 | Recommend: Increase Monitoring |
            | **R9** | Species = "Rhino" | Recommend: Notify Wildlife Authorities |
            | **R10** | Risk = "High" AND Conservation Status = "Critically Endangered" | Recommend: Immediate Conservation Action Required |
            """)

    # Interactive Task 2 Threat Rule Evaluator
    st.markdown("### ⚡ Interactive Threat & Sanctuary Rule Evaluator")
    with st.form("task2_evaluator"):
        c1, c2, c3, c4 = st.columns(4)
        status_input = c1.selectbox("Conservation Status", ["Least Concern", "Vulnerable", "Endangered", "Critically Endangered"], index=2)
        poaching_input = c2.selectbox("Previous Poaching?", ["No", "Yes"], index=1)
        camera_input = c3.selectbox("Camera Trap Status", ["Active", "Inactive", "Maintenance"], index=0)
        patrol_input = c4.selectbox("Patrol Frequency", ["High", "Medium", "Low"], index=2)

        c5, c6, c7, c8 = st.columns(4)
        temp_input = c5.number_input("Ambient Temperature (°C)", value=38.0, step=1.0)
        hum_input = c6.number_input("Humidity (%)", value=28.0, step=1.0)
        incidents_input = c7.number_input("Poaching Incidents (Last Year)", value=2, step=1)
        species_input = c8.selectbox("Target Species", ["Bengal Tiger", "Rhino", "Elephant", "Wild Boar", "Other"], index=0)

        eval_btn = st.form_submit_button("Evaluate Threat Rules")

    if eval_btn:
        fired = []
        recs = []
        risk = "Low"

        if status_input == "Endangered" and poaching_input == "Yes":
            fired.append("R1: Endangered + Previous Poaching")
            risk = "High"
            recs.append("Increase Ranger Patrols")
        if camera_input == "Inactive":
            fired.append("R2: Inactive Camera Trap")
            recs.append("Repair or Install Camera Trap")
        if patrol_input == "Low":
            fired.append("R3: Low Patrol Frequency")
            recs.append("Increase Patrol Frequency")
        if temp_input > 40:
            fired.append("R4: Temperature > 40°C")
            recs.append("Habitat Heat Warning")
        if hum_input < 30:
            fired.append("R5: Humidity < 30%")
            recs.append("Habitat Dryness Alert")
        if risk == "High":
            fired.append("R6: Escalated High Threat")
            recs.append("Deploy Drone Surveillance")
        if incidents_input >= 4:
            fired.append("R8: Poaching Incidents ≥ 4")
            recs.append("Increase Monitoring")
        if species_input == "Rhino":
            fired.append("R9: Priority Rhino Sighting")
            recs.append("Notify Wildlife Authorities")
        if risk == "High" and status_input == "Critically Endangered":
            fired.append("R10: Critical Risk Escalation")
            recs.append("Immediate Conservation Action Required")

        st.success(f"**Assessed Threat Risk:** {risk}")
        if fired:
            st.markdown(f"**Fired Rules ({len(fired)}):**")
            for r in fired:
                st.write(f"- ⚠️ `{r}`")
            st.markdown(f"**Recommended Conservation Actions:**")
            for a in set(recs):
                st.write(f"- ✅ **{a}**")
        else:
            st.info("No threat escalation rules fired. Sanctuary status normal.")

    # Task 4 Exploratory Data Analysis Gallery
    st.markdown("### 🗺️ Task 4: Exploratory Data Analysis & Quality Inspection")
    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        p_mat = os.path.join(ASSETS_DIR, "task4_plot_matrix.png")
        if os.path.exists(p_mat):
            st.image(p_mat, caption="Task 4: WEKA Multivariate Attribute Plot Matrix", use_container_width=True)
    with eda_col2:
        p_spat = os.path.join(ASSETS_DIR, "task4_spatial_distribution.png")
        if os.path.exists(p_spat):
            st.image(p_spat, caption="Task 4: Spatial Longitude vs. Latitude Survey Clusters (Grid_10_10)", use_container_width=True)

    with st.expander("View Additional WEKA Distribution & Quality Charts"):
        w_cols = st.columns(3)
        w_cols[0].image(os.path.join(ASSETS_DIR, "task4_attribute_dist_1.png"), caption="Terrain & Habitat Distributions", use_container_width=True)
        w_cols[1].image(os.path.join(ASSETS_DIR, "task4_attribute_dist_2.png"), caption="Coordinate Distributions", use_container_width=True)
        w_cols[2].image(os.path.join(ASSETS_DIR, "task4_extreme_value_indicator.png"), caption="WEKA IQR Extreme Value Flagging", use_container_width=True)


# =============================================================================
# SECTION 2: TIGER PRESENCE PREDICTOR (Tasks 5, 6, 7, 9)
# =============================================================================
with tab_predict:
    st.markdown("## 🎯 Live Tiger Presence/Absence Assessment")
    st.markdown("Real-time decision support combining **Ecological Heuristic Rules**, **XGBoost ML**, **Dense DNN DL**, and **Live SHAP Explanations**.")

    if not artifacts_ready:
        st.error(f"Prediction artifacts unavailable: {art.get('error', 'Unknown error')}")
        st.stop()

    enc = art["encoding"]
    cat_levels = enc["category_levels"]
    freq_maps = enc["freq_maps"]

    mode = st.radio("Select Input Mode:", ["📍 Manual Survey Point Form", "📂 Batch Survey CSV Upload"], horizontal=True)

    if mode == "📍 Manual Survey Point Form":
        with st.form("survey_form"):
            st.markdown("##### 1. Geographic & Environmental Context")
            g1, g2, g3 = st.columns(3)
            lat = g1.number_input("Latitude (°N)", min_value=25.0, max_value=32.0, value=27.5367, format="%.5f")
            lon = g2.number_input("Longitude (°E)", min_value=80.0, max_value=89.0, value=84.0750, format="%.5f")
            month = g3.selectbox("Survey Month", cat_levels["Survey_Month"], index=cat_levels["Survey_Month"].index("December") if "December" in cat_levels["Survey_Month"] else 0)

            e1, e2, e3 = st.columns(3)
            terrain = e1.selectbox("Terrain Type", cat_levels["Terrain"], index=cat_levels["Terrain"].index("Terai") if "Terai" in cat_levels["Terrain"] else 0)
            habitat = e2.selectbox("Habitat Type", cat_levels["Habitat_ty"], index=cat_levels["Habitat_ty"].index("Mf") if "Mf" in cat_levels["Habitat_ty"] else 0)
            landform = e3.selectbox("Featured Landform", cat_levels["Featured_F"], index=cat_levels["Featured_F"].index("Stream") if "Stream" in cat_levels["Featured_F"] else 0)

            s1, s2, s3 = st.columns(3)
            loc_opts = sorted(list(freq_maps["Location"].keys()))
            g2_opts = sorted(list(freq_maps["Grid_2_2"].keys()))
            g10_opts = sorted(list(freq_maps["Grid_10_10"].keys()))
            loc_val = s1.selectbox("Survey Sector / Location", loc_opts)
            grid2_val = s2.selectbox("Fine Grid (2x2 km)", g2_opts)
            grid10_val = s3.selectbox("Coarse Grid (10x10 km)", g10_opts)

            st.markdown("##### 2. Animal Sign Counts (Pugmarks, Scats, Scrapes)")
            st.caption("Prey and competitor species signs recorded within the transect cell.")
            
            p_cols = st.columns(6)
            spotted_deer = p_cols[0].number_input("Spotted Deer", min_value=0, max_value=50, value=3)
            wild_boar = p_cols[1].number_input("Wild Boar", min_value=0, max_value=50, value=2)
            sambar = p_cols[2].number_input("Sambar", min_value=0, max_value=50, value=1)
            barking_deer = p_cols[3].number_input("Barking Deer", min_value=0, max_value=50, value=0)
            blue_bull = p_cols[4].number_input("Blue Bull", min_value=0, max_value=50, value=0)
            gaur = p_cols[5].number_input("Gaur", min_value=0, max_value=50, value=1)

            c_cols = st.columns(3)
            elephant = c_cols[0].number_input("Elephant Sign", min_value=0, max_value=50, value=0)
            leopard = c_cols[1].number_input("Leopard Sign", min_value=0, max_value=50, value=0)
            other_carni = c_cols[2].number_input("Other Carnivore Sign", min_value=0, max_value=50, value=0)

            other_species = st.checkbox("Notable rare species sign present (Rhino, Sloth Bear, etc.)", value=False)

            assess_btn = st.form_submit_button("🚀 Run Comprehensive Tiger Assessment")

        if assess_btn:
            raw_input = {
                "Lattitude": lat,
                "Longitude": lon,
                "Survey_Month": month,
                "Terrain": terrain,
                "Habitat_ty": habitat,
                "Featured_F": landform,
                "Location": loc_val,
                "Grid_2_2": grid2_val,
                "Grid_10_10": grid10_val,
                "Spotted Deer": spotted_deer,
                "Wild Boar": wild_boar,
                "Sambar": sambar,
                "Barking Deer": barking_deer,
                "Blue Bull": blue_bull,
                "Gaur": gaur,
                "Elephant": elephant,
                "Leopard": leopard,
                "other_carni": other_carni,
                "Other_Species_Flag": 1 if other_species else 0
            }

            res = pipeline.run_full_assessment(raw_input, art)
            if not res["valid"]:
                st.error("Validation Error: " + "; ".join(res["errors"]))
            else:
                st.session_state["assessment_result"] = res
                st.session_state["assessment_raw"] = raw_input

        if "assessment_result" in st.session_state:
            res = st.session_state["assessment_result"]
            rec = res["recommendation"]
            ex = res["expert"]
            ml = res["ml"]
            dl = res["dl"]

            st.markdown("---")
            st.markdown("### 📋 Multi-Module Assessment Scorecard")

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Expert Heuristic Rules", ex["label"], f"Score: {ex['score']}")
            with sc2:
                st.metric("ML Model (XGBoost)", ml["label"], f"{ml['probability']:.1%}")
            with sc3:
                if dl.get("probability") is not None:
                    st.metric("DL Model (Dense DNN)", dl["label"], f"{dl['probability']:.1%}")
                else:
                    st.metric("DL Model (Dense DNN)", "Unavailable", "Model unready")
            with sc4:
                st.metric("Conservation Priority", rec["priority"], f"Avg: {rec['average_probability']:.1%}")

            # Priority Alert Box
            p_style = "badge-high" if rec["priority"] == "High" else ("badge-mod" if rec["priority"] == "Moderate" else "badge-low")
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-left: 6px solid #2d6a4f; padding: 16px 20px; border-radius: 6px; margin: 15px 0;">
                <span class="metric-badge {p_style}">Priority: {rec['priority']}</span>
                <span class="metric-badge badge-full">Consensus: {rec['agreement']}</span>
                <h4 style="color: #1b4332; margin: 10px 0 6px 0;">Recommendation & Field Intervention:</h4>
                <p style="font-size: 1.05rem; margin-bottom: 8px;"><strong>{rec['action']}</strong></p>
                <ul style="margin-bottom: 0;">
                    {''.join([f'<li>{note}</li>' for note in rec['notes']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # Local SHAP & Fired Rules Breakdown
            sh_col, r_col = st.columns([1, 1])

            with sh_col:
                st.markdown("##### 🔍 Why? Live Local SHAP Attribution (XGBoost)")
                sd = res["shap_contributions"].head(8).iloc[::-1]
                if len(sd) > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = ["#2d6a4f" if v > 0 else "#c9184a" for v in sd["shap"]]
                    ax.barh(sd["feature"], sd["shap"], color=colors)
                    ax.set_xlabel("SHAP Impact (Green = Pushes to Present, Red = Pushes to Absent)")
                    ax.set_title("Local Predictor Drivers for This Survey Point")
                    st.pyplot(fig)
                else:
                    st.info("SHAP attribution could not be computed for this prediction.")

            with r_col:
                st.markdown("##### 📜 Field-Ecology Rules Fired")
                if ex["rules_fired"]:
                    for pts, txt in ex["rules_fired"]:
                        symbol = "➕" if pts > 0 else "➖"
                        st.write(f"- {symbol} **{txt}**")
                else:
                    st.write("- No specific field rules triggered.")
                st.caption(f"Note: {ex['note']}")

            # Word Report Download
            st.markdown("##### 📄 Export Official Assessment Summary")
            doc_buf = pipeline.build_report_docx(res, st.session_state["assessment_raw"])
            st.download_button(
                label="📥 Download Conservation Field Report (.docx)",
                data=doc_buf.getvalue(),
                file_name=f"IWCDSS_Field_Report_{lat:.2f}_{lon:.2f}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    else:
        # Batch Mode
        st.markdown("##### Upload Multi-Row Field Survey CSV")
        st.markdown("""
        Required CSV headers: `Lattitude`, `Longitude`, `Terrain`, `Habitat_ty`, `Featured_F`, `Location`, `Grid_2_2`, `Grid_10_10`, 
        `Spotted Deer`, `Wild Boar`, `Sambar`, `Barking Deer`, `Blue Bull`, `Gaur`, `Elephant`, `Leopard`, `other_carni`.
        """)
        up_file = st.file_uploader("Select survey CSV file", type=["csv"])
        if up_file is not None:
            batch_in = pd.read_csv(up_file)
            st.write(f"Uploaded {len(batch_in)} survey rows.")
            if st.button("Process Batch Assessment"):
                with st.spinner("Processing transect surveys..."):
                    batch_out = pipeline.batch_assess(batch_in, art)
                st.dataframe(batch_out, use_container_width=True)
                
                # Priority breakdown
                if "priority" in batch_out.columns:
                    st.markdown("###### Priority Distribution:")
                    st.bar_chart(batch_out["priority"].value_counts())

                csv_data = batch_out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Results CSV",
                    data=csv_data,
                    file_name="IWCDSS_Batch_Assessment_Results.csv",
                    mime="text/csv"
                )


# =============================================================================
# SECTION 3: MODEL PERFORMANCE & BENCHMARKS (Tasks 5, 6, 7)
# =============================================================================
with tab_benchmark:
    st.markdown("## 📈 Machine Learning & Deep Learning Benchmarks")
    st.markdown("Official evaluations and reproducible comparative metrics preserved from project notebooks.")

    # Official Metrics Preserved (Requirement 3)
    st.markdown("### 🏆 Official Model Comparison Leaderboard")
    official_data = {
        "Model": ["XGBoost (Best Model)", "Decision Tree", "Random Forest", "Support Vector Machine (SVM)", "Logistic Regression", "Dense DNN (Deep Learning)"],
        "Type": ["Gradient Boosted Tree", "Decision Tree", "Ensemble Tree", "Kernel Maximum Margin", "Linear Baseline", "4-Layer Feedforward NN"],
        "Accuracy": [0.8745, 0.8856, 0.8794, 0.7842, 0.7687, 0.7749],
        "Precision": [0.6236, 0.6840, 0.6935, 0.4453, 0.4178, 0.4379],
        "Recall": [0.7629, 0.6770, 0.5911, 0.8110, 0.7251, 0.8899],
        "F1-Score": [0.6862, 0.6805, 0.6382, 0.5749, 0.5302, 0.5870],
        "ROC-AUC": [0.9292, 0.8036, 0.9193, 0.8810, 0.8305, 0.9062],
        "Primary Strength": [
            "Optimal F1 & ROC-AUC; best balanced presence detection",
            "Fast, interpretable baseline with good accuracy",
            "High precision (0.6935) with minimal false alarms",
            "High recall (0.8110) finding elusive presence",
            "Simple linear probability baseline",
            "Maximum Recall (0.8899); misses only 24 of 218 presences"
        ]
    }
    st.dataframe(pd.DataFrame(official_data).set_index("Model"), use_container_width=True)

    st.markdown("""
    > **Ecological Significance of Recall:** In tiger conservation, **False Negatives** (failing to detect an active tiger zone) are significantly more catastrophic than False Positives. An occupied forest pocket predicted as unoccupied risks losing ranger patrols, leaving tigers vulnerable to poaching. The **Dense DNN** achieves an extraordinary **88.99% Recall**, missing only 24 actual presences.
    """)

    # Visual Artifacts Display (Tasks 5, 6, 7)
    st.markdown("### 🖼️ Evaluation & Feature Engineering Visualizations")

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        m_chart = os.path.join(ASSETS_DIR, "model_comparison_chart.png")
        if os.path.exists(m_chart):
            st.image(m_chart, caption="Task 6: 5-Model Comparative Performance Metrics", use_container_width=True)
    with g_col2:
        roc_chart = os.path.join(ASSETS_DIR, "ROC_Curves.png")
        if os.path.exists(roc_chart):
            st.image(roc_chart, caption="Task 6: Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        cm_chart = os.path.join(ASSETS_DIR, "Confusion_Matrix.png")
        if os.path.exists(cm_chart):
            st.image(cm_chart, caption="Task 6: Confusion Matrices Across Supervised Models", use_container_width=True)
    with cm_col2:
        dnn_chart = os.path.join(ASSETS_DIR, "DNN_training_curves.png")
        if os.path.exists(dnn_chart):
            st.image(dnn_chart, caption="Task 7: Deep Neural Network (DNN) Loss & Accuracy Training History", use_container_width=True)

    with st.expander("View Additional Feature Engineering & Unsupervised Check (K-Means)"):
        f_cols = st.columns(3)
        f_cols[0].image(os.path.join(ASSETS_DIR, "feature_importance_chart.png"), caption="Task 5: Random Forest Feature Importance", use_container_width=True)
        f_cols[1].image(os.path.join(ASSETS_DIR, "correlation_matrix_heatmap.png"), caption="Task 5: Collinear Feature Correlation Heatmap", use_container_width=True)
        f_cols[2].image(os.path.join(ASSETS_DIR, "kmeans_clusters.png"), caption="Task 6: K-Means Spatial Clusters (Silhouette = 0.2032)", use_container_width=True)


# =============================================================================
# SECTION 4: EXPLAINABLE AI (XAI) (Task 8)
# =============================================================================
with tab_xai:
    st.markdown("## 🔍 Explainable Artificial Intelligence (XAI)")
    st.markdown("Demystifying the black-box XGBoost and DNN models for field rangers and conservation biologists using SHAP, LIME, Counterfactuals, and Fairness Audits.")

    st.markdown("### 🌍 Global Feature Attribution (Macro Insights)")
    st.write("""
    Global SHAP explains which ecological factors generally dictate tiger occupancy throughout the entire protected landscape:
    1. **Spatial Coordinates (`Lattitude`, `Longitude`):** Tigers exhibit strong home range fidelity and geographical clustering within core river valleys.
    2. **Sampling Effort (`Location_freq`, `Grid2_Sampling_Effort`):** Areas with systematic survey repetition provide higher signal resolution.
    3. **Prey Biomass (`Total_Prey_Sign`, `Gaur`, `Wild Boar`, `Sambar`):** Large ungulate availability directly governs apex predator territory carrying capacity.
    """)

    xai_col1, xai_col2 = st.columns(2)
    with xai_col1:
        f_task6 = os.path.join(ASSETS_DIR, "feature_importance_task6.png")
        if os.path.exists(f_task6):
            st.image(f_task6, caption="Task 8: Global Feature Importance Ranking (XGBoost)", use_container_width=True)
    with xai_col2:
        dnn_cm = os.path.join(ASSETS_DIR, "DNN_confusion_matrix.png")
        if os.path.exists(dnn_cm):
            st.image(dnn_cm, caption="Task 7/8: DNN Classification Matrix (Low False Negatives: 24)", use_container_width=True)

    st.markdown("### 🔬 Counterfactual & Local Reasoning Analysis")
    st.markdown("""
    **What would need to change for a missed tiger presence (False Negative) to flip to Present?**
    - In Task 8, counterfactual perturbation sweeps the primary drivers (`Total_Prey_Sign` and `Location_freq`) holding all other factors constant.
    - **Finding:** In low-effort grid cells where tiger sign was undetected, an increase of **+3 prey signs (Sambar/Chital)** or **2 additional patrol surveys** increases model confidence past the 0.5 decision boundary, proving that lack of detection in remote sectors is largely an artifact of lower survey intensity.
    """)

    st.markdown("### 🛡️ Trust Metrics & Ecological Subgroup Fairness")
    tf_c1, tf_c2 = st.columns(2)
    with tf_c1:
        st.markdown("""
        <div class="iwcdss-card">
            <div class="iwcdss-card-header">Model Trust Scores</div>
            <p><strong>Confidence Trust Metric: 83.2%</strong><br>
            Average distance of predicted probability from the uncertain 0.5 decision boundary.</p>
            <p><strong>Explanation Agreement: High</strong><br>
            Concordance between SHAP Shapley values and LIME local linear approximations on top-ranking features.</p>
        </div>
        """, unsafe_allow_html=True)
    with tf_c2:
        st.markdown("""
        <div class="iwcdss-card">
            <div class="iwcdss-card-header">Terrain Fairness Audit</div>
            <p><strong>Chure Hilly Terrain:</strong> Accuracy 86.8%, Recall 74.5%<br>
            <strong>Terai Plains:</strong> Accuracy 87.9%, Recall 77.1%<br>
            <strong>Riverbank / Streambed:</strong> Accuracy 88.2%, Recall 79.4%<br>
            <em>Conclusion: Consistent sensitivity across all terrain strata; no systematic geographic discrimination.</em></p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SECTION 5: PATROL ROUTE OPTIMIZER (Task 3)
# =============================================================================
with tab_patrol:
    st.markdown("## 🗺️ State Space Search & Patrol Route Optimization")
    st.markdown("Formulation of anti-poaching patrol routing across sanctuary zones comparing **Breadth-First Search (BFS)**, **Depth-First Search (DFS)**, and **A* Search**.")

    # State Space Graph & Heuristic Table from Task 3
    NODES = {
        "Entry Gate": {"h": 8, "coords": (0, 0)},
        "Observation Point": {"h": 6, "coords": (1, 2)},
        "Ranger Station": {"h": 7, "coords": (2, 0)},
        "Waterhole": {"h": 3, "coords": (3, 2)},
        "Camera Trap Zone": {"h": 4, "coords": (3, -1)},
        "Dense Forest": {"h": 1, "coords": (5, 1)},
        "Grassland": {"h": 2, "coords": (4, -2)},
        "Poaching Hotspot": {"h": 0, "coords": (6, 0)}  # Goal
    }

    EDGES = [
        ("Entry Gate", "Observation Point", 3),
        ("Entry Gate", "Ranger Station", 2),
        ("Observation Point", "Waterhole", 3),
        ("Observation Point", "Camera Trap Zone", 4),
        ("Ranger Station", "Camera Trap Zone", 3),
        ("Waterhole", "Dense Forest", 2),
        ("Camera Trap Zone", "Grassland", 3),
        ("Camera Trap Zone", "Dense Forest", 4),
        ("Grassland", "Poaching Hotspot", 4),
        ("Dense Forest", "Poaching Hotspot", 1)
    ]

    p_col1, p_col2 = st.columns([1, 1])
    with p_col1:
        st.markdown("""
        <div class="iwcdss-card">
            <div class="iwcdss-card-header">Task 3 State Space Model</div>
            <p>Ranger patrols must navigate from the <strong>Entry Gate</strong> to the high-risk <strong>Poaching Hotspot</strong> while minimizing total distance, fuel/wear, and surveillance blind spots.</p>
        </div>
        """, unsafe_allow_html=True)

        ss_img = os.path.join(ASSETS_DIR, "task3_state_space.png")
        if os.path.exists(ss_img):
            st.image(ss_img, caption="Task 3: State Space Diagram & Patrol Network Graph", use_container_width=True)

    with p_col2:
        st.markdown("##### 📍 Heuristic Values Table $h(n)$ to Poaching Hotspot")
        h_table = [{"Sanctuary Location": k, "Heuristic Distance h(n)": v["h"]} for k, v in NODES.items()]
        st.dataframe(pd.DataFrame(h_table).set_index("Sanctuary Location"), use_container_width=True)

        st.markdown(r"""
        | Strategy | Path Cost | Optimality | Time Complexity | Space Complexity |
        | :--- | :---: | :---: | :---: | :---: |
        | **BFS** | Not Cost Based | Yes (Unweighted) | $O(V+E)$ | High ($O(b^d)$) |
        | **DFS** | Not Cost Based | No | $O(V+E)$ | Low ($O(bm)$) |
        | **A\* Search** | **8 km (Optimal)** | **Yes (Optimal)** | Heuristic dependent | Moderate |
        """)

    # Interactive Route Search Runner
    st.markdown("### 🧭 Interactive Patrol Route Solver")
    with st.form("patrol_form"):
        sc1, sc2, sc3 = st.columns(3)
        start_node = sc1.selectbox("Patrol Origin", list(NODES.keys()), index=0)
        goal_node = sc2.selectbox("Surveillance Target", list(NODES.keys()), index=7)
        algo_choice = sc3.selectbox("Search Algorithm", ["A* Search (Heuristic + Cost)", "Breadth-First Search (BFS)", "Depth-First Search (DFS)"])
        run_route = st.form_submit_button("Find Patrol Route")

    if run_route:
        # Build adjacency graph
        adj = {n: [] for n in NODES}
        for u, v, w in EDGES:
            adj[u].append((v, w))
            adj[v].append((u, w))

        if algo_choice.startswith("A*"):
            # A* Search
            import heapq
            open_set = []
            heapq.heappush(open_set, (NODES[start_node]["h"], 0, start_node, [start_node]))
            visited = {}
            best_path = None
            total_cost = None

            while open_set:
                f, g, curr, path = heapq.heappop(open_set)
                if curr == goal_node:
                    best_path = path
                    total_cost = g
                    break
                if curr in visited and visited[curr] <= g:
                    continue
                visited[curr] = g
                for neighbor, weight in adj[curr]:
                    new_g = g + weight
                    h = NODES[neighbor]["h"]
                    heapq.heappush(open_set, (new_g + h, new_g, neighbor, path + [neighbor]))

            st.success(f"**A* Optimal Patrol Route:** {' ➔ '.join(best_path)}")
            st.info(f"**Total Patrol Distance:** {total_cost} km | **Heuristic Goal Estimate:** 0 km")

        elif algo_choice.startswith("Breadth"):
            from collections import deque
            queue = deque([(start_node, [start_node], 0)])
            visited = {start_node}
            best_path = None
            total_cost = 0

            while queue:
                curr, path, cost = queue.popleft()
                if curr == goal_node:
                    best_path = path
                    total_cost = cost
                    break
                for neighbor, weight in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor], cost + weight))

            st.success(f"**BFS Route (Level-wise exploration):** {' ➔ '.join(best_path)}")
            st.info(f"**Path Cost:** {total_cost} km | Guaranteed minimum hops in unweighted graphs.")

        else:
            # DFS
            stack = [(start_node, [start_node], 0)]
            visited = set()
            best_path = None
            total_cost = 0

            while stack:
                curr, path, cost = stack.pop()
                if curr == goal_node:
                    best_path = path
                    total_cost = cost
                    break
                if curr not in visited:
                    visited.add(curr)
                    for neighbor, weight in adj[curr]:
                        if neighbor not in visited:
                            stack.append((neighbor, path + [neighbor], cost + weight))

            st.success(f"**DFS Route (Depth-first traversal):** {' ➔ '.join(best_path)}")
            st.warning(f"**Path Cost:** {total_cost} km | Note: DFS is not guaranteed to find the minimum distance path.")

    with st.expander("View Task 3 Search Trees & Cost Curves from Report"):
        t3_cols = st.columns(3)
        t3_cols[0].image(os.path.join(ASSETS_DIR, "task3_bfs_tree.png"), caption="BFS Level-Wise Search Tree", use_container_width=True)
        t3_cols[1].image(os.path.join(ASSETS_DIR, "task3_dfs_tree.png"), caption="DFS Branch Search Tree", use_container_width=True)
        t3_cols[2].image(os.path.join(ASSETS_DIR, "task3_astar_tree.png"), caption="A* Optimal Heuristic Search Tree", use_container_width=True)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; font-size: 0.9rem; padding: 10px;">
    <strong>IWCDSS — Intelligent Wildlife Conservation Decision Support System</strong><br>
    Developed for Group 4 Project • Designed with Forest-Green & White Theme • All 9 CI Tasks Fully Integrated
</div>
""", unsafe_allow_html=True)
