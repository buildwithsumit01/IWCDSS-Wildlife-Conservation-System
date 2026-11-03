import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="IWCDSS - Tiger Decision Support", page_icon="🐅", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    return pipeline.load_artifacts(HERE)


art = get_artifacts()
enc = art["encoding"]

st.title("🐅 IWCDSS - Wildlife Conservation Decision Support System")
st.caption("Group 4 | Tiger presence/absence | Expert rules + XGBoost + DNN + SHAP + Recommendations")
tab_assess, tab_dash, tab_about = st.tabs(["Assess a Survey Point", "Dashboard", "About"])


def show_result(res):
    for w in res["warnings"]:
        st.warning(w)
    rec, ex, ml, dl = res["recommendation"], res["expert"], res["ml"], res["dl"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expert rules", ex["label"], f"score {ex['score']}")
    c2.metric("ML - XGBoost", ml["label"], f"{ml['probability']:.1%}")
    c3.metric("DL - DNN", dl["label"], "n/a" if dl["probability"] is None else f"{dl['probability']:.1%}")
    c4.metric("Priority", rec["priority"], f"avg {rec['average_probability']:.1%}")

    box = st.success if rec["priority"] in ("High", "Moderate") else st.info
    box(f"**Recommendation:** {rec['action']}  \n*{rec['agreement']} across modules.*")
    for n in rec["notes"]:
        st.write("- " + n)

    left, right = st.columns(2)
    with left:
        st.subheader("Why? (live SHAP, XGBoost)")
        sd = res["shap_contributions"].head(10).iloc[::-1]
        if len(sd):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(sd["feature"], sd["shap"], color=["#2e7d32" if v > 0 else "#c62828" for v in sd["shap"]])
            ax.set_xlabel("SHAP value (green = towards Present)")
            st.pyplot(fig)
        else:
            st.info("SHAP explanation unavailable.")
    with right:
        st.subheader("Expert-system rules fired")
        for _, txt in ex["rules_fired"] or [(0, "No rules fired")]:
            st.write("- " + txt)
        st.caption(ex["note"])

    st.download_button("📄 Download assessment report (.docx)", pipeline.build_report_docx(res),
                       file_name="IWCDSS_assessment_report.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


with tab_assess:
    mode = st.radio("Input mode", ["Manual form", "Batch CSV"], horizontal=True)
    if mode == "Manual form":
        with st.form("point"):
            a, b, c = st.columns(3)
            lat = a.number_input("Latitude", value=27.55, format="%.5f")
            lon = b.number_input("Longitude", value=84.10, format="%.5f")
            month = c.selectbox("Survey month", enc["category_levels"]["Survey_Month"])
            d, e, f = st.columns(3)
            terrain = d.selectbox("Terrain", enc["category_levels"]["Terrain"])
            habitat = e.selectbox("Habitat type", enc["category_levels"]["Habitat_ty"])
            feature = f.selectbox("Featured feature", enc["category_levels"]["Featured_F"])
            g, h, i = st.columns(3)
            location = g.selectbox("Location", sorted(enc["freq_maps"]["Location"]))
            grid2 = h.selectbox("Grid 2x2", sorted(enc["freq_maps"]["Grid_2_2"]))
            grid10 = i.selectbox("Grid 10x10", sorted(enc["freq_maps"]["Grid_10_10"]))
            st.markdown("**Sign counts**")
            cols = st.columns(5)
            counts = {k: cols[n % 5].number_input(k, min_value=0, value=0, step=1)
                      for n, k in enumerate(pipeline.COUNT_COLS)}
            other = st.checkbox("Other species recorded at this point")
            go = st.form_submit_button("Run assessment")
        if go:
            raw = {"Lattitude": lat, "Longitude": lon, "Survey_Month": month, "Terrain": terrain,
                   "Habitat_ty": habitat, "Featured_F": feature, "Location": location,
                   "Grid_2_2": grid2, "Grid_10_10": grid10, "Other_Species_Flag": other, **counts}
            res = pipeline.run_full_assessment(raw, art)
            if res["valid"]:
                st.session_state["last"] = res
            else:
                st.error("Invalid input: " + "; ".join(res["errors"]))
        if "last" in st.session_state:
            show_result(st.session_state["last"])
    else:
        st.write("Upload a CSV with columns: Lattitude, Longitude, Terrain, Habitat_ty, Featured_F, Location, "
                 "Grid_2_2, Grid_10_10, " + ", ".join(pipeline.COUNT_COLS) + " (optional: Survey_Month).")
        up = st.file_uploader("Survey CSV", type="csv")
        if up is not None:
            out = pipeline.batch_assess(pd.read_csv(up), art)
            st.session_state["batch"] = out
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results (.csv)", out.to_csv(index=False), "batch_results.csv", "text/csv")

with tab_dash:
    st.subheader("Model performance (Tasks 6-8)")
    csv = os.path.join(HERE, "Model_Comparison_Report.csv")
    if os.path.exists(csv):
        st.dataframe(pd.read_csv(csv), use_container_width=True)
    shown = False
    imgs = ["model_comparison_chart.png", "ROC_Curves.png", "DNN_training_curves.png",
            "feature_importance_recap.png", "SHAP_summary_bar_XGBoost.png", "SHAP_summary_bar_DNN.png"]
    cols = st.columns(2)
    for n, f in enumerate(imgs):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            cols[n % 2].image(p, caption=f, use_container_width=True)
            shown = True
    if not (shown or os.path.exists(csv)):
        st.info("No dashboard assets were bundled. Re-run the notebook's optional upload cell to add them.")
    if "batch" in st.session_state:
        st.subheader("Latest batch run")
        b = st.session_state["batch"]
        ok = b[b["status"] == "OK"]
        if len(ok):
            st.bar_chart(ok["priority"].value_counts())

with tab_about:
    st.markdown("""
**Pipeline:** input validation → feature engineering (Task 5) → expert rules → XGBoost (Task 6) → DNN (Task 7)
→ SHAP (Task 8) → recommendation engine → downloadable report.

**Notes**
- The expert system is a transparent rule-based heuristic, not a trained model.
- This is decision support only; confirm with field evidence before acting.
""")
