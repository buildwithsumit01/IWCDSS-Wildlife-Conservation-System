"""
IWCDSS - Intelligent Wildlife Conservation Decision Support System (Group 4)
Core pipeline: validation -> feature engineering -> expert system -> ML -> DL -> SHAP -> recommendation -> report.

No Streamlit calls live here, so every function can be tested on its own.
"""
import io
import os
import joblib
import numpy as np
import pandas as pd

PREY_COLS = ["Spotted Deer", "Wild Boar", "Sambar", "Barking Deer", "Blue Bull", "Gaur"]
COUNT_COLS = PREY_COLS + ["Elephant", "Leopard", "other_carni"]
CAT_COLS = ["Terrain", "Habitat_ty", "Featured_F", "Location"]
GRID_COLS = ["Grid_2_2", "Grid_10_10"]
TARGET = "Tiger"
THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_artifacts(bundle_dir="."):
    """Load every saved artifact from the app bundle folder."""
    p = lambda f: os.path.join(bundle_dir, f)
    art = {"bundle_dir": bundle_dir}
    art["encoding"] = joblib.load(p("encoding_reference.pkl"))
    art["selected_features"] = pd.read_csv(p("selected_features.csv"))["selected_feature"].tolist()
    fm = pd.read_csv(p("feature_matrix.csv"))
    art["all_features"] = [c for c in fm.columns if c != TARGET]
    art["feature_medians"] = fm.drop(columns=[TARGET]).median(numeric_only=True).to_dict()
    art["ml_model"] = joblib.load(p("xgboost.pkl"))

    art["dl_model"], art["dl_scaler"] = None, None
    if os.path.exists(p("IWCDSS_Tiger_DNN.h5")):
        try:
            from tensorflow.keras.models import load_model
            art["dl_model"] = load_model(p("IWCDSS_Tiger_DNN.h5"), compile=False)
            art["dl_scaler"] = joblib.load(p("IWCDSS_DNN_scaler.pkl"))
        except Exception as e:  # keep app usable if TF is unavailable
            art["dl_error"] = str(e)

    art["shap_explainer"] = None
    try:
        import shap
        art["shap_explainer"] = shap.TreeExplainer(art["ml_model"])
    except Exception as e:
        art["shap_error"] = str(e)
    return art


def _model_columns(obj, art):
    """Work out which feature columns (and order) a fitted model/scaler expects."""
    names = getattr(obj, "feature_names_in_", None)
    if names is None and hasattr(obj, "get_booster"):
        try:
            names = obj.get_booster().feature_names
        except Exception:
            names = None
    if names is not None:
        return list(names)
    n = getattr(obj, "n_features_in_", None)
    if n == len(art["all_features"]):
        return art["all_features"]
    return art["selected_features"]


# --------------------------------------------------------------------------- #
# 1. Input validation
# --------------------------------------------------------------------------- #
def _clean_text(s):
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return "Unknown"
    return str(s).strip().title()


def validate_input(raw, artifacts=None):
    """Returns dict: ok, errors, warnings, clean (cleaned copy of the record)."""
    errors, warnings = [], []
    clean = dict(raw)

    for col in ["Lattitude", "Longitude"]:
        try:
            clean[col] = float(raw.get(col))
        except (TypeError, ValueError):
            errors.append(f"{col} is missing or not a number.")
    if "Lattitude" in clean and isinstance(clean["Lattitude"], float) and not -90 <= clean["Lattitude"] <= 90:
        errors.append("Lattitude must be between -90 and 90.")
    if "Longitude" in clean and isinstance(clean["Longitude"], float) and not -180 <= clean["Longitude"] <= 180:
        errors.append("Longitude must be between -180 and 180.")

    for col in COUNT_COLS:
        try:
            v = float(raw.get(col, 0) or 0)
            if v < 0:
                errors.append(f"{col} cannot be negative.")
            clean[col] = v
        except (TypeError, ValueError):
            errors.append(f"{col} must be a number.")

    for col in CAT_COLS:
        clean[col] = _clean_text(raw.get(col))
    clean["Featured_F"] = {"Forest Raod": "Forest Road"}.get(clean["Featured_F"], clean["Featured_F"])
    for col in GRID_COLS:
        v = raw.get(col)
        clean[col] = "Unknown" if v is None or str(v).strip() == "" else str(v).strip()
    clean["Survey_Month"] = _clean_text(raw.get("Survey_Month"))
    clean["Other_Species_Flag"] = int(bool(raw.get("Other_Species_Flag", 0)))

    if artifacts is not None and not errors:
        enc = artifacts["encoding"]
        for col in ["Location", "Grid_2_2", "Grid_10_10"]:
            if clean[col] not in enc["freq_maps"][col]:
                warnings.append(f"{col} '{clean[col]}' was not in the training data; "
                                f"using the rarest-category frequency instead.")
        for col, levels in enc["category_levels"].items():
            if clean[col] not in levels:
                warnings.append(f"{col} '{clean[col]}' is unseen; treated as the baseline category.")
        if sum(clean[c] for c in PREY_COLS) == 0:
            warnings.append("No prey sign recorded at this point.")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "clean": clean}


# --------------------------------------------------------------------------- #
# 2. Feature engineering (reproduces Task 5)
# --------------------------------------------------------------------------- #
def engineer_features(clean, artifacts):
    """Turn one cleaned survey record into a one-row frame aligned with the Task 5 feature matrix."""
    enc = artifacts["encoding"]
    row = {
        "Lattitude": clean["Lattitude"], "Longitude": clean["Longitude"],
        "Elephant": clean["Elephant"], "Leopard": clean["Leopard"], "other_carni": clean["other_carni"],
    }
    for c in PREY_COLS:
        row[c] = clean[c]
    prey = [clean[c] for c in PREY_COLS]
    row["Total_Prey_Sign"] = sum(prey)
    row["Prey_Richness"] = sum(1 for v in prey if v > 0)
    row["Other_Carnivore_Sign"] = int(clean["Leopard"] > 0 or clean["other_carni"] > 0)
    row["Other_Species_Flag"] = clean["Other_Species_Flag"]
    row["Grid2_Sampling_Effort"] = enc["grid2_counts"].get(clean["Grid_2_2"], 1)  # same convention as Task 5 (count of surveys in the grid)

    for col in ["Location", "Grid_2_2", "Grid_10_10"]:
        row[col + "_freq"] = enc["freq_maps"][col].get(clean[col], enc["min_freq"][col])

    # one-hot with drop_first, same naming as pd.get_dummies
    for col, levels in enc["category_levels"].items():
        for lv in levels[1:]:
            row[f"{col}_{lv}"] = int(clean[col] == lv)

    X = pd.DataFrame([row])
    X = X.reindex(columns=artifacts["all_features"])
    for c in X.columns:                       # anything not computable -> training median
        if X[c].isna().any():
            X[c] = X[c].fillna(artifacts["feature_medians"].get(c, 0))
    return X.astype(float)


# --------------------------------------------------------------------------- #
# 3. Expert system (transparent rule-based stand-in)
# --------------------------------------------------------------------------- #
def expert_system(clean, X=None):
    """IF-THEN heuristic score. NOT a trained model - it encodes field-ecology rules of thumb."""
    score, fired = 0, []
    total_prey = sum(clean[c] for c in PREY_COLS)
    richness = sum(1 for c in PREY_COLS if clean[c] > 0)
    large_prey = clean["Sambar"] + clean["Gaur"] + clean["Spotted Deer"]

    def rule(cond, pts, text):
        nonlocal score
        if cond:
            score += pts
            fired.append((pts, text))

    rule(total_prey >= 6, 2, "R1: High total prey sign (>= 6) -> +2")
    rule(0 < total_prey < 6, 1, "R2: Some prey sign (1-5) -> +1")
    rule(total_prey == 0, -2, "R3: No prey sign -> -2")
    rule(richness >= 3, 2, "R4: Three or more prey species present -> +2")
    rule(large_prey >= 3, 1, "R5: Large prey (sambar/gaur/chital) sign >= 3 -> +1")
    rule(clean["Terrain"].lower() in ("streambed", "nala", "riverbed", "stream"), 1,
         "R6: Water-linked terrain (tigers use streambeds/nalas) -> +1")
    rule(clean["Featured_F"].lower() in ("stream", "waterhole", "forest road", "trail"), 1,
         "R7: Movement/water feature (stream, waterhole, road, trail) -> +1")
    rule(clean["Elephant"] >= 3, -1, "R8: Heavy elephant disturbance -> -1")
    rule(clean["Leopard"] > 0, -1, "R9: Leopard sign (competitor) -> -1")

    prob = float(np.clip(0.5 + score * 0.08, 0.02, 0.98))
    label = "Present" if score >= 3 else "Absent"
    return {"score": score, "probability": prob, "label": label, "rules_fired": fired,
            "note": "Rule-based heuristic (not a trained model)."}


# --------------------------------------------------------------------------- #
# 4/5. ML and DL predictions
# --------------------------------------------------------------------------- #
def _binary_proba(model, X):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0][1])
    return float(np.ravel(model.predict(X))[0])


def predict_ml(X, artifacts):
    model = artifacts["ml_model"]
    cols = _model_columns(model, artifacts)
    p = _binary_proba(model, X.reindex(columns=cols).fillna(0))
    return {"probability": p, "label": "Present" if p >= THRESHOLD else "Absent", "model": "XGBoost (Task 6)"}


def predict_dl(X, artifacts):
    if artifacts.get("dl_model") is None:
        return {"probability": None, "label": "Unavailable", "model": "DNN (Task 7)",
                "error": artifacts.get("dl_error", "DNN not loaded")}
    cols = _model_columns(artifacts["dl_scaler"], artifacts)
    Xs = artifacts["dl_scaler"].transform(X.reindex(columns=cols).fillna(0))
    p = float(np.ravel(artifacts["dl_model"].predict(Xs, verbose=0))[0])
    return {"probability": p, "label": "Present" if p >= THRESHOLD else "Absent", "model": "DNN (Task 7)"}


# --------------------------------------------------------------------------- #
# 6. Live SHAP explanation
# --------------------------------------------------------------------------- #
def explain_ml_prediction(X, artifacts):
    """Per-feature SHAP contributions for this one prediction (positive = pushes towards Tiger Present)."""
    expl = artifacts.get("shap_explainer")
    if expl is None:
        return pd.DataFrame(columns=["feature", "value", "shap"])
    cols = _model_columns(artifacts["ml_model"], artifacts)
    Xm = X.reindex(columns=cols).fillna(0)
    sv = expl.shap_values(Xm)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    df = pd.DataFrame({"feature": cols, "value": Xm.iloc[0].values, "shap": sv[0]})
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Recommendation engine
# --------------------------------------------------------------------------- #
def recommendation_engine(expert, ml, dl, shap_df, clean):
    probs = [ml["probability"]] + ([dl["probability"]] if dl.get("probability") is not None else [])
    avg = float(np.mean(probs))
    labels = {ml["label"], expert["label"]} | ({dl["label"]} if dl.get("probability") is not None else set())
    agreement = "Full agreement" if len(labels) == 1 else "Models disagree"

    if avg >= 0.75:
        level, action = "High", "Prioritise this point for monitoring: place camera traps, increase patrol frequency, protect the corridor."
    elif avg >= 0.5:
        level, action = "Moderate", "Likely tiger use: schedule a follow-up survey and consider a camera trap."
    elif avg >= 0.3:
        level, action = "Low", "Tiger use is uncertain: include in routine surveys and re-check after the next season."
    else:
        level, action = "Very low", "Tiger presence unlikely: keep as a low-priority monitoring point."

    notes = []
    if agreement == "Models disagree":
        notes.append("The expert rules, ML and DL models disagree - treat this as uncertain and verify in the field.")
    if clean["Elephant"] >= 3:
        notes.append("Heavy elephant activity: take care with patrol safety and camera placement.")
    if sum(clean[c] for c in PREY_COLS) == 0:
        notes.append("No prey sign here: consider a prey-base survey in the surrounding grid.")
    if clean["Leopard"] > 0:
        notes.append("Leopard sign recorded: possible competitor; interpret tiger absence with caution.")
    if len(shap_df):
        top = shap_df.iloc[0]
        notes.append(f"Most influential factor for the ML model: {top['feature']} "
                     f"({'raises' if top['shap'] > 0 else 'lowers'} tiger probability).")
    return {"average_probability": avg, "priority": level, "agreement": agreement,
            "action": action, "notes": notes}


# --------------------------------------------------------------------------- #
# Orchestrators
# --------------------------------------------------------------------------- #
def run_full_assessment(raw, artifacts):
    v = validate_input(raw, artifacts)
    if not v["ok"]:
        return {"valid": False, "errors": v["errors"], "warnings": v["warnings"]}
    clean = v["clean"]
    X = engineer_features(clean, artifacts)
    expert = expert_system(clean, X)
    ml = predict_ml(X, artifacts)
    dl = predict_dl(X, artifacts)
    shap_df = explain_ml_prediction(X, artifacts)
    rec = recommendation_engine(expert, ml, dl, shap_df, clean)
    return {"valid": True, "errors": [], "warnings": v["warnings"], "clean": clean, "features": X,
            "expert": expert, "ml": ml, "dl": dl, "shap_contributions": shap_df, "recommendation": rec}


def batch_assess(df, artifacts):
    """Assess every row of a CSV; returns a summary table (SHAP skipped for speed)."""
    rows = []
    for i, rec in enumerate(df.to_dict("records")):
        v = validate_input(rec, artifacts)
        if not v["ok"]:
            rows.append({"row": i + 1, "status": "Invalid: " + "; ".join(v["errors"])})
            continue
        X = engineer_features(v["clean"], artifacts)
        ex, ml, dl = expert_system(v["clean"], X), predict_ml(X, artifacts), predict_dl(X, artifacts)
        r = recommendation_engine(ex, ml, dl, pd.DataFrame(), v["clean"])
        rows.append({"row": i + 1, "status": "OK", "expert": ex["label"],
                     "ml_prob": round(ml["probability"], 3), "dl_prob": None if dl["probability"] is None else round(dl["probability"], 3),
                     "avg_prob": round(r["average_probability"], 3), "priority": r["priority"], "agreement": r["agreement"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Report generator (.docx)
# --------------------------------------------------------------------------- #
def build_report_docx(result, raw=None):
    """Return a BytesIO containing a conservation field report for one assessment."""
    from datetime import datetime
    from docx import Document
    from docx.shared import RGBColor

    c, rec = result["clean"], result["recommendation"]
    doc = Document()
    h = doc.add_heading("IWCDSS - Conservation Field Report", level=1)
    h.runs[0].font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    doc.add_paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}   |   Group 4 - Tiger Presence/Absence")

    doc.add_heading("1. Survey point", level=2)
    t = doc.add_table(rows=0, cols=2); t.style = "Light Grid Accent 1"
    for k in ["Lattitude", "Longitude", "Terrain", "Habitat_ty", "Featured_F", "Location", "Grid_2_2", "Grid_10_10", "Survey_Month"] + COUNT_COLS:
        r = t.add_row().cells; r[0].text, r[1].text = k, str(c[k])

    doc.add_heading("2. Predictions", level=2)
    t = doc.add_table(rows=1, cols=3); t.style = "Light Grid Accent 1"
    for i, s in enumerate(["Module", "Result", "Tiger probability"]):
        t.rows[0].cells[i].text = s
    for name, d in [("Expert system (rules)", result["expert"]), ("ML - XGBoost", result["ml"]), ("DL - DNN", result["dl"])]:
        r = t.add_row().cells
        r[0].text, r[1].text = name, d["label"]
        r[2].text = "n/a" if d.get("probability") is None else f"{d['probability']:.1%}"

    doc.add_heading("3. Recommendation", level=2)
    doc.add_paragraph(f"Priority: {rec['priority']}  (average probability {rec['average_probability']:.1%}; {rec['agreement']})")
    doc.add_paragraph(rec["action"])
    for n in rec["notes"]:
        doc.add_paragraph(n, style="List Bullet")

    doc.add_heading("4. Why? Top SHAP drivers (ML model)", level=2)
    for _, r in result["shap_contributions"].head(8).iterrows():
        doc.add_paragraph(f"{r['feature']} = {r['value']:.3g}: {'+' if r['shap'] > 0 else ''}{r['shap']:.3f} "
                          f"({'towards Present' if r['shap'] > 0 else 'towards Absent'})", style="List Bullet")

    doc.add_heading("5. Expert-system rules fired", level=2)
    for _, txt in result["expert"]["rules_fired"] or [(0, "No rules fired")]:
        doc.add_paragraph(txt, style="List Bullet")

    if result["warnings"]:
        doc.add_heading("6. Data warnings", level=2)
        for w in result["warnings"]:
            doc.add_paragraph(w, style="List Bullet")
    doc.add_paragraph("\nDisclaimer: decision support only. Predictions are statistical estimates and must be "
                      "confirmed by field evidence before conservation action is taken.")
    buf = io.BytesIO(); doc.save(buf); buf.seek(0)
    return buf
