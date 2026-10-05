import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"
FEATURES = [
    "num_rooms", "num_people", "housearea", "is_ac", "is_tv",
    "is_flat", "ave_monthly_income", "num_children", "is_urban"
]
METRICS = {
    "MAE": 54.04,
    "RMSE": 62.92,
    "R²": 0.88,
}

st.set_page_config(
    page_title="WattWise | Electricity Bill Estimator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Accessibility and theme ----------
st.sidebar.markdown("## ⚙️ Display & accessibility")
font_scale = st.sidebar.select_slider(
    "Text size", options=["Standard", "Large", "Extra large"], value="Large"
)
contrast = st.sidebar.toggle("High-contrast mode", value=False, key="high_contrast")
simple_mode = st.sidebar.toggle("Simplified instructions", value=True)
font_sizes = {"Standard": "16px", "Large": "18px", "Extra large": "21px"}
bg = "#070D18" if contrast else "#F5F7FB"
fg = "#FFFFFF" if contrast else "#172033"
panel = "#111827" if contrast else "#FFFFFF"
muted = "#D6DCE8" if contrast else "#596579"
border = "#FFFFFF" if contrast else "#DCE3EE"
accent = "#FFE600" if contrast else "#087F8C"
accent_text = "#000000" if contrast else "#FFFFFF"
button_hover = "#292929" if contrast else "#E5EDF7"
button_hover_text = "#FFFFFF" if contrast else "#172033"
primary_hover = "#D4C000" if contrast else "#066A73"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp {{ background: {bg}; color: {fg}; }}
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; font-size: {font_sizes[font_scale]}; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; color: {fg} !important; letter-spacing: -0.03em; }}
p, label, li, [data-testid="stMarkdownContainer"] {{ color: {fg}; }}
[data-testid="stSidebar"] {{ background: {panel}; border-right: 1px solid {border}; }}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {fg} !important; }}
div[data-testid="stMetric"] {{ background: {panel}; border: 1px solid {border}; padding: 18px 20px; border-radius: 18px; }}
div[data-testid="stMetricLabel"] p {{ color: {muted} !important; font-weight: 600; }}
div[data-testid="stMetricValue"] {{ color: {fg} !important; font-family: 'Space Grotesk', sans-serif; }}
div[data-testid="stForm"], div[data-testid="stExpander"] {{ border: 1px solid {border}; border-radius: 18px; }}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{ border-radius: 12px; font-weight: 700; min-height: 3rem; transition: background-color .15s ease, color .15s ease, border-color .15s ease; }}
.stButton > button:not(:disabled):hover, .stDownloadButton > button:not(:disabled):hover {{ background: {button_hover} !important; color: {button_hover_text} !important; border-color: {accent} !important; }}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, .stFormSubmitButton > button:focus-visible {{ outline: 3px solid {accent} !important; outline-offset: 2px; }}
.stFormSubmitButton > button {{ background: {accent} !important; color: {accent_text} !important; border: 0 !important; }}
.stFormSubmitButton > button:not(:disabled):hover {{ background: {primary_hover} !important; color: {accent_text} !important; border: 0 !important; }}
.stApp input, .stApp textarea, [data-baseweb="select"] > div {{ background: {panel} !important; color: {fg} !important; border-color: {border} !important; }}
[data-testid="stCaptionContainer"] {{ color: {muted} !important; }}
.hero {{ padding: 26px 28px; border-radius: 24px; background: linear-gradient(120deg, {panel}, {bg}); border: 1px solid {border}; margin-bottom: 18px; }}
.eyebrow {{ color: {accent} !important; text-transform: uppercase; font-weight: 700; letter-spacing: .13em; font-size: .78rem; }}
.hero-title {{ font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3.2rem); line-height: 1.08; font-weight: 700; margin: 6px 0 10px; }}
.hero-copy {{ color: {muted} !important; max-width: 720px; }}
.pill {{ display: inline-block; border: 1px solid {border}; padding: 5px 10px; border-radius: 99px; color: {muted}; font-size: .8rem; margin-right: 6px; }}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def call_prediction(payload):
    """Call the existing FastAPI endpoint using the same feature names as the model."""
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    if response.status_code == 200:
        data = response.json()
        return float(data["predicted_amount"]), data.get("note", "")
    try:
        detail = response.json().get("detail", "Prediction service returned an error.")
    except ValueError:
        detail = response.text or "Prediction service returned an error."
    raise RuntimeError(f"API error ({response.status_code}): {detail}")

def money(value):
    return f"₹{value:,.2f}"

def make_payload(rooms, people, area, children, income, has_ac, has_tv, is_flat, urban):
    return {
        "num_rooms": int(rooms),
        "num_people": int(people),
        "housearea": float(area),
        "is_ac": int(has_ac == "Yes") if isinstance(has_ac, str) else int(has_ac),
        "is_tv": int(has_tv == "Yes") if isinstance(has_tv, str) else int(has_tv),
        "is_flat": int(is_flat == "Yes") if isinstance(is_flat, str) else int(is_flat),
        "ave_monthly_income": float(income),
        "num_children": int(children),
        "is_urban": int(urban == "Yes") if isinstance(urban, str) else int(urban),
    }

def scenario_payload(base, **changes):
    result = dict(base)
    result.update(changes)
    return result

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">⚡ Household energy intelligence</div>
  <div class="hero-title">WattWise</div>
  <p class="hero-copy">Explore a model-based estimate of your monthly electricity bill, compare household scenarios, and understand how the prediction model is evaluated.</p>
  <span class="pill">Machine learning</span><span class="pill">Scenario explorer</span><span class="pill">Accessible design</span>
</div>
""", unsafe_allow_html=True)

# API status check
status_col1, status_col2 = st.columns([3, 1])
with status_col1:
    st.caption("A prediction requires the FastAPI backend to be running locally.")
with status_col2:
    if st.button("Check API status", use_container_width=True):
        try:
            health = requests.get(f"{API_URL}/health", timeout=3).json()
            if health.get("status") == "ok" and health.get("model_loaded"):
                st.success("API and model are available.")
            elif health.get("status") == "ok":
                st.warning("API is running, but the model file is missing.")
            else:
                st.warning(f"API response: {health}")
        except requests.RequestException:
            st.error("Cannot reach FastAPI. Start the backend first.")

tab_predict, tab_explore, tab_model = st.tabs(
    ["⚡ Predict bill", "📊 Explore scenarios", "🧠 Model insights"]
)

# ---------- Prediction ----------
with tab_predict:
    left, right = st.columns([1.35, 0.85], gap="large")
    with left:
        st.subheader("Tell us about your household")
        if simple_mode:
            st.write("Enter household details to estimate a monthly bill. Use the same area unit used in the training data.")
        with st.form("household_form"):
            c1, c2 = st.columns(2)
            with c1:
                rooms = st.number_input("Number of rooms", min_value=1, max_value=100, value=3, step=1)
                people = st.number_input("People living in the home", min_value=1, max_value=100, value=3, step=1)
                area = st.number_input("House area", min_value=1.0, max_value=1_000_000.0, value=800.0, step=50.0)
                children = st.number_input("Number of children", min_value=0, max_value=100, value=1, step=1)
            with c2:
                income = st.number_input("Average monthly household income (₹)", min_value=0.0, max_value=1_000_000_000.0, value=30000.0, step=1000.0)
                has_ac = st.radio("Air conditioner present?", ["No", "Yes"], horizontal=True)
                has_tv = st.radio("Television present?", ["No", "Yes"], horizontal=True)
                is_flat = st.radio("Home is a flat?", ["No", "Yes"], horizontal=True)
                urban = st.radio("Urban area?", ["No", "Yes"], horizontal=True)
            submitted = st.form_submit_button("⚡ Estimate monthly bill", type="primary", use_container_width=True)

    with right:
        st.subheader("What this model considers")
        st.markdown("""
        - 🏠 **Home:** rooms and area
        - 👨‍👩‍👧 **Household:** residents and children
        - 💡 **Appliances:** AC and TV presence
        - 🏙️ **Property:** flat and urban-area indicators
        - 💰 **Income:** average monthly household income
        """)
        st.info("The model estimates the amount paid in the training dataset. It does not calculate an official tariff-based bill.")

    if submitted:
        payload = make_payload(rooms, people, area, children, income, has_ac, has_tv, is_flat, urban)
        try:
            with st.spinner("Running the model…"):
                predicted, note = call_prediction(payload)
            st.session_state["last_payload"] = payload
            st.session_state["last_prediction"] = predicted
            st.session_state["last_note"] = note
            st.session_state["prediction_history"] = st.session_state.get("prediction_history", [])
            st.session_state["prediction_history"].append({
                "Prediction #": len(st.session_state["prediction_history"]) + 1,
                "Estimated monthly bill (₹)": round(predicted, 2),
                "Rooms": payload["num_rooms"],
                "People": payload["num_people"],
                "Area": payload["housearea"],
                "AC": "Yes" if payload["is_ac"] else "No",
                "TV": "Yes" if payload["is_tv"] else "No",
            })
        except requests.exceptions.ConnectionError:
            st.error("The prediction service is not running. Start FastAPI in a separate terminal.")
        except requests.exceptions.Timeout:
            st.error("The request took too long. Check the backend and try again.")
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
            st.error(f"Could not calculate the estimate: {exc}")

    if "last_prediction" in st.session_state:
        predicted = st.session_state["last_prediction"]
        payload = st.session_state["last_payload"]
        st.divider()
        st.markdown("### Your latest estimate")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated monthly bill", money(predicted))
        m2.metric("Household members", payload["num_people"])
        m3.metric("House area", f'{payload["housearea"]:,.0f}')
        st.caption(st.session_state.get("last_note", ""))
        st.warning("This is a model-based estimate, not a guaranteed or official electricity bill.")

        st.markdown("#### Household snapshot")
        st.caption("These charts use separate scales because room counts, area, and income have different units.")

        snap_left, snap_right = st.columns(2, gap="large")
        with snap_left:
            st.markdown("**People and rooms**")
            counts = pd.DataFrame({
                "Household factor": ["Rooms", "People", "Children"],
                "Value": [
                    payload["num_rooms"], payload["num_people"], payload["num_children"]
                ],
            })
            st.bar_chart(counts, x="Household factor", y="Value", horizontal=True,
                         color="#087F8C")
        with snap_right:
            st.markdown("**Home and income**")
            property_values = pd.DataFrame({
                "Household factor": ["House area", "Monthly income (₹)"],
                "Value": [payload["housearea"], payload["ave_monthly_income"]],
            })
            st.bar_chart(property_values, x="Household factor", y="Value",
                         horizontal=True, color="#6C63FF")

        st.markdown("**Appliances and property indicators**")
        st.caption("1 = Yes; 0 = No. These are the binary inputs sent to the model.")
        binary = pd.DataFrame({
            "Feature": ["Air conditioner", "Television", "Flat", "Urban area"],
            "Present / Yes": [
                payload["is_ac"], payload["is_tv"], payload["is_flat"], payload["is_urban"]
            ],
        })
        st.bar_chart(binary, x="Feature", y="Present / Yes", horizontal=True,
                     color="#E39A35")

        history = st.session_state.get("prediction_history", [])
        if history:
            st.markdown("#### Recent predictions")
            history_df = pd.DataFrame(history)
            st.dataframe(history_df.tail(5), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download prediction history (CSV)",
                history_df.to_csv(index=False).encode("utf-8"),
                file_name="electricity_prediction_history.csv",
                mime="text/csv",
            )

# ---------- Scenario explorer ----------
with tab_explore:
    st.subheader("What-if scenario explorer")
    st.write("Compare the current household with alternative inputs. Each scenario is sent to your actual prediction API.")
    if "last_payload" not in st.session_state:
        st.info("Make a prediction in the **Predict bill** tab first. Your latest household will be used as the baseline.")
    else:
        base = st.session_state["last_payload"]
        baseline_prediction = st.session_state["last_prediction"]
        st.metric("Baseline estimate", money(baseline_prediction))
        scenario_options = {
            "Baseline household": {},
            "One additional room": {"num_rooms": min(100, base["num_rooms"] + 1)},
            "One additional person": {"num_people": min(100, base["num_people"] + 1)},
            "Add an air conditioner": {"is_ac": 1},
            "Add a television": {"is_tv": 1},
            "Switch to urban = Yes": {"is_urban": 1},
        }
        selected = st.multiselect(
            "Choose scenarios to compare",
            options=list(scenario_options.keys()),
            default=["One additional room", "One additional person", "Add an air conditioner"],
        )
        if st.button("Run scenario comparison", type="primary", use_container_width=True):
            rows = [{"Scenario": "Baseline household", "Estimated bill (₹)": baseline_prediction}]
            errors = []
            with st.spinner("Comparing scenarios…"):
                for name in selected:
                    changes = scenario_options[name]
                    candidate = scenario_payload(base, **changes)
                    try:
                        amount, _ = call_prediction(candidate)
                        rows.append({"Scenario": name, "Estimated bill (₹)": amount})
                    except Exception as exc:
                        errors.append(f"{name}: {exc}")
            if len(rows) > 1:
                df = pd.DataFrame(rows).drop_duplicates(subset=["Scenario"])
                df["Change vs baseline (₹)"] = df["Estimated bill (₹)"] - baseline_prediction
                st.markdown("#### Estimated monthly bill by scenario")
                st.bar_chart(df.set_index("Scenario")[["Estimated bill (₹)"]])
                st.dataframe(
                    df.style.format({"Estimated bill (₹)": "₹{:,.2f}", "Change vs baseline (₹)": "₹{:+,.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Scenario differences show how this fitted model responds to changed inputs; they do not prove that changing one factor causes the same real-world bill change.")
            if errors:
                st.warning("Some scenarios could not be calculated: " + " | ".join(errors))

# ---------- Model insights ----------
with tab_model:
    st.subheader("Inside the model")
    st.write("Your notebook trains a **Linear Regression** model using nine household features to predict `amount_paid`. The notebook records these test-set metrics:")
    a, b, c = st.columns(3)
    a.metric("MAE", f"₹{METRICS['MAE']:.2f}", help="Mean Absolute Error: average absolute difference between predictions and actual test values.")
    b.metric("RMSE", f"₹{METRICS['RMSE']:.2f}", help="Root Mean Squared Error: gives larger errors more weight.")
    c.metric("R² score", f"{METRICS['R²']:.2f}", help="Coefficient of determination; 1 is a perfect fit on the evaluated data.")
    st.markdown("#### Error metrics at a glance")
    metric_df = pd.DataFrame({
        "Metric": ["MAE (₹)", "RMSE (₹)"],
        "Error value": [METRICS["MAE"], METRICS["RMSE"]],
    }).set_index("Metric")
    st.bar_chart(metric_df, horizontal=True)
    st.markdown("#### Features passed to the model")
    feature_descriptions = pd.DataFrame({
        "Feature": FEATURES,
        "Meaning": [
            "Number of rooms", "Number of people", "House area",
            "Air conditioner present (0/1)", "Television present (0/1)",
            "Home is a flat (0/1)", "Average monthly household income",
            "Number of children", "Urban area (0/1)"
        ],
    })
    st.dataframe(feature_descriptions, use_container_width=True, hide_index=True)
    with st.expander("How should I interpret these metrics?"):
        st.markdown("""
        - **MAE ₹54.04:** the notebook's test predictions differed from actual values by ₹54.04 on average in absolute terms.
        - **RMSE ₹62.92:** larger errors contribute more strongly to this metric; the value is also in the target's currency units.
        - **R² 0.88:** the model explained about 88% of the variation in the notebook's test target values under that split.
        """)
    st.warning("These metrics are copied from the uploaded notebook's recorded output. They are not recalculated by this UI, and may not describe `backend/model.pkl` if that file was trained differently. The notebook does not include the test predictions needed to draw a genuine actual-vs-predicted scatter plot here.")
    st.markdown("#### Data preparation notes from the notebook")
    st.markdown("- Dataset preview showed 1,000 rows.")
    st.markdown("- The notebook output showed no missing values and no duplicated rows.")
    st.caption("Model metrics describe the notebook's held-out test split (80/20 with random_state=42), not a guarantee of future performance.")

st.divider()
st.caption("WattWise • Built with Streamlit + FastAPI • Use estimates for exploration, not billing decisions.")
