# Electricity Bill Estimator — Streamlit + FastAPI

## Project structure
- `backend/main.py`: FastAPI prediction endpoint
- `backend/model.pkl`: your trained scikit-learn model (you create this)
- `frontend/app.py`: Streamlit user interface
- `requirements.txt`: dependencies

## 1. Save your model from the notebook
After fitting your `LinearRegression` model, run this in a notebook cell:

```python
import joblib
joblib.dump(model, "model.pkl")
```

Move/copy the resulting `model.pkl` into the `backend` folder. Ensure it is the same fitted model trained on these features, in this order:

```python
[
    "num_rooms", "num_people", "housearea", "is_ac", "is_tv",
    "is_flat", "ave_monthly_income", "num_children", "is_urban"
]
```

Also confirm the `housearea` unit and income currency/period match the dataset. The UI assumes income is monthly and displayed in rupees; change the label if the dataset uses something else.

## 2. Install dependencies (Windows PowerShell)
From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run FastAPI
Open Terminal 1 at the project root:

```powershell
uvicorn backend.main:app --reload
```

API docs: http://127.0.0.1:8000/docs  
Health check: http://127.0.0.1:8000/health

## 4. Run Streamlit
Open Terminal 2 at the project root:

```powershell
streamlit run frontend/app.py
```

The UI should open at http://localhost:8501.

## Accessibility choices included
- Adjustable text size
- High-contrast option
- Plain-language labels and help text
- Large primary action button
- Clear success/error messages
- Form grouping and reduced-detail mode
- Keyboard-accessible native Streamlit controls

## Important limitations / before deployment
- This is a starter implementation, not a certified WCAG audit. Test with keyboard-only navigation, screen readers, browser zoom, and users with disabilities.
- Streamlit's layout and accessibility semantics have limits; verify the app with NVDA/VoiceOver and avoid relying on color alone.
- The income feature may be sensitive. Explain why it is collected, do not log it, and consider whether the model truly needs it.
- Review the training data, validation scores, and prediction range before showing estimates to real users.
- `joblib` files should only be loaded if you trust the file source.
