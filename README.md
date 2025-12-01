
# Data Analytics Dashboard (Python)

Simple interactive data analytics dashboard built with Pandas, NumPy, Matplotlib and scikit-learn.  
Features:
- Load CSV/Excel dataset
- Data cleaning (duplicates removal, fill numeric NaNs with mean)
- Statistical summary (Pandas describe + NumPy)
- Visualizations (histogram, bar, pie, line, correlation heatmap)
- Basic prediction using Linear Regression (scikit-learn)

## How to run
1. Put your dataset (CSV) in the project folder, e.g. `student_scores.csv`.
2. Install dependencies: pip install pandas numpy matplotlib scikit-learn joblib
3. Run interactive dashboard: python dashboard.py
4. To run non-interactive demo (generates summary, plots and model): python run_demo.py

   
## Files
- `dashboard.py` — main interactive program
- `run_demo.py` — non-interactive demo script that auto-produces outputs in `demo_outputs/`
- `student_scores.csv` — example dataset (replace with your own)
- `demo_outputs/` — output folder for demo (plots, summary CSV, saved model)


- Implemented end-to-end analytics pipeline: data ingestion → cleaning → analysis → visualization → basic ML.
- Used Pandas/NumPy for ETL and statistics, Matplotlib for visualization, and scikit-learn for modeling.
- Wrote interactive CLI for demo and a non-interactive script for reproducible demo results.

## Future improvements
- Add logging and unit tests.
- Add more ML models (RandomForest, Ridge, cross-validation).
- Build a web dashboard (Streamlit/Flask) and deploy.
- Add configuration file (YAML) and command-line arguments for automation.
