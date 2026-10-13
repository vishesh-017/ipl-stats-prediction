import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="IPL Stats Explorer", page_icon="🏏", layout="wide")

DATA_FILE = "ball_by_ball.csv"

# Common column names used by different IPL ball-by-ball datasets.
ALIASES = {
    "match_id": ["match_id", "id", "matchid", "match"],
    "innings": ["innings", "inning", "innings_no", "inning_no"],
    "over": ["over", "overs", "over_number", "overnumber"],
    "total_runs": ["total_runs", "total_run", "totalruns", "runs_total"],
    "batter_runs": ["batter_runs", "batsman_runs", "batsman_run", "batter_run", "runs_off_bat"],
    "batter": ["batter", "batsman", "striker", "player_batting"],
    "bowler": ["bowler", "bowler_name"],
    "batting_team": ["batting_team", "battingteam", "team_batting"],
    "bowling_team": ["bowling_team", "bowlingteam", "team_bowling"],
    "wicket_flag": ["iswicketdelivery", "is_wicket_delivery", "is_wicket", "wicket", "wickets"],
    "player_out": ["player_out", "player_dismissed", "playerout"],
    "season": ["season", "year", "match_year", "season_id"],
    "super_over": ["is_super_over", "super_over", "issuperover"],
}

def clean_col_name(col):
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")

def normalize_columns(df):
    df = df.copy()
    df.columns = [clean_col_name(c) for c in df.columns]
    # Remove punctuation/underscores only for alias matching.
    lookup = {c.replace("_", ""): c for c in df.columns}
    rename = {}
    for standard, choices in ALIASES.items():
        for choice in choices:
            key = choice.replace("_", "")
            if key in lookup:
                rename[lookup[key]] = standard
                break
    return df.rename(columns=rename)

@st.cache_data
def load_data(path, modified_time):
    raw = pd.read_csv(path, low_memory=False)
    return normalize_columns(raw)

def numeric(series):
    return pd.to_numeric(series, errors="coerce")

def prepare_data(raw):
    df = raw.copy()
    required = ["match_id", "innings", "over", "total_runs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns after name matching: "
            + ", ".join(missing)
            + ". Please use a ball-by-ball IPL CSV containing match ID, innings, over, and total runs."
        )

    for col in ["innings", "over", "total_runs", "batter_runs", "wicket_flag"]:
        if col in df.columns:
            df[col] = numeric(df[col])

    df = df.dropna(subset=["match_id", "innings", "over", "total_runs"])
    df = df.drop_duplicates()
    # This dataset includes a dedicated is_super_over column. Exclude super overs.
    if "super_over" in df.columns:
        if df["super_over"].dtype == object:
            df["super_over"] = df["super_over"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[~df["super_over"].fillna(False).astype(bool)].copy()
    # Use first innings only for a consistent final-score regression task.
    df = df[df["innings"] == 1].copy()
    df = df[df["over"] >= 0]
    if df.empty:
        raise ValueError("No first-innings records were found. Check the innings column in your CSV.")

    # Use explicit wicket indicator when available; otherwise use dismissed-player field.
    if "wicket_flag" in df.columns:
        df["wicket_value"] = df["wicket_flag"].fillna(0).astype(float).clip(0, 1)
    elif "player_out" in df.columns:
        df["wicket_value"] = df["player_out"].notna().astype(float)
    else:
        df["wicket_value"] = 0.0

    # Some datasets store over numbers as 1-20; others store 0-19.
    # Preserve the source convention and cap to a standard 20-over match.
    df["over"] = df["over"].astype(int)
    df = df[df["over"] <= 20]
    return df

def build_over_snapshots(df):
    """Build one row per completed over; features use only information up to that over."""
    rows = []
    for match_id, group in df.groupby("match_id", sort=False):
        group = group.sort_values(["over"])
        per_over = group.groupby("over", as_index=False).agg(
            runs_in_over=("total_runs", "sum"),
            wickets_in_over=("wicket_value", "sum")
        )
        per_over["cumulative_runs"] = per_over["runs_in_over"].cumsum()
        per_over["cumulative_wickets"] = per_over["wickets_in_over"].cumsum().clip(0, 10)
        final_score = float(per_over["runs_in_over"].sum())
        for _, row in per_over.iterrows():
            over_num = int(row["over"])
            # Rolling total from current and previous four overs.
            recent = per_over[per_over["over"].between(over_num - 4, over_num)]["runs_in_over"].sum()
            # Dataset convention can be 0-19 or 1-20. Both work as elapsed-over input
            # as long as the convention is consistent within the dataset.
            rows.append({
                "match_id": match_id,
                "overs_completed": over_num + 1 if per_over["over"].min() == 0 else over_num,
                "current_score": float(row["cumulative_runs"]),
                "wickets_lost": float(row["cumulative_wickets"]),
                "runs_last_5_overs": float(recent),
                "target_score": final_score,
            })
    snapshots = pd.DataFrame(rows)
    # Do not train on the final-over snapshot: it already reveals the target score.
    snapshots = snapshots[snapshots["overs_completed"] < 20]
    snapshots = snapshots[snapshots["overs_completed"] > 0]
    return snapshots

def train_models(snapshots):
    features = ["overs_completed", "current_score", "wickets_lost", "runs_last_5_overs"]
    X = snapshots[features]
    y = snapshots["target_score"]
    groups = snapshots["match_id"]

    if groups.nunique() < 5:
        raise ValueError("At least 5 distinct matches are recommended for a train/test split.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_groups = groups.iloc[train_idx]

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]),
    }

    # Tune Random Forest using cross-validation, keeping matches grouped to reduce leakage.
    rf_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 8, 14],
        "model__min_samples_leaf": [1, 2, 4],
    }
    unique_train_matches = train_groups.nunique()
    cv_splits = min(3, unique_train_matches)
    if cv_splits >= 2:
        from sklearn.model_selection import GroupKFold
        search = RandomizedSearchCV(
            models["Random Forest"],
            param_distributions=rf_params,
            n_iter=5,
            scoring="neg_mean_absolute_error",
            cv=GroupKFold(n_splits=cv_splits),
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=train_groups)
        models["Random Forest"] = search.best_estimator_
        best_params = search.best_params_
    else:
        models["Random Forest"].fit(X_train, y_train)
        best_params = {"note": "Not enough training matches for grouped cross-validation tuning."}

    fitted = {}
    results = []
    for name, model in models.items():
        if name != "Random Forest" or not hasattr(model, "best_params_"):
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fitted[name] = model
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R²": r2_score(y_test, preds),
        })

    return fitted, pd.DataFrame(results).sort_values("MAE"), X_test, y_test, best_params, features

def show_overview(df):
    matches = df["match_id"].nunique()
    total_runs = int(df["total_runs"].sum())
    teams = df["batting_team"].nunique() if "batting_team" in df.columns else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches in loaded data", f"{matches:,}")
    c2.metric("Runs in first innings", f"{total_runs:,}")
    c3.metric("Teams found", f"{teams:,}" if teams is not None else "Not available")

    if "season" in df.columns:
        season_data = df.groupby("season", as_index=False)["total_runs"].sum()
        st.subheader("Runs by season")
        st.plotly_chart(px.bar(season_data, x="season", y="total_runs",
                               title="First-innings runs by season"), use_container_width=True)

    if "batting_team" in df.columns:
        team_runs = df.groupby("batting_team", as_index=False)["total_runs"].sum()
        team_runs = team_runs.sort_values("total_runs", ascending=False).head(10)
        st.subheader("Top teams by first-innings runs")
        st.plotly_chart(px.bar(team_runs, x="batting_team", y="total_runs",
                               title="Top 10 teams by first-innings runs",
                               labels={"batting_team": "Team", "total_runs": "Runs"}),
                        use_container_width=True)

def show_players(df):
    if "batter" not in df.columns or "batter_runs" not in df.columns:
        st.info("Player analysis needs batter/batsman and batter_runs/batsman_runs columns in the CSV.")
        return
    bat = df.groupby("batter", as_index=False)["batter_runs"].sum()
    bat = bat.sort_values("batter_runs", ascending=False)
    st.subheader("Top 10 run scorers in the loaded first-innings data")
    st.plotly_chart(px.bar(bat.head(10), x="batter_runs", y="batter", orientation="h",
                           labels={"batter_runs": "Runs", "batter": "Player"}),
                    use_container_width=True)
    player = st.selectbox("Choose a player", bat["batter"].sort_values().tolist())
    st.metric(f"{player} — runs in loaded first innings", f"{int(bat.loc[bat['batter'] == player, 'batter_runs'].iloc[0]):,}")
    if "season" in df.columns:
        season_player = df[df["batter"] == player].groupby("season", as_index=False)["batter_runs"].sum()
        st.plotly_chart(px.line(season_player, x="season", y="batter_runs", markers=True,
                                title=f"{player}: runs by season"),
                        use_container_width=True)

def main():
    st.title("🏏 IPL Stats Explorer & Score Predictor")
    st.caption("A simple, end-to-end machine learning project using Python and Streamlit.")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Player Statistics", "ML Score Prediction", "About the Project"])

    if not os.path.exists(DATA_FILE):
        st.warning(f"Dataset not found. Put your IPL ball-by-ball CSV in this folder and rename it to **{DATA_FILE}**.")
        st.markdown("""
        **Expected setup**
        1. Keep `app.py` and `requirements.txt` in one folder.
        2. Copy your IPL ball-by-ball CSV into that folder.
        3. Rename the CSV to `ball_by_ball.csv`.
        4. Run `streamlit run app.py` in the terminal.
        """)
        st.stop()

    try:
        raw = load_data(DATA_FILE, os.path.getmtime(DATA_FILE))
        df = prepare_data(raw)
    except Exception as e:
        st.error(f"Could not prepare the dataset: {e}")
        st.info("Open the CSV and check that it contains ball-by-ball records with match ID, innings, over, and total runs.")
        st.stop()

    if page == "Overview":
        st.header("IPL Overview")
        show_overview(df)
        st.caption("Overview currently summarizes first-innings records to keep the score-prediction task consistent.")
    elif page == "Player Statistics":
        st.header("Player Statistics")
        show_players(df)
    elif page == "ML Score Prediction":
        st.header("Predict the final first-innings score")
        st.write("Enter the match situation so far. The model estimates the final score using historical first-innings data.")
        try:
            snapshots = build_over_snapshots(df)
            if len(snapshots) < 30:
                st.warning("There are too few usable innings snapshots to train a reliable demonstration model.")
                st.stop()
            with st.spinner("Training and comparing models..."):
                models, results, X_test, y_test, best_params, features = train_models(snapshots)
            st.subheader("Model comparison on unseen matches")
            st.dataframe(results.round(3), use_container_width=True, hide_index=True)
            best_name = results.iloc[0]["Model"]
            st.success(f"Lowest test MAE in this run: {best_name}")
            st.write("**Random Forest tuning result:**")
            st.json(best_params)

            st.subheader("Try a prediction")
            a, b = st.columns(2)
            overs = a.slider("Overs completed", 1, 19, 10)
            score = b.number_input("Current score", min_value=0, max_value=350, value=85)
            c, d = st.columns(2)
            wickets = c.slider("Wickets lost", 0, 9, 3)
            recent = d.number_input("Runs in the last 5 overs", min_value=0, max_value=100, value=40)
            input_df = pd.DataFrame([{
                "overs_completed": overs,
                "current_score": score,
                "wickets_lost": wickets,
                "runs_last_5_overs": recent,
            }])
            chosen = st.selectbox("Prediction model", ["Random Forest", "Linear Regression"])
            if st.button("Predict final score", type="primary"):
                prediction = float(models[chosen].predict(input_df[features])[0])
                prediction = max(float(score), prediction)
                st.metric("Estimated final score", f"{round(prediction):,}")
                st.caption("This is a model estimate, not a guaranteed score. Accuracy depends on dataset quality and the match situation.")
            pred = models[best_name].predict(X_test)
            plot_df = pd.DataFrame({"Actual score": y_test, "Predicted score": pred})
            st.plotly_chart(px.scatter(plot_df, x="Actual score", y="Predicted score",
                                       title=f"Actual vs predicted scores — {best_name}",
                                       trendline="ols"), use_container_width=True)
        except Exception as e:
            st.error(f"Model training could not complete: {e}")
            st.info("Check the dataset columns and ensure it contains several full first innings from different matches.")
    else:
        st.header("About this project")
        st.markdown("""
        **Problem:** Cricket fans and analysts want to explore historical IPL performance and estimate an innings' final score from the match situation.

        **Input features for ML**
        - Overs completed
        - Current score
        - Wickets lost
        - Runs in the last five overs

        **Models compared**
        - Mean baseline: predicts the average final score from the training set.
        - Linear Regression: a simple, interpretable regression model.
        - Random Forest Regressor: combines decision trees to model non-linear patterns.

        **Evaluation:** MAE, RMSE, and R² on a held-out set of matches. Matches are separated between training and testing so snapshots from the same match do not appear in both sets.

        **Limitations:** The estimate depends on historical data and the available features. It does not account for live player form, pitch conditions, weather, or all tactical factors.
        """)

if __name__ == "__main__":
    main()
