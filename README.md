# IPL Stats Explorer & Score Predictor

## What this project does
- IPL first-innings overview and charts
- Player run statistics (when player columns exist)
- First-innings final-score prediction from the current match situation
- Comparison of a mean baseline, Linear Regression, and Random Forest
- Evaluation using MAE, RMSE, and R²
- Random Forest hyperparameter tuning using grouped cross-validation

## Files
- `app.py` — complete Streamlit application
- `requirements.txt` — Python packages
- `ball_by_ball.csv` — add your own IPL ball-by-ball CSV here

## Run on Windows
1. Install Python 3.10 or newer.
2. Put your IPL ball-by-ball CSV in this folder.
3. Rename it to `ball_by_ball.csv`.
4. Open Command Prompt/Terminal in this folder.
5. Run:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
6. Your browser should open the app.

If `py` does not work, use `python` instead.

## Dataset columns
The app tries to recognize common variations of:
- match ID (`match_id` or `id`)
- innings (`innings` or `inning`)
- over (`over` or `overs`)
- total runs (`total_runs` or `total_run`)
- batter (`batter` or `batsman`)
- batter runs (`batter_runs`, `batsman_runs`, or `batsman_run`)
- batting team (`batting_team` or `BattingTeam`)
- season (`season` or `year`)

If the app reports missing columns, inspect the CSV headers and update the alias list at the top of `app.py`.

## Notes for your report/viva
- This is a regression problem, so class balancing is not applicable.
- The final innings score is the target. Model inputs use only the match state at an earlier over.
- The final-over snapshot is removed because it would directly reveal the target.
- Train/test splitting is grouped by match ID to reduce data leakage.
- Do not report results until you have run the project on your actual dataset. The app displays measured results when it trains.


## Verified uploaded dataset
The supplied `ball_by_ball.csv` has 288,226 rows and 30 columns, 1,212 unique match IDs, and seasons 2008–2026. Its headers include `season_id`, `match_id`, `team_batting`, `team_bowling`, `over_number`, `batter_runs`, `total_runs`, `is_wicket`, `is_super_over`, and `innings`. The app maps these names and excludes super-over deliveries for the first-innings model.
