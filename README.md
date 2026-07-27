# Subsurface Fence Diagram

A Streamlit app for geotechnical boring logs and subsurface fence diagrams.

## Features
- Upload CSV or Excel boring logs
- Plot intervals by depth or elevation
- Show groundwater and N-values
- Use basic soil/rock patterns
- Download the chart as PNG

## Expected columns
Minimum:
- `borehole_id`
- `ground_elev` with `top_depth` and `bottom_depth`

Or:
- `borehole_id`
- `top_elev` and `bottom_elev`

Optional:
- `x`, `y`, `station`, `order`, `water_elev`, `water_depth`, `n_value`, `rqd`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub + Streamlit Cloud
1. Push this repository to GitHub.
2. On Streamlit Cloud, select this repo.
3. Set the main file path to `app.py`.
