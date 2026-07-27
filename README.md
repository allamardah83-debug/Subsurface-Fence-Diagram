# Subsurface Fence Diagram Generator

A Streamlit app for geotechnical boring logs and fence diagrams.

## Features
- Upload CSV or Excel boring logs
- Plot by ground elevation and interval depth/elevation
- Show SPT N-values, recovery, RQD, and water levels
- Legend and borehole spacing controls
- PNG download

## Expected columns
Minimum:
- `borehole_id`
- `ground_elev`
- `top_depth` and `bottom_depth` or `top_elev` and `bottom_elev`
- `unit` or `uscs`

Optional:
- `x`, `y`
- `water_elev`, `water_depth`
- `n_value`, `recovery`, `rqd`
- `description`, `formation`, `rock_type`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
