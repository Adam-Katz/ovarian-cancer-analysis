import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
async def _():
    import sys
    if 'pyodide' in sys.modules:
        import micropip
        await micropip.install(
            ['plotly', 'lifelines', 'statsmodels', 'pyarrow'],
            keep_going=True,
        )

    import pandas as pd
    import numpy as np
    import marimo as mo
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.stats import chi2_contingency
    import statsmodels.api as sm

    return chi2_contingency, go, mo, np, pd, px, sm


@app.cell
def _(mo):
    mo.md(r"""
    # Ovarian Cancer Treatment and Insurance Analysis

    **Research Question:** How do insurance characteristics (payer type, plan design) relate to treatment patterns for ovarian cancer patients receiving PARP inhibitors and/or bevacizumab?

    **Analysis Approach:**
    1. Define cohort and describe patient characteristics
    2. Characterize treatment patterns (PARP, bevacizumab, both, or neither)
    3. Examine treatment variation by insurance type
    4. Formal statistical testing of treatment-insurance relationships
    """)
    return


@app.cell
async def _(mo, pd):
    import sys as _sys
    import io as _io

    _DATA = 'data-filtered'

    if 'pyodide' in _sys.modules:
        from pyodide.http import pyfetch as _pyfetch

        async def _load(name):
            _r = await _pyfetch(f'{_DATA}/{name}')
            return pd.read_parquet(_io.BytesIO(await _r.bytes()))
    else:
        async def _load(name):
            return pd.read_parquet(f'{_DATA}/{name}')

    diagnosis_df = await _load('diagnosis.parquet')
    medication_ingredient_df = await _load('medication_ingredient.parquet')
    patient_df = await _load('patient.parquet')
    tumor_df = await _load('tumor.parquet')

    _enr_raw = await _load('member_enrollment.parquet')
    _cutoff = pd.Timestamp('2026-01-01')
    _bad = _enr_raw['termination_date'] >= _cutoff
    member_enrollment_df = _enr_raw[~_bad].copy()

    mo.md(
        f"**Enrollment filter:** removed {_bad.sum():,} of {len(_enr_raw):,} records "
        f"with `termination_date ≥ 2026-01-01` "
        f"({_bad.sum() / len(_enr_raw) * 100:.4f}%). "
        f"{len(member_enrollment_df):,} records remain."
    )
    return (
        diagnosis_df,
        medication_ingredient_df,
        member_enrollment_df,
        patient_df,
        tumor_df,
    )


@app.cell
def _(diagnosis_df, mo):
    # Identify patients with C56 (ovarian cancer) diagnosis, from 2012 onward
    earliest_c56 = (
        diagnosis_df[diagnosis_df['code'].str.startswith(('C56', '183.0'), na=False)]
        .sort_values('date')
        .groupby('patient_id', as_index=False)
        .first()
        [['patient_id', 'code', 'date']]
        .rename(columns={'date': 'earliest_diagnosis_date', 'code': 'diagnosis_code'})
        .loc[lambda df: df['earliest_diagnosis_date'] >= '2012-01-01']
    )

    mo.md(
        f"**Total patients with C56 diagnosis (≥2012):** {len(earliest_c56):,}\n\n"
        f"**Date range:** {earliest_c56['earliest_diagnosis_date'].min()} to {earliest_c56['earliest_diagnosis_date'].max()}"
    )
    return (earliest_c56,)


@app.cell
def _(earliest_c56, medication_ingredient_df, member_enrollment_df, mo, pd):
    # Filter enrollment to C56 cohort patients only
    c56_ids = earliest_c56['patient_id']
    c56_enrollment_df = member_enrollment_df[member_enrollment_df['patient_id'].isin(c56_ids)]

    # Medication codes
    PARP_CODES = ['1597982', '1918231', '1862579']
    BEVACIZUMAB_CODE = ['253337']

    # Helper function to match medications to enrollment periods
    def add_treatment_flag(enrollment_df, medication_df, codes, treatment_name):
        """Add boolean flag and start date for treatment within enrollment period."""
        meds = (
            medication_df[medication_df['code'].isin(codes)]
            [['patient_id', 'start_date']]
            .rename(columns={'start_date': f'{treatment_name}_start_date'})
        )

        merged = enrollment_df.merge(meds, on='patient_id', how='left')

        # Check if treatment started within enrollment period
        within_period = (
            (merged[f'{treatment_name}_start_date'] >= merged['effective_date']) &
            (merged[f'{treatment_name}_start_date'] <= merged['termination_date'])
        )

        merged[treatment_name] = within_period.fillna(False)
        merged.loc[~merged[treatment_name], f'{treatment_name}_start_date'] = pd.NaT

        return merged

    # Add treatment flags to enrollment data
    enrollment_with_treatments = (
        c56_enrollment_df
        .pipe(add_treatment_flag, medication_ingredient_df, PARP_CODES, 'PARP')
        .pipe(add_treatment_flag, medication_ingredient_df, BEVACIZUMAB_CODE, 'bevacizumab')
    )

    # Reorder columns for clarity
    treatment_cols = ['PARP', 'PARP_start_date', 'bevacizumab', 'bevacizumab_start_date']
    base_cols = ['patient_id', 'effective_date', 'termination_date']
    other_cols = [c for c in enrollment_with_treatments.columns if c not in base_cols + treatment_cols]

    enrollment_with_treatments = enrollment_with_treatments[base_cols + treatment_cols + other_cols]

    mo.md(
        f"**Total enrollment periods (before deduplication):** {len(enrollment_with_treatments):,}\n\n"
        f"**Periods with PARP:** {enrollment_with_treatments['PARP'].sum():,}\n\n"
        f"**Periods with bevacizumab:** {enrollment_with_treatments['bevacizumab'].sum():,}"
    )
    return (enrollment_with_treatments,)


@app.cell
def _(earliest_c56, enrollment_with_treatments, mo, np, pd):
    # Join enrollment data with diagnosis dates
    _enr = enrollment_with_treatments.merge(
        earliest_c56[['patient_id', 'earliest_diagnosis_date']],
        on='patient_id',
        how='inner'
    )

    # --- Insurance at diagnosis: enrollment period covering the diagnosis date ---
    _ins_cols = ['payer_type', 'plan_design', 'medical_eligible', 'rx_eligible', 'source_id']
    _at_diag = (
        _enr[
            (_enr['earliest_diagnosis_date'] >= _enr['effective_date']) &
            (_enr['earliest_diagnosis_date'] <= _enr['termination_date'])
        ]
        .sort_values('effective_date')
        .groupby('patient_id')
        .first()
        [_ins_cols]
        .reset_index()
    )

    # Fallback: for patients with no enrollment period covering diagnosis,
    # use the nearest enrollment period (earliest period starting after diagnosis,
    # or if none, the latest period before diagnosis)
    _all_patients = _enr[['patient_id']].drop_duplicates()
    _missing = _all_patients[~_all_patients['patient_id'].isin(_at_diag['patient_id'])]
    _n_exact = len(_at_diag)
    if len(_missing) > 0:
        _enr_missing = _enr[_enr['patient_id'].isin(_missing['patient_id'])].copy()
        _enr_missing['_days_after'] = (_enr_missing['effective_date'] - _enr_missing['earliest_diagnosis_date']).dt.days
        # Prefer nearest period starting after diagnosis; fall back to closest before
        _after = _enr_missing[_enr_missing['_days_after'] >= 0].sort_values('_days_after').groupby('patient_id').first()[_ins_cols].reset_index()
        _still_missing = _missing[~_missing['patient_id'].isin(_after['patient_id'])]
        _before = _enr_missing[_enr_missing['patient_id'].isin(_still_missing['patient_id'])].sort_values('_days_after', ascending=False).groupby('patient_id').first()[_ins_cols].reset_index()
        _at_diag = pd.concat([_at_diag, _after, _before], ignore_index=True)

    mo.md(
        f"**Patients with insurance at diagnosis (exact match):** {_n_exact:,}\n\n"
        f"**Patients with insurance from nearest period (fallback):** {len(_missing):,}"
    )

    # --- Aggregate treatment flags across ALL enrollment periods per patient ---
    _treatments = (
        _enr
        .groupby('patient_id')
        .agg({
            'PARP': 'any',
            'PARP_start_date': lambda x: x.dropna().min() if x.notna().any() else pd.NaT,
            'bevacizumab': 'any',
            'bevacizumab_start_date': lambda x: x.dropna().min() if x.notna().any() else pd.NaT,
        })
        .reset_index()
    )

    # --- Insurance at treatment: enrollment period where treatment was flagged ---
    _treated_rows = _enr[_enr['PARP'] | _enr['bevacizumab']].copy()
    _treated_rows['_treatment_date'] = _treated_rows[['PARP_start_date', 'bevacizumab_start_date']].min(axis=1)
    _at_treatment = (
        _treated_rows
        .sort_values('_treatment_date')
        .groupby('patient_id')[['payer_type', 'plan_design']]
        .first()
        .rename(columns={'payer_type': 'payer_type_at_treatment', 'plan_design': 'plan_design_at_treatment'})
        .reset_index()
    )

    # --- Combine: all patients + diagnosis insurance + treatment flags + treatment insurance ---
    enrollment_final = (
        _at_diag
        .merge(_treatments, on='patient_id', how='left')
        .merge(
            earliest_c56[['patient_id', 'earliest_diagnosis_date']],
            on='patient_id',
            how='left'
        )
        .merge(_at_treatment, on='patient_id', how='left')
    )

    # same_payer: True if diagnosis and treatment insurance match (NaN for untreated)
    enrollment_final['same_payer'] = np.where(
        enrollment_final['payer_type_at_treatment'].notna(),
        enrollment_final['payer_type'] == enrollment_final['payer_type_at_treatment'],
        np.nan
    )

    # Create treatment group classification
    conditions = [
        (enrollment_final['PARP'] & enrollment_final['bevacizumab']),
        (enrollment_final['PARP'] & ~enrollment_final['bevacizumab']),
        (~enrollment_final['PARP'] & enrollment_final['bevacizumab']),
    ]
    choices = ['Both', 'PARP only', 'Bevacizumab only']
    enrollment_final['treatment_group'] = np.select(conditions, choices, default='Neither')
    enrollment_final['treated'] = (enrollment_final['PARP'] | enrollment_final['bevacizumab']).astype(int)

    # Drop unknown payer type
    enrollment_final = enrollment_final[enrollment_final['payer_type'] != 'Unknown']

    mo.md(
        f"**Patients with insurance at diagnosis** (dropping Unknown): {len(enrollment_final):,}\n\n"
        f"**Treated:** {enrollment_final['treated'].sum():,} | "
        f"**Untreated:** {(~enrollment_final['treated'].astype(bool)).sum():,}"
    )
    return (enrollment_final,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Descriptive Statistics

    Overview of the ovarian cancer cohort: how patients were identified, demographic characteristics, and insurance coverage at the time of diagnosis.
    """)
    return


@app.cell
def _(
    earliest_c56,
    enrollment_final,
    enrollment_with_treatments,
    member_enrollment_df,
    mo,
):
    # Cohort attrition summary
    steps = [
        ("All enrolled patients (member_enrollment_df)", member_enrollment_df['patient_id'].nunique()),
        ("Patients with C56/183.0 diagnosis ≥2012 (earliest_c56)", len(earliest_c56)),
        ("C56 enrollment periods with treatment flags", enrollment_with_treatments['patient_id'].nunique()),
        ("  → Insurance assigned at diagnosis (enrollment_final)", len(enrollment_final)),
    ]

    _lines = [
        "**Cohort Attrition**\n",
        "| Step | N Patients | Lost |",
        "|------|-----------|------|",
    ]
    prev = None
    for label, n in steps:
        lost = f"-{prev - n:,}" if prev is not None and prev > n else ""
        _lines.append(f"| {label} | {n:,} | {lost} |")
        prev = n
    mo.md("\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Diagnosis Distribution
    """)
    return


@app.cell
def _(earliest_c56, go):
    # Distribution of specific C56 diagnosis codes
    _counts = earliest_c56['diagnosis_code'].value_counts().head(10)
    _fig = go.Figure(data=[
        go.Bar(
            y=_counts.index[::-1],
            x=_counts.values[::-1],
            orientation='h',
            marker=dict(color='steelblue'),
            hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
        )
    ])
    _fig.update_layout(
        title='Top 10 C56 Diagnosis Codes',
        xaxis_title='Number of Patients',
        yaxis_title='Diagnosis Code'
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Patient Demographics
    """)
    return


@app.cell
def _(earliest_c56, mo, patient_df):
    # Merge demographics with C56 cohort
    cohort_demographics = earliest_c56.merge(patient_df, on='patient_id')

    mo.md(
        f"**Cohort size:** {len(cohort_demographics):,} patients | "
        f"**Unique races:** {cohort_demographics['race'].nunique()} | "
        f"**Unique ethnicities:** {cohort_demographics['ethnicity'].nunique()}"
    )
    return (cohort_demographics,)


@app.cell
def _(cohort_demographics, go):
    # Demographics: Race
    _counts = cohort_demographics['race'].value_counts().head(10)
    _fig = go.Figure(data=[
        go.Bar(
            y=_counts.index[::-1],
            x=_counts.values[::-1],
            orientation='h',
            marker=dict(color='steelblue'),
            hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
        )
    ])
    _fig.update_layout(
        title='Patient Distribution by Race (Top 10)',
        xaxis_title='Number of Patients'
    )
    _fig
    return


@app.cell
def _(cohort_demographics, go):
    # Demographics: Ethnicity
    _counts = cohort_demographics['ethnicity'].value_counts().head(10)
    _fig = go.Figure(data=[
        go.Bar(
            y=_counts.index[::-1],
            x=_counts.values[::-1],
            orientation='h',
            marker=dict(color='steelblue'),
            hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
        )
    ])
    _fig.update_layout(
        title='Patient Distribution by Ethnicity (Top 10)',
        xaxis_title='Number of Patients'
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Insurance Coverage at Diagnosis
    """)
    return


@app.cell
def _(enrollment_final, go):
    # Payer Type Distribution plot
    _payer_counts = enrollment_final.payer_type.value_counts()
    _payer_pct = enrollment_final.payer_type.value_counts(normalize=True) * 100

    _fig = go.Figure(data=[
        go.Bar(
            x=_payer_counts.index,
            y=_payer_pct.values,
            text=[f"{pct:.1f}%<br>(n={count:,})" for pct, count in zip(_payer_pct.values, _payer_counts.values)],
            textposition='auto',
            marker=dict(color='steelblue'),
            hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata:,}<extra></extra>',
            customdata=_payer_counts.values
        )
    ])

    _fig.update_layout(
        title='Payer Type Distribution',
        xaxis_title='Payer Type',
        yaxis_title='Percentage (%)',
        showlegend=False
    )

    _fig
    return


@app.cell
def _(enrollment_final, mo):
    # Eligibility rates by treated vs not treated
    _treated_label = enrollment_final['treated'].map({1: 'Treated', 0: 'Not treated'})
    _eligibility_summary = (
        enrollment_final
        .assign(
            _label=_treated_label,
            medical_eligible=(lambda df: (df['medical_eligible'] == 'T').astype(int)),
            rx_eligible=(lambda df: (df['rx_eligible'] == 'T').astype(int))
        )
        .groupby('_label')[['medical_eligible', 'rx_eligible']]
        .mean()
        .reset_index()
        .rename(columns={'_label': 'group'})
    )

    _lines = [
        "**Eligibility Rates by Treatment Status**\n",
        "| Group | Medical Eligible | Rx Eligible |",
        "|-------|-----------------|-------------|",
    ]
    for _, _row in _eligibility_summary.iterrows():
        _lines.append(f"| {_row['group']} | {_row['medical_eligible']:.3f} | {_row['rx_eligible']:.3f} |")
    mo.md("\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Treatment Patterns

    Characterizing who receives PARP inhibitors and/or bevacizumab: overall treatment rates, trends over time, and time from diagnosis to first treatment.
    """)
    return


@app.cell
def _(enrollment_final, go):
    # Treatment Group Distribution plot
    _treatment_counts = enrollment_final.treatment_group.value_counts()
    _treatment_pct = enrollment_final.treatment_group.value_counts(normalize=True) * 100

    _fig = go.Figure(data=[
        go.Bar(
            x=_treatment_counts.index,
            y=_treatment_pct.values,
            text=[f"{pct:.1f}%<br>(n={count:,})" for pct, count in zip(_treatment_pct.values, _treatment_counts.values)],
            textposition='auto',
            marker=dict(color='steelblue'),
            hovertemplate='<b>%{x}</b><br>Percentage: %{y:.1f}%<br>Count: %{customdata:,}<extra></extra>',
            customdata=_treatment_counts.values
        )
    ])

    _fig.update_layout(
        title='Treatment Group Distribution',
        xaxis_title='Treatment Group',
        yaxis_title='Percentage (%)',
        showlegend=False
    )

    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Treatment Trends Over Time
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### PARP Inhibitor vs Bevacizumab Use Over Time
    """)
    return


@app.cell
def _(enrollment_final, go, mo):
    # PARP vs Bev use by diagnosis year
    # Numerator & denominator both keyed on year of diagnosis:
    # "Of patients diagnosed in year X, what % ever received each drug?"
    _df = enrollment_final[['patient_id', 'earliest_diagnosis_date', 'PARP', 'bevacizumab']].copy()
    _df['year'] = _df['earliest_diagnosis_date'].dt.year

    _yearly = _df.groupby('year').agg(
        n_diagnosed=('patient_id', 'count'),
        n_parp=('PARP', 'sum'),
        n_bev=('bevacizumab', 'sum'),
    ).reset_index()
    _yearly['parp_pct'] = _yearly['n_parp'] / _yearly['n_diagnosed'] * 100
    _yearly['bev_pct'] = _yearly['n_bev'] / _yearly['n_diagnosed'] * 100

    _lines = [
        "**PARP vs Bevacizumab Use by Diagnosis Year**\n",
        "*(Of patients diagnosed in each year, % who ever received each drug)*\n",
        "| Year | N Diagnosed | N PARP | PARP% | N Bev | Bev% |",
        "|------|------------|--------|-------|-------|------|",
    ]
    for _, _row in _yearly.iterrows():
        _lines.append(f"| {int(_row['year'])} | {int(_row['n_diagnosed']):,} | {int(_row['n_parp']):,} | "
                      f"{_row['parp_pct']:.1f}% | {int(_row['n_bev']):,} | {_row['bev_pct']:.1f}% |")

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_yearly['year'], y=_yearly['parp_pct'],
        mode='lines+markers', name='PARP Inhibitor',
        line=dict(color='steelblue', width=2),
        hovertemplate='<b>PARP Inhibitor</b><br>Diagnosis Year: %{x}<br>% ever treated: %{y:.1f}%<extra></extra>'
    ))
    _fig.add_trace(go.Scatter(
        x=_yearly['year'], y=_yearly['bev_pct'],
        mode='lines+markers', name='Bevacizumab',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='<b>Bevacizumab</b><br>Diagnosis Year: %{x}<br>% ever treated: %{y:.1f}%<extra></extra>'
    ))
    _fig.update_layout(
        title='PARP Inhibitor vs Bevacizumab Use by Diagnosis Year<br>'
              '<sub>Of patients diagnosed each year, % who ever received each drug</sub>',
        xaxis_title='Diagnosis Year',
        yaxis_title='Patients Ever Receiving Drug (%)',
        xaxis=dict(dtick=1),
        legend_title='Drug Type'
    )
    mo.vstack([mo.md("\n".join(_lines)), _fig])
    return


@app.cell
def _(enrollment_final, mo, pd):
    # Percentage of treated patients per year
    # For each patient-year (from diagnosis year to last year in data),
    # mark as treated if treatment started in or before that year
    _df = enrollment_final[['patient_id', 'earliest_diagnosis_date', 'PARP_start_date', 'bevacizumab_start_date']].copy()
    _df['diagnosis_year'] = _df['earliest_diagnosis_date'].dt.year

    # Earliest treatment date (whichever came first: PARP or bevacizumab)
    _df['treatment_year'] = _df[['PARP_start_date', 'bevacizumab_start_date']].min(axis=1).dt.year

    min_year = int(_df['diagnosis_year'].min())
    max_year = int(_df['diagnosis_year'].max())

    # Cross join patients with years, keep only years >= diagnosis year
    _years = pd.DataFrame({'year': range(min_year, max_year + 1)})
    _expanded = _df.merge(_years, how='cross')
    _expanded = _expanded[_expanded['year'] >= _expanded['diagnosis_year']]

    # Patient is treated in a given year if their treatment started in or before that year
    _expanded['treated'] = _expanded['treatment_year'].notna() & (_expanded['year'] >= _expanded['treatment_year'])

    # Aggregate per year
    _yearly = _expanded.groupby('year').agg(
        n_patients=('patient_id', 'count'),
        n_treated=('treated', 'sum')
    ).reset_index()
    _yearly['pct_treated'] = _yearly['n_treated'] / _yearly['n_patients'] * 100

    _lines = [
        "**Treated Patients by Year**\n",
        "| Year | N Patients | N Treated | % Treated |",
        "|------|-----------|-----------|-----------|",
    ]
    for _, _row in _yearly.iterrows():
        _lines.append(f"| {int(_row['year'])} | {int(_row['n_patients']):,} | {int(_row['n_treated']):,} | {_row['pct_treated']:.1f}% |")
    mo.md("\n".join(_lines))

    yearly_treatment = _yearly
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Time to Treatment
    """)
    return


@app.cell
def _(enrollment_final, mo, np):
    # Time gap between first diagnosis and first treatment
    _df = enrollment_final[enrollment_final['PARP'] | enrollment_final['bevacizumab']].copy()
    _df['treatment_date'] = _df[['PARP_start_date', 'bevacizumab_start_date']].min(axis=1)
    _df['days_to_treatment'] = (_df['treatment_date'] - _df['earliest_diagnosis_date']).dt.days
    _df['months_to_treatment'] = _df['days_to_treatment'] / 30.44

    _n_negative = (_df['days_to_treatment'] < 0).sum()
    _df = _df[_df['days_to_treatment'] >= 0]

    _percentiles = " | ".join(
        f"**P{q}:** {np.percentile(_df['months_to_treatment'].dropna(), q):.1f}m" for q in [25, 50, 75, 90]
    )
    mo.md(
        f"**Time from Diagnosis to First Treatment**\n\n"
        f"N treated patients: {len(_df) + _n_negative:,} | "
        f"Excluded (treatment before diagnosis): {_n_negative:,} | "
        f"N after exclusion: {len(_df):,}\n\n"
        f"| Stat | Value |\n|------|-------|\n"
        f"| Median | {_df['months_to_treatment'].median():.1f} months |\n"
        f"| Mean | {_df['months_to_treatment'].mean():.1f} months |\n"
        f"| Std | {_df['months_to_treatment'].std():.1f} months |\n"
        f"| Min | {_df['months_to_treatment'].min():.1f} months |\n"
        f"| Max | {_df['months_to_treatment'].max():.1f} months |\n\n"
        f"{_percentiles}"
    )

    time_to_treatment = _df[['patient_id', 'earliest_diagnosis_date', 'days_to_treatment', 'months_to_treatment']].copy()
    return (time_to_treatment,)


@app.cell
def _(px, time_to_treatment):
    # Box plot of time to treatment grouped by diagnosis year
    _df = time_to_treatment.copy()
    _df['diagnosis_year'] = _df['earliest_diagnosis_date'].dt.year.astype(str)
    _df = _df.sort_values('earliest_diagnosis_date')

    _fig = px.box(
        _df,
        x='diagnosis_year',
        y='months_to_treatment',
        title='Time from Diagnosis to First Treatment by Diagnosis Year',
        labels={
            'diagnosis_year': 'Year of Diagnosis',
            'months_to_treatment': 'Months to Treatment'
        },
        category_orders={'diagnosis_year': sorted(_df['diagnosis_year'].unique())}
    )
    _fig.update_traces(marker_color='steelblue', line_color='steelblue')
    _fig
    return


@app.cell
def _(enrollment_final, mo, np, pd):
    # Time from diagnosis to first drug start, by treatment group
    _df = enrollment_final[enrollment_final['treatment_group'] != 'Neither'].copy()
    _df['PARP_days'] = (_df['PARP_start_date'] - _df['earliest_diagnosis_date']).dt.days
    _df['bev_days'] = (_df['bevacizumab_start_date'] - _df['earliest_diagnosis_date']).dt.days

    def _days_to_first(row):
        if row['treatment_group'] == 'PARP only':
            return row['PARP_days']
        elif row['treatment_group'] == 'Bevacizumab only':
            return row['bev_days']
        else:  # Both
            vals = [v for v in [row['PARP_days'], row['bev_days']] if pd.notna(v)]
            return min(vals) if vals else np.nan

    _df['days_to_first'] = _df.apply(_days_to_first, axis=1)
    _df['months_to_treatment'] = _df['days_to_first'] / 30.44

    # Exclude negative values (treatment before diagnosis)
    _df = _df[_df['days_to_first'] >= 0]

    _lines = [
        "**Time to First Drug by Treatment Group**\n",
        "| Group | N | Median | Mean | P25 | P75 | P90 |",
        "|-------|---|--------|------|-----|-----|-----|",
    ]
    for _grp, _g in _df.groupby('treatment_group'):
        _m = _g['months_to_treatment'].dropna()
        if len(_m) == 0:
            continue
        _lines.append(f"| {_grp} | {len(_m):,} | {_m.median():.1f}m | {_m.mean():.1f}m | "
                      f"{np.percentile(_m, 25):.1f}m | {np.percentile(_m, 75):.1f}m | "
                      f"{np.percentile(_m, 90):.1f}m |")
    mo.md("\n".join(_lines))

    time_to_treatment_by_group = _df[['patient_id', 'treatment_group', 'months_to_treatment']].copy()
    return (time_to_treatment_by_group,)


@app.cell
def _(px, time_to_treatment_by_group):
    _fig = px.box(
        time_to_treatment_by_group,
        x='treatment_group',
        y='months_to_treatment',
        color='treatment_group',
        title='Time from Diagnosis to First Drug by Treatment Group',
        labels={
            'treatment_group': 'Treatment Group',
            'months_to_treatment': 'Months to First Drug'
        }
    )
    _fig.update_layout(showlegend=False)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Insurance and Treatment

    Examining how treatment rates vary by payer type and plan design. Insurance is assigned based on the enrollment period at the time of first diagnosis.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Insurance Continuity
    """)
    return


@app.cell
def _(enrollment_final, go, mo, pd):
    # Insurance continuity: same vs changed payer between diagnosis and treatment
    _treated = enrollment_final[enrollment_final['treated'] == 1].copy()
    _n_treated = len(_treated)
    _n_same = int(_treated['same_payer'].sum())
    _n_changed = _n_treated - _n_same

    _md_lines = [
        "**Insurance Continuity: Diagnosis vs Treatment**\n",
        f"Treated patients: {_n_treated:,}\n",
        f"- Same payer at treatment: {_n_same:,} ({_n_same/_n_treated*100:.1f}%)",
        f"- Changed payer at treatment: {_n_changed:,} ({_n_changed/_n_treated*100:.1f}%)",
    ]

    # Cross-tab of payer type transitions for patients who changed
    _changed = _treated[_treated['same_payer'] == False]
    if len(_changed) > 0:
        _transitions = pd.crosstab(
            _changed['payer_type'],
            _changed['payer_type_at_treatment'],
            margins=True, margins_name='Total'
        )
        _md_lines.append(f"\n**Payer Type Transitions** (n={len(_changed):,} who changed)\n")
        _md_lines.append("*Rows = at diagnosis, Columns = at treatment*\n")
        _cols = _transitions.columns.tolist()
        _md_lines.append("| | " + " | ".join(str(c) for c in _cols) + " |")
        _md_lines.append("|---|" + "|".join(["---"] * len(_cols)) + "|")
        for _idx, _row in _transitions.iterrows():
            _md_lines.append(f"| {_idx} | " + " | ".join(f"{int(v):,}" for v in _row.values) + " |")

    # Bar chart: same vs changed
    _labels = ['Same payer type', 'Changed payer type']
    _values = [_n_same, _n_changed]
    _fig = go.Figure(data=[
        go.Bar(
            x=_labels,
            y=_values,
            text=[f"{v:,} ({v/_n_treated*100:.1f}%)" for v in _values],
            textposition='auto',
            marker=dict(color=['steelblue', '#e74c3c'])
        )
    ])
    _fig.update_layout(
        title='Insurance Continuity: Same Payer Type at Diagnosis vs Treatment',
        yaxis_title='Number of Patients',
        showlegend=False
    )
    mo.vstack([mo.md("\n".join(_md_lines)), _fig])
    return


@app.cell
def _(enrollment_final, go, mo):
    # Treated vs Not Treated by Payer Type
    _treated_label = enrollment_final['treated'].map({1: 'Treated', 0: 'Not treated'})

    _summary = (
        enrollment_final
        .assign(_label=_treated_label)
        .groupby(['payer_type', '_label'])
        .size()
        .unstack(fill_value=0)
    )
    _summary['Total'] = _summary.sum(axis=1)
    _summary['Treatment Rate'] = _summary['Treated'] / _summary['Total'] * 100

    _lines = [
        "**Treated vs Not Treated by Payer Type**\n",
        "| Payer Type | Not Treated | Treated | Total | Treatment Rate |",
        "|------------|------------|---------|-------|---------------|",
    ]
    for _payer, _row in _summary.iterrows():
        _lines.append(f"| {_payer} | {int(_row.get('Not treated', 0)):,} | {int(_row.get('Treated', 0)):,} | {int(_row['Total']):,} | {_row['Treatment Rate']:.1f}% |")

    # Bar chart: treatment rate by payer type
    _rates = _summary[['Treatment Rate', 'Total']].reset_index().sort_values('Treatment Rate', ascending=False)
    _fig = go.Figure(data=[
        go.Bar(
            x=_rates['payer_type'],
            y=_rates['Treatment Rate'],
            text=[f"{r:.1f}%<br>(n={t:,})" for r, t in zip(_rates['Treatment Rate'], _rates['Total'])],
            textposition='auto',
            marker=dict(color='steelblue'),
            hovertemplate='<b>%{x}</b><br>Treatment Rate: %{y:.1f}%<br>Total: %{customdata:,}<extra></extra>',
            customdata=_rates['Total'].values
        )
    ])
    _fig.update_layout(
        title='Treatment Rate by Payer Type (at Diagnosis)',
        xaxis_title='Payer Type',
        yaxis_title='Treatment Rate (%)',
        showlegend=False
    )
    mo.vstack([mo.md("\n".join(_lines)), _fig])
    return


@app.cell
def _(chi2_contingency, enrollment_final, mo, np, pd):
    # Chi-square test: Treated vs Payer Type (by payer)
    _contingency = pd.crosstab(
        enrollment_final['treated'],
        enrollment_final['payer_type']
    )

    _chi2, _p, _dof, _expected = chi2_contingency(_contingency)

    _sig = "HIGHLY SIGNIFICANT (p < 0.001)" if _p < 0.001 else ("SIGNIFICANT (p < 0.05)" if _p < 0.05 else "NOT SIGNIFICANT (p ≥ 0.05)")

    _expected_df = pd.DataFrame(_expected, index=_contingency.index, columns=_contingency.columns).round(1)
    _std_resid = ((_contingency - _expected) / np.sqrt(_expected)).round(2)

    _obs_lines = ["| | " + " | ".join(str(c) for c in _contingency.columns) + " |",
                  "|---|" + "|".join(["---"] * len(_contingency.columns)) + "|"]
    for _idx, _row in _contingency.iterrows():
        _obs_lines.append(f"| {_idx} | " + " | ".join(f"{v:,}" for v in _row.values) + " |")

    _exp_lines = ["| | " + " | ".join(str(c) for c in _expected_df.columns) + " |",
                  "|---|" + "|".join(["---"] * len(_expected_df.columns)) + "|"]
    for _idx, _row in _expected_df.iterrows():
        _exp_lines.append(f"| {_idx} | " + " | ".join(f"{v:.1f}" for v in _row.values) + " |")

    _resid_lines = ["| | " + " | ".join(str(c) for c in _std_resid.columns) + " |",
                    "|---|" + "|".join(["---"] * len(_std_resid.columns)) + "|"]
    for _idx, _row in _std_resid.iterrows():
        _resid_lines.append(f"| {_idx} | " + " | ".join(f"{v:.2f}" for v in _row.values) + " |")

    mo.md(
        f"**Chi-Square Test: Treated × Payer Type**\n\n"
        f"χ² = {_chi2:.2f} | df = {_dof} | p-value = {_p:.6f}\n\n"
        f"**Result:** {_sig}\n\n"
        f"**Observed counts**\n\n" + "\n".join(_obs_lines) + "\n\n"
        f"**Expected counts**\n\n" + "\n".join(_exp_lines) + "\n\n"
        f"**Standardized residuals**\n\n" + "\n".join(_resid_lines)
    )
    return


@app.cell
def _(enrollment_final, mo, pd):
    # Crosstab: Treated vs not treated by payer type
    _treated_label = enrollment_final['treated'].map({1: 'Treated', 0: 'Not treated'})
    crosstab_payer = pd.crosstab(
        _treated_label,
        enrollment_final['payer_type'],
        margins=True,
        margins_name='Total'
    )

    crosstab_payer_pct = pd.crosstab(
        _treated_label,
        enrollment_final['payer_type'],
        normalize='columns'
    ) * 100

    _cols = crosstab_payer.columns.tolist()
    _count_lines = ["**Treated vs Not Treated by Payer Type**\n",
                    "**Counts:**\n",
                    "| | " + " | ".join(str(c) for c in _cols) + " |",
                    "|---|" + "|".join(["---"] * len(_cols)) + "|"]
    for _idx, _row in crosstab_payer.iterrows():
        _count_lines.append(f"| {_idx} | " + " | ".join(f"{int(v):,}" for v in _row.values) + " |")

    _pct_lines = ["\n**Column Percentages:**\n",
                  "| | " + " | ".join(str(c) for c in crosstab_payer_pct.columns) + " |",
                  "|---|" + "|".join(["---"] * len(crosstab_payer_pct.columns)) + "|"]
    for _idx, _row in crosstab_payer_pct.round(1).iterrows():
        _pct_lines.append(f"| {_idx} | " + " | ".join(f"{v:.1f}%" for v in _row.values) + " |")

    mo.md("\n".join(_count_lines + _pct_lines))
    return


@app.cell
def _(chi2_contingency, enrollment_final, mo, pd):
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests

    _contingency = pd.crosstab(
        enrollment_final['treated'],
        enrollment_final['payer_type']
    )
    _payers = _contingency.columns.tolist()

    # Pairwise chi-square tests
    _rows = []
    for _a, _b in combinations(_payers, 2):
        _sub = _contingency[[_a, _b]]
        _chi2, _p, _, _ = chi2_contingency(_sub)
        _rate_a = _sub.loc[1, _a] / _sub[_a].sum() * 100
        _rate_b = _sub.loc[1, _b] / _sub[_b].sum() * 100
        _rows.append({'payer_a': _a, 'payer_b': _b, 'rate_a': _rate_a, 'rate_b': _rate_b, 'chi2': _chi2, 'p_raw': _p})

    _pairwise = pd.DataFrame(_rows)

    # Bonferroni correction
    _, _p_adj, _, _ = multipletests(_pairwise['p_raw'], method='bonferroni')
    _pairwise['p_adj'] = _p_adj
    _pairwise['significant'] = _pairwise['p_adj'] < 0.05

    _pairwise = _pairwise.sort_values('p_adj')

    _lines = [
        "**Pairwise Chi-Square: Treated × Payer Type (Bonferroni-corrected)**\n",
        "| Payer A | Payer B | Rate A | Rate B | χ² | p (raw) | p (adj) | Sig |",
        "|---------|---------|--------|--------|-----|---------|---------|-----|",
    ]
    for _, _r in _pairwise.iterrows():
        _sig = "\\*\\*\\*" if _r['p_adj'] < 0.001 else ("\\*" if _r['significant'] else "")
        _lines.append(f"| {_r['payer_a']} | {_r['payer_b']} | {_r['rate_a']:.1f}% | {_r['rate_b']:.1f}% | {_r['chi2']:.2f} | {_r['p_raw']:.6f} | {_r['p_adj']:.6f} | {_sig} |")
    mo.md("\n".join(_lines))
    return


@app.cell
def _(enrollment_final):
    # Treated vs Not Treated by Payer Type, grouped by diagnosis year
    _df = enrollment_final.copy()
    _df['diagnosis_year'] = _df['earliest_diagnosis_date'].dt.year

    treatment_by_payer_year = (
        _df.groupby(['diagnosis_year', 'payer_type'])
        .agg(
            n_total=('patient_id', 'count'),
            n_treated=('treated', 'sum')
        )
        .reset_index()
    )
    treatment_by_payer_year['n_not_treated'] = treatment_by_payer_year['n_total'] - treatment_by_payer_year['n_treated']
    treatment_by_payer_year['treatment_rate_pct'] = (treatment_by_payer_year['n_treated'] / treatment_by_payer_year['n_total'] * 100).round(1)

    _lines = [
        "**Treated vs Not Treated by Payer Type (by Year)**\n",
        "| Year | Payer Type | Total | Treated | Not Treated | Rate |",
        "|------|-----------|-------|---------|-------------|------|",
    ]
    # for _, _row in treatment_by_payer_year.iterrows():
    #     _lines.append(f"| {int(_row['diagnosis_year'])} | {_row['payer_type']} | {int(_row['n_total']):,} | {int(_row['n_treated']):,} | {int(_row['n_not_treated']):,} | {_row['treatment_rate_pct']:.1f}% |")
    # mo.md("\n".join(_lines))
    return (treatment_by_payer_year,)


@app.cell
def _(px, treatment_by_payer_year):
    _fig = px.line(
        treatment_by_payer_year,
        x='diagnosis_year',
        y='treatment_rate_pct',
        color='payer_type',
        markers=True,
        title='Treatment Rate by Payer Type Over Time',
        labels={
            'diagnosis_year': 'Year of Diagnosis',
            'treatment_rate_pct': 'Treatment Rate (%)',
            'payer_type': 'Payer Type'
        },
        custom_data=['n_treated', 'n_total']
    )
    _fig.update_layout(xaxis=dict(dtick=1))
    _fig.update_traces(
        hovertemplate=(
            '<b>%{fullData.name}</b><br>'
            'Year: %{x}<br>'
            'Treatment Rate: %{y:.1f}%<br>'
            'Treated: %{customdata[0]:,} / %{customdata[1]:,}<extra></extra>'
        )
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Statistical Analysis

    Formal tests of the relationship between insurance characteristics and treatment status.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Chi-Square Tests of Independence

    Testing whether treatment status (treated vs not treated) is independent of insurance characteristics.
    """)
    return


@app.cell
def _(chi2_contingency, enrollment_final, mo, pd):
    # Chi-square test: Treated vs payer type
    contingency_payer = pd.crosstab(
        enrollment_final['treated'],
        enrollment_final['payer_type']
    )

    chi2_payer, p_payer, dof_payer, _ = chi2_contingency(contingency_payer)

    _sig = "HIGHLY SIGNIFICANT (p < 0.001)" if p_payer < 0.001 else ("SIGNIFICANT (p < 0.05)" if p_payer < 0.05 else "NOT SIGNIFICANT (p ≥ 0.05)")
    mo.md(
        f"**Chi-Square Test: Treated × Payer Type**\n\n"
        f"χ² = {chi2_payer:.2f} | df = {dof_payer} | p-value = {p_payer:.6f}\n\n"
        f"**Result:** {_sig}"
    )
    return chi2_payer, p_payer


@app.cell
def _(chi2_contingency, enrollment_final, mo, pd):
    # Chi-square test: Treated vs plan design
    contingency_plan = pd.crosstab(
        enrollment_final['treated'],
        enrollment_final['plan_design']
    )

    chi2_plan, p_plan, dof_plan, _ = chi2_contingency(contingency_plan)

    _sig = "HIGHLY SIGNIFICANT (p < 0.001)" if p_plan < 0.001 else ("SIGNIFICANT (p < 0.05)" if p_plan < 0.05 else "NOT SIGNIFICANT (p ≥ 0.05)")
    mo.md(
        f"**Chi-Square Test: Treated × Plan Design**\n\n"
        f"χ² = {chi2_plan:.2f} | df = {dof_plan} | p-value = {p_plan:.6f}\n\n"
        f"**Result:** {_sig}"
    )
    return chi2_plan, p_plan


@app.cell
def _(mo):
    mo.md(r"""
    ### Logistic Regression

    Examining odds of receiving any treatment (PARP or bevacizumab), controlling for insurance characteristics and eligibility.
    """)
    return


@app.cell
def _(enrollment_final, mo, pd):
    # Prepare regression dataset with dummy variables for payer_type only
    regression_df = (
        enrollment_final
        .assign(
            rx_eligible_binary=(lambda df: (df['rx_eligible'] == 'T').astype(int))
        )
        .pipe(pd.get_dummies, columns=['payer_type'], drop_first=True, dtype=int)
    )

    # Extract predictor columns (exclude _at_treatment columns)
    payer_predictors = [c for c in regression_df.columns
                        if c.startswith('payer_type_') and '_at_treatment' not in c]
    all_predictors = payer_predictors + ['rx_eligible_binary']

    mo.md(
        f"**Regression dataset:** {len(regression_df):,} observations\n\n"
        f"**Predictors:** {len(all_predictors)} ({len(payer_predictors)} payer type + 1 rx eligibility)"
    )
    return all_predictors, regression_df


@app.cell
def _(all_predictors, mo, np, pd, regression_df, sm):
    # Logistic regression: Any treatment (PARP or bevacizumab)
    X = sm.add_constant(regression_df[all_predictors])
    y = regression_df['treated'].astype(int)

    model_treated = sm.Logit(y, X).fit(disp=0)

    # Extract odds ratios
    _results = pd.DataFrame({
        'Variable': model_treated.params.index,
        'Odds Ratio': np.exp(model_treated.params.values),
        'OR 95% CI Lower': np.exp(model_treated.conf_int()[0]),
        'OR 95% CI Upper': np.exp(model_treated.conf_int()[1]),
        'P-value': model_treated.pvalues.values,
        'Significant': model_treated.pvalues.values < 0.05
    })

    treated_odds_ratios = _results[_results['Variable'] != 'const'].sort_values('Odds Ratio', ascending=False)

    _lines = [
        "**Logistic Regression: Any Treatment (PARP or Bevacizumab)**\n",
        f"Pseudo R²: {model_treated.prsquared:.4f} | Log-Likelihood: {model_treated.llf:.2f}\n",
        "**Odds Ratios (sorted):**\n",
        "| Variable | Odds Ratio | P-value | Significant |",
        "|----------|-----------|---------|-------------|",
    ]
    for _, _row in treated_odds_ratios.iterrows():
        _lines.append(f"| {_row['Variable']} | {_row['Odds Ratio']:.3f} | {_row['P-value']:.4f} | {_row['Significant']} |")
    mo.md("\n".join(_lines))
    return model_treated, treated_odds_ratios


@app.cell
def _(go, np, treated_odds_ratios):
    # Forest plot: Treatment odds ratios
    _valid_mask = (
        (treated_odds_ratios['Odds Ratio'] > 0) &
        (treated_odds_ratios['Odds Ratio'] < np.inf) &
        (treated_odds_ratios['OR 95% CI Lower'] > 0) &
        (treated_odds_ratios['OR 95% CI Upper'] < np.inf)
    )
    _plot_data = treated_odds_ratios[_valid_mask].copy().sort_values('Odds Ratio')

    _colors = ['steelblue' if sig else 'lightgray' for sig in _plot_data['Significant']]

    _fig = go.Figure()
    _fig.add_shape(type='line', x0=1, x1=1, y0=-0.5, y1=len(_plot_data) - 0.5,
                   line=dict(color='red', dash='dash'))

    _fig.add_trace(go.Scatter(
        x=_plot_data['Odds Ratio'],
        y=_plot_data['Variable'],
        mode='markers',
        marker=dict(size=10, color=_colors),
        error_x=dict(
            type='data',
            symmetric=False,
            array=(_plot_data['OR 95% CI Upper'] - _plot_data['Odds Ratio']).values,
            arrayminus=(_plot_data['Odds Ratio'] - _plot_data['OR 95% CI Lower']).values,
            color='gray'
        ),
        hovertemplate=(
            '<b>%{y}</b><br>'
            'OR: %{x:.3f}<br>'
            'CI: %{customdata[0]:.3f} – %{customdata[1]:.3f}<br>'
            'p: %{customdata[2]:.4f}<extra></extra>'
        ),
        customdata=_plot_data[['OR 95% CI Lower', 'OR 95% CI Upper', 'P-value']].values
    ))

    _fig.update_layout(
        title='Any Treatment: Odds Ratios with 95% Confidence Intervals',
        xaxis_title='Odds Ratio (log scale)',
        xaxis_type='log',
        showlegend=False,
        height=400
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Survival Analysis

    Time from diagnosis to death, stratified by payer type.

    **Event:** death recorded in `month_year_death` (month precision → first of month used).
    **Cross-validation:** if a patient has any enrollment period whose `termination_date` falls
    after their recorded death month, the death record is considered unreliable and the patient
    is censored at their last enrollment end date instead.
    **Censoring:** patients with no death record are censored at their last enrollment `termination_date`.
    **2-year / 5-year survival:** all patients are included; follow-up is capped at 730 or 1825 days respectively. Patients censored before the cap contribute their observed time.
    """)
    return


@app.cell
def _(enrollment_final, member_enrollment_df, mo, patient_df, pd):
    # ── 1. Last enrollment date per patient ──────────────────────────────────
    # Cap sentinel far-future termination dates at today
    _today = pd.Timestamp.today().normalize()
    _enr = member_enrollment_df[member_enrollment_df['patient_id'].isin(enrollment_final['patient_id'])].copy()
    _enr['termination_date'] = _enr['termination_date'].clip(upper=_today)
    _last_enr = (
        _enr
        .groupby('patient_id')['termination_date']
        .max()
        .reset_index()
        .rename(columns={'termination_date': 'last_enrollment_date'})
    )

    # ── 2. Death info from patient_df ────────────────────────────────────────
    _death = patient_df[['patient_id', 'month_year_death', 'year_of_birth']].copy()
    # month_year_death is month-precision → use first of month
    _death['death_date'] = _death['month_year_death'].dt.to_period('M').dt.to_timestamp()

    # ── 3. Merge everything ───────────────────────────────────────────────────
    _surv = (
        enrollment_final[['patient_id', 'payer_type', 'earliest_diagnosis_date', 'treated']]
        .merge(_last_enr, on='patient_id', how='left')
        .merge(_death[['patient_id', 'death_date', 'year_of_birth']], on='patient_id', how='left')
    )

    # ── 4. Cross-validate: enrollment after death → censor ───────────────────
    _has_enr_after_death = (
        _surv['last_enrollment_date'].notna() &
        _surv['death_date'].notna() &
        (_surv['last_enrollment_date'] > _surv['death_date'])
    )
    _surv['event'] = (_surv['death_date'].notna() & ~_has_enr_after_death).astype(int)

    # ── 5. Survival / censoring time (days from diagnosis) ───────────────────
    _surv['end_date'] = _surv['death_date'].where(_surv['event'] == 1, _surv['last_enrollment_date'])
    _surv['survival_days'] = (_surv['end_date'] - _surv['earliest_diagnosis_date']).dt.days
    _surv['survival_months'] = _surv['survival_days'] / 30.44

    # Drop patients with negative or zero survival time (data issues)
    _surv = _surv[_surv['survival_days'] > 0].copy()

    # ── 6. Age at diagnosis ───────────────────────────────────────────────────
    _surv['age_at_diagnosis'] = (
        _surv['earliest_diagnosis_date'].dt.year - _surv['year_of_birth'].astype('Int64')
    )

    n_events = int(_surv['event'].sum())
    n_censored = int((1 - _surv['event']).sum())
    mo.md(
        f"**Survival Dataset Summary**\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Total patients | {len(_surv):,} |\n"
        f"| Deaths (events) | {n_events:,} ({n_events/len(_surv)*100:.1f}%) |\n"
        f"| Censored | {n_censored:,} ({n_censored/len(_surv)*100:.1f}%) |\n"
        f"| Median survival | {_surv['survival_months'].median():.1f} months |\n"
        f"| Cross-validated (enrollment after death → censored) | {_has_enr_after_death.sum():,} |\n"
        f"| Age at diagnosis — median | {_surv['age_at_diagnosis'].median():.0f} |\n"
        f"| Age at diagnosis — range | {_surv['age_at_diagnosis'].min():.0f}–{_surv['age_at_diagnosis'].max():.0f} |"
    )

    survival_df = _surv[['patient_id', 'payer_type', 'treated', 'event',
                          'survival_days', 'survival_months', 'age_at_diagnosis',
                          'earliest_diagnosis_date']].copy()
    return (survival_df,)


@app.cell
def _(member_enrollment_df, mo, pd, survival_df):
    # ── 2-year survival rate overall and by payer type ────────────────────────
    # Sentinel far-future dates (open enrollments) are capped at today
    _today = pd.Timestamp.today().normalize()
    _max_date = member_enrollment_df['termination_date'].clip(upper=_today).max()
    _surv2 = survival_df.copy()

    # Cap follow-up at 24 months (730 days); re-derive event within window
    _surv2['event_2yr'] = ((_surv2['event'] == 1) & (_surv2['survival_days'] <= 730)).astype(int)
    _surv2['time_2yr'] = _surv2['survival_days'].clip(upper=730)

    # Overall 2-year survival (naive rate — does not account for censoring;
    # the KM curve below provides the censoring-adjusted estimate)
    _n = len(_surv2)
    _died_2yr = int(_surv2['event_2yr'].sum())
    _censored_2yr = int(((_surv2['event_2yr'] == 0) & (_surv2['time_2yr'] < 730)).sum())
    _overall_2yr = (1 - _died_2yr / _n) * 100

    _lines = [
        f"**2-Year Survival (all patients, follow-up capped at 2 years)**\n",
        f"Eligible patients: {_n:,} | Deaths within 2 yrs: {_died_2yr:,} | Censored before 2 yrs: {_censored_2yr:,}\n",
        f"Naive 2-year survival rate: **{_overall_2yr:.1f}%** *(does not adjust for censoring — see KM curve below)*\n",
        "| Payer Type | N | Deaths (2yr) | Naive 2yr Surv |",
        "|------------|---|-------------|---------------|",
    ]
    for _payer, _g in _surv2.groupby('payer_type'):
        _nd = int(_g['event_2yr'].sum())
        _rate = (1 - _nd / len(_g)) * 100
        _lines.append(f"| {_payer} | {len(_g):,} | {_nd:,} | {_rate:.1f}% |")
    mo.md("\n".join(_lines))

    surv2yr_df = _surv2
    max_date = _max_date
    return (surv2yr_df,)


@app.cell
def _(go, mo, surv2yr_df):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    _fig = go.Figure()
    _colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    _payers = sorted(surv2yr_df['payer_type'].unique())
    _all_sf_vals = []

    for _i, _payer in enumerate(_payers):
        _g = surv2yr_df[surv2yr_df['payer_type'] == _payer]
        _kmf = KaplanMeierFitter()
        _kmf.fit(_g['time_2yr'], event_observed=_g['event_2yr'], label=_payer)
        # Use fitted step-function data (at event times) — avoids missing method in lifelines 0.30
        _sf = _kmf.survival_function_
        _ci = _kmf.confidence_interval_survival_function_
        _t = _sf.index.tolist()
        _col = _colors[_i % len(_colors)]
        _n = len(_g)

        _sf_pct = (_sf.iloc[:, 0].values * 100).tolist()
        _ci_lo_pct = (_ci.iloc[:, 0].values * 100).tolist()
        _all_sf_vals.extend(_sf_pct + _ci_lo_pct)

        _fig.add_trace(go.Scatter(
            x=_t, y=_sf_pct,
            mode='lines', name=f'{_payer} (n={_n:,})',
            line=dict(color=_col, width=2), line_shape='hv',
            hovertemplate=f'<b>{_payer}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
        ))
        _fig.add_trace(go.Scatter(
            x=_t + _t[::-1],
            y=_ci_lo_pct + (_ci.iloc[:, 1].values * 100).tolist()[::-1],
            fill='toself', fillcolor=_col, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip'
        ))

    _y_min = max(0, min(_all_sf_vals) - 2)
    _y_max = 102

    # Overall log-rank test across all payers
    _lr = multivariate_logrank_test(
        surv2yr_df['time_2yr'], surv2yr_df['payer_type'], surv2yr_df['event_2yr']
    )
    _p_lr = _lr.p_value

    _fig.update_layout(
        title=f'Kaplan-Meier: 2-Year Survival by Payer Type<br>'
              f'<sup>Log-rank p = {_p_lr:.4f} | All patients, follow-up capped at 2 years</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[_y_min, _y_max]),
        legend_title='Payer Type',
        hovermode='x unified'
    )

    mo.vstack([
        _fig,
        mo.md(f"Overall log-rank test across payer types: χ² = {_lr.test_statistic:.2f}, p = {_p_lr:.4f}")
    ])
    return


@app.cell
def _(go, mo, surv5yr_df):
    from lifelines import KaplanMeierFitter as _KMF5p
    from lifelines.statistics import multivariate_logrank_test as _mlrt5p

    _fig5p = go.Figure()
    _colors5p = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]
    _payers5p = sorted(surv5yr_df['payer_type'].unique())
    _all_sf_vals5p = []

    for _i5p, _payer5p in enumerate(_payers5p):
        _g5p = surv5yr_df[surv5yr_df['payer_type'] == _payer5p]
        _kmf5p = _KMF5p()
        _kmf5p.fit(_g5p['time_5yr'], event_observed=_g5p['event_5yr'], label=_payer5p)
        _sf5p = _kmf5p.survival_function_
        _ci5p = _kmf5p.confidence_interval_survival_function_
        _t5p = _sf5p.index.tolist()
        _col5p = _colors5p[_i5p % len(_colors5p)]
        _n5p = len(_g5p)

        _sf_pct5p = (_sf5p.iloc[:, 0].values * 100).tolist()
        _ci_lo_pct5p = (_ci5p.iloc[:, 0].values * 100).tolist()
        _all_sf_vals5p.extend(_sf_pct5p + _ci_lo_pct5p)

        _fig5p.add_trace(go.Scatter(
            x=_t5p, y=_sf_pct5p,
            mode='lines', name=f'{_payer5p} (n={_n5p:,})',
            line=dict(color=_col5p, width=2), line_shape='hv',
            hovertemplate=f'<b>{_payer5p}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
        ))
        _fig5p.add_trace(go.Scatter(
            x=_t5p + _t5p[::-1],
            y=_ci_lo_pct5p + (_ci5p.iloc[:, 1].values * 100).tolist()[::-1],
            fill='toself', fillcolor=_col5p, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip'
        ))

    _y_min5p = max(0, min(_all_sf_vals5p) - 2)

    _lr5p = _mlrt5p(
        surv5yr_df['time_5yr'], surv5yr_df['payer_type'], surv5yr_df['event_5yr']
    )
    _p_lr5p = _lr5p.p_value

    _fig5p.update_layout(
        title=f'Kaplan-Meier: 5-Year Survival by Payer Type<br>'
              f'<sup>Log-rank p = {_p_lr5p:.4f} | All patients, follow-up capped at 5 years</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[_y_min5p, 102]),
        legend_title='Payer Type',
        hovermode='x unified'
    )

    mo.vstack([
        _fig5p,
        mo.md(f"Overall log-rank test across payer types: χ² = {_lr5p.test_statistic:.2f}, p = {_p_lr5p:.4f}")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive 5-Year KM Plot

    Toggle the segmentation dimensions (both can be on at once), optionally edit age-bin edges, then click **Calculate** to render.
    """)
    return


@app.cell
def _(mo):
    km_payer_toggle = mo.ui.switch(value=True, label='Segment by Payer Type')
    km_age_toggle = mo.ui.switch(value=False, label='Segment by Age Bin')
    km_age_bins_input = mo.ui.text(
        value='0,50,65,75,120',
        label='Age bin edges (comma-separated)',
        full_width=True,
    )
    km_run_btn = mo.ui.run_button(label='Calculate', kind='success')
    mo.vstack([km_payer_toggle, km_age_toggle, km_age_bins_input, km_run_btn])
    return km_age_bins_input, km_age_toggle, km_payer_toggle, km_run_btn


@app.cell
def _(
    go,
    km_age_bins_input,
    km_age_toggle,
    km_payer_toggle,
    km_run_btn,
    mo,
    pd,
    surv5yr_df,
):
    from lifelines import KaplanMeierFitter as _KMFi
    from lifelines.statistics import multivariate_logrank_test as _mlrti

    mo.stop(
        not km_run_btn.value,
        mo.md("*Configure options above and click **Calculate** to render the plot.*"),
    )
    mo.stop(
        not (km_payer_toggle.value or km_age_toggle.value),
        mo.md("⚠️ Turn on at least one segmentation toggle (Payer Type and/or Age Bin)."),
    )

    _df_i = surv5yr_df.copy()
    _dims = []

    if km_age_toggle.value:
        try:
            _edges = sorted({float(x.strip()) for x in km_age_bins_input.value.split(',') if x.strip()})
        except ValueError:
            _edges = []
        mo.stop(
            len(_edges) < 2,
            mo.md("⚠️ Please enter at least 2 numeric bin edges separated by commas."),
        )
        _labels = [f"{int(_edges[_k])} <= age < {int(_edges[_k+1])}" for _k in range(len(_edges) - 1)]
        _df_i['_age_bin'] = pd.cut(
            _df_i['age_at_diagnosis'].astype(float),
            bins=_edges,
            labels=_labels,
            include_lowest=True,
            right=False,
        ).astype(object)
        _dims.append(('Age Bin', '_age_bin'))

    if km_payer_toggle.value:
        _dims.append(('Payer Type', 'payer_type'))

    if len(_dims) == 1:
        _label, _col = _dims[0]
        _df_i['_group'] = _df_i[_col]
        _group_col = _label
    else:
        _df_i['_group'] = _df_i[_dims[0][1]].astype(str) + ' | ' + _df_i[_dims[1][1]].astype(str)
        _df_i.loc[_df_i[_dims[0][1]].isna() | _df_i[_dims[1][1]].isna(), '_group'] = pd.NA
        _group_col = ' × '.join(_d[0] for _d in _dims)

    _df_i = _df_i[_df_i['_group'].notna()].copy()

    _palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    _groups_i = sorted(_df_i['_group'].unique().tolist(), key=str)
    _all_vals = []
    _fig_i = go.Figure()

    for _idx, _grp in enumerate(_groups_i):
        _gd = _df_i[_df_i['_group'] == _grp]
        if len(_gd) == 0:
            continue
        _kmf_i = _KMFi()
        _kmf_i.fit(_gd['time_5yr'], event_observed=_gd['event_5yr'], label=str(_grp))
        _sf_i = _kmf_i.survival_function_
        _ci_i = _kmf_i.confidence_interval_survival_function_
        _t_i = _sf_i.index.tolist()
        _color = _palette[_idx % len(_palette)]

        _sf_pct = (_sf_i.iloc[:, 0].values * 100).tolist()
        _ci_lo = (_ci_i.iloc[:, 0].values * 100).tolist()
        _ci_hi = (_ci_i.iloc[:, 1].values * 100).tolist()
        _all_vals.extend(_sf_pct + _ci_lo)

        _fig_i.add_trace(go.Scatter(
            x=_t_i, y=_sf_pct,
            mode='lines', name=f'{_grp} (n={len(_gd):,})',
            line=dict(color=_color, width=2), line_shape='hv',
            hovertemplate=f'<b>{_grp}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>',
        ))
        _fig_i.add_trace(go.Scatter(
            x=_t_i + _t_i[::-1],
            y=_ci_lo + _ci_hi[::-1],
            fill='toself', fillcolor=_color, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip',
        ))

    _y_min_i = max(0, min(_all_vals) - 2) if _all_vals else 0

    if _df_i['_group'].nunique() >= 2:
        _lr_i = _mlrti(_df_i['time_5yr'], _df_i['_group'].astype(str), _df_i['event_5yr'])
        _title_sup = f'Log-rank p = {_lr_i.p_value:.4f} | follow-up capped at 5 years'
        _lr_md = mo.md(
            f"Overall log-rank test across {_group_col.lower()} groups: "
            f"χ² = {_lr_i.test_statistic:.2f}, p = {_lr_i.p_value:.4f}"
        )
    else:
        _title_sup = 'follow-up capped at 5 years'
        _lr_md = mo.md("*Need ≥2 groups for a log-rank test.*")

    _fig_i.update_layout(
        title=f'Kaplan-Meier: 5-Year Survival by {_group_col}<br><sup>{_title_sup}</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[_y_min_i, 102]),
        legend_title=_group_col,
        hovermode='x unified',
    )

    mo.vstack([_fig_i, _lr_md])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2-Year Survival by Treatment Status
    """)
    return


@app.cell
def _(go, mo, survival_df):
    from lifelines import KaplanMeierFitter as _KMF2t
    from lifelines.statistics import multivariate_logrank_test as _mlrt2t

    _surv2t = survival_df.copy()
    _surv2t['event_2t'] = ((_surv2t['event'] == 1) & (_surv2t['survival_days'] <= 730)).astype(int)
    _surv2t['time_2t']  = _surv2t['survival_days'].clip(upper=730)
    _surv2t['treatment_group'] = _surv2t['treated'].map({1: 'Treated', 0: 'Not Treated'})

    _colors2t = {'Treated': '#1f77b4', 'Not Treated': '#ff7f0e'}
    _fig2t = go.Figure()
    _all_sf_vals2t = []

    for _grp2t in ['Treated', 'Not Treated']:
        _g2t = _surv2t[_surv2t['treatment_group'] == _grp2t]
        _kmf2t = _KMF2t()
        _kmf2t.fit(_g2t['time_2t'], event_observed=_g2t['event_2t'])
        _sf2t = _kmf2t.survival_function_
        _ci2t = _kmf2t.confidence_interval_survival_function_
        _t2t  = _sf2t.index.tolist()
        _col2t = _colors2t[_grp2t]

        _sf_pct2t   = (_sf2t.iloc[:, 0].values * 100).tolist()
        _ci_lo_pct2t = (_ci2t.iloc[:, 0].values * 100).tolist()
        _all_sf_vals2t.extend(_sf_pct2t + _ci_lo_pct2t)

        _fig2t.add_trace(go.Scatter(
            x=_t2t, y=_sf_pct2t,
            mode='lines', name=f'{_grp2t} (n={len(_g2t):,})',
            line=dict(color=_col2t, width=2), line_shape='hv',
            hovertemplate=f'<b>{_grp2t}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
        ))
        _fig2t.add_trace(go.Scatter(
            x=_t2t + _t2t[::-1],
            y=_ci_lo_pct2t + (_ci2t.iloc[:, 1].values * 100).tolist()[::-1],
            fill='toself', fillcolor=_col2t, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip'
        ))

    _lr2t = _mlrt2t(_surv2t['time_2t'], _surv2t['treatment_group'], _surv2t['event_2t'])
    _fig2t.update_layout(
        title=f'Kaplan-Meier: 2-Year Survival by Treatment Status<br>'
              f'<sup>Log-rank p = {_lr2t.p_value:.4f} | All patients, follow-up capped at 2 years</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[max(0, min(_all_sf_vals2t) - 2), 102]),
        legend_title='Treatment Status',
        hovermode='x unified'
    )

    mo.vstack([
        _fig2t,
        mo.md(f"Log-rank test (Treated vs Not Treated): χ² = {_lr2t.test_statistic:.2f}, p = {_lr2t.p_value:.4f}")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5-Year Survival by Treatment Type
    """)
    return


@app.cell
def _(mo, survival_df):
    _surv5 = survival_df.copy()

    # Cap follow-up at 1825 days (5 years); re-derive event within window
    _surv5['event_5yr'] = ((_surv5['event'] == 1) & (_surv5['survival_days'] <= 1825)).astype(int)
    _surv5['time_5yr'] = _surv5['survival_days'].clip(upper=1825)

    _surv5['treatment_group'] = _surv5['treated'].map({1: 'Treated', 0: 'Not Treated'})

    _n = len(_surv5)
    _died_5yr = int(_surv5['event_5yr'].sum())
    _overall_5yr = (1 - _died_5yr / _n) * 100

    _lines = [
        f"**5-Year Survival (all patients, follow-up capped at 5 years)**\n",
        f"Eligible patients: {_n:,} | Deaths within 5 yrs: {_died_5yr:,} | Overall 5-year survival rate: **{_overall_5yr:.1f}%**\n",
        "| Treatment Group | N | Deaths (5yr) | 5yr Survival |",
        "|----------------|---|-------------|-------------|",
    ]
    for _grp, _g in _surv5.groupby('treatment_group'):
        _nd = int(_g['event_5yr'].sum())
        _rate = (1 - _nd / len(_g)) * 100
        _lines.append(f"| {_grp} | {len(_g):,} | {_nd:,} | {_rate:.1f}% |")
    mo.md("\n".join(_lines))

    surv5yr_df = _surv5
    return (surv5yr_df,)


@app.cell
def _(go, mo, surv5yr_df):
    from lifelines import KaplanMeierFitter as _KMF5
    from lifelines.statistics import multivariate_logrank_test as _mlrt5

    _fig = go.Figure()
    _colors5 = {'Treated': '#1f77b4', 'Not Treated': '#ff7f0e'}
    _groups5 = ['Treated', 'Not Treated']
    _all_sf_vals5 = []

    for _grp5 in _groups5:
        _g5 = surv5yr_df[surv5yr_df['treatment_group'] == _grp5]
        _kmf5 = _KMF5()
        _kmf5.fit(_g5['time_5yr'], event_observed=_g5['event_5yr'], label=_grp5)
        _sf5 = _kmf5.survival_function_
        _ci5 = _kmf5.confidence_interval_survival_function_
        _t5 = _sf5.index.tolist()
        _col5 = _colors5[_grp5]
        _n5 = len(_g5)

        _sf_pct5 = (_sf5.iloc[:, 0].values * 100).tolist()
        _ci_lo_pct5 = (_ci5.iloc[:, 0].values * 100).tolist()
        _all_sf_vals5.extend(_sf_pct5 + _ci_lo_pct5)

        _fig.add_trace(go.Scatter(
            x=_t5, y=_sf_pct5,
            mode='lines', name=f'{_grp5} (n={_n5:,})',
            line=dict(color=_col5, width=2), line_shape='hv',
            hovertemplate=f'<b>{_grp5}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
        ))
        _fig.add_trace(go.Scatter(
            x=_t5 + _t5[::-1],
            y=_ci_lo_pct5 + (_ci5.iloc[:, 1].values * 100).tolist()[::-1],
            fill='toself', fillcolor=_col5, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip'
        ))

    _y_min5 = max(0, min(_all_sf_vals5) - 2)
    _y_max5 = 102

    # Filter out NaN treatment groups before log-rank test
    _valid5 = surv5yr_df[surv5yr_df['treatment_group'].notna()]
    _lr5 = _mlrt5(_valid5['time_5yr'], _valid5['treatment_group'], _valid5['event_5yr'])
    _p_lr5 = _lr5.p_value

    _fig.update_layout(
        title=f'Kaplan-Meier: 5-Year Survival by Treatment Group<br>'
              f'<sup>Log-rank p = {_p_lr5:.4f} | All patients, follow-up capped at 5 years</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[_y_min5, _y_max5]),
        legend_title='Treatment Group',
        hovermode='x unified'
    )

    mo.vstack([
        _fig,
        mo.md(f"Overall log-rank test across treatment groups: χ² = {_lr5.test_statistic:.2f}, p = {_p_lr5:.4f}")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5-Year Survival by Treatment Status, Stratified by Age Group
    """)
    return


@app.cell
def _(go, mo, surv5yr_df):
    from lifelines import KaplanMeierFitter as _KMFage
    from lifelines.statistics import logrank_test as _lrt_age

    _age_groups = [
        ('<45',   lambda df: df['age_at_diagnosis'] < 45),
        ('45–60', lambda df: (df['age_at_diagnosis'] >= 45) & (df['age_at_diagnosis'] < 60)),
        ('60–70', lambda df: (df['age_at_diagnosis'] >= 60) & (df['age_at_diagnosis'] < 70)),
        ('70+',   lambda df: df['age_at_diagnosis'] >= 70),
    ]
    _colors_age = {'Treated': '#1f77b4', 'Not Treated': '#ff7f0e'}

    _figs_age = []
    for _age_label, _age_mask in _age_groups:
        _subset = surv5yr_df[_age_mask(surv5yr_df)].copy()
        _fig_age = go.Figure()
        _all_sf_age = []

        for _grp, _gval in [('Treated', 1), ('Not Treated', 0)]:
            _g = _subset[_subset['treated'] == _gval]
            if len(_g) < 5:
                continue
            _kmf = _KMFage()
            _kmf.fit(_g['time_5yr'], event_observed=_g['event_5yr'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors_age[_grp]
            _n   = len(_g)

            _sf_pct = (_sf.iloc[:, 0].values * 100).tolist()
            _ci_lo  = (_ci.iloc[:, 0].values * 100).tolist()
            _all_sf_age.extend(_sf_pct + _ci_lo)

            _fig_age.add_trace(go.Scatter(
                x=_t, y=_sf_pct,
                mode='lines', name=f'{_grp} (n={_n:,})',
                line=dict(color=_col, width=2), line_shape='hv',
                hovertemplate=f'<b>{_grp}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
            ))
            _fig_age.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=_ci_lo + (_ci.iloc[:, 1].values * 100).tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.08,
                line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
                showlegend=False, hoverinfo='skip'
            ))

        # Log-rank test between treated and not treated within age group
        _t_grp  = _subset[_subset['treated'] == 1]
        _nt_grp = _subset[_subset['treated'] == 0]
        _lr_age = None
        if len(_t_grp) >= 5 and len(_nt_grp) >= 5:
            _lr_age = _lrt_age(
                _t_grp['time_5yr'],  _nt_grp['time_5yr'],
                event_observed_A=_t_grp['event_5yr'],
                event_observed_B=_nt_grp['event_5yr'],
            )

        _y_min_age = max(0, min(_all_sf_age) - 2) if _all_sf_age else 0
        _lr_note = (f"Log-rank p = {_lr_age.p_value:.4f}" if _lr_age else "insufficient data for log-rank")
        _fig_age.update_layout(
            title=f'5-Year KM by Treatment Status — Age {_age_label} (n={len(_subset):,})<br>'
                  f'<sup>{_lr_note}</sup>',
            xaxis_title='Days from Diagnosis',
            yaxis_title='Survival Probability (%)',
            yaxis=dict(range=[_y_min_age, 102]),
            legend_title='Treatment',
            hovermode='x unified',
        )
        _figs_age.append(_fig_age)

    mo.vstack([_fig for _fig in _figs_age])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5-Year Survival by Treatment Status and Age Group, by Payer Type
    """)
    return


@app.cell
def _(go, mo, surv5yr_df):
    from lifelines import KaplanMeierFitter as _KMFagep
    from lifelines.statistics import logrank_test as _lrt_agep

    _age_groups_p = [
        ('<45',   lambda df: df['age_at_diagnosis'] < 45),
        ('45–60', lambda df: (df['age_at_diagnosis'] >= 45) & (df['age_at_diagnosis'] < 60)),
        ('60–70', lambda df: (df['age_at_diagnosis'] >= 60) & (df['age_at_diagnosis'] < 70)),
        ('70+',   lambda df: df['age_at_diagnosis'] >= 70),
    ]
    _colors_agep = {'Treated': '#1f77b4', 'Not Treated': '#ff7f0e'}
    _payers_agep = sorted(surv5yr_df['payer_type'].dropna().unique())

    _all_figs_agep = []
    for _payer_agep in _payers_agep:
        _payer_df = surv5yr_df[surv5yr_df['payer_type'] == _payer_agep]
        _figs_agep = []
        for _age_label_p, _age_mask_p in _age_groups_p:
            _subset_p = _payer_df[_age_mask_p(_payer_df)].copy()
            _fig_agep = go.Figure()
            _all_sf_agep = []

            for _grp_p, _gval_p in [('Treated', 1), ('Not Treated', 0)]:
                _g_p = _subset_p[_subset_p['treated'] == _gval_p]
                if len(_g_p) < 5:
                    continue
                _kmf_p = _KMFagep()
                _kmf_p.fit(_g_p['time_5yr'], event_observed=_g_p['event_5yr'])
                _sf_p  = _kmf_p.survival_function_
                _ci_p  = _kmf_p.confidence_interval_survival_function_
                _t_p   = _sf_p.index.tolist()
                _col_p = _colors_agep[_grp_p]

                _sf_pct_p = (_sf_p.iloc[:, 0].values * 100).tolist()
                _ci_lo_p  = (_ci_p.iloc[:, 0].values * 100).tolist()
                _all_sf_agep.extend(_sf_pct_p + _ci_lo_p)

                _fig_agep.add_trace(go.Scatter(
                    x=_t_p, y=_sf_pct_p,
                    mode='lines', name=f'{_grp_p} (n={len(_g_p):,})',
                    line=dict(color=_col_p, width=2), line_shape='hv',
                    hovertemplate=f'<b>{_grp_p}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
                ))
                _fig_agep.add_trace(go.Scatter(
                    x=_t_p + _t_p[::-1],
                    y=_ci_lo_p + (_ci_p.iloc[:, 1].values * 100).tolist()[::-1],
                    fill='toself', fillcolor=_col_p, opacity=0.08,
                    line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
                    showlegend=False, hoverinfo='skip'
                ))

            _t_g_p  = _subset_p[_subset_p['treated'] == 1]
            _nt_g_p = _subset_p[_subset_p['treated'] == 0]
            _lr_note_p = 'n<5'
            if len(_t_g_p) >= 5 and len(_nt_g_p) >= 5:
                _lr_p = _lrt_agep(
                    _t_g_p['time_5yr'],  _nt_g_p['time_5yr'],
                    event_observed_A=_t_g_p['event_5yr'],
                    event_observed_B=_nt_g_p['event_5yr'],
                )
                _lr_note_p = f'Log-rank p = {_lr_p.p_value:.4f}'

            _y_min_agep = max(0, min(_all_sf_agep) - 2) if _all_sf_agep else 0
            _fig_agep.update_layout(
                title=f'{_payer_agep} | Age {_age_label_p} (n={len(_subset_p):,})<br>'
                      f'<sup>{_lr_note_p}</sup>',
                xaxis_title='Days from Diagnosis',
                yaxis_title='Survival Probability (%)',
                yaxis=dict(range=[_y_min_agep, 102]),
                legend_title='Treatment',
                hovermode='x unified',
            )
            _figs_agep.append(_fig_agep)

        _all_figs_agep.extend(_figs_agep)

    mo.vstack([_fig for _fig in _all_figs_agep])
    return


@app.cell
def _(mo, pd, surv5yr_df):
    # Summary table: N by payer type × age group × treatment status
    _surv_tbl = surv5yr_df.copy()
    _surv_tbl['age_group'] = pd.cut(
        _surv_tbl['age_at_diagnosis'],
        bins=[0, 45, 60, 70, 200],
        labels=['<45', '45–60', '60–70', '70+'],
        right=False
    )
    _surv_tbl['treatment_label'] = _surv_tbl['treated'].map({1: 'Treated', 0: 'Not Treated'})

    _summary_tbl = (
        _surv_tbl
        .groupby(['payer_type', 'age_group', 'treatment_label'], observed=True)
        .size()
        .reset_index(name='n')
    )

    _lines_tbl = [
        "**N by Payer Type × Age Group × Treatment Status**\n",
        "| Payer Type | Age Group | Treated | Not Treated |",
        "|------------|-----------|---------|-------------|",
    ]
    for (_payer_t, _age_t), _grp_t in _summary_tbl.groupby(['payer_type', 'age_group'], observed=True):
        _n_treated   = int(_grp_t.loc[_grp_t['treatment_label'] == 'Treated',   'n'].sum())
        _n_untreated = int(_grp_t.loc[_grp_t['treatment_label'] == 'Not Treated', 'n'].sum())
        _lines_tbl.append(f"| {_payer_t} | {_age_t} | {_n_treated:,} | {_n_untreated:,} |")

    mo.md('\n'.join(_lines_tbl))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2-Year Survival by Insurance Category and Treatment Status
    """)
    return


@app.cell
def _(go, mo, surv2yr_df):
    from lifelines import KaplanMeierFitter as _KMF2pt
    from lifelines.statistics import multivariate_logrank_test as _mlrt2pt

    _surv2pt = surv2yr_df.copy()
    _surv2pt['treated_label'] = _surv2pt['treated'].map({1: 'Treated', 0: 'Untreated'})
    _surv2pt['payer_treatment'] = _surv2pt['payer_type'] + ' – ' + _surv2pt['treated_label']

    _fig2pt = go.Figure()
    _colors2pt = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
        '#2ca02c', '#98df8a', '#d62728', '#ff9896'
    ]
    _groups2pt = sorted(_surv2pt['payer_treatment'].dropna().unique())
    _all_sf_vals2pt = []

    for _i2pt, _grp2pt in enumerate(_groups2pt):
        _g2pt = _surv2pt[_surv2pt['payer_treatment'] == _grp2pt]
        if len(_g2pt) < 5:
            continue
        _kmf2pt = _KMF2pt()
        _kmf2pt.fit(_g2pt['time_2yr'], event_observed=_g2pt['event_2yr'], label=_grp2pt)
        _sf2pt = _kmf2pt.survival_function_
        _ci2pt = _kmf2pt.confidence_interval_survival_function_
        _t2pt = _sf2pt.index.tolist()
        _col2pt = _colors2pt[_i2pt % len(_colors2pt)]
        _n2pt = len(_g2pt)

        _sf_pct2pt = (_sf2pt.iloc[:, 0].values * 100).tolist()
        _ci_lo_pct2pt = (_ci2pt.iloc[:, 0].values * 100).tolist()
        _all_sf_vals2pt.extend(_sf_pct2pt + _ci_lo_pct2pt)

        _fig2pt.add_trace(go.Scatter(
            x=_t2pt, y=_sf_pct2pt,
            mode='lines', name=f'{_grp2pt} (n={_n2pt:,})',
            line=dict(color=_col2pt, width=2), line_shape='hv',
            hovertemplate=f'<b>{_grp2pt}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
        ))
        _fig2pt.add_trace(go.Scatter(
            x=_t2pt + _t2pt[::-1],
            y=_ci_lo_pct2pt + (_ci2pt.iloc[:, 1].values * 100).tolist()[::-1],
            fill='toself', fillcolor=_col2pt, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip'
        ))

    _y_min2pt = max(0, min(_all_sf_vals2pt) - 2) if _all_sf_vals2pt else 0

    _valid2pt = _surv2pt[_surv2pt['payer_treatment'].notna()]
    _lr2pt = _mlrt2pt(_valid2pt['time_2yr'], _valid2pt['payer_treatment'], _valid2pt['event_2yr'])
    _p_lr2pt = _lr2pt.p_value

    _fig2pt.update_layout(
        title=f'Kaplan-Meier: 2-Year Survival by Payer Type × Treatment Status<br>'
              f'<sup>Log-rank p = {_p_lr2pt:.4f}</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[_y_min2pt, 102]),
        legend_title='Group',
        hovermode='x unified'
    )

    mo.vstack([
        _fig2pt,
        mo.md(f"Overall log-rank test across payer×treatment groups: χ² = {_lr2pt.test_statistic:.2f}, p = {_p_lr2pt:.4f}")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 5-Year Survival by Insurance Category and Treatment Status
    """)
    return


@app.cell
def _(go, mo, surv5yr_df):
    from lifelines import KaplanMeierFitter as _KMF6
    from lifelines.statistics import multivariate_logrank_test as _mlrt6

    _surv6 = surv5yr_df.copy()
    _surv6['treated_label'] = _surv6['treated'].map({1: 'Treated', 0: 'Untreated'})
    _surv6['payer_treatment'] = _surv6['payer_type'] + ' \u2013 ' + _surv6['treated_label']

    _fig = go.Figure()
    _colors6 = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
        '#2ca02c', '#98df8a', '#d62728', '#ff9896'
    ]
    _groups6 = sorted(_surv6['payer_treatment'].dropna().unique())
    _all_sf_vals6 = []

    for _i6, _grp6 in enumerate(_groups6):
        _g6 = _surv6[_surv6['payer_treatment'] == _grp6]
        if len(_g6) < 5:
            continue
        _kmf6 = _KMF6()
        _kmf6.fit(_g6['time_5yr'], event_observed=_g6['event_5yr'], label=_grp6)
        _sf6 = _kmf6.survival_function_
        _ci6 = _kmf6.confidence_interval_survival_function_
        _t6 = _sf6.index.tolist()
        _col6 = _colors6[_i6 % len(_colors6)]
        _n6 = len(_g6)

        _sf_pct6 = (_sf6.iloc[:, 0].values * 100).tolist()
        _ci_lo_pct6 = (_ci6.iloc[:, 0].values * 100).tolist()
        _all_sf_vals6.extend(_sf_pct6 + _ci_lo_pct6)

        _fig.add_trace(go.Scatter(
            x=_t6, y=_sf_pct6,
            mode='lines', name=f'{_grp6} (n={_n6:,})',
            line=dict(color=_col6, width=2), line_shape='hv',
            hovertemplate=f'<b>{_grp6}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
        ))
        _fig.add_trace(go.Scatter(
            x=_t6 + _t6[::-1],
            y=_ci_lo_pct6 + (_ci6.iloc[:, 1].values * 100).tolist()[::-1],
            fill='toself', fillcolor=_col6, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip'
        ))

    _y_min6 = max(0, min(_all_sf_vals6) - 2) if _all_sf_vals6 else 0
    _y_max6 = 102

    _valid6 = _surv6[_surv6['payer_treatment'].notna()]
    _lr6 = _mlrt6(_valid6['time_5yr'], _valid6['payer_treatment'], _valid6['event_5yr'])
    _p_lr6 = _lr6.p_value

    _fig.update_layout(
        title=f'Kaplan-Meier: 5-Year Survival by Payer Type \u00d7 Treatment Status<br>'
              f'<sup>Log-rank p = {_p_lr6:.4f}</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[_y_min6, _y_max6]),
        legend_title='Group',
        hovermode='x unified'
    )

    mo.vstack([
        _fig,
        mo.md(f"Overall log-rank test across payer×treatment groups: χ² = {_lr6.test_statistic:.2f}, p = {_p_lr6:.4f}")
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Cox Proportional Hazards Models

    A **Cox model** estimates the relationship between covariates and the *rate* at which events (deaths) occur over time, without assuming any particular shape for the survival curve.

    The key output is the **Hazard Ratio (HR)**:
    - **HR > 1** → higher risk of death (shorter survival) compared to the reference group
    - **HR < 1** → lower risk of death (longer survival) compared to the reference group
    - **HR = 1** → no difference

    The reference group for payer type is the first alphabetically (e.g. *Commercial*) — all other payer HRs are relative to it.

    **Model 1** tests whether payer type alone predicts survival.
    **Model 2** adds `treated` and `age_at_diagnosis` as covariates to check whether any payer effect
    is explained by differences in treatment rates or patient age across payer types.
    If a payer's HR shrinks or loses significance in Model 2, it suggests the survival difference
    was driven by treatment access or age rather than the insurance type itself.

    Models are run separately for **2-year** and **5-year** survival horizons.
    """)
    return


@app.cell
def _(mo, pd, survival_df):
    from lifelines import CoxPHFitter as _CoxPHFitter2yr

    _base2 = survival_df.copy()
    _base2['event_h'] = ((_base2['event'] == 1) & (_base2['survival_days'] <= 730)).astype(int)
    _base2['time_h']  = _base2['survival_days'].clip(upper=730)

    def _format_cox(label, cph):
        s = cph.summary.copy()
        s.index = s.index.str.replace('_', ' ')
        cols = ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
        s = s[cols].rename(columns={
            'exp(coef)': 'HR', 'exp(coef) lower 95%': 'CI_lo',
            'exp(coef) upper 95%': 'CI_hi', 'p': 'p_value'
        })
        _lines = [
            f"**{label}**\n",
            f"N={cph._n_examples:,} | Events={int(cph.event_observed.sum()):,} | "
            f"Concordance={cph.concordance_index_:.3f} | Log-likelihood={cph.log_likelihood_:.1f}\n",
            "| Covariate | HR | CI_lo | CI_hi | p | Sig |",
            "|-----------|-----|-------|-------|---|-----|",
        ]
        for var, row in s.iterrows():
            sig = "\\*\\*\\*" if row['p_value'] < 0.001 else ("\\*" if row['p_value'] < 0.05 else "")
            _lines.append(f"| {var} | {row['HR']:.3f} | {row['CI_lo']:.3f} | {row['CI_hi']:.3f} | {row['p_value']:.4f} | {sig} |")
        return "\n".join(_lines)

    # Model 1: payer_type only
    _cox_df1 = pd.get_dummies(
        _base2[['time_h', 'event_h', 'payer_type']].dropna(),
        columns=['payer_type'], drop_first=True, dtype=float
    )
    _cox_df1.columns = [c.replace(' ', '_') for c in _cox_df1.columns]
    _payer_cols1 = [c for c in _cox_df1.columns if c.startswith('payer_type_')]
    _cph2yr_1 = _CoxPHFitter2yr()
    _cph2yr_1.fit(_cox_df1, duration_col='time_h', event_col='event_h',
                  formula=' + '.join(_payer_cols1))

    # Model 2: payer_type + treated + age_at_diagnosis
    _cox_df2 = pd.get_dummies(
        _base2[['time_h', 'event_h', 'payer_type', 'treated', 'age_at_diagnosis']].dropna(),
        columns=['payer_type'], drop_first=True, dtype=float
    )
    _cox_df2.columns = [c.replace(' ', '_') for c in _cox_df2.columns]
    _payer_cols2 = [c for c in _cox_df2.columns if c.startswith('payer_type_')]
    _cph2yr_2 = _CoxPHFitter2yr()
    _cph2yr_2.fit(_cox_df2, duration_col='time_h', event_col='event_h',
                  formula=' + '.join(_payer_cols2 + ['treated', 'age_at_diagnosis']))

    cox_2yr_model1_summary = _cph2yr_1.summary.copy()
    cox_2yr_model2_summary = _cph2yr_2.summary.copy()
    mo.output.replace(mo.md(
        "#### 2-Year Horizon\n\n" +
        _format_cox("Cox Model 1 (2yr): payer_type only", _cph2yr_1) + "\n\n---\n\n" +
        _format_cox("Cox Model 2 (2yr): payer_type + treated + age_at_diagnosis", _cph2yr_2)
    ))
    return


@app.cell
def _(mo, pd, survival_df):
    from lifelines import CoxPHFitter as _CoxPHFitter5yr

    _base5 = survival_df.copy()
    _base5['event_h'] = ((_base5['event'] == 1) & (_base5['survival_days'] <= 1825)).astype(int)
    _base5['time_h']  = _base5['survival_days'].clip(upper=1825)

    def _format_cox(label, cph):
        s = cph.summary.copy()
        s.index = s.index.str.replace('_', ' ')
        cols = ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
        s = s[cols].rename(columns={
            'exp(coef)': 'HR', 'exp(coef) lower 95%': 'CI_lo',
            'exp(coef) upper 95%': 'CI_hi', 'p': 'p_value'
        })
        _lines = [
            f"**{label}**\n",
            f"N={cph._n_examples:,} | Events={int(cph.event_observed.sum()):,} | "
            f"Concordance={cph.concordance_index_:.3f} | Log-likelihood={cph.log_likelihood_:.1f}\n",
            "| Covariate | HR | CI_lo | CI_hi | p | Sig |",
            "|-----------|-----|-------|-------|---|-----|",
        ]
        for var, row in s.iterrows():
            sig = "\\*\\*\\*" if row['p_value'] < 0.001 else ("\\*" if row['p_value'] < 0.05 else "")
            _lines.append(f"| {var} | {row['HR']:.3f} | {row['CI_lo']:.3f} | {row['CI_hi']:.3f} | {row['p_value']:.4f} | {sig} |")
        return "\n".join(_lines)

    # Model 1: payer_type only
    _cox_df1 = pd.get_dummies(
        _base5[['time_h', 'event_h', 'payer_type']].dropna(),
        columns=['payer_type'], drop_first=True, dtype=float
    )
    _cox_df1.columns = [c.replace(' ', '_') for c in _cox_df1.columns]
    _payer_cols1 = [c for c in _cox_df1.columns if c.startswith('payer_type_')]
    _cph5yr_1 = _CoxPHFitter5yr()
    _cph5yr_1.fit(_cox_df1, duration_col='time_h', event_col='event_h',
                  formula=' + '.join(_payer_cols1))

    # Model 2: payer_type + treated + age_at_diagnosis
    _cox_df2 = pd.get_dummies(
        _base5[['time_h', 'event_h', 'payer_type', 'treated', 'age_at_diagnosis']].dropna(),
        columns=['payer_type'], drop_first=True, dtype=float
    )
    _cox_df2.columns = [c.replace(' ', '_') for c in _cox_df2.columns]
    _payer_cols2 = [c for c in _cox_df2.columns if c.startswith('payer_type_')]
    _cph5yr_2 = _CoxPHFitter5yr()
    _cph5yr_2.fit(_cox_df2, duration_col='time_h', event_col='event_h',
                  formula=' + '.join(_payer_cols2 + ['treated', 'age_at_diagnosis']))

    cox_model1_summary = _cph5yr_1.summary.copy()
    cox_model2_summary = _cph5yr_2.summary.copy()
    mo.output.replace(mo.md(
        "#### 5-Year Horizon\n\n" +
        _format_cox("Cox Model 1 (5yr): payer_type only", _cph5yr_1) + "\n\n---\n\n" +
        _format_cox("Cox Model 2 (5yr): payer_type + treated + age_at_diagnosis", _cph5yr_2)
    ))
    return cox_model1_summary, cox_model2_summary


@app.cell
def _(cox_model2_summary, go):
    # ── Forest plot of Model 2 HRs ────────────────────────────────────────────
    _summary = cox_model2_summary.reset_index().rename(columns={
        'index': 'covariate', 'exp(coef)': 'HR',
        'exp(coef) lower 95%': 'HR_lo', 'exp(coef) upper 95%': 'HR_hi', 'p': 'p_val'
    })
    _summary['covariate'] = _summary['covariate'].str.replace('_', ' ')
    _summary['significant'] = _summary['p_val'] < 0.05
    _summary = _summary.sort_values('HR')

    _colors_fp = ['steelblue' if s else 'lightgray' for s in _summary['significant']]

    _fig = go.Figure()
    _fig.add_shape(type='line', x0=1, x1=1, y0=-0.5, y1=len(_summary) - 0.5,
                   line=dict(color='red', dash='dash'))
    _fig.add_trace(go.Scatter(
        x=_summary['HR'], y=_summary['covariate'],
        mode='markers', marker=dict(size=10, color=_colors_fp),
        error_x=dict(
            type='data', symmetric=False,
            array=((_summary['HR_hi'] - _summary['HR']).clip(upper=10)).values,
            arrayminus=((_summary['HR'] - _summary['HR_lo']).clip(upper=10)).values,
            color='gray'
        ),
        hovertemplate=(
            '<b>%{y}</b><br>HR: %{x:.3f}<br>'
            'CI: %{customdata[0]:.3f}–%{customdata[1]:.3f}<br>'
            'p: %{customdata[2]:.4f}<extra></extra>'
        ),
        customdata=_summary[['HR_lo', 'HR_hi', 'p_val']].values
    ))
    _fig.update_layout(
        title='Cox Model 2: Hazard Ratios with 95% CI<br>'
              '<sup>Blue = significant (p < 0.05); reference = first payer type alphabetically</sup>',
        xaxis_title='Hazard Ratio (log scale)',
        xaxis_type='log',
        showlegend=False,
        height=max(350, len(_summary) * 40)
    )
    _fig
    return


@app.cell
def _(cox_model1_summary, cox_model2_summary, mo):
    # ── Interpretation of Cox Models ──────────────────────────────────────────
    _s1 = cox_model1_summary.copy()
    _s1.index = _s1.index.str.replace('_', ' ')
    _s2 = cox_model2_summary.copy()
    _s2.index = _s2.index.str.replace('_', ' ')

    # Find Medicare Advantage HR in both models — look up each index independently
    # so a different reference category in model 2 (due to dropna on extra columns) doesn't KeyError
    _ma_vars1 = [idx for idx in _s1.index if 'Medicare' in idx and 'Advantage' in idx]
    _ma_vars2 = [idx for idx in _s2.index if 'Medicare' in idx and 'Advantage' in idx]
    _hr1 = _s1.loc[_ma_vars1[0], 'exp(coef)'] if _ma_vars1 else float('nan')
    _p1  = _s1.loc[_ma_vars1[0], 'p']         if _ma_vars1 else float('nan')
    _hr2 = _s2.loc[_ma_vars2[0], 'exp(coef)'] if _ma_vars2 else float('nan')
    _p2  = _s2.loc[_ma_vars2[0], 'p']         if _ma_vars2 else float('nan')

    # Age HR from Model 2
    _age_key = [idx for idx in _s2.index if 'age' in idx.lower()]
    if _age_key:
        _hr_age = _s2.loc[_age_key[0], 'exp(coef)']
        _p_age = _s2.loc[_age_key[0], 'p']
    else:
        _hr_age, _p_age = float('nan'), float('nan')

    # Treated HR from Model 2
    _treat_key = [idx for idx in _s2.index if idx.strip().lower() == 'treated']
    if _treat_key:
        _hr_treat = _s2.loc[_treat_key[0], 'exp(coef)']
        _p_treat = _s2.loc[_treat_key[0], 'p']
    else:
        _hr_treat, _p_treat = float('nan'), float('nan')

    _ten_yr_risk = ((_hr_age ** 10) - 1) * 100

    mo.md(
        f"### Interpretation: Confounding by Age\n\n"
        f"**Model 1** (payer type only) shows Medicare Advantage with HR = **{_hr1:.2f}** "
        f"(p = {_p1:.4f}), suggesting substantially higher mortality risk vs the reference payer.\n\n"
        f"**Model 2** adds `treated` and `age_at_diagnosis` as covariates. Medicare Advantage HR drops to "
        f"**{_hr2:.2f}** (p = {_p2:.4f}) — the apparent payer effect largely disappears.\n\n"
        f"**Why?** This is a classic example of **confounding by age**:\n\n"
        f"- Medicare enrollees are predominantly aged 65+, so \"Medicare Advantage\" is a proxy for older age.\n"
        f"- Age independently predicts mortality: HR = **{_hr_age:.3f}** per year (p = {_p_age:.4f}). "
        f"Each additional year adds ~{(_hr_age - 1) * 100:.1f}% hazard; a 10-year age gap corresponds to "
        f"~{_ten_yr_risk:.0f}% higher risk.\n"
        f"- Once age is in the model, the payer type coefficient captures only the *residual* association — "
        f"which is near null.\n\n"
        f"**Treatment effect:** `treated` HR = **{_hr_treat:.3f}** (p = {_p_treat:.4f}). "
        + ("Treatment is significantly associated with lower mortality risk after adjusting for age and payer type."
           if _p_treat < 0.05 else
           "Treatment does not reach significance after adjusting for age and payer type.")
        + "\n\n"
        f"**Conclusion:** The survival difference across payer types observed in Model 1 was driven by "
        f"age composition, not by the insurance type itself. After adjusting for age (and treatment), "
        f"payer type has minimal independent effect on survival."
    )
    return


@app.cell
def _(
    chi2_payer,
    chi2_plan,
    enrollment_final,
    mo,
    model_treated,
    p_payer,
    p_plan,
    time_to_treatment,
    treated_odds_ratios,
):
    _n_total = len(enrollment_final)
    _n_treated = int(enrollment_final['treated'].sum())
    _n_untreated = _n_total - _n_treated
    _pct_treated = _n_treated / _n_total * 100
    _median_months = time_to_treatment['months_to_treatment'].median()
    _top_payer = enrollment_final.groupby('payer_type').size().idxmax()
    _top_payer_pct = enrollment_final['payer_type'].value_counts(normalize=True).iloc[0] * 100

    _tg = enrollment_final['treatment_group'].value_counts()
    def _fg(name):
        c = _tg.get(name, 0)
        return str(c) + " (" + f"{c/_n_total*100:.1f}" + "%)"

    def _sig(p):
        if p < 0.001:
            return "p < 0.001"
        elif p < 0.05:
            return "p = " + f"{p:.4f}"
        else:
            return "p = " + f"{p:.3f}" + ", not significant"

    def _or_line(row):
        v = row['Variable']
        o = row['Odds Ratio']
        lo = row['OR 95% CI Lower']
        hi = row['OR 95% CI Upper']
        return "- **" + v + "** (OR " + f"{o:.2f}" + ", 95% CI " + f"{lo:.2f}" + "\u2013" + f"{hi:.2f}" + ")"

    _sig_treated = treated_odds_ratios[treated_odds_ratios['Significant']].sort_values('Odds Ratio', ascending=False)
    _treated_hi = "\n    ".join(_or_line(r) for _, r in _sig_treated.head(3).iterrows()) if len(_sig_treated) > 0 else "- No significant predictors"
    _treated_lo = "\n    ".join(_or_line(r) for _, r in _sig_treated.tail(3).iterrows()) if len(_sig_treated) > 3 else ""
    _treated_lo_section = "\n\n    Lowest odds:\n    " + _treated_lo if _treated_lo else ""

    _assoc_note = (
        "Insurance type is significantly associated with treatment status, "
        "suggesting coverage and formulary differences may influence prescribing."
        if p_payer < 0.05
        else "No significant association was found between insurance type and treatment status."
    )

    # Treatment rates by payer type
    _payer_rates = (
        enrollment_final
        .groupby('payer_type')
        .agg(_n=('patient_id', 'count'), _rate=('treated', 'mean'))
        .sort_values('_rate', ascending=False)
        .reset_index()
    )
    _payer_rows = []
    for _, r in _payer_rates.iterrows():
        _payer_rows.append(
            "| " + r['payer_type'] + " | " + f"{r['_n']:,}" + " | " + f"{r['_rate']*100:.2f}" + "% |"
        )

    # Treatment rates by plan design
    _plan_rates = (
        enrollment_final
        .groupby('plan_design')
        .agg(_n=('patient_id', 'count'), _rate=('treated', 'mean'))
        .sort_values('_rate', ascending=False)
        .reset_index()
    )
    _plan_rows = []
    for _, r in _plan_rates.iterrows():
        _plan_rows.append(
            "| " + r['plan_design'] + " | " + f"{r['_n']:,}" + " | " + f"{r['_rate']*100:.2f}" + "% |"
        )

    _lines = [
        "## 5. Summary and Key Findings",
        "",
        "---",
        "",
        "### Cohort Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        "| Patients analyzed | " + f"{_n_total:,}" + " |",
        "| Treated (PARP/bevacizumab) | " + f"{_n_treated:,}" + " (" + f"{_pct_treated:.1f}" + "%) |",
        "| Not treated | " + f"{_n_untreated:,}" + " (" + f"{_n_untreated/_n_total*100:.1f}" + "%) |",
        "| Dominant payer type | " + _top_payer + " (" + f"{_top_payer_pct:.1f}" + "%) |",
        "| Median time to treatment | " + f"{_median_months:.1f}" + " months |",
        "",
        "### Treatment Breakdown",
        "",
        "| Group | N (%) |",
        "|-------|-------|",
        "| PARP only | " + _fg('PARP only') + " |",
        "| Bevacizumab only | " + _fg('Bevacizumab only') + " |",
        "| Both | " + _fg('Both') + " |",
        "| Neither | " + _fg('Neither') + " |",
        "",
        "### Treatment Rates by Payer Type",
        "",
        "| Payer Type | N | Treatment Rate |",
        "|------------|---|---------------|",
    ] + _payer_rows + [
        "",
        "### Treatment Rates by Plan Design",
        "",
        "| Plan Design | N | Treatment Rate |",
        "|-------------|---|---------------|",
    ] + _plan_rows + [
        "",
        "### Statistical Associations (Chi-Square)",
        "",
        "| Test | Chi-square | Result |",
        "|------|-----------|--------|",
        "| Treated x Payer Type | " + f"{chi2_payer:.2f}" + " | " + _sig(p_payer) + " |",
        "| Treated x Plan Design | " + f"{chi2_plan:.2f}" + " | " + _sig(p_plan) + " |",
        "",
        "### Logistic Regression: Key Predictors",
        "",
        "**Any Treatment** (Pseudo R\u00b2 = " + f"{model_treated.prsquared:.4f}" + ")",
        "",
        "Highest odds:",
        _treated_hi,
        _treated_lo_section,
        "",
        "### Interpretation",
        "",
        "- The majority of ovarian cancer patients (" + _fg('Neither') + ") did not receive PARP inhibitors or bevacizumab during observed enrollment periods.",
        "- " + _assoc_note,
        "- **Highest treatment rate by payer:** " + _payer_rates.iloc[0]['payer_type'] + " (" + f"{_payer_rates.iloc[0]['_rate']*100:.2f}" + "%). **Lowest:** " + _payer_rates.iloc[-1]['payer_type'] + " (" + f"{_payer_rates.iloc[-1]['_rate']*100:.2f}" + "%).",
        "- **Highest treatment rate by plan:** " + _plan_rates.iloc[0]['plan_design'] + " (" + f"{_plan_rates.iloc[0]['_rate']*100:.2f}" + "%). **Lowest:** " + _plan_rates.iloc[-1]['plan_design'] + " (" + f"{_plan_rates.iloc[-1]['_rate']*100:.2f}" + "%).",
        "- The low Pseudo R\u00b2 (" + f"{model_treated.prsquared:.4f}" + ") indicates that insurance characteristics alone explain only a small portion of treatment variation \u2014 clinical factors (stage, biomarkers, physician preference) likely dominate.",
    ]

    mo.md("\n".join(_lines))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Tumor Sub-Analysis

    Patients who appear in `tumor.parquet` linked to their C56 diagnosis. Examines whether having tumor data corresponds to demographic/treatment differences, and tests whether targeted therapy use varies by stage and histology subtype.
    """)
    return


@app.cell
def _(enrollment_final, mo, tumor_df):
    # Filter to ovarian cancer tumors only (C56 in ICD-10; 56.x in ICD-O-3 topography)
    _cohort_ids = enrollment_final['patient_id']
    _all_cohort_tumors = tumor_df[tumor_df['patient_id'].isin(_cohort_ids)]
    _ovarian_mask = (
        _all_cohort_tumors['tumor_site_code'].str.startswith('C56', na=False) |
        _all_cohort_tumors['tumor_site_code'].str.match(r'^56[.\d]*$', na=False)
    )
    _ovarian_tumors = _all_cohort_tumors[_ovarian_mask]

    _site_counts = _all_cohort_tumors['tumor_site_code'].value_counts().head(10)
    _site_lines = ["| Site Code | Count |", "|-----------|-------|"]
    for _code, _cnt in _site_counts.items():
        _site_lines.append(f"| {_code} | {_cnt:,} |")

    # Keep first ovarian tumor record per patient
    _tumor_first = (
        _ovarian_tumors
        .sort_values('diagnosis_date')
        .groupby('patient_id')
        .first()
        .reset_index()
    )

    tumor_cohort = enrollment_final.merge(_tumor_first, on='patient_id', how='inner')
    tumor_cohort_missing = enrollment_final[
        ~enrollment_final['patient_id'].isin(tumor_cohort['patient_id'])
    ]

    mo.md(
        f"**Tumor records for cohort patients (all sites):** {len(_all_cohort_tumors):,}\n\n"
        f"**Ovarian tumor records (C56 / 56.x):** {len(_ovarian_tumors):,}\n\n"
        f"**Top tumor site codes:**\n\n" + "\n".join(_site_lines) + "\n\n"
        f"---\n\n"
        f"**Tumor Cohort (ovarian tumors only)**\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Enrollment final patients | {len(enrollment_final):,} |\n"
        f"| Patients with tumor data | {len(tumor_cohort):,} ({len(tumor_cohort)/len(enrollment_final)*100:.1f}%) |\n"
        f"| Patients without tumor data | {len(tumor_cohort_missing):,} ({len(tumor_cohort_missing)/len(enrollment_final)*100:.1f}%) |"
    )
    return (tumor_cohort,)


@app.cell
def _(chi2_contingency, enrollment_final, mo, pd, tumor_cohort):
    # Compare with vs without tumor data on treatment rates and payer type
    _df = enrollment_final.copy()
    _df['tumor_data'] = _df['patient_id'].isin(tumor_cohort['patient_id']).map(
        {True: 'With tumor data', False: 'Without tumor data'}
    )

    _treat_summary = (
        _df.groupby('tumor_data')
        .agg(n=('patient_id', 'count'), n_treated=('treated', 'sum'), treat_rate=('treated', 'mean'))
        .reset_index()
    )
    _treat_summary['treat_pct'] = (_treat_summary['treat_rate'] * 100).round(1)

    _lines = [
        "**Descriptives: With vs Without Tumor Data**\n",
        "| Group | N | N Treated | Treat % |",
        "|-------|---|-----------|---------|",
    ]
    for _, _row in _treat_summary.iterrows():
        _lines.append(f"| {_row['tumor_data']} | {int(_row['n']):,} | {int(_row['n_treated']):,} | {_row['treat_pct']:.1f}% |")

    _payer_cross = pd.crosstab(_df['tumor_data'], _df['payer_type'], normalize='index') * 100
    _payer_cross = _payer_cross.round(1)
    _pcols = _payer_cross.columns.tolist()
    _lines.append("\n**Payer type distribution (%)**\n")
    _lines.append("| | " + " | ".join(str(c) for c in _pcols) + " |")
    _lines.append("|---|" + "|".join(["---"] * len(_pcols)) + "|")
    for _idx, _row in _payer_cross.iterrows():
        _lines.append(f"| {_idx} | " + " | ".join(f"{v:.1f}%" for v in _row.values) + " |")

    _ct_treat = pd.crosstab(_df['tumor_data'], _df['treated'])
    _chi2_t, _p_t, _, _ = chi2_contingency(_ct_treat)

    _ct_payer = pd.crosstab(_df['tumor_data'], _df['payer_type'])
    _chi2_p, _p_p, _, _ = chi2_contingency(_ct_payer)

    _lines.append(f"\nChi-square (tumor data × treated): χ² = {_chi2_t:.2f}, p = {_p_t:.4f}")
    _lines.append(f"\nChi-square (tumor data × payer_type): χ² = {_chi2_p:.2f}, p = {_p_p:.4f}")

    mo.md("\n".join(_lines))
    return


@app.cell
def _(enrollment_final, mo, px, tumor_cohort):
    # Percent treated by year, split by tumor data cohort
    _df = enrollment_final.copy()
    _df['tumor_data'] = _df['patient_id'].isin(tumor_cohort['patient_id']).map(
        {True: 'With tumor data', False: 'Without tumor data'}
    )
    _df['year'] = _df['earliest_diagnosis_date'].dt.year

    _yearly = (
        _df.groupby(['year', 'tumor_data'])
        .agg(n=('patient_id', 'count'), n_treated=('treated', 'sum'))
        .reset_index()
    )
    _yearly['pct_treated'] = _yearly['n_treated'] / _yearly['n'] * 100

    _fig = px.bar(
        _yearly,
        x='year',
        y='pct_treated',
        color='tumor_data',
        barmode='group',
        labels={'year': 'Diagnosis Year', 'pct_treated': 'Treated (%)', 'tumor_data': 'Cohort'},
        title='Percent Treated by Year: With vs Without Tumor Data',
        text=_yearly['pct_treated'].round(1).astype(str) + '%',
    )
    _fig.update_traces(textposition='outside', textfont_size=10)
    _fig.update_layout(yaxis_range=[0, _yearly['pct_treated'].max() * 1.15], xaxis=dict(dtick=1))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(enrollment_final, mo, px, tumor_cohort):
    # Number diagnosed per year, split by tumor data cohort
    _df2 = enrollment_final.copy()
    _df2['tumor_data'] = _df2['patient_id'].isin(tumor_cohort['patient_id']).map(
        {True: 'With tumor data', False: 'Without tumor data'}
    )
    _df2['year'] = _df2['earliest_diagnosis_date'].dt.year

    _yearly2 = (
        _df2.groupby(['year', 'tumor_data'])
        .agg(n=('patient_id', 'count'))
        .reset_index()
    )

    _fig2 = px.bar(
        _yearly2,
        x='year',
        y='n',
        color='tumor_data',
        barmode='group',
        labels={'year': 'Diagnosis Year', 'n': 'Patients Diagnosed', 'tumor_data': 'Cohort'},
        title='Patients Diagnosed per Year: With vs Without Tumor Data',
        text='n',
    )
    _fig2.update_traces(textposition='outside', textfont_size=10)
    _fig2.update_layout(yaxis_range=[0, _yearly2['n'].max() * 1.15], xaxis=dict(dtick=1))
    mo.ui.plotly(_fig2)
    return


@app.cell
def _(enrollment_final, mo, px, tumor_cohort):
    # Year-over-year percent change in patients diagnosed, by cohort
    _df3 = enrollment_final.copy()
    _df3['tumor_data'] = _df3['patient_id'].isin(tumor_cohort['patient_id']).map(
        {True: 'With tumor data', False: 'Without tumor data'}
    )
    _df3['year'] = _df3['earliest_diagnosis_date'].dt.year

    _yearly3 = (
        _df3.groupby(['year', 'tumor_data'])
        .agg(n=('patient_id', 'count'))
        .reset_index()
    )
    _yearly3 = _yearly3.sort_values(['tumor_data', 'year'])
    _yearly3['yoy_pct'] = _yearly3.groupby('tumor_data')['n'].pct_change() * 100

    _fig3 = px.bar(
        _yearly3.dropna(subset=['yoy_pct']),
        x='year',
        y='yoy_pct',
        color='tumor_data',
        barmode='group',
        labels={'year': 'Diagnosis Year', 'yoy_pct': 'YoY Change (%)', 'tumor_data': 'Cohort'},
        title='Year-over-Year Change in Patients Diagnosed: With vs Without Tumor Data',
        text=_yearly3.dropna(subset=['yoy_pct'])['yoy_pct'].round(1).astype(str) + '%',
    )
    _fig3.update_traces(textposition='outside', textfont_size=10)
    _fig3.update_layout(xaxis=dict(dtick=1))
    mo.ui.plotly(_fig3)
    return


@app.cell
def _(go, mo, survival_df, tumor_cohort):
    from lifelines import KaplanMeierFitter as _KMF
    from lifelines.statistics import multivariate_logrank_test as _mlrt

    # Tag each patient as with / without tumor data
    _surv = survival_df.copy()
    _surv['cohort'] = _surv['patient_id'].isin(tumor_cohort['patient_id']).map(
        {True: 'With tumor data', False: 'Without tumor data'}
    )

    _colors = {'With tumor data': '#636EFA', 'Without tumor data': '#EF553B'}
    _results = {}

    _fig_surv = go.Figure()

    for _days, _tag, _label in [(730, '2yr', '2-Year'), (1825, '5yr', '5-Year')]:
        _s = _surv.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= _days)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=_days)

        _lr = _mlrt(_s['time_w'], _s['event_w'], _s['cohort'])
        _point_ests = {}

        for _cohort in ['With tumor data', 'Without tumor data']:
            _g = _s[_s['cohort'] == _cohort]
            _kmf = _KMF()
            _kmf.fit(_g['time_w'], event_observed=_g['event_w'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors[_cohort]
            _dash = 'solid' if _cohort == 'With tumor data' else 'dash'

            _fig_surv.add_trace(go.Scatter(
                x=_t, y=_sf.iloc[:, 0].tolist(),
                mode='lines', name=_cohort,
                line=dict(color=_col, width=2, dash=_dash),
                legendgroup=_cohort, showlegend=(_tag == '2yr'),
            ))
            _fig_surv.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=_ci.iloc[:, 0].tolist() + _ci.iloc[:, 1].tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.12,
                line=dict(width=0), hoverinfo='skip',
                legendgroup=_cohort, showlegend=False,
            ))

            _idx = max(_sf.index.searchsorted(_days, side='right') - 1, 0)
            _est = float(_sf.iloc[_idx, 0])
            _lo  = float(_ci.iloc[_idx, 0])
            _hi  = float(_ci.iloc[_idx, 1])
            _n_g = len(_g)
            _ev_g = int(_g['event_w'].sum())
            _cens_g = _n_g - _ev_g
            _point_ests[_cohort] = (_est, _lo, _hi, _n_g, _ev_g, _cens_g)

        _results[_tag] = (_label, _lr, _point_ests)

    for _x, _lbl in [(730, '2yr'), (1825, '5yr')]:
        _fig_surv.add_vline(x=_x, line_dash='dot', line_color='grey',
                            annotation_text=_lbl, annotation_position='top right')

    _fig_surv.update_layout(
        title='Kaplan-Meier Survival: With vs Without Tumor Data',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability',
        yaxis=dict(range=[0, 1.05]),
        legend_title='Cohort',
    )

    _rows = ['### Survival Summary\n',
             '| Horizon | Cohort | N | Events | Censored | KM Estimate | 95% CI |',
             '|---------|--------|---|--------|----------|-------------|--------|']
    for _tag, (_label, _lr, _ests) in _results.items():
        for _cohort, (_est, _lo, _hi, _n, _ev, _cens) in _ests.items():
            _rows.append(
                f"| {_label} | {_cohort} | {_n:,} | {_ev:,} | {_cens:,} | "
                f"{_est*100:.1f}% | [{_lo*100:.1f}%, {_hi*100:.1f}%] |"
            )
        _sig = '**significant difference**' if _lr.p_value < 0.05 else 'no significant difference'
        _rows.append(f"\n**{_label} log-rank:** χ² = {_lr.test_statistic:.2f}, p = {_lr.p_value:.4f} — {_sig}\n")

    mo.vstack([mo.ui.plotly(_fig_surv), mo.md('\n'.join(_rows))])
    return


@app.cell
def _(tumor_cohort):
    tumor_cohort
    return


@app.cell
def _(mo, tumor_cohort):
    # Explore stage codes and map to readable labels
    _stage_counts = tumor_cohort['stage_code'].value_counts()
    _sc_lines = ["**Stage Code Value Counts**\n",
                 "| Stage Code | Count |", "|------------|-------|"]
    for _code, _cnt in _stage_counts.items():
        _sc_lines.append(f"| {_code} | {_cnt:,} |")

    _stage_map = {
        'S1': 'Stage I', 'S1A': 'Stage I', 'S1B': 'Stage I', 'S1C': 'Stage I',
        'S2': 'Stage II', 'S2A': 'Stage II', 'S2B': 'Stage II', 'S2C': 'Stage II',
        'S3': 'Stage III', 'S3A': 'Stage III', 'S3B': 'Stage III', 'S3C': 'Stage III',
        'S4': 'Stage IV', 'S4A': 'Stage IV', 'S4B': 'Stage IV',
    }
    tumor_cohort_staged = tumor_cohort[tumor_cohort['stage_code'].isin(_stage_map)].copy()
    tumor_cohort_staged['stage_label'] = tumor_cohort_staged['stage_code'].map(_stage_map)

    _stage_order = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    _sl_counts = tumor_cohort_staged['stage_label'].value_counts().reindex(_stage_order)
    _sl_lines = [f"\nPatients with mappable stage: {len(tumor_cohort_staged):,} / {len(tumor_cohort):,}\n",
                 "**Stage Label Counts**\n",
                 "| Stage | Count |", "|-------|-------|"]
    for _label, _cnt in _sl_counts.items():
        _sl_lines.append(f"| {_label} | {_cnt:,} |")

    mo.md("\n".join(_sc_lines + _sl_lines))
    return (tumor_cohort_staged,)


@app.cell
def _(chi2_contingency, mo, pd, px, tumor_cohort_staged):
    # Treatment rate by payer_type × stage
    _cross5 = (
        tumor_cohort_staged
        .groupby(['stage_label', 'payer_type'])
        .agg(n=('patient_id', 'count'), n_treated=('treated', 'sum'))
        .reset_index()
    )
    _cross5['treat_pct'] = (_cross5['n_treated'] / _cross5['n'] * 100).round(1)

    _fig5 = px.bar(
        _cross5,
        x='stage_label',
        y='treat_pct',
        color='payer_type',
        barmode='group',
        title='Treatment Rate by Stage and Insurance Type',
        labels={
            'stage_label': 'Tumor Stage',
            'treat_pct': 'Treatment Rate (%)',
            'payer_type': 'Payer Type'
        },
        custom_data=['n', 'n_treated']
    )
    _fig5.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Stage: %{x}<br>'
                      'Treatment Rate: %{y:.1f}%<br>N: %{customdata[0]:,} | Treated: %{customdata[1]:,}<extra></extra>'
    )

    _ct5 = pd.crosstab(tumor_cohort_staged['stage_label'], tumor_cohort_staged['treated'])
    _chi2_5, _p_5, _dof_5, _ = chi2_contingency(_ct5)

    mo.vstack([
        _fig5,
        mo.md(f"Chi-square (treatment × stage): χ² = {_chi2_5:.2f}, df = {_dof_5}, p = {_p_5:.4f}")
    ])
    return


@app.cell
def _(mo, tumor_cohort):
    _n_missing = tumor_cohort['tumor_size'].isna().sum()
    _n_total = len(tumor_cohort)
    _v_counts = tumor_cohort['tumor_size'].value_counts()
    if len(_v_counts) > 0:
        _size_note = (
            f"The `tumor_size` field is missing for {_n_missing:,} of {_n_total:,} patients.  \n"
            f"The remaining {_v_counts.iloc[0]:,} all have a value of **{_v_counts.index[0]}**."
        )
    else:
        _size_note = f"The `tumor_size` field is null for all {_n_total:,} patients."
    mo.md(f"**Tumor size not available.** {_size_note}")
    return


@app.cell
def _(mo, pd, px, tumor_cohort):
    # Morphology (histology) exploration and treatment rate by histology group
    _morph_counts = tumor_cohort['morphology_code'].value_counts().head(20)
    _mc_lines = ["**Morphology Code Top 20**\n",
                 "| Morphology Code | Count |", "|----------------|-------|"]
    for _code, _cnt in _morph_counts.items():
        _mc_lines.append(f"| {_code} | {_cnt:,} |")

    # Group into 4 broad categories
    _non_serous = {'8310', '8380', '8480'}  # clear cell, endometrioid, mucinous

    def _map_morph(code):
        if pd.isna(code):
            return 'Other'
        s = str(code).strip().split('/')[0].split('.')[0]
        if s == '8441':
            return 'High-grade serous'
        if s == '8460':
            return 'Low-grade serous'
        if s in _non_serous:
            return 'Non-serous epithelial'
        return 'Other'

    _df7 = tumor_cohort.copy()
    _df7['histology'] = _df7['morphology_code'].apply(_map_morph)

    _hist_order = ['High-grade serous', 'Low-grade serous', 'Non-serous epithelial', 'Other']

    _cross7 = (
        _df7.groupby('histology')
        .agg(n=('patient_id', 'count'), n_treated=('treated', 'sum'))
        .reset_index()
    )
    _cross7['treat_pct'] = (_cross7['n_treated'] / _cross7['n'] * 100).round(1)
    _cross7['histology'] = pd.Categorical(_cross7['histology'], categories=_hist_order, ordered=True)
    _cross7 = _cross7.sort_values('histology')

    _fig7 = px.bar(
        _cross7,
        x='histology',
        y='treat_pct',
        title='Treatment Rate by Histology Group',
        labels={'histology': 'Histology Group', 'treat_pct': 'Treatment Rate (%)'},
        custom_data=['n', 'n_treated'],
        text='treat_pct',
    )
    _fig7.update_traces(
        texttemplate='%{text:.1f}%', textposition='outside',
        hovertemplate='<b>%{x}</b><br>Treatment Rate: %{y:.1f}%<br>'
                      'N: %{customdata[0]:,} | Treated: %{customdata[1]:,}<extra></extra>'
    )
    _fig7.update_layout(yaxis_range=[0, _cross7['treat_pct'].max() * 1.15])
    mo.vstack([mo.md("\n".join(_mc_lines)), _fig7])
    return


@app.cell
def _(chi2_contingency, mo, pd, px, tumor_cohort):
    # Treatment rate by payer type, faceted by histology group
    _non_serous2 = {'8310', '8380', '8480'}

    def _map_morph2(code):
        if pd.isna(code):
            return 'Other'
        s = str(code).strip().split('/')[0].split('.')[0]
        if s == '8441':
            return 'High-grade serous'
        if s == '8460':
            return 'Low-grade serous'
        if s in _non_serous2:
            return 'Non-serous epithelial'
        return 'Other'

    _df8 = tumor_cohort.copy()
    _df8['histology'] = _df8['morphology_code'].apply(_map_morph2)
    _hist_order2 = ['High-grade serous', 'Low-grade serous', 'Non-serous epithelial', 'Other']
    _df8['histology'] = pd.Categorical(_df8['histology'], categories=_hist_order2, ordered=True)

    _cross8 = (
        _df8.groupby(['histology', 'payer_type'])
        .agg(n=('patient_id', 'count'), n_treated=('treated', 'sum'))
        .reset_index()
    )
    _cross8['treat_pct'] = (_cross8['n_treated'] / _cross8['n'] * 100).round(1)

    _fig8 = px.bar(
        _cross8,
        x='payer_type',
        y='treat_pct',
        facet_col='histology',
        facet_col_wrap=2,
        title='Treatment Rate by Payer Type, by Histology Group',
        labels={'payer_type': 'Payer Type', 'treat_pct': 'Treatment Rate (%)', 'histology': 'Histology'},
        custom_data=['n', 'n_treated'],
        text='treat_pct',
    )
    _fig8.update_traces(
        texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False,
        hovertemplate='<b>%{x}</b><br>Treatment Rate: %{y:.1f}%<br>'
                      'N: %{customdata[0]:,} | Treated: %{customdata[1]:,}<extra></extra>'
    )
    _fig8.update_xaxes(tickangle=35)
    _fig8.update_yaxes(matches=None, range=[0, _cross8['treat_pct'].max() * 1.25])
    _fig8.update_layout(showlegend=False, height=650)

    # Chi-square test within each histology group
    _chi_rows = ['**Chi-square: treatment × payer type within each histology group**\n',
                 '| Histology | N | χ² | p | Significant |',
                 '|-----------|---|----|---|-------------|']
    for _hist in _hist_order2:
        _g8 = _df8[_df8['histology'] == _hist]
        _ct8 = pd.crosstab(_g8['payer_type'], _g8['treated'])
        if _ct8.shape[0] < 2 or _ct8.shape[1] < 2:
            _chi_rows.append(f"| {_hist} | {len(_g8):,} | — | — | insufficient data |")
            continue
        _chi2_8, _p_8, _, _ = chi2_contingency(_ct8)
        _sig8 = '**Yes**' if _p_8 < 0.05 else 'No'
        _chi_rows.append(f"| {_hist} | {len(_g8):,} | {_chi2_8:.2f} | {_p_8:.4f} | {_sig8} |")

    mo.vstack([_fig8, mo.md('\n'.join(_chi_rows))])
    return


@app.cell
def _(go, mo, pd, survival_df, tumor_cohort):
    from lifelines import KaplanMeierFitter as _KMFh
    from lifelines.statistics import multivariate_logrank_test as _mlrth

    _non_serous3 = {'8310', '8380', '8480'}

    def _map_morph3(code):
        if pd.isna(code):
            return 'Other'
        s = str(code).strip().split('/')[0].split('.')[0]
        if s == '8441': return 'High-grade serous'
        if s == '8460': return 'Low-grade serous'
        if s in _non_serous3: return 'Non-serous epithelial'
        return 'Other'

    _hist_order3 = ['High-grade serous', 'Low-grade serous', 'Non-serous epithelial', 'Other']
    _colors_h = {
        'High-grade serous':     '#636EFA',
        'Low-grade serous':      '#EF553B',
        'Non-serous epithelial': '#00CC96',
        'Other':                 '#AB63FA',
    }

    # Attach histology to survival_df via tumor_cohort
    _tc = tumor_cohort[['patient_id', 'morphology_code']].copy()
    _tc['histology'] = _tc['morphology_code'].apply(_map_morph3)
    _sdf = survival_df.merge(_tc[['patient_id', 'histology']], on='patient_id', how='inner')

    _results_h = {}
    _fig_h = go.Figure()

    for _days, _tag, _label in [(730, '2yr', '2-Year'), (1825, '5yr', '5-Year')]:
        _s = _sdf.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= _days)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=_days)

        _lr_h = _mlrth(_s['time_w'], _s['event_w'], _s['histology'])
        _point_ests_h = {}

        for _hist in _hist_order3:
            _g = _s[_s['histology'] == _hist]
            if len(_g) < 5:
                continue
            _kmf = _KMFh()
            _kmf.fit(_g['time_w'], event_observed=_g['event_w'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors_h[_hist]
            _dash = 'solid' if _tag == '2yr' else 'dash'

            _fig_h.add_trace(go.Scatter(
                x=_t, y=(_sf.iloc[:, 0] * 100).tolist(),
                mode='lines', name=_hist,
                line=dict(color=_col, width=2, dash=_dash),
                legendgroup=_hist, showlegend=(_tag == '2yr'),
            ))
            _fig_h.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=(_ci.iloc[:, 0] * 100).tolist() + (_ci.iloc[:, 1] * 100).tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.10,
                line=dict(width=0), hoverinfo='skip',
                legendgroup=_hist, showlegend=False,
            ))

            _idx = max(_sf.index.searchsorted(_days, side='right') - 1, 0)
            _point_ests_h[_hist] = (
                float(_sf.iloc[_idx, 0]),
                float(_ci.iloc[_idx, 0]),
                float(_ci.iloc[_idx, 1]),
                len(_g), int(_g['event_w'].sum()),
            )

        _results_h[_tag] = (_label, _lr_h, _point_ests_h)

    for _x, _lbl in [(730, '2yr'), (1825, '5yr')]:
        _fig_h.add_vline(x=_x, line_dash='dot', line_color='grey',
                         annotation_text=_lbl, annotation_position='top right')

    _fig_h.update_layout(
        title='Kaplan-Meier Survival by Histology Group',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[40, 100]),
        legend_title='Histology (solid=2yr, dash=5yr)',
    )

    _rows_h = ['### Survival by Histology Group\n',
               '| Horizon | Histology | N | Events | KM Estimate | 95% CI |',
               '|---------|-----------|---|--------|-------------|--------|']
    for _tag, (_label, _lr_h, _ests) in _results_h.items():
        for _hist, (_est, _lo, _hi, _n, _ev) in _ests.items():
            _rows_h.append(
                f"| {_label} | {_hist} | {_n:,} | {_ev:,} | "
                f"{_est*100:.1f}% | [{_lo*100:.1f}%, {_hi*100:.1f}%] |"
            )
        _sig = '**significant**' if _lr_h.p_value < 0.05 else 'not significant'
        _rows_h.append(f"\n**{_label} log-rank:** χ² = {_lr_h.test_statistic:.2f}, "
                       f"p = {_lr_h.p_value:.4f} — {_sig}\n")

    mo.vstack([mo.ui.plotly(_fig_h), mo.md('\n'.join(_rows_h))])
    return


@app.cell
def _(go, mo, np, pd, survival_df, tumor_cohort):
    from lifelines import CoxPHFitter as _CoxHist
    _np = np

    _non_serous4 = {'8310', '8380', '8480'}

    def _map_morph4(code):
        if pd.isna(code):
            return 'Other'
        s = str(code).strip().split('/')[0].split('.')[0]
        if s == '8441': return 'High-grade serous'
        if s == '8460': return 'Low-grade serous'
        if s in _non_serous4: return 'Non-serous epithelial'
        return 'Other'

    _tc4 = tumor_cohort[['patient_id', 'morphology_code']].copy()
    _tc4['histology'] = _tc4['morphology_code'].apply(_map_morph4)
    _sdf4 = survival_df.merge(_tc4[['patient_id', 'histology']], on='patient_id', how='inner')

    def _run_cox_hist(sdf, days, label):
        _s = sdf.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= days)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=days)

        _cox_df = pd.get_dummies(
            _s[['time_w', 'event_w', 'payer_type', 'histology', 'age_at_diagnosis', 'treated']].dropna(),
            columns=['payer_type', 'histology'], drop_first=True, dtype=float
        )
        _cox_df.columns = [c.replace(' ', '_').replace('-', '_') for c in _cox_df.columns]
        _covars = [c for c in _cox_df.columns if c not in ('time_w', 'event_w')]

        _cph = _CoxHist()
        _cph.fit(_cox_df, duration_col='time_w', event_col='event_w',
                 formula=' + '.join(_covars))
        return _cph, len(_s)

    _cph_2yr, _n_2yr = _run_cox_hist(_sdf4, 730, '2-Year')
    _cph_5yr, _n_5yr = _run_cox_hist(_sdf4, 1825, '5-Year')

    def _cox_table(cph, label, n):
        s = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
        s.index = s.index.str.replace('_', ' ')
        s = s.rename(columns={'exp(coef)': 'HR', 'exp(coef) lower 95%': 'CI_lo',
                               'exp(coef) upper 95%': 'CI_hi', 'p': 'p_value'})
        lines = [
            f"**{label}** — N={n:,} | Events={int(cph.event_observed.sum()):,} | "
            f"Concordance={cph.concordance_index_:.3f}\n",
            "| Covariate | HR | 95% CI | p | Sig |",
            "|-----------|-----|--------|---|-----|",
        ]
        for var, row in s.iterrows():
            sig = '\\*\\*\\*' if row['p_value'] < 0.001 else ('\\*' if row['p_value'] < 0.05 else '')
            lines.append(f"| {var} | {row['HR']:.3f} | [{row['CI_lo']:.3f}, {row['CI_hi']:.3f}] "
                         f"| {row['p_value']:.4f} | {sig} |")
        return '\n'.join(lines)

    def _forest_plot(cph, title):
        _s = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
        _s.index = _s.index.str.replace('_', ' ')
        _s = _s.reset_index().rename(columns={
            'index': 'covariate', 'exp(coef)': 'HR',
            'exp(coef) lower 95%': 'HR_lo', 'exp(coef) upper 95%': 'HR_hi', 'p': 'p_val'
        })
        _s = _s.sort_values('HR').reset_index(drop=True)

        # Display range: 10th–90th percentile of HR, padded × 1.5, floored/capped
        _x_lo = max(0.1,  _s['HR'].quantile(0.10) / 1.5)
        _x_hi = min(10.0, _s['HR'].quantile(0.90) * 1.5)

        # Clip HR and CI to display range; track which rows are clipped
        _s['HR_disp']   = _s['HR'].clip(_x_lo, _x_hi)
        _s['HR_lo_disp'] = _s['HR_lo'].clip(_x_lo, _x_hi)
        _s['HR_hi_disp'] = _s['HR_hi'].clip(_x_lo, _x_hi)
        _s['clipped'] = (_s['HR'] < _x_lo) | (_s['HR'] > _x_hi)

        _colors_fp = ['steelblue' if p < 0.05 else 'lightgray' for p in _s['p_val']]
        _symbols   = ['arrow-right' if row['HR'] > _x_hi else
                      ('arrow-left' if row['HR'] < _x_lo else 'circle')
                      for _, row in _s.iterrows()]

        _fig = go.Figure()
        _fig.add_vline(x=1, line_dash='dash', line_color='red')
        _fig.add_trace(go.Scatter(
            x=_s['HR_disp'],
            y=_s['covariate'],
            mode='markers',
            marker=dict(color=_colors_fp, size=10, symbol=_symbols),
            error_x=dict(
                type='data', symmetric=False,
                array=(_s['HR_hi_disp'] - _s['HR_disp']).clip(lower=0).values,
                arrayminus=(_s['HR_disp'] - _s['HR_lo_disp']).clip(lower=0).values,
                color='gray',
            ),
            hovertemplate='<b>%{y}</b><br>HR: %{customdata[0]:.3f}<br>'
                          'CI: %{customdata[1]:.3f}–%{customdata[2]:.3f}<br>'
                          'p: %{customdata[3]:.4f}%{customdata[4]}<extra></extra>',
            customdata=_np.column_stack([
                _s['HR'], _s['HR_lo'], _s['HR_hi'], _s['p_val'],
                [' ⚠ clipped' if c else '' for c in _s['clipped']],
            ]),
            showlegend=False,
        ))
        _fig.update_layout(
            title=title + '<br><sup>Blue = p < 0.05 | Arrow = outlier clipped to axis</sup>',
            xaxis=dict(title='Hazard Ratio (log scale)', type='log',
                       range=[_np.log10(_x_lo), _np.log10(_x_hi)]),
            height=max(400, len(_s) * 38),
            margin=dict(l=200),
        )
        return _fig

    mo.vstack([
        mo.ui.plotly(_forest_plot(_cph_2yr, 'Cox PH Forest Plot — 2-Year Horizon')),
        mo.ui.plotly(_forest_plot(_cph_5yr, 'Cox PH Forest Plot — 5-Year Horizon')),
        mo.md(_cox_table(_cph_2yr, 'Cox Model: 2-Year Horizon (age + payer type + histology + treated)', _n_2yr)),
        mo.md('---'),
        mo.md(_cox_table(_cph_5yr, 'Cox Model: 5-Year Horizon (age + payer type + histology + treated)', _n_5yr)),
    ])
    return


@app.cell
def _(go, mo, survival_df, tumor_cohort):
    from lifelines import KaplanMeierFitter as _KMFp
    from lifelines.statistics import multivariate_logrank_test as _mlrtp

    # Restrict to patients with histology data (in tumor_cohort)
    _sdf_p = survival_df[survival_df['patient_id'].isin(tumor_cohort['patient_id'])].copy()

    _colors_p = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
    ]

    def _km_by_payer(sdf, days, label):
        _s = sdf.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= days)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=days)

        _payers = sorted(_s['payer_type'].dropna().unique())
        _fig = go.Figure()
        _all_sf = []
        _rows = [f'| Payer Type | N | Events | KM at {label} | 95% CI |',
                 '|------------|---|--------|--------------|--------|']

        for _i, _payer in enumerate(_payers):
            _g = _s[_s['payer_type'] == _payer]
            _kmf = _KMFp()
            _kmf.fit(_g['time_w'], event_observed=_g['event_w'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors_p[_i % len(_colors_p)]

            _sf_pct = (_sf.iloc[:, 0] * 100).tolist()
            _all_sf.extend(_sf_pct + (_ci.iloc[:, 0] * 100).tolist())

            _fig.add_trace(go.Scatter(
                x=_t, y=_sf_pct,
                mode='lines', name=f'{_payer} (n={len(_g):,})',
                line=dict(color=_col, width=2), line_shape='hv',
                hovertemplate=f'<b>{_payer}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
            ))
            _fig.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=(_ci.iloc[:, 0] * 100).tolist() + (_ci.iloc[:, 1] * 100).tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.08,
                line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
                showlegend=False, hoverinfo='skip'
            ))

            _idx = max(_sf.index.searchsorted(days, side='right') - 1, 0)
            _est = float(_sf.iloc[_idx, 0])
            _lo  = float(_ci.iloc[_idx, 0])
            _hi  = float(_ci.iloc[_idx, 1])
            _ev  = int(_g['event_w'].sum())
            _rows.append(f'| {_payer} | {len(_g):,} | {_ev:,} | {_est*100:.1f}% | [{_lo*100:.1f}%, {_hi*100:.1f}%] |')

        _lr = _mlrtp(_s['time_w'], _s['event_w'], _s['payer_type'])
        _y_min = max(0, min(_all_sf) - 2)
        _fig.update_layout(
            title=f'KM {label} Survival by Payer Type (histology cohort)<br>'
                  f'<sup>Log-rank p = {_lr.p_value:.4f} | All patients, follow-up capped at {label}</sup>',
            xaxis_title='Days from Diagnosis',
            yaxis_title='Survival Probability (%)',
            yaxis=dict(range=[_y_min, 102]),
            legend_title='Payer Type',
            hovermode='x unified',
        )
        _sig = '**significant**' if _lr.p_value < 0.05 else 'not significant'
        _summary = f'\n**Log-rank across payer types:** χ² = {_lr.test_statistic:.2f}, p = {_lr.p_value:.4f} — {_sig}\n'
        return _fig, '\n'.join(_rows) + _summary

    _fig_2p, _tbl_2p = _km_by_payer(_sdf_p, 730,  '2-Year')
    _fig_5p, _tbl_5p = _km_by_payer(_sdf_p, 1825, '5-Year')

    mo.vstack([
        mo.ui.plotly(_fig_2p),
        mo.md(_tbl_2p),
        mo.ui.plotly(_fig_5p),
        mo.md(_tbl_5p),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Survival Analyses by Stage and by Treatment Status (Tumor Cohort)
    """)
    return


@app.cell
def _(go, mo, survival_df, tumor_cohort_staged):
    from lifelines import KaplanMeierFitter as _KMFstage
    from lifelines.statistics import multivariate_logrank_test as _mlrtstage

    _stage_order = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
    _colors_stage = {
        'Stage I':   '#636EFA',
        'Stage II':  '#EF553B',
        'Stage III': '#00CC96',
        'Stage IV':  '#AB63FA',
    }

    _tc_stage = tumor_cohort_staged[['patient_id', 'stage_label']].drop_duplicates('patient_id')
    _sdf_stage = survival_df.merge(_tc_stage, on='patient_id', how='inner')

    _fig_stage = go.Figure()
    _results_stage = {}

    for _days_s, _tag_s, _label_s in [(730, '2yr', '2-Year'), (1825, '5yr', '5-Year')]:
        _s = _sdf_stage.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= _days_s)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=_days_s)

        _lr_s = _mlrtstage(_s['time_w'], _s['event_w'], _s['stage_label'])
        _point_ests_s = {}

        for _stage in _stage_order:
            _g = _s[_s['stage_label'] == _stage]
            if len(_g) < 5:
                continue
            _kmf = _KMFstage()
            _kmf.fit(_g['time_w'], event_observed=_g['event_w'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors_stage[_stage]
            _dash_s = 'solid' if _tag_s == '2yr' else 'dash'

            _sf_pct = (_sf.iloc[:, 0] * 100).tolist()
            _ci_lo  = (_ci.iloc[:, 0] * 100).tolist()

            _fig_stage.add_trace(go.Scatter(
                x=_t, y=_sf_pct,
                mode='lines', name=_stage,
                line=dict(color=_col, width=2, dash=_dash_s),
                legendgroup=_stage, showlegend=(_tag_s == '2yr'),
                hovertemplate=f'<b>{_stage} ({_label_s})</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
            ))
            _fig_stage.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=_ci_lo + (_ci.iloc[:, 1] * 100).tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.10,
                line=dict(width=0), hoverinfo='skip',
                legendgroup=_stage, showlegend=False,
            ))

            _idx = max(_sf.index.searchsorted(_days_s, side='right') - 1, 0)
            _point_ests_s[_stage] = (
                float(_sf.iloc[_idx, 0]),
                float(_ci.iloc[_idx, 0]),
                float(_ci.iloc[_idx, 1]),
                len(_g), int(_g['event_w'].sum()),
            )

        _results_stage[_tag_s] = (_label_s, _lr_s, _point_ests_s)

    for _x_s, _lbl_s in [(730, '2yr'), (1825, '5yr')]:
        _fig_stage.add_vline(x=_x_s, line_dash='dot', line_color='grey',
                             annotation_text=_lbl_s, annotation_position='top right')

    _fig_stage.update_layout(
        title='Kaplan-Meier Survival by Stage (Tumor Cohort)',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[0, 105]),
        legend_title='Stage (solid=2yr, dash=5yr)',
    )

    _rows_stage = ['### Survival by Stage\n',
                   '| Horizon | Stage | N | Events | KM Estimate | 95% CI |',
                   '|---------|-------|---|--------|-------------|--------|']
    for _tag_s, (_label_s, _lr_s, _ests_s) in _results_stage.items():
        for _stage, (_est, _lo, _hi, _n, _ev) in _ests_s.items():
            _rows_stage.append(
                f"| {_label_s} | {_stage} | {_n:,} | {_ev:,} | "
                f"{_est*100:.1f}% | [{_lo*100:.1f}%, {_hi*100:.1f}%] |"
            )
        _sig_s = '**significant**' if _lr_s.p_value < 0.05 else 'not significant'
        _rows_stage.append(f"\n**{_label_s} log-rank:** χ² = {_lr_s.test_statistic:.2f}, "
                           f"p = {_lr_s.p_value:.4f} — {_sig_s}\n")

    mo.vstack([mo.ui.plotly(_fig_stage), mo.md('\n'.join(_rows_stage))])
    return


@app.cell
def _(go, mo, survival_df, tumor_cohort):
    from lifelines import KaplanMeierFitter as _KMFpt
    from lifelines.statistics import multivariate_logrank_test as _mlrtpt

    _sdf_pt = survival_df[survival_df['patient_id'].isin(tumor_cohort['patient_id'])].copy()
    _sdf_pt['treated_label'] = _sdf_pt['treated'].map({1: 'Treated', 0: 'Untreated'})
    _sdf_pt['payer_treatment'] = _sdf_pt['payer_type'] + ' – ' + _sdf_pt['treated_label']

    _colors_pt = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
        '#2ca02c', '#98df8a', '#d62728', '#ff9896',
        '#9467bd', '#c5b0d5',
    ]

    def _km_payer_treat(sdf, days, label):
        _s = sdf.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= days)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=days)
        _groups = sorted(_s['payer_treatment'].dropna().unique())
        _fig = go.Figure()
        _all_sf = []
        _rows = [f'| Group | N | Events | KM at {label} | 95% CI |',
                 '|-------|---|--------|--------------|--------|']
        for _i, _grp in enumerate(_groups):
            _g = _s[_s['payer_treatment'] == _grp]
            if len(_g) < 5:
                continue
            _kmf = _KMFpt()
            _kmf.fit(_g['time_w'], event_observed=_g['event_w'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors_pt[_i % len(_colors_pt)]
            _sf_pct = (_sf.iloc[:, 0] * 100).tolist()
            _all_sf.extend(_sf_pct + (_ci.iloc[:, 0] * 100).tolist())
            _fig.add_trace(go.Scatter(
                x=_t, y=_sf_pct, mode='lines', name=f'{_grp} (n={len(_g):,})',
                line=dict(color=_col, width=2), line_shape='hv',
                hovertemplate=f'<b>{_grp}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
            ))
            _fig.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=(_ci.iloc[:, 0] * 100).tolist() + (_ci.iloc[:, 1] * 100).tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.08,
                line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
                showlegend=False, hoverinfo='skip'
            ))
            _idx = max(_sf.index.searchsorted(days, side='right') - 1, 0)
            _est = float(_sf.iloc[_idx, 0])
            _lo  = float(_ci.iloc[_idx, 0])
            _hi  = float(_ci.iloc[_idx, 1])
            _rows.append(f'| {_grp} | {len(_g):,} | {int(_g["event_w"].sum()):,} | '
                         f'{_est*100:.1f}% | [{_lo*100:.1f}%, {_hi*100:.1f}%] |')
        _lr = _mlrtpt(_s['time_w'], _s['event_w'], _s['payer_treatment'])
        _y_min = max(0, min(_all_sf) - 2) if _all_sf else 0
        _fig.update_layout(
            title=f'KM {label} Survival by Payer Type × Treatment Status (Tumor Cohort)<br>'
                  f'<sup>Log-rank p = {_lr.p_value:.4f}</sup>',
            xaxis_title='Days from Diagnosis',
            yaxis_title='Survival Probability (%)',
            yaxis=dict(range=[_y_min, 102]),
            legend_title='Group', hovermode='x unified',
        )
        _sig = '**significant**' if _lr.p_value < 0.05 else 'not significant'
        _summary = f'\n**Log-rank:** χ² = {_lr.test_statistic:.2f}, p = {_lr.p_value:.4f} — {_sig}\n'
        return _fig, '\n'.join(_rows) + _summary

    _fig_pt2, _tbl_pt2 = _km_payer_treat(_sdf_pt, 730,  '2-Year')
    _fig_pt5, _tbl_pt5 = _km_payer_treat(_sdf_pt, 1825, '5-Year')

    mo.vstack([
        mo.ui.plotly(_fig_pt2), mo.md(_tbl_pt2),
        mo.ui.plotly(_fig_pt5), mo.md(_tbl_pt5),
    ])
    return


@app.cell
def _(go, mo, pd, survival_df, tumor_cohort):
    from lifelines import KaplanMeierFitter as _KMFht
    from lifelines.statistics import multivariate_logrank_test as _mlrtht

    _non_serous_ht = {'8310', '8380', '8480'}

    def _map_morph_ht(code):
        if pd.isna(code):
            return 'Other'
        s = str(code).strip().split('/')[0].split('.')[0]
        if s == '8441': return 'High-grade serous'
        if s == '8460': return 'Low-grade serous'
        if s in _non_serous_ht: return 'Non-serous epithelial'
        return 'Other'

    _tc_ht = tumor_cohort[['patient_id', 'morphology_code']].copy()
    _tc_ht['histology'] = _tc_ht['morphology_code'].apply(_map_morph_ht)
    _sdf_ht = survival_df.merge(_tc_ht[['patient_id', 'histology']], on='patient_id', how='inner')
    _sdf_ht['treated_label'] = _sdf_ht['treated'].map({1: 'Treated', 0: 'Untreated'})
    _sdf_ht['histology_treatment'] = _sdf_ht['histology'] + ' – ' + _sdf_ht['treated_label']

    _colors_ht = [
        '#636EFA', '#aaaaff', '#EF553B', '#ffaaaa',
        '#00CC96', '#aaffee', '#AB63FA', '#ddaaff',
    ]

    def _km_hist_treat(sdf, days, label):
        _s = sdf.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= days)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=days)
        _hist_order = ['High-grade serous – Treated', 'High-grade serous – Untreated',
                       'Low-grade serous – Treated', 'Low-grade serous – Untreated',
                       'Non-serous epithelial – Treated', 'Non-serous epithelial – Untreated',
                       'Other – Treated', 'Other – Untreated']
        _groups = [g for g in _hist_order if g in _s['histology_treatment'].unique()]
        _fig = go.Figure()
        _all_sf = []
        _rows = [f'| Group | N | Events | KM at {label} | 95% CI |',
                 '|-------|---|--------|--------------|--------|']
        for _i, _grp in enumerate(_groups):
            _g = _s[_s['histology_treatment'] == _grp]
            if len(_g) < 5:
                continue
            _kmf = _KMFht()
            _kmf.fit(_g['time_w'], event_observed=_g['event_w'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors_ht[_i % len(_colors_ht)]
            _sf_pct = (_sf.iloc[:, 0] * 100).tolist()
            _all_sf.extend(_sf_pct + (_ci.iloc[:, 0] * 100).tolist())
            _fig.add_trace(go.Scatter(
                x=_t, y=_sf_pct, mode='lines', name=f'{_grp} (n={len(_g):,})',
                line=dict(color=_col, width=2), line_shape='hv',
                hovertemplate=f'<b>{_grp}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
            ))
            _fig.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=(_ci.iloc[:, 0] * 100).tolist() + (_ci.iloc[:, 1] * 100).tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.08,
                line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
                showlegend=False, hoverinfo='skip'
            ))
            _idx = max(_sf.index.searchsorted(days, side='right') - 1, 0)
            _est = float(_sf.iloc[_idx, 0])
            _lo  = float(_ci.iloc[_idx, 0])
            _hi  = float(_ci.iloc[_idx, 1])
            _rows.append(f'| {_grp} | {len(_g):,} | {int(_g["event_w"].sum()):,} | '
                         f'{_est*100:.1f}% | [{_lo*100:.1f}%, {_hi*100:.1f}%] |')
        _lr = _mlrtht(_s['time_w'], _s['event_w'], _s['histology_treatment'])
        _y_min = max(0, min(_all_sf) - 2) if _all_sf else 0
        _fig.update_layout(
            title=f'KM {label} Survival by Histology × Treatment Status (Tumor Cohort)<br>'
                  f'<sup>Log-rank p = {_lr.p_value:.4f}</sup>',
            xaxis_title='Days from Diagnosis',
            yaxis_title='Survival Probability (%)',
            yaxis=dict(range=[_y_min, 102]),
            legend_title='Group', hovermode='x unified',
        )
        _sig = '**significant**' if _lr.p_value < 0.05 else 'not significant'
        _summary = f'\n**Log-rank:** χ² = {_lr.test_statistic:.2f}, p = {_lr.p_value:.4f} — {_sig}\n'
        return _fig, '\n'.join(_rows) + _summary

    _fig_ht2, _tbl_ht2 = _km_hist_treat(_sdf_ht, 730,  '2-Year')
    _fig_ht5, _tbl_ht5 = _km_hist_treat(_sdf_ht, 1825, '5-Year')

    mo.vstack([
        mo.ui.plotly(_fig_ht2), mo.md(_tbl_ht2),
        mo.ui.plotly(_fig_ht5), mo.md(_tbl_ht5),
    ])
    return


@app.cell
def _(go, mo, survival_df, tumor_cohort_staged):
    from lifelines import KaplanMeierFitter as _KMFst
    from lifelines.statistics import multivariate_logrank_test as _mlrtst

    _tc_st = tumor_cohort_staged[['patient_id', 'stage_label']].drop_duplicates('patient_id')
    _sdf_st = survival_df.merge(_tc_st, on='patient_id', how='inner')
    _sdf_st['treated_label'] = _sdf_st['treated'].map({1: 'Treated', 0: 'Untreated'})
    _sdf_st['stage_treatment'] = _sdf_st['stage_label'] + ' – ' + _sdf_st['treated_label']

    _colors_st = [
        '#636EFA', '#aaaaff', '#EF553B', '#ffaaaa',
        '#00CC96', '#aaffee', '#AB63FA', '#ddaaff',
    ]

    def _km_stage_treat(sdf, days, label):
        _s = sdf.copy()
        _s['event_w'] = ((_s['event'] == 1) & (_s['survival_days'] <= days)).astype(int)
        _s['time_w']  = _s['survival_days'].clip(upper=days)
        _stage_treat_order = ['Stage I – Treated', 'Stage I – Untreated',
                              'Stage II – Treated', 'Stage II – Untreated',
                              'Stage III – Treated', 'Stage III – Untreated',
                              'Stage IV – Treated', 'Stage IV – Untreated']
        _groups = [g for g in _stage_treat_order if g in _s['stage_treatment'].unique()]
        _fig = go.Figure()
        _all_sf = []
        _rows = [f'| Group | N | Events | KM at {label} | 95% CI |',
                 '|-------|---|--------|--------------|--------|']
        for _i, _grp in enumerate(_groups):
            _g = _s[_s['stage_treatment'] == _grp]
            if len(_g) < 5:
                continue
            _kmf = _KMFst()
            _kmf.fit(_g['time_w'], event_observed=_g['event_w'])
            _sf  = _kmf.survival_function_
            _ci  = _kmf.confidence_interval_survival_function_
            _t   = _sf.index.tolist()
            _col = _colors_st[_i % len(_colors_st)]
            _sf_pct = (_sf.iloc[:, 0] * 100).tolist()
            _all_sf.extend(_sf_pct + (_ci.iloc[:, 0] * 100).tolist())
            _fig.add_trace(go.Scatter(
                x=_t, y=_sf_pct, mode='lines', name=f'{_grp} (n={len(_g):,})',
                line=dict(color=_col, width=2), line_shape='hv',
                hovertemplate=f'<b>{_grp}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>'
            ))
            _fig.add_trace(go.Scatter(
                x=_t + _t[::-1],
                y=(_ci.iloc[:, 0] * 100).tolist() + (_ci.iloc[:, 1] * 100).tolist()[::-1],
                fill='toself', fillcolor=_col, opacity=0.08,
                line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
                showlegend=False, hoverinfo='skip'
            ))
            _idx = max(_sf.index.searchsorted(days, side='right') - 1, 0)
            _est = float(_sf.iloc[_idx, 0])
            _lo  = float(_ci.iloc[_idx, 0])
            _hi  = float(_ci.iloc[_idx, 1])
            _rows.append(f'| {_grp} | {len(_g):,} | {int(_g["event_w"].sum()):,} | '
                         f'{_est*100:.1f}% | [{_lo*100:.1f}%, {_hi*100:.1f}%] |')
        _lr = _mlrtst(_s['time_w'], _s['event_w'], _s['stage_treatment'])
        _y_min = max(0, min(_all_sf) - 2) if _all_sf else 0
        _fig.update_layout(
            title=f'KM {label} Survival by Stage × Treatment Status (Tumor Cohort)<br>'
                  f'<sup>Log-rank p = {_lr.p_value:.4f}</sup>',
            xaxis_title='Days from Diagnosis',
            yaxis_title='Survival Probability (%)',
            yaxis=dict(range=[_y_min, 102]),
            legend_title='Group', hovermode='x unified',
        )
        _sig = '**significant**' if _lr.p_value < 0.05 else 'not significant'
        _summary = f'\n**Log-rank:** χ² = {_lr.test_statistic:.2f}, p = {_lr.p_value:.4f} — {_sig}\n'
        return _fig, '\n'.join(_rows) + _summary

    _fig_st2, _tbl_st2 = _km_stage_treat(_sdf_st, 730,  '2-Year')
    _fig_st5, _tbl_st5 = _km_stage_treat(_sdf_st, 1825, '5-Year')

    mo.vstack([
        mo.ui.plotly(_fig_st2), mo.md(_tbl_st2),
        mo.ui.plotly(_fig_st5), mo.md(_tbl_st5),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive Tumor KM Plot

    Toggle segmentation by tumor **Stage** and/or **Histology subtype**, then click **Calculate** to render. Both can be on at once for a cross-product breakdown.
    """)
    return


@app.cell
def _(mo):
    km_t_stage_toggle = mo.ui.switch(value=True, label='Segment by Stage')
    km_t_hist_toggle = mo.ui.switch(value=False, label='Segment by Histology')
    km_t_treated_toggle = mo.ui.switch(value=False, label='Segment by Treated')
    km_t_payer_toggle = mo.ui.switch(value=False, label='Segment by Payer Type')
    km_t_age_toggle = mo.ui.switch(value=False, label='Segment by Age Bin')
    km_t_age_bins_input = mo.ui.text(
        value='0,50,65,75,120',
        label='Age bin edges (comma-separated)',
        full_width=True,
    )
    km_t_run_btn = mo.ui.run_button(label='Calculate', kind='success')
    mo.vstack([km_t_stage_toggle, km_t_hist_toggle, km_t_treated_toggle, km_t_payer_toggle, km_t_age_toggle, km_t_age_bins_input, km_t_run_btn])
    return (
        km_t_age_bins_input,
        km_t_age_toggle,
        km_t_hist_toggle,
        km_t_payer_toggle,
        km_t_run_btn,
        km_t_stage_toggle,
        km_t_treated_toggle,
    )


@app.cell
def _(
    go,
    km_t_age_bins_input,
    km_t_age_toggle,
    km_t_hist_toggle,
    km_t_payer_toggle,
    km_t_run_btn,
    km_t_stage_toggle,
    km_t_treated_toggle,
    mo,
    pd,
    survival_df,
    tumor_cohort,
    tumor_cohort_staged,
):
    from lifelines import KaplanMeierFitter as _KMFt
    from lifelines.statistics import multivariate_logrank_test as _mlrtt

    mo.stop(
        not km_t_run_btn.value,
        mo.md("*Configure options above and click **Calculate** to render the plot.*"),
    )
    mo.stop(
        not (km_t_stage_toggle.value or km_t_hist_toggle.value or km_t_treated_toggle.value or km_t_payer_toggle.value or km_t_age_toggle.value),
        mo.md("⚠️ Turn on at least one segmentation toggle (Stage, Histology, Treated, Payer Type, and/or Age Bin)."),
    )

    _non_serous_t = {'8310', '8380', '8480'}

    def _map_morph_t(code):
        if pd.isna(code):
            return None
        s = str(code).strip().split('/')[0].split('.')[0]
        if s == '8441': return 'High-grade serous'
        if s == '8460': return 'Low-grade serous'
        if s in _non_serous_t: return 'Non-serous epithelial'
        return 'Other'

    _base_t = survival_df[['patient_id', 'survival_days', 'event', 'treated', 'age_at_diagnosis', 'payer_type']].copy()
    _base_t['event_w'] = ((_base_t['event'] == 1) & (_base_t['survival_days'] <= 1825)).astype(int)
    _base_t['time_w'] = _base_t['survival_days'].clip(upper=1825)

    _dims_t = []

    if km_t_stage_toggle.value:
        _tc_s = tumor_cohort_staged[['patient_id', 'stage_label']].drop_duplicates('patient_id')
        _base_t = _base_t.merge(_tc_s, on='patient_id', how='inner')
        _dims_t.append(('Stage', 'stage_label'))

    if km_t_hist_toggle.value:
        _tc_h = tumor_cohort[['patient_id', 'morphology_code']].copy()
        _tc_h['histology'] = _tc_h['morphology_code'].apply(_map_morph_t)
        _base_t = _base_t.merge(_tc_h[['patient_id', 'histology']], on='patient_id', how='inner')
        _dims_t.append(('Histology', 'histology'))

    if km_t_treated_toggle.value:
        _base_t['treated_label'] = _base_t['treated'].map({1: 'Treated', 0: 'Untreated'})
        _dims_t.append(('Treated', 'treated_label'))

    if km_t_payer_toggle.value:
        _dims_t.append(('Payer Type', 'payer_type'))

    if km_t_age_toggle.value:
        try:
            _edges_t = sorted({float(x.strip()) for x in km_t_age_bins_input.value.split(',') if x.strip()})
        except ValueError:
            _edges_t = []
        mo.stop(
            len(_edges_t) < 2,
            mo.md("⚠️ Please enter at least 2 numeric bin edges separated by commas."),
        )
        _age_labels_t = [f"{int(_edges_t[_k])} <= age < {int(_edges_t[_k+1])}" for _k in range(len(_edges_t) - 1)]
        _base_t['age_bin'] = pd.cut(
            _base_t['age_at_diagnosis'].astype(float),
            bins=_edges_t,
            labels=_age_labels_t,
            include_lowest=True,
            right=False,
        ).astype(object)
        _dims_t.append(('Age Bin', 'age_bin'))

    if len(_dims_t) == 1:
        _label_t, _col_t = _dims_t[0]
        _base_t['_group'] = _base_t[_col_t]
        _group_col_t = _label_t
    else:
        _base_t['_group'] = _base_t[_dims_t[0][1]].astype(str)
        for _d in _dims_t[1:]:
            _base_t['_group'] = _base_t['_group'] + ' | ' + _base_t[_d[1]].astype(str)
        _null_mask = pd.Series(False, index=_base_t.index)
        for _d in _dims_t:
            _null_mask = _null_mask | _base_t[_d[1]].isna()
        _base_t.loc[_null_mask, '_group'] = pd.NA
        _group_col_t = ' × '.join(_d[0] for _d in _dims_t)
        _group_col_t = ' × '.join(_d[0] for _d in _dims_t)

    _base_t = _base_t[_base_t['_group'].notna()].copy()

    _palette_t = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
    _groups_t = sorted(_base_t['_group'].unique().tolist(), key=str)
    _all_vals_t = []
    _fig_t = go.Figure()

    for _idx_t, _grp_t in enumerate(_groups_t):
        _gd_t = _base_t[_base_t['_group'] == _grp_t]
        if len(_gd_t) < 5:
            continue
        _kmf_t = _KMFt()
        _kmf_t.fit(_gd_t['time_w'], event_observed=_gd_t['event_w'], label=str(_grp_t))
        _sf_t = _kmf_t.survival_function_
        _ci_t = _kmf_t.confidence_interval_survival_function_
        _t_t = _sf_t.index.tolist()
        _color_t = _palette_t[_idx_t % len(_palette_t)]

        _sf_pct_t = (_sf_t.iloc[:, 0].values * 100).tolist()
        _ci_lo_t = (_ci_t.iloc[:, 0].values * 100).tolist()
        _ci_hi_t = (_ci_t.iloc[:, 1].values * 100).tolist()
        _all_vals_t.extend(_sf_pct_t + _ci_lo_t)

        _fig_t.add_trace(go.Scatter(
            x=_t_t, y=_sf_pct_t,
            mode='lines', name=f'{_grp_t} (n={len(_gd_t):,})',
            line=dict(color=_color_t, width=2), line_shape='hv',
            hovertemplate=f'<b>{_grp_t}</b><br>Days: %{{x}}<br>Survival: %{{y:.1f}}%<extra></extra>',
        ))
        _fig_t.add_trace(go.Scatter(
            x=_t_t + _t_t[::-1],
            y=_ci_lo_t + _ci_hi_t[::-1],
            fill='toself', fillcolor=_color_t, opacity=0.08,
            line=dict(color='rgba(0,0,0,0)'), line_shape='hv',
            showlegend=False, hoverinfo='skip',
        ))

    _y_min_t = max(0, min(_all_vals_t) - 2) if _all_vals_t else 0

    if _base_t['_group'].nunique() >= 2:
        _lr_t = _mlrtt(_base_t['time_w'], _base_t['_group'].astype(str), _base_t['event_w'])
        _title_sup_t = f'Log-rank p = {_lr_t.p_value:.4f} | follow-up capped at 5 years | tumor cohort'
        _lr_md_t = mo.md(
            f"Overall log-rank test across {_group_col_t.lower()} groups: "
            f"χ² = {_lr_t.test_statistic:.2f}, p = {_lr_t.p_value:.4f}"
        )
    else:
        _title_sup_t = 'follow-up capped at 5 years | tumor cohort'
        _lr_md_t = mo.md("*Need ≥2 groups for a log-rank test.*")

    _fig_t.update_layout(
        title=f'Kaplan-Meier: 5-Year Survival by {_group_col_t} (Tumor Cohort)<br><sup>{_title_sup_t}</sup>',
        xaxis_title='Days from Diagnosis',
        yaxis_title='Survival Probability (%)',
        yaxis=dict(range=[_y_min_t, 102]),
        legend_title=_group_col_t,
        hovermode='x unified',
    )

    mo.vstack([mo.ui.plotly(_fig_t), _lr_md_t])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Mediation Analysis

    Tests whether payer type's association with survival operates **through treatment** (indirect/mediated effect)
    or **directly** (independent of treatment).

    **Causal diagram:** Payer type → Treatment (a-path) → Survival (b-path); Payer type → Survival directly (c'-path).

    - **ACME** (Average Causal Mediation Effect): indirect effect operating through treatment
    - **ADE** (Average Direct Effect): direct payer → survival effect not through treatment
    - **Total effect** = ACME + ADE
    - **Proportion mediated** = ACME / Total

    Payer type is multi-categorical; K-1 dummy variables are created (reference = most common payer).
    Mediation is run separately for each payer dummy using `statsmodels.stats.mediation.Mediation`
    (parametric, n_rep=200). Outcomes: binary event (primary) and continuous survival days (sensitivity).
    """)
    return


@app.cell
def _(enrollment_final, mo, pd, survival_df):
    # ── Mediation data prep (entire cohort) ──────────────────────────────────
    _med = survival_df.copy()
    _med['event_2yr'] = ((_med['event'] == 1) & (_med['survival_days'] <= 730)).astype(int)
    _med['event_5yr'] = ((_med['event'] == 1) & (_med['survival_days'] <= 1825)).astype(int)

    # Merge plan_design from enrollment_final (not in survival_df)
    _med = _med.merge(
        enrollment_final[['patient_id', 'payer_type']].drop_duplicates('patient_id'),
        on='patient_id', how='left', suffixes=('', '_enr')
    )

    # Reference payer = most common
    ref_payer = _med['payer_type'].value_counts().idxmax()
    _med['_payer_clean'] = _med['payer_type'].str.replace(' ', '_').str.replace('-', '_')
    ref_col = 'payer_' + ref_payer.replace(' ', '_').replace('-', '_')

    _dummies = pd.get_dummies(_med['_payer_clean'], prefix='payer', dtype=int)
    _med = pd.concat([_med, _dummies], axis=1)
    if ref_col in _med.columns:
        _med = _med.drop(columns=[ref_col])

    payer_cols = sorted([c for c in _med.columns if c.startswith('payer_')])
    _med = _med.dropna(subset=['age_at_diagnosis'] + payer_cols)

    mediation_df = _med
    mo.md(
        f"**Mediation dataset:** {len(mediation_df):,} patients | "
        f"**Reference payer:** {ref_payer} | "
        f"**Payer dummies:** {', '.join(payer_cols)}"
    )
    return mediation_df, payer_cols, ref_payer


@app.cell
def _(mediation_df, mo, np, payer_cols, ref_payer):
    import statsmodels.formula.api as _smf

    def _med_boot(df, outcome_col, payer_col, n_rep=200):
        _acmes, _ades, _totals = [], [], []
        for _ in range(n_rep):
            _b = df.sample(n=len(df), replace=True).reset_index(drop=True)
            try:
                _mfit = _smf.logit(
                    f'treated ~ {payer_col} + age_at_diagnosis', data=_b
                ).fit(disp=0)
                _ofit = _smf.logit(
                    f'{outcome_col} ~ treated + {payer_col} + age_at_diagnosis', data=_b
                ).fit(disp=0)
            except Exception:
                continue
            _d0 = df.assign(**{payer_col: 0})
            _d1 = df.assign(**{payer_col: 1})
            _m0s = np.random.binomial(1, np.clip(_mfit.predict(_d0).values, 0, 1))
            _m1s = np.random.binomial(1, np.clip(_mfit.predict(_d1).values, 0, 1))
            _y00 = float(_ofit.predict(_d0.assign(treated=_m0s)).mean())
            _y01 = float(_ofit.predict(_d0.assign(treated=_m1s)).mean())
            _y10 = float(_ofit.predict(_d1.assign(treated=_m0s)).mean())
            _y11 = float(_ofit.predict(_d1.assign(treated=_m1s)).mean())
            _acmes.append(_y11 - _y10)
            _ades.append(_y11 - _y01)
            _totals.append(_y11 - _y00)
        _a = np.array(_acmes); _d = np.array(_ades); _t = np.array(_totals)
        if len(_a) < 10:
            raise RuntimeError('too many bootstrap failures')
        _acme = float(_a.mean()); _ade = float(_d.mean()); _total = float(_t.mean())
        _prop = _acme / _total if abs(_total) > 1e-10 else float('nan')
        _lo = float(np.percentile(_a, 2.5)); _hi = float(np.percentile(_a, 97.5))
        _p = float(max(2 * min((_a > 0).mean(), (_a < 0).mean()), 1 / n_rep))
        return _acme, _ade, _total, _prop, _lo, _hi, _p

    _lines = [
        f"**Mediation Analysis: Entire Cohort** (reference payer: {ref_payer})\n",
        "*Outcome models: Logistic (binary event). Bootstrap n=200. ACME = indirect effect through treatment.*\n",
    ]
    for _horizon, _outcome_col in [('2-Year', 'event_2yr'), ('5-Year', 'event_5yr')]:
        _rows_med = [
            f"\n**{_horizon} Horizon** (`{_outcome_col}`)\n",
            "| Payer (vs ref) | ACME | ADE | Total | Prop. Med. | ACME 95% CI | p(ACME) |",
            "|----------------|------|-----|-------|-----------|-------------|---------|",
        ]
        for _pc in payer_cols:
            _label = _pc.replace('payer_', '').replace('_', ' ')
            try:
                _acme, _ade, _total, _prop, _lo, _hi, _p = _med_boot(
                    mediation_df, _outcome_col, _pc
                )
                _sig = '\\*' if _p < 0.05 else ''
                _rows_med.append(
                    f"| {_label} | {_acme:.4f} | {_ade:.4f} | {_total:.4f} | "
                    f"{_prop:.3f} | [{_lo:.4f}, {_hi:.4f}] | {_p:.4f}{_sig} |"
                )
            except Exception as _e:
                _rows_med.append(f"| {_label} | error: {_e} | | | | | |")
        _lines.extend(_rows_med)

    mo.md('\n'.join(_lines))
    return


@app.cell
def _(mediation_df, mo, np, payer_cols, ref_payer):
    import statsmodels.formula.api as _smf_ols

    def _med_boot_ols(df, payer_col, n_rep=200):
        _acmes, _ades, _totals = [], [], []
        for _ in range(n_rep):
            _b = df.sample(n=len(df), replace=True).reset_index(drop=True)
            try:
                _mfit = _smf_ols.logit(
                    f'treated ~ {payer_col} + age_at_diagnosis', data=_b
                ).fit(disp=0)
                _ofit = _smf_ols.ols(
                    f'survival_days ~ treated + {payer_col} + age_at_diagnosis', data=_b
                ).fit()
            except Exception:
                continue
            _d0 = df.assign(**{payer_col: 0})
            _d1 = df.assign(**{payer_col: 1})
            _m0s = np.random.binomial(1, np.clip(_mfit.predict(_d0).values, 0, 1))
            _m1s = np.random.binomial(1, np.clip(_mfit.predict(_d1).values, 0, 1))
            _y00 = float(_ofit.predict(_d0.assign(treated=_m0s)).mean())
            _y01 = float(_ofit.predict(_d0.assign(treated=_m1s)).mean())
            _y10 = float(_ofit.predict(_d1.assign(treated=_m0s)).mean())
            _y11 = float(_ofit.predict(_d1.assign(treated=_m1s)).mean())
            _acmes.append(_y11 - _y10)
            _ades.append(_y11 - _y01)
            _totals.append(_y11 - _y00)
        _a = np.array(_acmes); _d = np.array(_ades); _t = np.array(_totals)
        if len(_a) < 10:
            raise RuntimeError('too many bootstrap failures')
        _acme = float(_a.mean()); _ade = float(_d.mean()); _total = float(_t.mean())
        _prop = _acme / _total if abs(_total) > 1e-10 else float('nan')
        _lo = float(np.percentile(_a, 2.5)); _hi = float(np.percentile(_a, 97.5))
        _p = float(max(2 * min((_a > 0).mean(), (_a < 0).mean()), 1 / n_rep))
        return _acme, _ade, _total, _prop, _lo, _hi, _p

    _lines_ols = [
        f"**Sensitivity: Continuous Outcome (survival_days)** — Entire Cohort (reference: {ref_payer})\n",
        "*⚠ Censored patients have truncated survival_days; interpret directionally.*\n",
        "| Payer (vs ref) | ACME (days) | ADE (days) | Total (days) | Prop. Med. | ACME 95% CI | p(ACME) |",
        "|----------------|-------------|-----------|--------------|-----------|-------------|---------|",
    ]
    for _pc in payer_cols:
        _label = _pc.replace('payer_', '').replace('_', ' ')
        try:
            _acme, _ade, _total, _prop, _lo, _hi, _p = _med_boot_ols(mediation_df, _pc)
            _sig = '\\*' if _p < 0.05 else ''
            _lines_ols.append(
                f"| {_label} | {_acme:.1f} | {_ade:.1f} | {_total:.1f} | {_prop:.3f} | "
                f"[{_lo:.1f}, {_hi:.1f}] | {_p:.4f}{_sig} |"
            )
        except Exception as _e:
            _lines_ols.append(f"| {_label} | error: {_e} | | | | | |")

    mo.md('\n'.join(_lines_ols))
    return


@app.cell
def _(mo, pd, survival_df, tumor_cohort, tumor_cohort_staged):
    _non_serous_tm = {'8310', '8380', '8480'}

    def _map_morph_tm(code):
        if pd.isna(code):
            return 'Other'
        s = str(code).strip().split('/')[0].split('.')[0]
        if s == '8441': return 'High_grade_serous'
        if s == '8460': return 'Low_grade_serous'
        if s in _non_serous_tm: return 'Non_serous_epithelial'
        return 'Other'

    _tc_tm = tumor_cohort[['patient_id', 'morphology_code']].copy()
    _tc_tm['histology'] = _tc_tm['morphology_code'].apply(_map_morph_tm)

    _staged_tm = tumor_cohort_staged[['patient_id', 'stage_label']].drop_duplicates('patient_id').copy()
    _staged_tm['stage_clean'] = _staged_tm['stage_label'].str.replace(' ', '_')

    _tmed = (
        survival_df
        .merge(_tc_tm[['patient_id', 'histology']], on='patient_id', how='inner')
        .merge(_staged_tm[['patient_id', 'stage_clean']], on='patient_id', how='inner')
    )
    _tmed['event_2yr'] = ((_tmed['event'] == 1) & (_tmed['survival_days'] <= 730)).astype(int)
    _tmed['event_5yr'] = ((_tmed['event'] == 1) & (_tmed['survival_days'] <= 1825)).astype(int)

    ref_payer_t = _tmed['payer_type'].value_counts().idxmax()
    _tmed['_payer_clean'] = _tmed['payer_type'].str.replace(' ', '_').str.replace('-', '_')
    ref_col_t = 'payer_' + ref_payer_t.replace(' ', '_').replace('-', '_')
    _payer_dummies_t = pd.get_dummies(_tmed['_payer_clean'], prefix='payer', dtype=int)
    _hist_dummies_t  = pd.get_dummies(_tmed['histology'],   prefix='hist',  dtype=int)
    _stage_dummies_t = pd.get_dummies(_tmed['stage_clean'], prefix='stage', dtype=int)
    _tmed = pd.concat([_tmed, _payer_dummies_t, _hist_dummies_t, _stage_dummies_t], axis=1)

    if ref_col_t in _tmed.columns:
        _tmed = _tmed.drop(columns=[ref_col_t])

    payer_cols_t    = sorted([c for c in _tmed.columns if c.startswith('payer_')])
    histology_cols  = sorted([c for c in _tmed.columns if c.startswith('hist_')])
    stage_cols      = sorted([c for c in _tmed.columns if c.startswith('stage_')])

    # Drop reference histology (most common) and Stage_I
    _hist_ref = 'hist_' + _tmed['histology'].value_counts().idxmax()
    if _hist_ref in histology_cols:
        histology_cols = [c for c in histology_cols if c != _hist_ref]
        _tmed = _tmed.drop(columns=[_hist_ref])
    if 'stage_Stage_I' in stage_cols:
        stage_cols = [c for c in stage_cols if c != 'stage_Stage_I']
        _tmed = _tmed.drop(columns=['stage_Stage_I'])

    _tmed = _tmed.dropna(subset=['age_at_diagnosis'] + payer_cols_t)
    tumor_mediation_df = _tmed

    mo.md(
        f"**Tumor Cohort Mediation Dataset:** {len(tumor_mediation_df):,} patients (with stage + histology)\n\n"
        f"Reference payer: {ref_payer_t} | "
        f"Payer dummies: {len(payer_cols_t)} | "
        f"Histology dummies: {len(histology_cols)} | "
        f"Stage dummies: {len(stage_cols)}"
    )
    return (
        histology_cols,
        payer_cols_t,
        ref_payer_t,
        stage_cols,
        tumor_mediation_df,
    )


@app.cell
def _(
    histology_cols,
    mo,
    np,
    payer_cols_t,
    ref_payer_t,
    stage_cols,
    tumor_mediation_df,
):
    import statsmodels.formula.api as _smf_t

    _extra_cols_t = histology_cols + stage_cols
    _cov_t = 'age_at_diagnosis' + (' + ' + ' + '.join(_extra_cols_t) if _extra_cols_t else '')

    def _med_boot_t(df, outcome_col, payer_col, cov, n_rep=200):
        _acmes, _ades, _totals = [], [], []
        for _ in range(n_rep):
            _b = df.sample(n=len(df), replace=True).reset_index(drop=True)
            try:
                _mfit = _smf_t.logit(
                    f'treated ~ {payer_col} + {cov}', data=_b
                ).fit(disp=0)
                _ofit = _smf_t.logit(
                    f'{outcome_col} ~ treated + {payer_col} + {cov}', data=_b
                ).fit(disp=0)
            except Exception:
                continue
            _d0 = df.assign(**{payer_col: 0})
            _d1 = df.assign(**{payer_col: 1})
            _m0s = np.random.binomial(1, np.clip(_mfit.predict(_d0).values, 0, 1))
            _m1s = np.random.binomial(1, np.clip(_mfit.predict(_d1).values, 0, 1))
            _y00 = float(_ofit.predict(_d0.assign(treated=_m0s)).mean())
            _y01 = float(_ofit.predict(_d0.assign(treated=_m1s)).mean())
            _y10 = float(_ofit.predict(_d1.assign(treated=_m0s)).mean())
            _y11 = float(_ofit.predict(_d1.assign(treated=_m1s)).mean())
            _acmes.append(_y11 - _y10)
            _ades.append(_y11 - _y01)
            _totals.append(_y11 - _y00)
        _a = np.array(_acmes); _d = np.array(_ades); _t = np.array(_totals)
        if len(_a) < 10:
            raise RuntimeError('too many bootstrap failures')
        _acme = float(_a.mean()); _ade = float(_d.mean()); _total = float(_t.mean())
        _prop = _acme / _total if abs(_total) > 1e-10 else float('nan')
        _lo = float(np.percentile(_a, 2.5)); _hi = float(np.percentile(_a, 97.5))
        _p = float(max(2 * min((_a > 0).mean(), (_a < 0).mean()), 1 / n_rep))
        return _acme, _ade, _total, _prop, _lo, _hi, _p

    _lines_t = [
        f"**Mediation Analysis: Tumor Cohort** (reference payer: {ref_payer_t})\n",
        "*Controls: age, histology, stage. Bootstrap n=200. ACME = indirect effect through treatment.*\n",
    ]
    for _horizon_t, _outcome_t in [('2-Year', 'event_2yr'), ('5-Year', 'event_5yr')]:
        _rows_t = [
            f"\n**{_horizon_t} Horizon** (`{_outcome_t}`)\n",
            "| Payer (vs ref) | ACME | ADE | Total | Prop. Med. | ACME 95% CI | p(ACME) |",
            "|----------------|------|-----|-------|-----------|-------------|---------|",
        ]
        for _pc_t in payer_cols_t:
            _label_t = _pc_t.replace('payer_', '').replace('_', ' ')
            try:
                _acme, _ade, _total, _prop, _lo, _hi, _p = _med_boot_t(
                    tumor_mediation_df, _outcome_t, _pc_t, _cov_t
                )
                _sig = '\\*' if _p < 0.05 else ''
                _rows_t.append(
                    f"| {_label_t} | {_acme:.4f} | {_ade:.4f} | {_total:.4f} | "
                    f"{_prop:.3f} | [{_lo:.4f}, {_hi:.4f}] | {_p:.4f}{_sig} |"
                )
            except Exception as _e:
                _rows_t.append(f"| {_label_t} | error: {_e} | | | | | |")
        _lines_t.extend(_rows_t)

    mo.md('\n'.join(_lines_t))
    return


@app.cell
def _(
    histology_cols,
    mo,
    np,
    payer_cols_t,
    ref_payer_t,
    stage_cols,
    tumor_mediation_df,
):
    import statsmodels.formula.api as _smf_tols

    _extra_cols_tols = histology_cols + stage_cols
    _cov_tols = 'age_at_diagnosis' + (' + ' + ' + '.join(_extra_cols_tols) if _extra_cols_tols else '')

    def _med_boot_tols(df, payer_col, cov, n_rep=200):
        _acmes, _ades, _totals = [], [], []
        for _ in range(n_rep):
            _b = df.sample(n=len(df), replace=True).reset_index(drop=True)
            try:
                _mfit = _smf_tols.logit(
                    f'treated ~ {payer_col} + {cov}', data=_b
                ).fit(disp=0)
                _ofit = _smf_tols.ols(
                    f'survival_days ~ treated + {payer_col} + {cov}', data=_b
                ).fit()
            except Exception:
                continue
            _d0 = df.assign(**{payer_col: 0})
            _d1 = df.assign(**{payer_col: 1})
            _m0s = np.random.binomial(1, np.clip(_mfit.predict(_d0).values, 0, 1))
            _m1s = np.random.binomial(1, np.clip(_mfit.predict(_d1).values, 0, 1))
            _y00 = float(_ofit.predict(_d0.assign(treated=_m0s)).mean())
            _y01 = float(_ofit.predict(_d0.assign(treated=_m1s)).mean())
            _y10 = float(_ofit.predict(_d1.assign(treated=_m0s)).mean())
            _y11 = float(_ofit.predict(_d1.assign(treated=_m1s)).mean())
            _acmes.append(_y11 - _y10)
            _ades.append(_y11 - _y01)
            _totals.append(_y11 - _y00)
        _a = np.array(_acmes); _d = np.array(_ades); _t = np.array(_totals)
        if len(_a) < 10:
            raise RuntimeError('too many bootstrap failures')
        _acme = float(_a.mean()); _ade = float(_d.mean()); _total = float(_t.mean())
        _prop = _acme / _total if abs(_total) > 1e-10 else float('nan')
        _lo = float(np.percentile(_a, 2.5)); _hi = float(np.percentile(_a, 97.5))
        _p = float(max(2 * min((_a > 0).mean(), (_a < 0).mean()), 1 / n_rep))
        return _acme, _ade, _total, _prop, _lo, _hi, _p

    _lines_tols = [
        f"**Sensitivity: Continuous Outcome (survival_days)** — Tumor Cohort (reference: {ref_payer_t})\n",
        "*Controls: age, histology, stage. ⚠ Censored patients have truncated survival_days.*\n",
        "| Payer (vs ref) | ACME (days) | ADE (days) | Total (days) | Prop. Med. | ACME 95% CI | p(ACME) |",
        "|----------------|-------------|-----------|--------------|-----------|-------------|---------|",
    ]
    for _pc_tols in payer_cols_t:
        _label_tols = _pc_tols.replace('payer_', '').replace('_', ' ')
        try:
            _acme, _ade, _total, _prop, _lo, _hi, _p = _med_boot_tols(
                tumor_mediation_df, _pc_tols, _cov_tols
            )
            _sig = '\\*' if _p < 0.05 else ''
            _lines_tols.append(
                f"| {_label_tols} | {_acme:.1f} | {_ade:.1f} | {_total:.1f} | {_prop:.3f} | "
                f"[{_lo:.1f}, {_hi:.1f}] | {_p:.4f}{_sig} |"
            )
        except Exception as _e:
            _lines_tols.append(f"| {_label_tols} | error: {_e} | | | | | |")

    mo.md('\n'.join(_lines_tols))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Enrollment / Death Signal Consistency

    Two data quality checks on the relationship between enrollment records and death signals:

    1. **Death with active enrollment**: Patients who have a recorded death date but still have an enrollment period *starting* after their death date.
    2. **Enrollment lapse without death**: Patients with no death signal whose last enrollment terminated more than `LAPSE_YEARS` before the end of the observation window — potential unreported deaths or permanent disenrollment.
    """)
    return


@app.cell
def _(mo):
    lapse_years_slider = mo.ui.slider(start=1, stop=5, step=1, value=1, label='Lapse threshold (years)')
    lapse_years_slider
    return (lapse_years_slider,)


@app.cell
def _(
    enrollment_final,
    lapse_years_slider,
    member_enrollment_df,
    mo,
    patient_df,
    pd,
):
    _lapse_years = lapse_years_slider.value

    _cohort_ids = set(enrollment_final['patient_id'])

    _enr = member_enrollment_df[member_enrollment_df['patient_id'].isin(_cohort_ids)][
        ['patient_id', 'effective_date', 'termination_date']
    ].copy()

    _deaths = (
        patient_df[patient_df['patient_id'].isin(_cohort_ids)][['patient_id', 'month_year_death']]
        .copy()
        .loc[lambda df: df['month_year_death'].notna()]
    )

    # ── Check 1: enrollment period starting after death date ────────────────────
    _enr_dead = _enr.merge(_deaths, on='patient_id', how='inner')
    _pts_enr_starts_after = (
        _enr_dead[_enr_dead['effective_date'] > _enr_dead['month_year_death']]
        ['patient_id'].nunique()
    )
    _total_with_death = _deaths['patient_id'].nunique()

    # ── Check 2: no death signal, last enrollment ended > LAPSE_YEARS before obs end
    _dead_ids = set(_deaths['patient_id'])
    _enr_no_death = _enr[~_enr['patient_id'].isin(_dead_ids)]
    _last_term = (
        _enr_no_death.groupby('patient_id')['termination_date'].max()
        .reset_index()
        .rename(columns={'termination_date': 'last_termination'})
    )
    _obs_end = _enr['termination_date'].max()
    _lapse_cutoff = _obs_end - pd.Timedelta(days=int(_lapse_years * 365.25))
    _lapsed = _last_term[_last_term['last_termination'] < _lapse_cutoff]
    _total_no_death = len(_last_term)

    _lines = [
        f"**Cohort:** {len(_cohort_ids):,} patients | "
        f"Observation window end: {_obs_end.date()} | "
        f"Lapse threshold: {_lapse_years} year(s)\n",
        "### Check 1: Death signal with enrollment starting after death\n",
        "| Metric | N | % of patients with death signal |",
        "|--------|---|---------------------------------|",
        f"| Patients with recorded death date | {_total_with_death:,} | 100% |",
        f"| ... with an enrollment period **starting after** death date | "
        f"{_pts_enr_starts_after:,} | {_pts_enr_starts_after / _total_with_death * 100:.1f}% |",
        "",
        f"### Check 2: Enrollment lapse ≥ {_lapse_years} year(s) without death signal\n",
        "| Metric | N | % of patients without death signal |",
        "|--------|---|-------------------------------------|",
        f"| Patients without recorded death date | {_total_no_death:,} | 100% |",
        f"| ... last enrollment ended > {_lapse_years} year(s) before obs window end | "
        f"{len(_lapsed):,} | {len(_lapsed) / _total_no_death * 100:.1f}% |",
    ]

    mo.md('\n'.join(_lines))
    return


@app.cell
def _(enrollment_final, member_enrollment_df, mo, patient_df, pd):
    _cohort_ids2 = set(enrollment_final['patient_id'])

    _enr2 = member_enrollment_df[member_enrollment_df['patient_id'].isin(_cohort_ids2)][
        ['patient_id', 'effective_date', 'termination_date']
    ].copy()

    _dead_ids2 = set(
        patient_df[
            patient_df['patient_id'].isin(_cohort_ids2) &
            patient_df['month_year_death'].notna()
        ]['patient_id']
    )

    _obs_end2 = _enr2['termination_date'].max()
    _one_year_ago = _obs_end2 - pd.Timedelta(days=365)

    # No death signal, and has at least one enrollment period that overlaps the last year:
    # effective_date <= obs_end AND termination_date >= one_year_ago
    _enr_no_death2 = _enr2[~_enr2['patient_id'].isin(_dead_ids2)]
    _active_last_year = (
        _enr_no_death2[
            (_enr_no_death2['effective_date'] <= _obs_end2) &
            (_enr_no_death2['termination_date'] >= _one_year_ago)
        ]['patient_id'].nunique()
    )
    _total_no_death2 = _enr_no_death2['patient_id'].nunique()

    mo.md(
        f"**Diagnostic — No death date, active enrollment in last year of obs window**\n\n"
        f"Observation window end: {_obs_end2.date()} | Last-year window: {_one_year_ago.date()} → {_obs_end2.date()}\n\n"
        f"| Metric | N |\n|--------|---|\n"
        f"| Cohort patients without death date | {_total_no_death2:,} |\n"
        f"| ... with enrollment overlapping last year of obs window | {_active_last_year:,} ({_active_last_year / _total_no_death2 * 100:.1f}%) |"
    )
    return


@app.cell
def _(enrollment_final, go, member_enrollment_df, mo, pd):
    _cohort_ids3 = set(enrollment_final['patient_id'])
    _enr3 = member_enrollment_df[member_enrollment_df['patient_id'].isin(_cohort_ids3)].copy()
    _enr3['duration_days'] = (_enr3['termination_date'] - _enr3['effective_date']).dt.days
    _enr3['duration_months'] = _enr3['duration_days'] / 30.44

    _d = _enr3['duration_days']
    _stats = (
        f"**Enrollment period duration (days)** — {len(_enr3):,} periods across {_enr3['patient_id'].nunique():,} patients\n\n"
        f"| | |\n|---|---|\n"
        f"| Median | {_d.median():.0f} days ({_d.median()/30.44:.1f} mo) |\n"
        f"| Mean | {_d.mean():.0f} days ({_d.mean()/30.44:.1f} mo) |\n"
        f"| Min | {_d.min():.0f} days |\n"
        f"| Max | {_d.max():.0f} days ({_d.max()/365.25:.1f} yr) |\n"
        f"| 25th pct | {_d.quantile(0.25):.0f} days |\n"
        f"| 75th pct | {_d.quantile(0.75):.0f} days |"
    )

    _plot_data = _enr3[_enr3['duration_months'] >= 0].copy()
    _plot_data['duration_years'] = _plot_data['duration_months'] / 12
    _edges_yr = list(range(0, 11))
    _labels_yr = [f'{_edges_yr[i]}–{_edges_yr[i+1]}' for i in range(len(_edges_yr) - 1)] + ['10+']
    _plot_data['bin'] = pd.cut(
        _plot_data['duration_years'],
        bins=_edges_yr + [float('inf')],
        labels=_labels_yr,
        right=False,
    )
    _bin_counts = _plot_data['bin'].value_counts().reindex(_labels_yr).fillna(0).reset_index()
    _bin_counts.columns = ['bin', 'count']

    _fig = go.Figure(go.Bar(
        x=_bin_counts['bin'],
        y=_bin_counts['count'],
        marker_line_width=0.3,
        marker_line_color='white',
    ))
    _fig.update_layout(
        title='Distribution of Enrollment Period Duration',
        xaxis_title='Duration (years)',
        yaxis_title='Number of enrollment periods',
        bargap=0.02,
    )

    mo.vstack([mo.md(_stats), _fig])
    return


@app.cell
def _(enrollment_final, member_enrollment_df, mo):
    _cohort_ids_diag = set(enrollment_final['patient_id'])
    _enr_diag = member_enrollment_df[member_enrollment_df['patient_id'].isin(_cohort_ids_diag)].copy()
    _enr_diag['duration_days'] = (_enr_diag['termination_date'] - _enr_diag['effective_date']).dt.days
    _enr_diag['duration_years'] = (_enr_diag['duration_days'] / 365.25).round(1)
    _top = _enr_diag.sort_values('duration_days', ascending=False).head(300)
    mo.vstack([
        mo.md(f"**Top 300 longest enrollment periods** — max is {_enr_diag['duration_days'].max():,} days ({_enr_diag['duration_days'].max()/365.25:.1f} yr)"),
        mo.ui.table(_top[['patient_id', 'effective_date', 'termination_date', 'duration_days', 'duration_years', 'payer_type', 'plan_design']]),
    ])
    return


@app.cell
def _(enrollment_final, member_enrollment_df, mo):
    _cohort_ids4 = set(enrollment_final['patient_id'])
    _enr4 = member_enrollment_df[member_enrollment_df['patient_id'].isin(_cohort_ids4)].copy()
    _enr4['duration_days'] = (_enr4['termination_date'] - _enr4['effective_date']).dt.days

    _negative = _enr4[_enr4['duration_days'] < 0].sort_values('duration_days')
    _over_24m = _enr4[_enr4['duration_days'] > 24 * 30.44].sort_values('duration_days', ascending=False)

    mo.vstack([
        mo.md(f"**Negative duration periods: {len(_negative):,}**"),
        mo.ui.table(_negative),
        mo.md(f"**Periods over 24 months: {len(_over_24m):,}**"),
        mo.ui.table(_over_24m),
    ])
    return


@app.cell
def _(enrollment_final, member_enrollment_df, mo):
    _cohort_ids5 = set(enrollment_final['patient_id'])
    _enr5 = member_enrollment_df[member_enrollment_df['patient_id'].isin(_cohort_ids5)].copy()
    _enr5['duration_years'] = (_enr5['termination_date'] - _enr5['effective_date']).dt.days / 365.25

    _long = _enr5[_enr5['duration_years'] > 6]
    _grouped = (
        _long.groupby(['effective_date', 'termination_date'], as_index=False)
        .size()
        .rename(columns={'size': 'count'})
        .sort_values('count', ascending=False)
    )
    _grouped['duration_years'] = ((_grouped['termination_date'] - _grouped['effective_date']).dt.days / 365.25).round(1)
    mo.vstack([
        mo.md(f"**Enrollment periods with duration > 6 years: {len(_long):,} records — {_grouped['effective_date'].nunique()} unique (effective, termination) pairs**"),
        mo.ui.table(_grouped[['effective_date', 'termination_date', 'duration_years', 'count']]),
    ])
    return


if __name__ == "__main__":
    app.run()
