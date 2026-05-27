"""
ECS TechOps - Shift Capacity Planning Web Application
=====================================================
Streamlit-based web app for partner leads to self-service capacity forecasting.
Upload your SPC Excel export and get instant demand forecast + capacity plan.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import io
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="ECS TechOps - Shift Capacity Planning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING (Enhanced: handles multi-track, mixed formats)
# =============================================================================

def load_excel_data(uploaded_file):
    """Load partner data from SPC Excel export, auto-detecting format.
    
    Handles:
    - Single-track files (one sheet per month)
    - Multi-track files (Basis/SM/DB as separate sheets)
    - Mixed formats within a sheet (header row 0 + header row 4 blocks)
    - Repeated headers within monthly blocks
    - NTT/other partner contamination
    """
    xls = pd.ExcelFile(uploaded_file)
    tracks_found = {}
    
    for sheet in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        
        # Find all header rows in this sheet (blocks of monthly data)
        header_positions = []
        for idx in range(len(raw)):
            row_vals = [str(v) for v in raw.iloc[idx] if pd.notna(v)]
            if 'Team ID' in row_vals:
                header_positions.append(idx)
        
        if not header_positions:
            continue
        
        # Parse each block
        sheet_frames = []
        for block_idx, hdr_row in enumerate(header_positions):
            # Determine end of this block
            end_row = header_positions[block_idx + 1] if block_idx + 1 < len(header_positions) else len(raw)
            
            # Find where data starts (skip blank rows between header and data)
            # Determine column offset (some formats have extra NaN column 0)
            header_vals = raw.iloc[hdr_row].tolist()
            col_offset = 0
            for ci, val in enumerate(header_vals):
                if str(val).strip() == 'Team ID':
                    col_offset = ci
                    break
            
            # Extract block
            chunk = raw.iloc[hdr_row:end_row].copy().reset_index(drop=True)
            
            # Set proper column names from header row
            chunk.columns = chunk.iloc[0]
            chunk = chunk.iloc[1:]
            
            # Drop unnamed/NaN columns
            chunk = chunk.loc[:, chunk.columns.notna()]
            chunk = chunk.loc[:, ~chunk.columns.astype(str).str.startswith('Unnamed')]
            
            # Filter to valid data rows
            if 'Team ID' in chunk.columns and 'Shift' in chunk.columns:
                chunk = chunk[chunk['Shift'].isin(['S1', 'S2', 'S3'])]
                chunk = chunk[chunk['Team ID'].astype(str).str.startswith('ACE')]
            
            if len(chunk) > 0:
                sheet_frames.append(chunk)
        
        if sheet_frames:
            sheet_df = pd.concat(sheet_frames, ignore_index=True)
            
            # Identify unique team IDs in this sheet
            team_ids = sheet_df['Team ID'].value_counts()
            
            for tid in team_ids.index:
                tdf = sheet_df[sheet_df['Team ID'] == tid].copy()
                if len(tdf) < 20:  # Skip if too few rows
                    continue
                
                # Determine track from team name
                team_name = tdf['Team Name'].iloc[0] if 'Team Name' in tdf.columns else ''
                track = detect_track(team_name)
                partner = detect_partner(team_name)
                
                key = f"{partner}|{track}|{tid}"
                if key not in tracks_found:
                    tracks_found[key] = {
                        'data': tdf,
                        'team_id': tid,
                        'team_name': team_name,
                        'track': track,
                        'partner': partner,
                        'sheet': sheet,
                        'rows': len(tdf),
                    }
                else:
                    # Combine data from multiple sheets/blocks for same track
                    tracks_found[key]['data'] = pd.concat(
                        [tracks_found[key]['data'], tdf], ignore_index=True)
                    tracks_found[key]['rows'] = len(tracks_found[key]['data'])
    
    return tracks_found


def detect_track(team_name):
    """Detect track from team name."""
    team_upper = str(team_name).upper()
    if 'SM L2' in team_upper or 'SM' in team_upper.split(':')[1] if ':' in team_upper else False:
        return 'SM'
    elif 'DB' in team_upper:
        return 'DB'
    else:
        return 'Basis'


def detect_partner(team_name):
    """Extract partner name from team name."""
    name = str(team_name)
    for prefix in ['ECS SR Delivery: Basis - ', 'ECS SR Delivery: SM L2 - ',
                   'ECS SR Delivery: SM - ', 'ECS SR Delivery: DB - ',
                   'ECS SR Delivery: ']:
        if name.startswith(prefix):
            return name[len(prefix):]
    # Fallback: take everything after the last ' - '
    if ' - ' in name:
        return name.split(' - ')[-1]
    return name


def prepare_data(df):
    """Add computed columns for analysis."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Ensure numeric columns
    df['Ʃ Total Demand Hour(s)'] = pd.to_numeric(df['Ʃ Total Demand Hour(s)'], errors='coerce')
    df['Ʃ Total Capacity Hour(s)'] = pd.to_numeric(df['Ʃ Total Capacity Hour(s)'], errors='coerce')
    df = df.dropna(subset=['Ʃ Total Demand Hour(s)', 'Ʃ Total Capacity Hour(s)'])
    
    df['DateOnly'] = df['Date'].dt.date
    df['DayOfWeekNum'] = df['Date'].dt.dayofweek
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['MonthNum'] = df['Date'].dt.month
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df['WeekOfMonth'] = ((df['Date'].dt.day - 1) // 7) + 1
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    shift_hours = {'S1': 8, 'S2': 9, 'S3': 7}
    df['Shift_Hours'] = df['Shift'].map(shift_hours)
    df['Executors'] = df['Ʃ Total Capacity Hour(s)'] / df['Shift_Hours']
    df['Utilization_Pct'] = (df['Ʃ Total Demand Hour(s)'] / df['Ʃ Total Capacity Hour(s)']) * 100
    
    return df


# =============================================================================
# DEMAND FORECASTING
# =============================================================================

class DemandForecaster:
    """Forecasts demand based on day-of-week + shift + week-of-month patterns."""
    
    def __init__(self, data):
        self.data = data
        self.profiles = {}
        self.wom_factors = {}
        self.trend_factor = 1.0
        self._fit()
    
    def _fit(self):
        for shift in ['S1', 'S2', 'S3']:
            for dow in range(7):
                subset = self.data[(self.data['Shift'] == shift) & (self.data['DayOfWeekNum'] == dow)]
                if len(subset) > 0:
                    self.profiles[(dow, shift)] = {
                        'mean': subset['Ʃ Total Demand Hour(s)'].mean(),
                        'std': subset['Ʃ Total Demand Hour(s)'].std(),
                        'p75': subset['Ʃ Total Demand Hour(s)'].quantile(0.75),
                        'p90': subset['Ʃ Total Demand Hour(s)'].quantile(0.90),
                        'p95': subset['Ʃ Total Demand Hour(s)'].quantile(0.95),
                    }
        
        overall_mean = self.data['Ʃ Total Demand Hour(s)'].mean()
        for wom in range(1, 6):
            subset = self.data[self.data['WeekOfMonth'] == wom]
            if len(subset) > 0:
                self.wom_factors[wom] = subset['Ʃ Total Demand Hour(s)'].mean() / overall_mean
            else:
                self.wom_factors[wom] = 1.0
        
        months = sorted(self.data['YearMonth'].unique())
        if len(months) >= 4:
            recent = self.data[self.data['YearMonth'].isin(months[-3:])]
            earlier = self.data[self.data['YearMonth'].isin(months[:3])]
            if len(earlier) > 0 and earlier['Ʃ Total Demand Hour(s)'].mean() > 0:
                self.trend_factor = recent['Ʃ Total Demand Hour(s)'].mean() / earlier['Ʃ Total Demand Hour(s)'].mean()
            self.trend_factor = min(max(self.trend_factor, 0.8), 1.3)
    
    def forecast(self, date, shift):
        dow = date.weekday()
        wom = min(((date.day - 1) // 7) + 1, 5)
        
        profile = self.profiles.get((dow, shift))
        if profile is None:
            return {'mean': 50, 'std': 15, 'p75': 60, 'p90': 70, 'p95': 75}
        
        wom_adj = self.wom_factors.get(wom, 1.0)
        trend_adj = min(self.trend_factor, 1.15)
        
        return {
            'mean': profile['mean'] * wom_adj * trend_adj,
            'std': profile['std'],
            'p75': profile['p75'] * wom_adj * trend_adj,
            'p90': profile['p90'] * wom_adj * trend_adj,
            'p95': profile['p95'] * wom_adj * trend_adj,
        }


# =============================================================================
# CAPACITY PLANNING
# =============================================================================

def compute_capacity_plan(forecaster, start_date, days=90, target_util=0.65):
    shift_hours = {'S1': 8, 'S2': 9, 'S3': 7}
    plan = []
    
    for d in range(days):
        date = start_date + timedelta(days=d)
        for shift in ['S1', 'S2', 'S3']:
            fc = forecaster.forecast(date, shift)
            sh = shift_hours[shift]
            
            required_cap = fc['p90'] / target_util
            recommended_exec = int(np.ceil(required_cap / sh))
            
            required_cap_cons = fc['p95'] / target_util
            conservative_exec = int(np.ceil(required_cap_cons / sh))
            
            min_cap = fc['mean'] / 0.70
            min_exec = int(np.ceil(min_cap / sh))
            
            total_cap = recommended_exec * sh
            util_mean = (fc['mean'] / total_cap) * 100 if total_cap > 0 else 0
            util_p90 = (fc['p90'] / total_cap) * 100 if total_cap > 0 else 0
            buffer = total_cap - fc['p90']
            
            plan.append({
                'Date': date,
                'Day': date.strftime('%A'),
                'DayOfWeekNum': date.weekday(),
                'Shift': shift,
                'Shift_Timing': {'S1': '6AM-2PM IST', 'S2': '2PM-11PM IST', 'S3': '11PM-6AM IST'}[shift],
                'Forecast_Mean': round(fc['mean'], 1),
                'Forecast_P75': round(fc['p75'], 1),
                'Forecast_P90': round(fc['p90'], 1),
                'Forecast_P95': round(fc['p95'], 1),
                'Min_Executors': min_exec,
                'Recommended_Executors': recommended_exec,
                'Conservative_Executors': conservative_exec,
                'Total_Capacity_Hrs': total_cap,
                'Expected_Util_Mean': round(util_mean, 1),
                'Expected_Util_P90': round(util_p90, 1),
                'Buffer_Hours': round(buffer, 1),
            })
    
    return pd.DataFrame(plan)


# =============================================================================
# HTML DASHBOARD GENERATION
# =============================================================================

def generate_html_dashboard(data, plan_df, forecaster, team_id, team_name, partner_name):
    """Generate interactive HTML dashboard and return as string."""
    
    shift_hours_map = {'S1': 8, 'S2': 9, 'S3': 7}
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    colors = {
        'primary': '#0070F2', 'secondary': '#4CB1FF', 'success': '#36A41D',
        'warning': '#E76500', 'danger': '#BB0000', 'S1': '#0070F2', 'S2': '#E76500', 'S3': '#36A41D',
    }
    
    # Chart 1: Demand by Day
    fig_dow = go.Figure()
    for shift in ['S1', 'S2', 'S3']:
        s_data = data[data['Shift'] == shift]
        means = s_data.groupby('DayOfWeekNum')['Ʃ Total Demand Hour(s)'].mean()
        fig_dow.add_trace(go.Bar(name=shift, x=dow_names,
            y=[means.get(i, 0) for i in range(7)], marker_color=colors[shift]))
    
    avg_cap = data.groupby('DayOfWeekNum')['Ʃ Total Capacity Hour(s)'].mean()
    threshold_line = [avg_cap.get(i, 0) * 0.70 for i in range(7)]
    fig_dow.add_trace(go.Scatter(name='70% Threshold', x=dow_names, y=threshold_line,
        mode='lines+markers', line=dict(color=colors['danger'], dash='dash', width=2)))
    fig_dow.update_layout(title=dict(text='Historical Demand by Day & Shift', x=0.5),
        barmode='group', yaxis_title='Demand Hours', height=380, margin=dict(t=60, b=40, l=60, r=20),
        legend=dict(orientation='h', y=-0.15))
    
    # Chart 2: Resource Gap
    gap_data = []
    for dow in range(7):
        for shift in ['S1', 'S2', 'S3']:
            current = data[(data['DayOfWeekNum'] == dow) & (data['Shift'] == shift)]['Executors'].mean()
            recommended = plan_df[(plan_df['DayOfWeekNum'] == dow) & (plan_df['Shift'] == shift)]['Recommended_Executors'].mean()
            gap_data.append({'Day': dow_names[dow][:3], 'Shift': shift, 'Gap': recommended - current})
    
    gap_df = pd.DataFrame(gap_data)
    fig_gap = go.Figure()
    for shift in ['S1', 'S2', 'S3']:
        s_gap = gap_df[gap_df['Shift'] == shift]
        fig_gap.add_trace(go.Bar(name=shift, x=s_gap['Day'], y=s_gap['Gap'],
            marker_color=colors[shift],
            text=[f'+{v:.0f}' if v > 0 else f'{v:.0f}' for v in s_gap['Gap']], textposition='outside'))
    fig_gap.update_layout(title=dict(text='Resource GAP: Additional Executors Needed', x=0.5),
        barmode='group', yaxis_title='Additional Executors', height=380,
        margin=dict(t=60, b=40, l=60, r=20), legend=dict(orientation='h', y=-0.15))
    fig_gap.add_hline(y=0, line_color='black', line_width=1)
    
    # KPIs
    total_shifts = len(data)
    pct_exceeded = len(data[data['Total Average Utilization Status'] == 'Exceeded']) / total_shifts * 100
    pct_threshold = len(data[data['Total Average Utilization Status'] == 'Threshold Exceeded']) / total_shifts * 100
    pct_sufficient = len(data[data['Total Average Utilization Status'] == 'Sufficient']) / total_shifts * 100
    avg_util = data['Utilization_Pct'].mean()
    avg_demand = data['Ʃ Total Demand Hour(s)'].mean()
    avg_cap_val = data['Ʃ Total Capacity Hour(s)'].mean()
    avg_exec = data['Executors'].mean()
    cap_demand_ratio = avg_cap_val / avg_demand
    
    sim_sufficient = len(plan_df[plan_df['Expected_Util_P90'] < 70])
    sim_pct_ok = sim_sufficient / len(plan_df) * 100
    
    # Weekly template table
    weekly_template = plan_df.groupby(['DayOfWeekNum', 'Day', 'Shift']).agg({
        'Min_Executors': 'mean', 'Recommended_Executors': 'mean',
        'Conservative_Executors': 'mean', 'Forecast_Mean': 'mean', 'Forecast_P90': 'mean',
    }).reset_index().round(0).sort_values(['DayOfWeekNum', 'Shift'])
    
    table_rows = ""
    for _, row in weekly_template.iterrows():
        shift_time = {'S1': '6AM-2PM', 'S2': '2PM-11PM', 'S3': '11PM-6AM'}[row['Shift']]
        current_avg = data[(data['DayOfWeekNum'] == row['DayOfWeekNum']) & (data['Shift'] == row['Shift'])]['Executors'].mean()
        gap = row['Recommended_Executors'] - current_avg
        gap_color = colors['danger'] if gap > 3 else (colors['warning'] if gap > 1 else colors['success'])
        table_rows += f"""
        <tr>
            <td>{row['Day']}</td>
            <td>{row['Shift']} ({shift_time})</td>
            <td>{current_avg:.1f}</td>
            <td><strong>{int(row['Recommended_Executors'])}</strong></td>
            <td style="color:{gap_color}; font-weight:bold;">{'+' if gap > 0 else ''}{gap:.1f}</td>
            <td>{int(row['Forecast_Mean'])}h</td>
            <td>{int(row['Forecast_P90'])}h</td>
        </tr>"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECS TechOps - Shift Capacity Report | {partner_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: '72', 'Segoe UI', Arial, sans-serif; background: #F5F6F7; color: #000; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #0070F2, #0054B4); color: white; padding: 30px 40px; border-radius: 12px; margin-bottom: 24px; }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .subtitle {{ font-size: 14px; opacity: 0.9; }}
        .header .meta {{ display: flex; gap: 30px; margin-top: 12px; font-size: 13px; opacity: 0.85; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-top: 3px solid #0070F2; }}
        .kpi-card.danger {{ border-top-color: #BB0000; }}
        .kpi-card.warning {{ border-top-color: #E76500; }}
        .kpi-card.success {{ border-top-color: #36A41D; }}
        .kpi-card .value {{ font-size: 28px; font-weight: bold; margin: 8px 0; }}
        .kpi-card .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 24px; }}
        .chart-card {{ background: white; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .section-title {{ font-size: 18px; font-weight: bold; margin: 30px 0 16px 0; padding-bottom: 8px; border-bottom: 2px solid #0070F2; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ background: #0070F2; color: white; padding: 10px 12px; text-align: center; }}
        td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #E1E2E6; }}
        tr:nth-child(even) {{ background: #F8F9FA; }}
        tr:hover {{ background: #E1F4FF; }}
        .insight-box {{ background: #E1F4FF; border-left: 4px solid #0070F2; padding: 16px 20px; border-radius: 0 8px 8px 0; margin: 16px 0; font-size: 14px; }}
        .insight-box.alert {{ background: #FFF3E0; border-left-color: #E76500; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; font-size: 12px; color: #666; }}
        @media (max-width: 900px) {{ .chart-grid {{ grid-template-columns: 1fr; }} .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    </style>
</head>
<body>
<div class="header">
    <h1>ECS TechOps - Shift Capacity Planning Report</h1>
    <div class="subtitle">{team_name} | Team ID: {team_id}</div>
    <div class="meta">
        <span>Analysis Period: {data['Date'].min().strftime('%b %Y')} - {data['Date'].max().strftime('%b %Y')}</span>
        <span>Total Shifts Analyzed: {total_shifts}</span>
        <span>Report Generated: {datetime.now().strftime('%d %b %Y')}</span>
    </div>
</div>
<div class="kpi-grid">
    <div class="kpi-card danger"><div class="label">Exceeded Shifts</div><div class="value">{pct_exceeded:.1f}%</div></div>
    <div class="kpi-card warning"><div class="label">Threshold Exceeded</div><div class="value">{pct_threshold:.1f}%</div></div>
    <div class="kpi-card success"><div class="label">Sufficient Shifts</div><div class="value">{pct_sufficient:.1f}%</div></div>
    <div class="kpi-card"><div class="label">Avg Utilization</div><div class="value">{avg_util:.0f}%</div></div>
    <div class="kpi-card"><div class="label">Cap/Demand Ratio</div><div class="value">{cap_demand_ratio:.2f}x</div></div>
    <div class="kpi-card"><div class="label">Avg Executors</div><div class="value">{avg_exec:.1f}</div></div>
    <div class="kpi-card success"><div class="label">Projected Slot Avail.</div><div class="value">{sim_pct_ok:.0f}%</div></div>
</div>
<div class="insight-box alert">
    <strong>Key Finding:</strong> {pct_exceeded + pct_threshold:.0f}% of shifts are currently problematic.
    With the recommended executor allocation, projected slot availability improves to {sim_pct_ok:.0f}%.
</div>
<div class="section-title">Demand Patterns & Resource Gap</div>
<div class="chart-grid">
    <div class="chart-card"><div id="chart_dow"></div></div>
    <div class="chart-card"><div id="chart_gap"></div></div>
</div>
<div class="section-title">Weekly Executor Planning Template</div>
<div class="insight-box">
    Use this table as the baseline for weekly shift planning. The "Recommended" column ensures
    90% of shifts remain below the 70% threshold.
</div>
<div class="chart-card" style="margin-top:16px; overflow-x:auto;">
    <table>
        <thead><tr><th>Day</th><th>Shift</th><th>Current Avg</th><th>Recommended</th><th>Gap (+)</th><th>Forecast Mean</th><th>Forecast P90</th></tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
</div>
<div class="footer">
    ECS TechOps Shift Capacity Planning Tool | Model: Day-of-Week + Shift + Week-of-Month | Target: P90 demand at 65% utilization
</div>
<script>
    var chart_dow = {fig_dow.to_json()};
    var chart_gap = {fig_gap.to_json()};
    Plotly.newPlot('chart_dow', chart_dow.data, chart_dow.layout, {{responsive: true}});
    Plotly.newPlot('chart_gap', chart_gap.data, chart_gap.layout, {{responsive: true}});
</script>
</body>
</html>"""
    
    return html_content


# =============================================================================
# EXCEL OUTPUT
# =============================================================================

def generate_excel_output(data, plan_df, forecaster):
    """Generate Excel capacity plan and return as bytes."""
    output = io.BytesIO()
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: 90-Day Plan
        plan_export = plan_df[['Date', 'Day', 'Shift', 'Shift_Timing',
                               'Recommended_Executors', 'Min_Executors', 'Conservative_Executors',
                               'Total_Capacity_Hrs', 'Forecast_Mean', 'Forecast_P90', 'Forecast_P95',
                               'Expected_Util_Mean', 'Expected_Util_P90', 'Buffer_Hours']].copy()
        plan_export.to_excel(writer, sheet_name='90Day_Capacity_Plan', index=False)
        
        # Sheet 2: Weekly Template
        template_data = []
        for dow in range(7):
            for shift in ['S1', 'S2', 'S3']:
                subset = plan_df[(plan_df['DayOfWeekNum'] == dow) & (plan_df['Shift'] == shift)]
                hist = data[(data['DayOfWeekNum'] == dow) & (data['Shift'] == shift)]
                template_data.append({
                    'Day': dow_names[dow], 'Shift': shift,
                    'Shift_Timing': {'S1': '6AM-2PM IST', 'S2': '2PM-11PM IST', 'S3': '11PM-6AM IST'}[shift],
                    'Current_Avg_Executors': round(hist['Executors'].mean(), 1) if len(hist) > 0 else 0,
                    'Recommended_Executors': int(subset['Recommended_Executors'].mean()),
                    'Conservative_Executors': int(subset['Conservative_Executors'].mean()),
                    'Gap': round(subset['Recommended_Executors'].mean() - hist['Executors'].mean(), 1) if len(hist) > 0 else 0,
                    'Avg_Demand_Hrs': round(subset['Forecast_Mean'].mean(), 1),
                    'P90_Demand_Hrs': round(subset['Forecast_P90'].mean(), 1),
                })
        pd.DataFrame(template_data).to_excel(writer, sheet_name='Weekly_Template', index=False)
        
        # Sheet 3: Weekly Summary
        plan_df_copy = plan_df.copy()
        plan_df_copy['Week_Start'] = plan_df_copy['Date'] - pd.to_timedelta(plan_df_copy['Date'].dt.dayofweek, unit='D')
        weekly_agg = plan_df_copy.groupby(['Week_Start', 'Shift']).agg({
            'Recommended_Executors': ['mean', 'max'], 'Forecast_Mean': 'mean',
            'Forecast_P90': 'mean', 'Expected_Util_Mean': 'mean',
        }).round(1)
        weekly_agg.columns = ['Avg_Executors', 'Peak_Executors', 'Avg_Demand', 'P90_Demand', 'Avg_Util%']
        weekly_agg.reset_index().to_excel(writer, sheet_name='Weekly_Summary', index=False)
        
        # Sheet 4: Historical Analysis
        hist_summary = []
        for dow in range(7):
            for shift in ['S1', 'S2', 'S3']:
                subset = data[(data['DayOfWeekNum'] == dow) & (data['Shift'] == shift)]
                if len(subset) > 0:
                    hist_summary.append({
                        'Day': dow_names[dow], 'Shift': shift,
                        'Demand_Mean': round(subset['Ʃ Total Demand Hour(s)'].mean(), 1),
                        'Demand_Std': round(subset['Ʃ Total Demand Hour(s)'].std(), 1),
                        'Demand_P90': round(subset['Ʃ Total Demand Hour(s)'].quantile(0.9), 1),
                        'Capacity_Mean': round(subset['Ʃ Total Capacity Hour(s)'].mean(), 1),
                        'Utilization_Mean%': round(subset['Utilization_Pct'].mean(), 1),
                        'Current_Executors_Avg': round(subset['Executors'].mean(), 1),
                        'Exceeded_Count': len(subset[subset['Total Average Utilization Status'] == 'Exceeded']),
                        'Threshold_Count': len(subset[subset['Total Average Utilization Status'] == 'Threshold Exceeded']),
                        'Total_Shifts': len(subset),
                    })
        pd.DataFrame(hist_summary).to_excel(writer, sheet_name='Historical_Analysis', index=False)
        
        # Sheet 5: Monthly Trend
        monthly = data.groupby('YearMonth').agg({
            'Ʃ Total Demand Hour(s)': ['mean', 'sum'], 'Ʃ Total Capacity Hour(s)': ['mean', 'sum'],
            'Utilization_Pct': 'mean', 'Executors': 'mean',
        }).round(1)
        monthly.columns = ['Avg_Demand', 'Total_Demand', 'Avg_Capacity', 'Total_Capacity', 'Avg_Util%', 'Avg_Executors']
        monthly.reset_index().to_excel(writer, sheet_name='Monthly_Trend', index=False)
    
    return output.getvalue()


# =============================================================================
# STREAMLIT UI
# =============================================================================

def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header { font-size: 28px; font-weight: bold; color: #0070F2; margin-bottom: 4px; }
        .sub-header { font-size: 14px; color: #666; margin-bottom: 20px; }
        .track-card { background: #F7FAFF; border: 1px solid #E1E2E6; border-radius: 8px; padding: 16px; margin: 8px 0; }
        .metric-row { display: flex; gap: 16px; flex-wrap: wrap; }
        .stDownloadButton button { background-color: #0070F2 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">ECS TechOps - Shift Capacity Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload your SPC Excel export to generate demand forecast and capacity plan</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.sap.com/dam/application/shared/logos/sap-logo-svg.svg", width=100)
        st.markdown("### Instructions")
        st.markdown("""
        1. Download the **Capacity Demand Overview** export from SPC (6-7 months of data)
        2. Upload the Excel file below
        3. Select the track to analyze
        4. Download the HTML dashboard and Excel plan
        
        ---
        **Supported Formats:**
        - Single-track files (one partner, one track)
        - Multi-track files (Basis/SM/DB sheets)
        - All 9 MSM partners
        
        ---
        **Shift Schedule:**
        - S1: 6AM - 2PM IST (8h)
        - S2: 2PM - 11PM IST (9h)
        - S3: 11PM - 6AM IST (7h)
        """)
        
        st.markdown("---")
        st.markdown("**Target:** P90 demand at 65% utilization")
        target_util = st.slider("Target Utilization %", 50, 80, 65, 5) / 100
        forecast_days = st.slider("Forecast Days", 30, 120, 90, 15)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload SPC Excel Export",
        type=['xlsx', 'xls'],
        help="Upload the Capacity Demand Overview export from SPC. Supports single or multi-track files."
    )
    
    if uploaded_file is not None:
        with st.spinner("Analyzing file structure..."):
            tracks_found = load_excel_data(uploaded_file)
        
        if not tracks_found:
            st.error("Could not find valid shift data in the uploaded file. Please ensure it contains columns: Team ID, Date, Shift, Capacity Hours, Demand Hours.")
            return
        
        # Show detected tracks
        st.success(f"Detected **{len(tracks_found)} track(s)** in the uploaded file")
        
        # Let user select which track to analyze
        track_options = {}
        for key, info in tracks_found.items():
            label = f"{info['partner']} - {info['track']} ({info['rows']} shifts, {info['team_id']})"
            track_options[label] = key
        
        if len(track_options) == 1:
            selected_label = list(track_options.keys())[0]
            st.info(f"Processing: **{selected_label}**")
        else:
            selected_label = st.selectbox(
                "Select track to analyze:",
                options=list(track_options.keys()),
                help="Choose which partner/track combination to generate the forecast for"
            )
        
        selected_key = track_options[selected_label]
        track_info = tracks_found[selected_key]
        
        # Process button
        if st.button("Generate Forecast & Capacity Plan", type="primary", use_container_width=True):
            with st.spinner("Building demand model and generating forecast..."):
                # Prepare data
                data = prepare_data(track_info['data'])
                
                if len(data) < 30:
                    st.error(f"Insufficient data: only {len(data)} valid shifts found. Need at least 30 for reliable forecasting.")
                    return
                
                # Build forecast
                forecaster = DemandForecaster(data)
                
                # Generate plan
                start_date = data['Date'].max().normalize() + timedelta(days=1)
                plan_df = compute_capacity_plan(forecaster, start_date.to_pydatetime(), 
                                                days=forecast_days, target_util=target_util)
                
                # Store in session state
                st.session_state['data'] = data
                st.session_state['plan_df'] = plan_df
                st.session_state['forecaster'] = forecaster
                st.session_state['track_info'] = track_info
                st.session_state['processed'] = True
        
        # Display results if processed
        if st.session_state.get('processed', False):
            data = st.session_state['data']
            plan_df = st.session_state['plan_df']
            forecaster = st.session_state['forecaster']
            track_info = st.session_state['track_info']
            
            partner_name = track_info['partner']
            team_id = track_info['team_id']
            team_name = track_info['team_name']
            track = track_info['track']
            
            st.markdown("---")
            st.markdown(f"### Results: {partner_name} - {track}")
            
            # KPI Metrics
            total_shifts = len(data)
            pct_exceeded = len(data[data['Total Average Utilization Status'] == 'Exceeded']) / total_shifts * 100
            pct_threshold = len(data[data['Total Average Utilization Status'] == 'Threshold Exceeded']) / total_shifts * 100
            pct_sufficient = len(data[data['Total Average Utilization Status'] == 'Sufficient']) / total_shifts * 100
            avg_util = data['Utilization_Pct'].mean()
            cap_ratio = data['Ʃ Total Capacity Hour(s)'].mean() / data['Ʃ Total Demand Hour(s)'].mean()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Exceeded", f"{pct_exceeded:.1f}%")
            col2.metric("Threshold Exc.", f"{pct_threshold:.1f}%")
            col3.metric("Sufficient", f"{pct_sufficient:.1f}%")
            col4.metric("Avg Utilization", f"{avg_util:.0f}%")
            col5.metric("Cap/Demand Ratio", f"{cap_ratio:.2f}x")
            
            st.markdown(f"**Period:** {data['Date'].min().strftime('%b %Y')} - {data['Date'].max().strftime('%b %Y')} | "
                       f"**Shifts analyzed:** {total_shifts} | **Trend factor:** {forecaster.trend_factor:.3f}")
            
            # Charts
            col_left, col_right = st.columns(2)
            
            with col_left:
                dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                fig_dow = go.Figure()
                for shift in ['S1', 'S2', 'S3']:
                    s_data = data[data['Shift'] == shift]
                    means = s_data.groupby('DayOfWeekNum')['Ʃ Total Demand Hour(s)'].mean()
                    fig_dow.add_trace(go.Bar(name=shift, x=dow_names,
                        y=[means.get(i, 0) for i in range(7)]))
                avg_cap_day = data.groupby('DayOfWeekNum')['Ʃ Total Capacity Hour(s)'].mean()
                fig_dow.add_trace(go.Scatter(name='70% Threshold', x=dow_names,
                    y=[avg_cap_day.get(i, 0) * 0.70 for i in range(7)],
                    mode='lines+markers', line=dict(color='red', dash='dash', width=2)))
                fig_dow.update_layout(title='Demand by Day & Shift', barmode='group',
                    yaxis_title='Hours', height=350, margin=dict(t=40, b=20))
                st.plotly_chart(fig_dow, use_container_width=True)
            
            with col_right:
                gap_data = []
                for dow in range(7):
                    for shift in ['S1', 'S2', 'S3']:
                        current = data[(data['DayOfWeekNum'] == dow) & (data['Shift'] == shift)]['Executors'].mean()
                        recommended = plan_df[(plan_df['DayOfWeekNum'] == dow) & (plan_df['Shift'] == shift)]['Recommended_Executors'].mean()
                        gap_data.append({'Day': dow_names[dow][:3], 'Shift': shift, 'Gap': recommended - current})
                gap_df_chart = pd.DataFrame(gap_data)
                fig_gap = go.Figure()
                for shift in ['S1', 'S2', 'S3']:
                    s_gap = gap_df_chart[gap_df_chart['Shift'] == shift]
                    fig_gap.add_trace(go.Bar(name=shift, x=s_gap['Day'], y=s_gap['Gap']))
                fig_gap.add_hline(y=0, line_color='black', line_width=1)
                fig_gap.update_layout(title='Resource Gap (Recommended - Current)', barmode='group',
                    yaxis_title='Executors', height=350, margin=dict(t=40, b=20))
                st.plotly_chart(fig_gap, use_container_width=True)
            
            # Weekly Template Table
            st.markdown("#### Weekly Executor Planning Template")
            template_data = []
            for dow in range(7):
                for shift in ['S1', 'S2', 'S3']:
                    subset = plan_df[(plan_df['DayOfWeekNum'] == dow) & (plan_df['Shift'] == shift)]
                    hist = data[(data['DayOfWeekNum'] == dow) & (data['Shift'] == shift)]
                    current = hist['Executors'].mean() if len(hist) > 0 else 0
                    recommended = int(subset['Recommended_Executors'].mean())
                    gap = recommended - current
                    template_data.append({
                        'Day': dow_names[dow], 'Shift': shift,
                        'Current': round(current, 1), 'Recommended': recommended,
                        'Gap': f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}",
                        'Forecast P90': f"{subset['Forecast_P90'].mean():.0f}h"
                    })
            
            st.dataframe(pd.DataFrame(template_data), use_container_width=True, hide_index=True)
            
            # Downloads
            st.markdown("---")
            st.markdown("#### Download Outputs")
            
            dl_col1, dl_col2 = st.columns(2)
            
            with dl_col1:
                html_content = generate_html_dashboard(data, plan_df, forecaster, team_id, team_name, partner_name)
                safe_name = re.sub(r'[<>:"/\\|?*]', '', partner_name).strip().replace(' ', '_')
                st.download_button(
                    label="Download HTML Dashboard",
                    data=html_content.encode('utf-8'),
                    file_name=f"ECS_TechOps_Report_{safe_name}_{track}.html",
                    mime="text/html",
                    use_container_width=True
                )
            
            with dl_col2:
                excel_bytes = generate_excel_output(data, plan_df, forecaster)
                st.download_button(
                    label="Download Excel Capacity Plan",
                    data=excel_bytes,
                    file_name=f"ECS_TechOps_Plan_{safe_name}_{track}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


if __name__ == '__main__':
    main()
