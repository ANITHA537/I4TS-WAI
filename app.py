import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optimization import run_optimization
from ai_model import train_delay_model, predict_delay_risk
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, mean_squared_error, roc_curve, precision_score, recall_score, f1_score

# --- Page Config ---
st.set_page_config(
    page_title="Intelligent Logistics - AI Optimization",
    page_icon="🚛",
    layout="wide"
)

# --- Corporate / Professional Theme Styling ---
st.markdown("""
<style>
    /* Professional Corporate Palette */
    :root {
        --primary-color: #0a6ed1; /* Deep Blue */
        --secondary-color: #0854a0;
        --accent-color: #f0ab00; /* Gold/Amber */
        --bg-color: #f5f7fa;
        --card-bg: #ffffff;
        --text-color: #32363a;
        --text-light: #6a6d70;
        --font-family: "Segoe UI", "Helvetica", "Arial", sans-serif;
        --header-bg: #2c3e50; /* Dark Slate */
    }
    
    /* Global App Styling */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: var(--font-family);
    }
    
    /* Custom Shell Header */
    .app-shell-header {
        background-color: var(--header-bg);
        color: white;
        padding: 1rem 2rem;
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        border-bottom: 4px solid var(--accent-color);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .app-shell-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-left: 1rem;
        letter-spacing: 0.5px;
    }
    
    /* Streamlit Headers Overrides */
    h1, h2, h3 {
        color: var(--header-bg);
        font-family: var(--font-family);
        font-weight: 700;
    }
    
    /* Metrics / KPI Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: var(--primary-color) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-light) !important;
        font-size: 0.9rem !important;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .stDataFrame, .stPlotlyChart {
        background-color: var(--card-bg);
        border: 1px solid #e5e5e5;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 1.25rem;
        transition: box-shadow 0.2s ease-in-out;
    }
    .stPlotlyChart:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d9d9d9;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 0.25rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: var(--secondary-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: var(--primary-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header (Corporate Style) ---
st.markdown("""
<div class="app-shell-header">
    <span style="font-size: 24px;">🌐</span>
    <span class="app-shell-title">Intelligent Logistics Management System</span>
</div>
""", unsafe_allow_html=True)

st.markdown("### AI-Enabled Multimodal Optimization workspace")
st.markdown("---")

# --- Initialize AI Model (Cached) ---
@st.cache_resource
def load_ai_model_data():
    return train_delay_model()

ai_model, X_test, y_test, y_bin_test = load_ai_model_data()

# --- Sidebar: Inputs ---
st.sidebar.header("📦 Shipment Parameters")

total_volume = st.sidebar.number_input("Total Shipment Volume (Tons)", min_value=100, value=1000, step=50)

st.sidebar.subheader("💰 Costs ($/Ton)")
cost_road = st.sidebar.number_input("Road Cost", value=50.0)
cost_rail = st.sidebar.number_input("Rail Cost", value=30.0)
cost_coastal = st.sidebar.number_input("Coastal Cost", value=20.0)

st.sidebar.subheader("🏭 Capacities (Tons)")
cap_road = st.sidebar.number_input("Road Capacity", value=800.0)
cap_rail = st.sidebar.number_input("Rail Capacity", value=1500.0)
cap_coastal = st.sidebar.number_input("Coastal Capacity", value=2000.0)

st.sidebar.subheader("🌍 Sustainability & Constraints")
emission_cost_per_ton_co2 = st.sidebar.number_input("Emission Cost ($/Ton CO2)", value=100.0)
# Emission factors (tons CO2 per ton of cargo) - Typical approx values
ef_road = 0.15
ef_rail = 0.05
ef_coastal = 0.02
min_low_carbon = st.sidebar.slider("Min % Low-Carbon (Rail+Coastal)", 0, 100, 30)
min_fast_response = st.sidebar.slider("Min % Fast-Response (Road)", 0, 100, 20)

st.sidebar.subheader("🚥 ITS Real-Time Data")
congestion_level = st.sidebar.slider("Congestion/Disruption Index (0-100)", 0, 100, 45)
delay_penalty = st.sidebar.number_input("Delay Penalty ($/Ton)", value=25.0)

# --- AI Predictions ---
# Predict delay risk for current volume (simplified: assuming proportional split for prediction)
# In reality, this is iterative, but for demo we predict based on average expected volume per mode or unit.
risk_road = predict_delay_risk(ai_model, congestion_level, total_volume/3, 'Road')
risk_rail = predict_delay_risk(ai_model, congestion_level, total_volume/3, 'Rail')
risk_coastal = predict_delay_risk(ai_model, congestion_level, total_volume/3, 'Coastal')

ai_risks = {'Road': risk_road, 'Rail': risk_rail, 'Coastal': risk_coastal}

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⚙️ Model Setup", "🚀 Optimization Results", "🤖 AI Delay Prediction", "📊 Scenario Comparison", "📈 Advanced Analysis", "🧪 Model Performance"])

with tab1:
    st.subheader("Current Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Total Demand:** {total_volume} Tons")
        st.write("Constraint Check:")
        st.write(f"- Min Low-Carbon: {min_low_carbon}%")
        st.write(f"- Min Fast-Response: {min_fast_response}%")
        
    with col2:
        st.warning(f"**Current Context:**")
        st.write(f"- Congestion Index: {congestion_level}/100")
        st.write(f"- CO2 Cost: ${emission_cost_per_ton_co2}/ton")
        
    st.markdown("#### Mode Characteristics")
    df_modes = pd.DataFrame({
        'Mode': ['Road', 'Rail', 'Coastal'],
        'Base Cost ($)': [cost_road, cost_rail, cost_coastal],
        'Capacity': [cap_road, cap_rail, cap_coastal],
        'Emission Factor': [ef_road, ef_rail, ef_coastal],
        'AI Predicted Delay Risk': [f"{risk_road:.1%}", f"{risk_rail:.1%}", f"{risk_coastal:.1%}"]
    })
    st.dataframe(df_modes, hide_index=True)

with tab2:
    st.subheader("Optimal Transport Allocation")
    
    if st.button("Run Optimization Solver", type="primary"):
        costs_dict = {'Road': cost_road, 'Rail': cost_rail, 'Coastal': cost_coastal}
        caps_dict = {'Road': cap_road, 'Rail': cap_rail, 'Coastal': cap_coastal}
        # Emission cost per ton of CARGO = factor * penalty
        em_costs_dict = {
            'Road': ef_road * emission_cost_per_ton_co2,
            'Rail': ef_rail * emission_cost_per_ton_co2,
            'Coastal': ef_coastal * emission_cost_per_ton_co2
        }
        
        results = run_optimization(
            total_volume, costs_dict, caps_dict, em_costs_dict,
            congestion_level, delay_penalty,
            min_low_carbon, min_fast_response,
            ai_risks
        )
        
        if results['status'] == 'Optimal':
            st.success(f"Optimization Status: {results['status']}")
            
            # Check for Soft Constraint Violations
            slacks = results.get('slacks', {})
            if slacks.get('low_carbon', 0) > 0.1:
                st.warning(f"⚠️ **Target Missed:** Could not fully meet the {min_low_carbon}% Low-Carbon requirement due to capacity limits. Missed by {slacks['low_carbon']:.1f} tons.")
            
            if slacks.get('fast_response', 0) > 0.1:
                st.warning(f"⚠️ **Target Missed:** Could not fully meet the {min_fast_response}% Fast-Response (Road) requirement due to capacity limits. Missed by {slacks['fast_response']:.1f} tons.")
            
            # 1. Allocation Chart
            alloc = results['allocation']
            df_alloc = pd.DataFrame(list(alloc.items()), columns=['Mode', 'Tons'])
            
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_alloc = px.pie(df_alloc, values='Tons', names='Mode', title='Optimal Mode Share', hole=0.4,
                                   color='Mode', color_discrete_map={'Road':'#e74c3c', 'Rail':'#f1c40f', 'Coastal':'#3498db'})
                st.plotly_chart(fig_alloc, width="stretch")
            
            with c2:
                st.metric("Total Logistics Cost", f"${results['total_cost']:,.2f}")
                st.write("**Allocation Breakdown (Tons):**")
                st.dataframe(df_alloc, hide_index=True)

            # 2. Cost Breakdown
            st.markdown("#### Cost Structure Analysis")
            breakdown_data = []
            for m, vals in results['breakdown'].items():
                breakdown_data.append({'Mode': m, 'Type': 'Transport', 'Cost': vals['Transport']})
                breakdown_data.append({'Mode': m, 'Type': 'Delay Penalty', 'Cost': vals['Delay_Penalty']})
                breakdown_data.append({'Mode': m, 'Type': 'Emission', 'Cost': vals['Emission']})
            
            df_cost = pd.DataFrame(breakdown_data)
            fig_cost = px.bar(df_cost, x='Mode', y='Cost', color='Type', title="Cost Components per Mode",
                              color_discrete_map={'Transport':'#34495e', 'Delay Penalty':'#e74c3c', 'Emission':'#2ecc71'})
            st.plotly_chart(fig_cost, width="stretch")
            
        else:
            st.error(f"Optimization Failed: {results['status']}. Check constraints (e.g., Capacity vs Demand).")

with tab3:
    st.subheader("🤖 AI-Driven Delay Probabilities")
    st.markdown("Explaining how Intelligent Transportation Systems (ITS) data affects dynamic planning.")
    
    # Visualization of Risk vs Congestion for each mode
    x_vals = list(range(0, 101, 5))
    df_trends = []
    
    for c in x_vals:
        for m in ['Road', 'Rail', 'Coastal']:
            # approximate volume for sensitivity check
            r = predict_delay_risk(ai_model, c, 500, m)
            df_trends.append({'Congestion': c, 'Risk': r, 'Mode': m})
            
    df_trends = pd.DataFrame(df_trends)
    
    fig_risk = px.line(df_trends, x='Congestion', y='Risk', color='Mode', title="Impact of Congestion on Delay Probability",
                       markers=True, color_discrete_map={'Road':'#e74c3c', 'Rail':'#f1c40f', 'Coastal':'#3498db'})
    st.plotly_chart(fig_risk, width="stretch")
    
    st.info("Notice how 'Road' is most sensitive to congestion, while 'Rail' and 'Coastal' are more stable but have different base risks.")

with tab4:
    st.subheader("Scenario: Optmized vs Static Planning")
    st.markdown("Comparing the AI-optimized result against a traditional fixed planning strategy (e.g., 70% Road, 20% Rail, 10% Coastal).")
    
    if st.button("Compare Scenarios"):
        # Run Optimization again to get current optimal
        costs_dict = {'Road': cost_road, 'Rail': cost_rail, 'Coastal': cost_coastal}
        caps_dict = {'Road': cap_road, 'Rail': cap_rail, 'Coastal': cap_coastal}
        em_costs_dict = {
            'Road': ef_road * emission_cost_per_ton_co2,
            'Rail': ef_rail * emission_cost_per_ton_co2,
            'Coastal': ef_coastal * emission_cost_per_ton_co2
        }
        res_opt = run_optimization(total_volume, costs_dict, caps_dict, em_costs_dict, congestion_level, delay_penalty, min_low_carbon, min_fast_response, ai_risks)
        
        # Calculate Static
        static_mix = {'Road': 0.7, 'Rail': 0.2, 'Coastal': 0.1}
        
        # Check capacity for static
        static_cost = 0
        static_valid = True
        
        for m, pct in static_mix.items():
            qty = total_volume * pct
            if qty > caps_dict[m]:
                static_valid = False
            
            # Calculate cost components
            trans = qty * costs_dict[m]
            delay = qty * delay_penalty * ai_risks[m]
            emis = qty * em_costs_dict[m]
            static_cost += (trans + delay + emis)
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("AI Optimized Cost", f"${res_opt.get('total_cost', 0):,.2f}")
            if res_opt['status'] != 'Optimal':
                st.error("Optimization failed.")
                
        with col2:
            st.metric("Static Plan Cost", f"${static_cost:,.2f}")
            if not static_valid:
                st.error("Static plan violates capacity constraints!")
            else:
                diff = static_cost - res_opt.get('total_cost', 0)
                st.success(f"Savings: ${diff:,.2f} ({diff/static_cost:.1%})")
        
        # Chart comparison
        comp_df = pd.DataFrame({
            'Scenario': ['AI Optimization', 'Static (70/20/10)'],
            'Total Cost': [res_opt.get('total_cost', 0), static_cost]
        })
        
        fig_comp = px.bar(comp_df, x='Scenario', y='Total Cost', color='Scenario', title="Financial Impact Analysis")
        st.plotly_chart(fig_comp, width="stretch")

with tab5:
    st.subheader("📈 Sensitivity & Trade-off Analysis")
    st.markdown("Simulate how changes in congestion affect the Total Logistics Cost and Emission levels.")
    
    if st.button("Run Sensitivity Analysis", type="primary"):
        # We will loop through congestion levels 0 to 100 in steps of 10
        param_ranges = list(range(0, 101, 10))
        sensitivity_results = []
        
        # Setup static dictionaries for the loop
        costs_dict = {'Road': cost_road, 'Rail': cost_rail, 'Coastal': cost_coastal}
        caps_dict = {'Road': cap_road, 'Rail': cap_rail, 'Coastal': cap_coastal}
        em_costs_dict = {
            'Road': ef_road * emission_cost_per_ton_co2,
            'Rail': ef_rail * emission_cost_per_ton_co2,
            'Coastal': ef_coastal * emission_cost_per_ton_co2
        }
        
        progress_bar = st.progress(0)
        
        for i, c_level in enumerate(param_ranges):
            # 1. Update AI risks for this congestion level
            r_road = predict_delay_risk(ai_model, c_level, total_volume/3, 'Road')
            r_rail = predict_delay_risk(ai_model, c_level, total_volume/3, 'Rail')
            r_coastal = predict_delay_risk(ai_model, c_level, total_volume/3, 'Coastal')
            loop_risks = {'Road': r_road, 'Rail': r_rail, 'Coastal': r_coastal}
            
            # 2. Run Optimization
            res = run_optimization(total_volume, costs_dict, caps_dict, em_costs_dict, 
                                   c_level, delay_penalty, min_low_carbon, min_fast_response, loop_risks)
            
            if res['status'] == 'Optimal':
                # Calculate Total CO2 Emissions (Tons) - separate from cost
                total_emissions = 0
                for m, qty in res['allocation'].items():
                     # ef_mode is ton CO2 / ton cargo
                    factor = ef_road if m == 'Road' else (ef_rail if m == 'Rail' else ef_coastal)
                    total_emissions += qty * factor
                
                sensitivity_results.append({
                    'Congestion Index': c_level,
                    'Total Cost ($)': res['total_cost'],
                    'Total Emissions (Tons CO2)': total_emissions,
                    'Road Allocation': res['allocation']['Road'],
                    'Rail Allocation': res['allocation']['Rail'],
                    'Coastal Allocation': res['allocation']['Coastal']
                })
            progress_bar.progress((i + 1) / len(param_ranges))
            
        df_sens = pd.DataFrame(sensitivity_results)
        
        # --- Visualization 1: Cost vs Congestion ---
        st.markdown("#### 1. Cost Sensitivity to Congestion")
        fig_sens = px.area(df_sens, x='Congestion Index', y='Total Cost ($)', 
                           title="Total Cost Increase as Congestion Rises",
                           color_discrete_sequence=['#2C3E50'])
        st.plotly_chart(fig_sens, width="stretch")
        
        # --- Visualization 2: Modal Shift ---
        st.markdown("#### 2. Adaptive Modal Shift")
        df_melt = df_sens.melt(id_vars=['Congestion Index'], 
                               value_vars=['Road Allocation', 'Rail Allocation', 'Coastal Allocation'],
                               var_name='Mode', value_name='Tons Allocated')
        
        fig_shift = px.bar(df_melt, x='Congestion Index', y='Tons Allocated', color='Mode',
                           title="How the AI Re-allocates Freight as Congestion Worsens",
                           color_discrete_map={'Road Allocation':'#e74c3c', 'Rail Allocation':'#f1c40f', 'Coastal Allocation':'#3498db'})
        st.plotly_chart(fig_shift, width="stretch")
        
        # --- Visualization 3: Cost vs Environment Tradeoff ---
        st.markdown("#### 3. Economic vs Environmental Impact")
        fig_bubble = px.scatter(df_sens, x='Total Emissions (Tons CO2)', y='Total Cost ($)',
                                size='Congestion Index', color='Congestion Index',
                                title="Trade-off: Cost vs Emissions across Congestion Levels",
                                hover_data=['Congestion Index'])
        st.plotly_chart(fig_bubble, width="stretch")
        
        st.success("Analysis Complete. The AI model dynamically shifts volume away from Road to Rail/Coastal as congestion makes road transport prohibitively expensive due to delay penalties.")

with tab6:
    st.subheader("🧪 AI Model Performance Analysis")
    st.markdown("Evaluation of the Random Forest model on unseen test data (20% split).")

    # Predictions on Test Set
    # Regression model predicts probability (risk)
    y_pred_prob = ai_model.predict(X_test)
    # Binary classification for metrics (Threshold > 0.5)
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # Metrics
    # y_bin_test is the actual binary delay event (simulated ground truth)
    # y_test is the actual probability (simulated ground truth risk)
    acc = accuracy_score(y_bin_test, y_pred_bin)
    auc = roc_auc_score(y_bin_test, y_pred_prob)
    mse = mean_squared_error(y_test, y_pred_prob) # pred prob vs actual prob (Brier Score equivalent)
    f1 = f1_score(y_bin_test, y_pred_bin)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.1%}")
    c2.metric("ROC-AUC Score", f"{auc:.3f}")
    c3.metric("MSE (Probabilities)", f"{mse:.4f}")
    c4.metric("F1 Score", f"{f1:.3f}")

    col_grp1, col_grp2 = st.columns(2)

    with col_grp1:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_bin_test, y_pred_prob)
        fig_roc = px.area(
            x=fpr, y=tpr, title=f'ROC Curve (AUC={auc:.3f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
        )
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_layout(xaxis_constrain='domain', yaxis_scaleanchor="x")
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_grp2:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_bin_test, y_pred_bin)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
            x=['No Delay', 'Delay'],
            y=['No Delay', 'Delay'],
            title="Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    st.markdown("#### Error Analysis (MSE)")
    st.markdown("The **Mean Squared Error (MSE)** here represents the average squared difference between the *Predicted Delay Probability* and the *Actual Delay Probability* (Ground Truth Risk). Lower is better.")
    
    # Residuals plot
    residuals = y_test - y_pred_prob
    fig_res = px.histogram(residuals, nbins=30, title="Prediction Error Distribution (Residuals)",
                           labels={'value': 'Error (Actual Risk - Predicted Risk)'},
                           color_discrete_sequence=['#e74c3c'])
    st.plotly_chart(fig_res, use_container_width=True)

st.markdown("---")
st.caption("© 2026 Intelligent Logistics Management System - Prototype")
