import pulp

def run_optimization(total_volume, costs, capacities, emissions, 
                     congestion_level, delay_penalty_per_ton, 
                     min_low_carbon_pct, min_fast_response_pct,
                     ai_delay_risks):
    """
    Runs the optimization model using PuLP.
    
    ai_delay_risks: dict with keys 'Road', 'Rail', 'Coastal' -> probability (0-1)
    """
    
    # 1. Initialize Problem
    prob = pulp.LpProblem("Transport_Optimization", pulp.LpMinimize)

    # 2. Decision Variables (Amount in Tons)
    modes = ['Road', 'Rail', 'Coastal']
    x = pulp.LpVariable.dicts("Mode_Alloc", modes, lowBound=0, cat='Continuous')

    # 3. Objective Function
    # Cost = (Transport Cost) + (Delay Penalty * Risk) + (Emission Cost)
    # Note: Emission Cost is handled if 'emissions' input contains cost, 
    # but prompt implies 'Emission cost per ton CO2' is a global parameter 
    # and 'emissions' dict contains factors. 
    # Let's assume 'emissions' dict contains Cost-equivalent or we calculate it.
    # Re-reading prompt: "Emission cost per ton CO2" is an input.
    # Let's assume 'emissions' passed here implies CO2 factor * Cost/tonCO2.
    
    # 3. Objective Function (Initialize with 0)
    total_cost = 0
    
    # Initialize Slack Variables (for Soft Constraints)
    # These allow the model to violate constraints if strictly necessary, at a high cost
    slack_low_carbon = pulp.LpVariable("Slack_Low_Carbon", lowBound=0, cat='Continuous')
    slack_fast_response = pulp.LpVariable("Slack_Fast_Response", lowBound=0, cat='Continuous')
    
    # Penalty for violating constraints (make it high enough to avoid violation if possible)
    # e.g., $1,000,000 per ton of violation
    constraint_violation_penalty = 100000 
    
    for m in modes:
        # Transport Cost
        transport_c = costs[m] * x[m]
        
        # Delay Penalty = Penalty * Risk * Allocation
        delay_c = delay_penalty_per_ton * ai_delay_risks.get(m, 0) * x[m]
        
        # Emission Cost
        emission_c = emissions[m] * x[m]
        
        total_cost += transport_c + delay_c + emission_c
        
    # Add Penalties for Slack to Objective
    total_cost += (slack_low_carbon * constraint_violation_penalty) + (slack_fast_response * constraint_violation_penalty)

    prob += total_cost, "Total_Logistics_Cost"

    # 4. Constraints
    
    # Demand Satisfaction (Hard Constraint)
    prob += pulp.lpSum([x[m] for m in modes]) == total_volume, "Meet_Total_Demand"
    
    # Capacity Constraints (Hard Constraint)
    for m in modes:
        prob += x[m] <= capacities[m], f"Capacity_{m}"
        
    # Low Carbon Requirement (Soft Constraint)
    # (Rail + Coastal) + Slack >= % * Total
    prob += x['Rail'] + x['Coastal'] + slack_low_carbon >= (min_low_carbon_pct / 100.0) * total_volume, "Min_Low_Carbon"
    
    # Fast Response Requirement (Soft Constraint)
    # Road + Slack >= % * Total
    prob += x['Road'] + slack_fast_response >= (min_fast_response_pct / 100.0) * total_volume, "Min_Fast_Response"

    # 5. Solve
    import sys
    
    solver = None
    solver_logs = []
    
    try:
        import highspy
        solver = pulp.getSolver('HiGHS')
        solver_logs.append("Selected solver: HiGHS (via highspy)")
    except Exception as e_highs:
        solver_logs.append(f"HiGHS availability check failed: {e_highs}")

    if solver is None:
        solver_logs.append("Attempting Fallback to default CBC...")
        pass

    try:
        if solver:
            prob.solve(solver)
        else:
            prob.solve()
    except Exception as e:
        solver_logs.append(f"Primary solve attempt failed: {e}")
        return {
            'status': 'SolverError',
            'error_msg': str(e),
            'debug_logs': '\n'.join(solver_logs),
            'allocation': {m: 0 for m in modes},
            'total_cost': 0,
            'breakdown': {m: {'Transport':0, 'Delay_Penalty':0, 'Emission':0} for m in modes},
            'slacks': {'low_carbon':0, 'fast_response':0}
        }
    
    # 6. Results
    status = pulp.LpStatus[prob.status]
    
    # Calculate real cost without penalties for the user display
    real_cost = 0
    breakdown = {}
    
    # Check Slacks
    low_carbon_violation = slack_low_carbon.varValue
    fast_response_violation = slack_fast_response.varValue
    
    results = {
        'status': status,
        'allocation': {m: x[m].varValue for m in modes},
        # total_cost in `prob.objective` includes the huge penalty. 
        # We need to re-sum natural costs for display.
        'total_cost': 0, 
        'breakdown': {},
        'slacks': {
            'low_carbon': low_carbon_violation,
            'fast_response': fast_response_violation
        }
    }
    
    # Re-calculate clean costs
    for m in modes:
        val = x[m].varValue
        t_cost = costs[m] * val
        d_cost = delay_penalty_per_ton * ai_delay_risks.get(m, 0) * val
        e_cost = emissions[m] * val
        
        results['breakdown'][m] = {
            'Transport': t_cost,
            'Delay_Penalty': d_cost,
            'Emission': e_cost
        }
        results['total_cost'] += t_cost + d_cost + e_cost
        
    return results
