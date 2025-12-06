#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class RBCAgent:
    """
    Enhanced Rule-Based Controller with heuristics:
    - Reduce consumption during peak hours (9am-9pm)
    - Respond to electricity pricing
    - Maintain comfort bounds
    - Adapt to outdoor temperature
    """
    
    def __init__(self, action_space=5):
        """
        Initialize RBC agent
        
        Args:
            action_space: Number of discrete actions (default 5)
        """
        self.action_space = action_space
        # Action mapping: 0=-1.0, 1=-0.5, 2=0.0, 3=0.5, 4=1.0
        
    def predict(self, state):
        """
        Generate action based on heuristic rules
        
        Args:
            state: 17-dimensional state vector from building
                [0]: Month (normalized)
                [1]: outdoor_dry_bulb_temperature
                [7]: indoor_dry_bulb_temperature
                [12]: cooling_demand
                [13]: heating_demand
                [14]: electricity_pricing
                
        Returns:
            action: Discrete action index [0-4]
        """
        # Extract relevant features
        hour = int((state[0] * 12) * 730) % 24  # Rough hour estimate from month
        outdoor_temp = state[1] if len(state) > 1 else 20
        indoor_temp = state[7] if len(state) > 7 else 22
        electricity_price = state[14] if len(state) > 14 else 0.12
        
        # Default action: no change (action 2 = 0.0 continuous)
        action = 2
        
        # Rule 1: Peak hours (9am-9pm) - reduce consumption slightly
        if 9 <= hour < 21:
            action = 1  # -0.5 (reduce demand)
        else:
            # Off-peak - maintain baseline
            action = 2  # 0.0 (no change)
        
        # Rule 2: High electricity price - reduce more aggressively
        avg_price = 0.12  # Typical average price
        if electricity_price > avg_price * 1.3:
            action = 0  # -1.0 (reduce significantly)
        
        # Rule 3: Low electricity price - can afford more comfort
        elif electricity_price < avg_price * 0.7:
            action = 3  # +0.5 (increase comfort)
        
        # Rule 4: Outdoor temperature is comfortable - reduce HVAC
        if 18 <= outdoor_temp <= 24:
            # Don't increase demand when outdoor temp is nice
            action = min(action, 2)
        
        # Rule 5: Indoor temperature bounds (maintain comfort)
        if indoor_temp < 19:  # Too cold - increase heating
            action = 4  # +1.0
        elif indoor_temp > 27:  # Too hot - increase cooling
            action = 4  # +1.0
        
        return action
    
    def __repr__(self):
        return "RBCAgent(rule_based_controller)"
