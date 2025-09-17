from typing import List, Dict, Tuple
import itertools
import time
import random
import numpy as np

class RidePricingOptimizer:
    def __init__(self):
        self.cost_per_ride = 3
        self.price_options = [9, 12, 15, 18, 20.5]  # price options
        
    def optimize_prices(self, base_demands, algorithm="forward_greedy"):
        """
        Main optimization function with algorithm selection
        
        Args:
            base_demands: Expected number of ride requests for each time period
                         Example: [20, 50, 80, 60, 40] for 5 periods
            algorithm: "forward_greedy", "dynamic_programming", or "simple"
        
        Returns:
            List of optimal prices for each period
            Example: [8, 12, 15, 10, 7] 
        """
        if algorithm == "forward_greedy":
            return self.forward_greedy_lookahead(base_demands, lookahead=3)
        elif algorithm == "dynamic_programming":
            return self.dynamic_programming(base_demands)
        elif algorithm == "simple":
            return self.simple_greedy(base_demands)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def forward_greedy_lookahead(self, base_demands, lookahead=3):
        """
        Forward-Looking Greedy Algorithm with Lookahead Window
        
        For each time period:
        1. Consider all possible prices for current period
        2. Look ahead 'lookahead' periods to see impact
        3. Choose price that maximizes profit over the window
        
        This balances immediate profit with future spillover effects
        """
        n_periods = len(base_demands)
        optimal_prices = []
        accumulated_spillover = 0.0
        # loops through all periods
        for t in range(n_periods):
            best_window_profit = 0
            
            # loops thru prices
            for current_price in self.price_options:
                # find window profit 
                window_profit = 0.0
                temp_spillover = accumulated_spillover
                
                # lookahead periods 
                for k in range(min(lookahead, n_periods - t)):
                    # total index of lookahead period 
                    period_idx = t + k
                    
                    # calculate total demand per period 
                    period_demand = base_demands[period_idx] + temp_spillover
                    
                    if k == 0:
                        # Current period - use the price we're testing
                        period_price = current_price
                    else:
                        # price for future periods ? 
                        period_price = optimal_prices[-1] if optimal_prices else np.median(self.price_options)
                    
                    # get customers + spillover
                    actual_customers, spillover = self._calculate_demand_and_spillover(
                        period_demand, period_price
                    )
                    
                    period_profit = (period_price - self.cost_per_ride) * actual_customers
                    window_profit += period_profit
                    
                    # change spillover for next period within lookahead
                    if period_idx < n_periods - 1:
                        temp_spillover = spillover
                    else:
                        temp_spillover = 0  # no spillover if last period
                
                # optimize window_profit to find current_price 
                if window_profit > best_window_profit:
                    best_window_profit = window_profit
                    optimized_price = current_price
            
            # see the best price for this period
            optimal_prices.append(optimized_price)
            
            # udpate actual spillover for next period
            total_demand = base_demands[t] + accumulated_spillover
            _, spillover = self._calculate_demand_and_spillover(total_demand, optimized_price)
            print("spillover " + str(spillover))
            if t < n_periods - 1:
                accumulated_spillover = spillover
            else:
                accumulated_spillover = 0
        
        return optimal_prices
    
    def dynamic_programming(self, base_demands):
        """
        Dynamic Programming Solution
        
        State: (period, spillover_level)
        Decision: price at each period
        
        Uses backward induction to find optimal pricing strategy
        Discretizes spillover into buckets for computational tractability
        """
        n_periods = len(base_demands)
        
        # Discretize spillover states
        # Maximum possible spillover is 30% of total demand
        max_possible_spillover = sum(base_demands) * 0.3
        spillover_buckets = 20  # Number of spillover states to consider
        spillover_step = max_possible_spillover / spillover_buckets
        spillover_states = [i * spillover_step for i in range(spillover_buckets + 1)]
        
        # DP table: dp[period][spillover_bucket] = (max_profit, best_price)
        # Initialize with zeros
        dp = {}
        
        # Backward induction - start from last period
        for period in range(n_periods - 1, -1, -1):
            for spill_idx, spillover in enumerate(spillover_states):
                best_profit = float('-inf')
                best_price = self.price_options[0]
                
                # Total demand for this period
                total_demand = base_demands[period] + spillover
                
                # Try all possible prices
                for price in self.price_options:
                    # Calculate immediate profit
                    actual_customers, next_spillover = self._calculate_demand_and_spillover(
                        total_demand, price
                    )
                    immediate_profit = (price - self.cost_per_ride) * actual_customers
                    
                    if period == n_periods - 1:
                        # Last period - no future profit
                        total_profit = immediate_profit
                    else:
                        # Add future profit from next period
                        # Find closest spillover bucket for next period
                        next_spill_idx = min(
                            range(len(spillover_states)),
                            key=lambda i: abs(spillover_states[i] - next_spillover)
                        )
                        
                        # Get future profit from DP table
                        future_profit = dp.get((period + 1, next_spill_idx), (0, 0))[0]
                        total_profit = immediate_profit + future_profit
                    
                    # Update best if this is better
                    if total_profit > best_profit:
                        best_profit = total_profit
                        best_price = price
                
                # Store in DP table
                dp[(period, spill_idx)] = (best_profit, best_price)
        
        # Forward pass to extract optimal prices
        optimal_prices = []
        current_spillover = 0.0
        
        for period in range(n_periods):
            # Find closest spillover bucket
            spill_idx = min(
                range(len(spillover_states)),
                key=lambda i: abs(spillover_states[i] - current_spillover)
            )
            
            # Get optimal price from DP table
            _, price = dp.get((period, spill_idx), (0, self.price_options[0]))
            optimal_prices.append(price)
            
            # Calculate spillover for next period
            total_demand = base_demands[period] + current_spillover
            _, current_spillover = self._calculate_demand_and_spillover(total_demand, price)
            
            # Reset spillover if last period
            if period == n_periods - 1:
                current_spillover = 0
        
        return optimal_prices
    
    def simple_greedy(self, base_demands):
        """
        Simple greedy algorithm for baseline comparison
        Sets prices based on demand level without considering spillover
        """
        prices = []
        for demand in base_demands:
            # Simple heuristic: higher price for higher demand
            if demand > 70:
                prices.append(20)
            elif demand > 50:
                prices.append(15)
            elif demand > 30:
                prices.append(12)
            else:
                prices.append(8)
        return prices
    
    def simulate_day(self, base_demands, prices, penalty_per_deferred=0.5, loss = 0.2):
      total_profit = 0
      deferred_customers = 0  
      
      for period in range(len(base_demands)):
          p_demand = base_demands[period] + deferred_customers
          customers, new_deferred = self._calculate_demand_and_spillover(p_demand, prices[period])
          
          # from served customers
          total_profit += (prices[period] - self.cost_per_ride) * customers
          
          # penalty for deferring
          total_profit -= penalty_per_deferred * new_deferred
          if period < len(base_demands) - 1:
            deferred_customers = new_deferred * (1 - loss)
          else:
            deferred_customers = 0

      
      return total_profit
    
    def _calculate_demand_and_spillover(self, total_demand, price):
        """
        - Low prices (<$10): Everyone rides, no deferrals, no loss
        - Medium prices ($10-$19): 10% defer, but 5% of those are lost
        - High prices ($20+): 30% defer, but 15% of those are lost
        """
        if price < 10:
            # all riders pass
            return total_demand, 0

        elif price < 20:
            raw_deferred = total_demand * 0.10
            lost = raw_deferred * 0.50   # half of deferred switch services
            deferred = raw_deferred - lost
            current = total_demand - raw_deferred
            return current, deferred

        else:  # price >= 20
            raw_deferred = total_demand * 0.30
            lost = raw_deferred * 0.50   
            deferred = raw_deferred - lost
            current = total_demand - raw_deferred
            return current, deferred   

    def compare_algorithms(self, base_demands):
        results = {}
        algorithms = ["simple", "forward_greedy", "dynamic_programming"]
        
        for alg in algorithms:
            start_time = time.time()
            prices = self.optimize_prices(base_demands, algorithm=alg)
            execution_time = time.time() - start_time
            
            profit = self.simulate_day(base_demands, prices)
            
            results[alg] = {
                'prices': prices,
                'profit': profit,
                'execution_time': execution_time
            }
        
        return results
    
    def analyze_spillover_impact(self, base_demands, prices):
        """
        return analytics on the impact of the decided price point on spillover 
        """
        analysis = []
        deferred_customers = 0
        
        for period in range(len(base_demands)):
            total_demand = base_demands[period] + deferred_customers
            actual_customers, new_deferred = self._calculate_demand_and_spillover(
                total_demand, prices[period]
            )
            
            period_profit = (prices[period] - self.cost_per_ride) * actual_customers
            
            analysis.append({
                'period': period + 1,
                'base_demand': base_demands[period],
                'spillover_in': deferred_customers,
                'total_demand': total_demand,
                'price': prices[period],
                'actual_customers': actual_customers,
                'spillover_out': new_deferred if period < len(base_demands) - 1 else 0,
                'profit': period_profit
            })
            
            deferred_customers = new_deferred if period < len(base_demands) - 1 else 0
        
        return analysis



def run_tests():
    
    optimizer = RidePricingOptimizer()
    
    # varying cases
    test_cases = {
        "Morning Rush": [500, 1700, 10000, 25000, 15000, 7000, 8800],
        "Slow Day": [25, 30, 35, 45, 65, 80, 70],
        "Steady Demand": [10000, 10000, 10000, 10000, 10000, 10000],
        "Variable Pattern": [10000, 5000, 10, 100, 15040, 53000],
        "Short Burst": [1000, 1203, 30, 25]
    }
    
    for scenario_name, demands in test_cases.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"Base Demands: {demands}")
        print(f"{'='*60}")
        
        # compare the algorithms using each demand
        results = optimizer.compare_algorithms(demands)
        
        # sorted by profit
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['profit'], 
                              reverse=True)
        
        print("\nAlgorithm Performance:")
        print("-" * 50)
        
        for algo_name, metrics in sorted_results:
            print(f"\n{algo_name.upper()}:")
            print(f"  Profit: ${metrics['profit']:.2f}")
            print(f"  Execution Time: {metrics['execution_time']*1000:.2f} ms")
            print(f"  Prices: {metrics['prices']}")
        
        # analysis of best algorithm
        best_algo = sorted_results[0][0]
        best_prices = sorted_results[0][1]['prices']
        
        print(f"\n{'='*60}")
        print(f"Detailed Analysis - Best Algorithm: {best_algo.upper()}")
        print(f"{'='*60}")
        
        analysis = optimizer.analyze_spillover_impact(demands, best_prices)
        
        print("\nPeriod | Base | Spillover In | Price | Customers | Spillover Out | Profit")
        print("-" * 75)
        
        for period_data in analysis:
            print(f"  {period_data['period']:2d}   | {period_data['base_demand']:3.0f}  |"
                  f"     {period_data['spillover_in']:5.1f}    | ${period_data['price']:2.0f}   |"
                  f"    {period_data['actual_customers']:5.1f}   |"
                  f"      {period_data['spillover_out']:5.1f}     |"
                  f" ${period_data['profit']:6.2f}")
        
        total_profit = sum(p['profit'] for p in analysis)
        print("-" * 75)
        print(f"TOTAL PROFIT: ${total_profit:.2f}")


run_tests()