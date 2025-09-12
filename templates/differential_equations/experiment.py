import argparse
import json
import os
import numpy as np
from scipy.integrate import odeint
import sympy as sp
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Differential Equation Solver Template
# This template provides a framework for solving differential equations analytically
# and comparing with numerical solutions using AI-Scientist automation.
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run differential equation experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()

if __name__ == "__main__":
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    def solve_differential_equation():
        """
        Solve a differential equation analytically using symbolic computation.
        This is a template function that can be modified for different equations.
        """
        # Define symbols
        x = sp.Symbol('x')
        y = sp.Function('y')
        
        # Example: Solve dy/dx = x*y (separable equation)
        # This can be modified for different differential equations
        ode = sp.Eq(y(x).diff(x), x * y(x))
        
        # Solve analytically
        analytical_solution = sp.dsolve(ode, y(x))
        
        # Apply initial condition y(0) = 1
        C1 = sp.Symbol('C1')
        initial_condition = analytical_solution.subs(x, 0) - 1
        C1_value = sp.solve(initial_condition, C1)[0]
        particular_solution = analytical_solution.subs(C1, C1_value)
        
        return particular_solution, ode
    
    def numerical_solution(x_range, initial_condition):
        """
        Solve the same differential equation numerically for comparison.
        """
        def dydx(y, x):
            return x * y
        
        x_vals = np.linspace(x_range[0], x_range[1], 100)
        y_vals = odeint(dydx, initial_condition, x_vals)
        return x_vals, y_vals.flatten()
    
    def evaluate_solution(solution, x_vals):
        """
        Evaluate the analytical solution at given x values.
        """
        # Convert symbolic solution to numerical function
        solution_func = sp.lambdify(x, solution.rhs, 'numpy')
        return solution_func(x_vals)
    
    # Solve the differential equation
    analytical_solution, original_ode = solve_differential_equation()
    
    # Define solution domain
    x_range = (0, 5)
    initial_condition = 1.0
    
    # Get numerical solution
    x_numerical, y_numerical = numerical_solution(x_range, initial_condition)
    
    # Get analytical solution at same points
    y_analytical = evaluate_solution(analytical_solution, x_numerical)
    
    # Calculate accuracy metrics
    mse = np.mean((y_analytical - y_numerical)**2)
    max_error = np.max(np.abs(y_analytical - y_numerical))
    relative_error = np.mean(np.abs((y_analytical - y_numerical) / y_analytical)) * 100
    
    # Calculate solution characteristics
    solution_characteristics = {
        "max_value": float(np.max(y_analytical)),
        "min_value": float(np.min(y_analytical)),
        "final_value": float(y_analytical[-1]),
        "growth_rate": float(np.mean(np.diff(y_analytical) / np.diff(x_numerical))),
        "convergence_rate": float(np.mean(np.abs(np.diff(y_analytical))))
    }
    
    # Store results
    means = {
        "mse": mse,
        "max_error": max_error,
        "relative_error": relative_error,
        "solution_characteristics": solution_characteristics
    }
    
    final_info = {
        "differential_equation": {
            "means": means,
            "analytical_solution": str(analytical_solution),
            "original_equation": str(original_ode),
            "x_values": x_numerical.tolist(),
            "y_analytical": y_analytical.tolist(),
            "y_numerical": y_numerical.tolist(),
            "solution_accuracy": "high" if relative_error < 1.0 else "medium" if relative_error < 5.0 else "low"
        }
    }
    
    # Save results
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)
    
    print(f"Experiment completed. Results saved to {out_dir}/final_info.json")
    print(f"Analytical solution: {analytical_solution}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Relative Error: {relative_error:.2f}%")
