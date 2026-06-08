import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------

# Equations of motion ------------
def three_body(t, state, G, m1, m2, m3):
    p1, p2, p3 = state[:6].reshape(3, 2)
    v1, v2, v3 = state[6:].reshape(3, 2)
    r12 = np.linalg.norm(p2 - p1)
    r13 = np.linalg.norm(p3 - p1)
    r23 = np.linalg.norm(p3 - p2)
    a1 = G * m2 * (p2 - p1) / r12**3 + G * m3 * (p3 - p1) / r13**3
    a2 = G * m1 * (p1 - p2) / r12**3 + G * m3 * (p3 - p2) / r23**3
    a3 = G * m1 * (p1 - p3) / r13**3 + G * m2 * (p2 - p3) / r23**3
    return np.concatenate([v1, v2, v3, a1, a2, a3])

# -----------------------------------------------------------------------------

# One-time rendering function for a given configuration, used for single mode -
def single_render(time, g, im1, im2, im3, initial_state, save):
    t_span = (0, time)
    t_eval = np.linspace(t_span[0], t_span[1], time * 2000)
    solution = solve_ivp(three_body, t_span, initial_state, t_eval=t_eval, method='RK45', args=(g, im1, im2, im3))
    x1, y1 = solution.y[0], solution.y[1]
    x2, y2 = solution.y[2], solution.y[3]
    x3, y3 = solution.y[4], solution.y[5]
    
    if save:
        plt.figure(figsize=(8, 8), dpi=700)
    else:
        plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(x1, y1, color='crimson', label='Body 1')
    plt.plot(x2, y2, color='royalblue', label='Body 2')
    plt.plot(x3, y3, color='forestgreen', label='Body 3')
    plt.scatter(x1[-1], y1[-1], color='crimson', s=50)
    plt.scatter(x2[-1], y2[-1], color='royalblue', s=50)
    plt.scatter(x3[-1], y3[-1], color='forestgreen', s=50)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.axis('off')

    if save:
        plt.savefig('single_render.png')
    else:
        plt.show()
    plt.close()

# -----------------------------------------------------------------------------

# Infinite loop to find interesting configurations ----------------------------
def infinite_config_finder():
    # Make directories if they don't already exist
    os.makedirs(os.path.join(cwd, 'unsure/'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'interesting/'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'rejected/'), exist_ok=True)

    # Define constant parameters
    t_span = (0, 1000)
    t_eval = np.linspace(t_span[0], t_span[1], 2000000)
    G = 1.0


    n = 0
    while True:
        n+=1
        # Define random parameters for this simulation
        m1, m2, m3 = np.random.uniform(0.5, 2.0, 3)
        initial_state = np.random.uniform(-1.0, 1.0, 12)
        # Start timer for simulation
        start_time = time.monotonic()

        solution = solve_ivp(three_body, t_span, initial_state, t_eval=t_eval, method='RK45', args=(G, m1, m2, m3))
        x1, y1 = solution.y[0], solution.y[1]
        x2, y2 = solution.y[2], solution.y[3]
        x3, y3 = solution.y[4], solution.y[5]

        plt.figure(figsize=(8, 8), dpi=700)
        plt.plot(x1, y1, color='crimson', label='Body 1')
        plt.plot(x2, y2, color='royalblue', label='Body 2')
        plt.plot(x3, y3, color='forestgreen', label='Body 3')
        plt.scatter(x1[-1], y1[-1], color='crimson', s=50)
        plt.scatter(x2[-1], y2[-1], color='royalblue', s=50)
        plt.scatter(x3[-1], y3[-1], color='forestgreen', s=50)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axis('equal')
        plt.axis('off')
        plt.figtext(
            0.5, 0.02,
            f"Masses: {m1:.6f}, {m2:.6f}, {m3:.6f} \n Pos: {initial_state[:6].round(2).tolist()}\n Vel: {initial_state[6:].round(2).tolist()}",
            ha='center',
            fontsize=4,
            color='gray',
            style='italic'
        )
        results, message = judging(solution)

        if results:
            if "UNSURE" in message.upper():
                save_path = os.path.join(cwd, 'unsure/')
            else: 
                save_path = os.path.join(cwd, 'interesting/')
        else:
            save_path = os.path.join(cwd, 'rejected/')
        plt.savefig(os.path.join(save_path, f'{random.randint(0, 1000000000000000000)}.png'))
        plt.close()

        # End time and print out rate of simulations
        end_time = time.monotonic()
        print(f"Simulation {n} completed in {end_time - start_time:.2f} seconds, written to {save_path} with message: {message}")
        speed = 1 / (end_time - start_time)
        print(f"Speed: {speed:.2f} simulations per second.")

# Judging of interesting configurations; helper for infinite_config_finder ----
def judging(solution):
    positions = solution.y[:6, :]
    r1 = np.linalg.norm(positions[0:2, :], axis=0)
    r2 = np.linalg.norm(positions[2:4, :], axis=0)
    r3 = np.linalg.norm(positions[4:6, :], axis=0)
    max_distance = max(np.max(r1), np.max(r2), np.max(r3))
    if max_distance > 15.0:  # If they drift too far out, they escaped
        return False, "Escaped into deep space"

    initial_state = solution.y[:, 0]
    differences = solution.y.T - initial_state
    distances_to_start = np.linalg.norm(differences, axis=1)
    ignore_warmup = int(len(distances_to_start) * 0.15)
    best_return_idx = np.argmin(distances_to_start[ignore_warmup:]) + ignore_warmup
    closest_approach = distances_to_start[best_return_idx]
    if closest_approach < 0.2:
        return True, f"MATCH! Periodic Loop Found (Error: {closest_approach:.3f})"
    if max_distance < 3.0:
        return True, f"Bounded Geometric Structure (Max Dist: {max_distance:.2f})"

    return True, "Unsure"

# -----------------------------------------------------------------------------

# Command-line argument parsing -----------------------------------------------
def parse():
    parser = argparse.ArgumentParser(description="3-Body Problem Simulator")
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'infinite', 'random-search'], help='Mode of operation')
    parser.add_argument('--time', type=float, default=1000.0, help='Simulation time for single and random-search modes')
    parser.add_argument('--g', type=float, default=1.0, help='Gravitational constant for all modes')
    parser.add_argument('--m1', type=float, default=1.0, help='Mass of body 1 for single and infinite mode')
    parser.add_argument('--m2', type=float, default=1.0, help='Mass of body 2 for single and infinite modes')
    parser.add_argument('--m3', type=float, default=1.0, help='Mass of body 3 for single and infinite modes')
    parser.add_argument('--save', action='store_true', help='Whether to save the single_render for single mode')
    parser.add_argument('--initial-state', type=float, nargs=12, help='Initial state (positions and velocities) for single mode and infinite')
    parser.add_argument('--save_path', type=str, help = 'save path for single_renders')
    args = parser.parse_args()
    return args




# -----------------------------------------------------------------------------

# Main loop -------------------------------------------------------------------
if __name__ == "__main__":
    args = parse()
    cwd = os.path.expanduser(args.save_path) if args.save_path else os.getcwd()
    if args.mode == 'single':
        initial_state = args.initial_state if args.initial_state else np.random.uniform(-1.0, 1.0, 12)
        single_render(int(args.time), args.g, args.m1, args.m2, args.m3, initial_state, args.save)
        print("Single simulation completed.")
        print(f"Parameters: Time={args.time}, G={args.g}, m1={args.m1}, m2={args.m2}, m3={args.m3}")
        print(f"Initial State: {initial_state}")
    if args.mode == 'infinite':
        pass
    if args.mode == 'random-search':
        infinite_config_finder()
