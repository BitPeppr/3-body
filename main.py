import argparse
import os
import random
import shutil
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------

# Equations of motion ------------
def three_body(t, state, G, m1, m2, m3):
    p1, p2, p3 = state[:6].reshape(3, 2)
    v1, v2, v3 = state[6:].reshape(3, 2)
    dr12 = p2 - p1; r12_sq = dr12 @ dr12 + 1e-3
    dr13 = p3 - p1; r13_sq = dr13 @ dr13 + 1e-3
    dr23 = p3 - p2; r23_sq = dr23 @ dr23 + 1e-3
    a1 = G * m2 * dr12 / (r12_sq * np.sqrt(r12_sq)) + G * m3 * dr13 / (r13_sq * np.sqrt(r13_sq))
    a2 = G * m1 * (-dr12) / (r12_sq * np.sqrt(r12_sq)) + G * m3 * dr23 / (r23_sq * np.sqrt(r23_sq))
    a3 = G * m1 * (-dr13) / (r13_sq * np.sqrt(r13_sq)) + G * m2 * (-dr23) / (r23_sq * np.sqrt(r23_sq))
    return np.concatenate([v1, v2, v3, a1, a2, a3])

# Single-step RK4 (for infinite rendering) ------------------------------------
def rk4_step(f, t, y, h, *args):
    k1 = f(t, y, *args)
    k2 = f(t + h / 2, y + h * k1 / 2, *args)
    k3 = f(t + h / 2, y + h * k2 / 2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# -----------------------------------------------------------------------------

# Get terminal size -----------------------------------------------------------
def get_terminal_size():
    size = shutil.get_terminal_size()
    return size.columns, size.lines

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


# Infinite terminal rendering function ----------------------------------------
def infinite_render(time_step, g, m1, m2, m3, initial_state, columns, lines, trail_length):
    state = initial_state
    steps_per_cycle = 1
    pad = 0.5
    cam_cx = 0.0
    cam_cy = 0.0
    cam_hw = 3.0
    min_cam_hw = 3.5

    trails = [deque(maxlen=trail_length) for _ in range(3)]

    print("\033[?25l", end="", flush=True)
    
    try:
        while True:
            for _ in range(steps_per_cycle):
                state = rk4_step(three_body, 0, state, time_step, g, m1, m2, m3)
            for i in range(3):
                trails[i].append((state[i*2], state[i*2+1]))

            cur_x = [trail[-1][0] for trail in trails]
            cur_y = [trail[-1][1] for trail in trails]

            target_cx = (m1 * cur_x[0] + m2 * cur_x[1] + m3 * cur_x[2]) / (m1 + m2 + m3) * 0.9 + ((min(cur_x) + max(cur_x)) / 2) * 0.1
            target_cy = (m1 * cur_y[0] + m2 * cur_y[1] + m3 * cur_y[2]) / (m1 + m2 + m3) * 0.9 + ((min(cur_y) + max(cur_y)) / 2) * 0.1
            spread = max(max(cur_x) - min(cur_x), max(cur_y) - min(cur_y), 0.5)
            target_hw = max(spread / 2 + pad, min_cam_hw)

            x0, x1 = cam_cx - cam_hw, cam_cx + cam_hw
            y0, y1 = cam_cy - cam_hw, cam_cy + cam_hw

            inside = all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in zip(cur_x, cur_y))
            smoothing = 1.0 if not inside else 0.1

            cam_cx = cam_cx * (1 - smoothing) + target_cx * smoothing
            cam_cy = cam_cy * (1 - smoothing) + target_cy * smoothing
            cam_hw = max(cam_hw * (1 - smoothing) + target_hw * smoothing, min_cam_hw)

            x_min, x_max = cam_cx - cam_hw, cam_cx + cam_hw
            y_min, y_max = cam_cy - cam_hw, cam_cy + cam_hw

            frame = build_frame(trails, columns, lines, x_min, x_max, y_min, y_max)
            print('\033[H' + frame, end='', flush=True)

            time.sleep(time_step)
    finally:
        print("\033[?25h", end="", flush=True)

# Helper functions to actually build ASCII / braille frame ---------------------
def build_frame(trails, columns, rows, x_min, x_max, y_min, y_max):
    bits = [[0] * columns for _ in range(rows)]
    colors = [[(0, 0, 0) for _ in range(columns)] for _ in range(rows)]
    offsets = [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 0), (-2, 0), (0, 2), (0, -2)]
    light_offsets = [(1, -2), (1, 2), (-1, -2), (-1, 2), (2, -1), (2, 1), (-2, -1), (-2, 1)]
    
    for age in range(len(trails[0])):
        for body in range(3):
            brightness = min((age + 1) / 80, 1.0)
            r_, g_, b_ = BODY_COLOURS[body]
            colour = (int(r_ * brightness), int(g_ * brightness), int(b_ * brightness))

            xx, yy = to_dot(trails[body][age][0], trails[body][age][1], x_min, y_min, x_max, y_max, columns, rows)

            row, col, dotx, doty = dot_to_braille(xx, yy)

            if 0 <= row < rows and 0 <= col < columns:
                bit_index = doty + dotx * 3 if doty < 3 else 6 + dotx
                bits[row][col] |= (1 << bit_index)
                colors[row][col] = colour
    

    for body in range(3):
        r_, g_, b_ = BODY_COLOURS[body]
        colour = (int(r_ * 0.3), int(g_ * 0.3), int(b_ * 0.3))
        xx, yy = to_dot(trails[body][-1][0], trails[body][-1][1], x_min, y_min, x_max, y_max, columns, rows)
        
        for dx, dy in light_offsets:
            row, col, dotx, doty = dot_to_braille(xx + dx, yy + dy)
            if 0 <= row < rows and 0 <= col < columns:
                bit_index = doty + dotx * 3 if doty < 3 else 6 + dotx
                bits[row][col] |= (1 << bit_index)
                colors[row][col] = colour
    for body in range(3):
        r_, g_, b_ = BODY_COLOURS[body]
        colour = (r_, g_, b_)
        xx, yy = to_dot(trails[body][-1][0], trails[body][-1][1], x_min, y_min, x_max, y_max, columns, rows)
        
        for dx, dy in offsets:
            row, col, dotx, doty = dot_to_braille(xx + dx, yy + dy)
            if 0 <= row < rows and 0 <= col < columns:
                bit_index = doty + dotx * 3 if doty < 3 else 6 + dotx
                bits[row][col] |= (1 << bit_index)
                colors[row][col] = colour




    lines_out = []
    for row in range(rows):
        line_parts = []
        for col in range(columns):
            if bits[row][col] != 0:
                r_, g_, b_ = colors[row][col]
                line_parts.append(f"\033[38;2;{r_};{g_};{b_}m{chr(0x2800 + bits[row][col])}\033[0m")
            else:
                line_parts.append(chr(0x2800))
        lines_out.append(''.join(line_parts))
    return '\n'.join(lines_out)

def to_dot(x, y, x_min, y_min, x_max, y_max, width, height):
    x_scaled = (x - x_min) / (x_max - x_min) * (width * 2 - 1)
    y_scaled = (y - y_min) / (y_max - y_min) * (height * 4 - 1)
    return x_scaled, y_scaled

def dot_to_braille(x, y):
    cellx = int(x // 2)
    celly = int(y // 4)
    dotx = int(x % 2)
    doty = int(y % 4)
    return celly, cellx, dotx, doty


# -----------------------------------------------------------------------------

# Infinite loop to find interesting configurations ----------------------------
def infinite_config_finder(sim_time):
    # Make directories if they don't already exist
    os.makedirs(os.path.join(cwd, 'unsure/'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'interesting/'), exist_ok=True)
    os.makedirs(os.path.join(cwd, 'rejected/'), exist_ok=True)

    # Define constant parameters
    t_span = (0, sim_time)
    t_eval = np.linspace(t_span[0], t_span[1], sim_time * 2000)
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
    parser.add_argument('--time', type=float, default=100.0, help='Simulation time for single and random-search modes')
    parser.add_argument('--g', type=float, default=1.0, help='Gravitational constant for all modes')
    parser.add_argument('--m1', type=float, default=1.0, help='Mass of body 1 for single and infinite mode')
    parser.add_argument('--m2', type=float, default=1.0, help='Mass of body 2 for single and infinite modes')
    parser.add_argument('--m3', type=float, default=1.0, help='Mass of body 3 for single and infinite modes')
    parser.add_argument('--save', action='store_true', help='Whether to save the single_render for single mode')
    parser.add_argument('--initial-state', type=float, nargs=12, help='Initial state (positions and velocities) for single mode and infinite')
    parser.add_argument('--red', type=int, nargs=3, help='RGB colour to use as red')
    parser.add_argument('--blue', type=int, nargs=3, help='RGB colour to use as blue')
    parser.add_argument('--green', type=int, nargs=3, help='RGB colour to use as green')
    parser.add_argument('--save_path', type=str, help = 'save path for single_renders')
    parser.add_argument('--time_step', type=float, default=0.01, help='Time step for infinite mode')
    parser.add_argument('--trail_length', type=int, default=80, help='Trail length for infinite mode')
    args = parser.parse_args()
    return args




# -----------------------------------------------------------------------------

# Main loop -------------------------------------------------------------------
if __name__ == "__main__":
    args = parse()
    red = args.red if args.red else (255, 140, 160)
    blue = args.blue if args.blue else (130, 175, 255)
    green = args.green if args.green else (120, 220, 120)
    BODY_COLOURS = [red, blue, green]
    cwd = os.path.expanduser(args.save_path) if args.save_path else os.getcwd()
    if args.mode == 'single':
        initial_state = np.array(args.initial_state) if args.initial_state else np.random.uniform(-1.0, 1.0, 12)
        single_render(int(args.time), args.g, args.m1, args.m2, args.m3, initial_state, args.save)
        print("Single simulation completed.")
        print(f"Parameters: Time={args.time}, G={args.g}, m1={args.m1}, m2={args.m2}, m3={args.m3}")
        print(f"Initial State: {initial_state}")
    if args.mode == 'infinite':
        columns, lines = get_terminal_size()
        initial_state = np.array(args.initial_state) if args.initial_state else np.random.uniform(-1.0, 1.0, 12)
        infinite_render(args.time_step, args.g, args.m1, args.m2, args.m3, initial_state, columns, lines, args.trail_length)
    if args.mode == 'random-search':
        infinite_config_finder(int(args.time))
