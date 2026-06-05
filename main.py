import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

#   # Constants ------------
#   G = 1.0
#   m1, m2, m3 = 1.0, 1.0, 1.0
#   initial_state = np.array([
#       -1.0, 0.0, # Body 1 starting position
#       1.0, 0.0,  # Body 2 starting position
#       0.0, 1.0,  # Body 3 starting position
#       0.0, -0.5, # Body 1 starting velocity
#       0.0, 0.5,  # Body 2 starting velocity
#       0.5, 0.0   # Body 3 starting velocity
#       ])
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 20000)


# Equations of motion ------------
def three_body(t, state):
    p1, p2, p3 = state[:6].reshape(3, 2)  # Positions
    v1, v2, v3 = state[6:].reshape(3, 2)  # Velocities
    r12 = np.linalg.norm(p2 - p1)
    r13 = np.linalg.norm(p3 - p1)
    r23 = np.linalg.norm(p3 - p2)
    a1 = G * m2 * (p2 - p1) / r12**3 + G * m3 * (p3 - p1) / r13**3
    a2 = G * m1 * (p1 - p2) / r12**3 + G * m3 * (p3 - p2) / r23**3
    a3 = G * m1 * (p1 - p3) / r13**3 + G * m2 * (p2 - p3) / r23**3
    return np.concatenate([v1, v2, v3, a1, a2, a3])

def main():
    start_time = time.monotonic()
    solution = solve_ivp(three_body, t_span, initial_state, t_eval=t_eval, method='RK45')
    x1, y1 = solution.y[0], solution.y[1]
    x2, y2 = solution.y[2], solution.y[3]
    x3, y3 = solution.y[4], solution.y[5]

    plt.figure(figsize=(8, 8))
    plt.plot(x1, y1, color='crimson', label='Body 1')
    plt.plot(x2, y2, color='royalblue', label='Body 2')
    plt.plot(x3, y3, color='forestgreen', label='Body 3')
    plt.scatter(x1[-1], y1[-1], color='crimson', s=50)
    plt.scatter(x2[-1], y2[-1], color='royalblue', s=50)
    plt.scatter(x3[-1], y3[-1], color='forestgreen', s=50)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(f'{n}.png')
    plt.close()
    end_time = time.monotonic()
    print(f"Simulation {n} completed in {end_time - start_time:.2f} seconds.")
    speed = 1 / (end_time - start_time)
    print(f"Speed: {speed:.2f} simulations per second.")

n = 0
while True:
    n += 1
    G = 1.0
    m1, m2, m3 = np.random.uniform(0.5, 2.0, 3)
    initial_state = np.random.uniform(-1.0, 1.0, 12)
    try:
        main()
    except Exception as e:        print(f"An error occurred: {e}")
