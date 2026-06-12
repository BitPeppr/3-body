# TriSolaris

A command-line based 3-body problem visualiser, simulator, and renderer, with live terminal-visual modes, png-render output, or infinite brute-force configuration finder functions. Highly customisable; every aspect from masses, gravity, positions and velocities, simulation time steps, etc. can be customised with extensive CLI flags. Great as a screen saver, or for accurate (rk4) renders of three bodies.

To use:

```python
python main.py # Default, random single render
python main.py --initial-state ... # Custom initial state, with masses, positions, velocities
python main.py --mode infinite # Infinite, terminal-based braille visualisation, updating live.
python main.py -h # For more options and details, check out help menu
```
