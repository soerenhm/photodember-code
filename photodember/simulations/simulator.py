import gc
import pathlib
from ..src.transport.simulation import *


def advance(pde: OdeFunction, init_state: SimulationState, ti: float, dt: float) -> SimulationState:
    gc.collect() # sometimes memory consumption keeps rising, dunno why... this helps...
    return solve_pde(pde, init_state, [ti, ti + dt])[-1]

def save_state_in(f, state: SimulationState, t: float):
    s = np.append([t], state.array).astype(np.float64).tobytes()
    f.write(s)

def simulation_step(f, pde: OdeFunction, init_state: SimulationState, ti: float, tf: float):
    new_state = advance(pde, init_state, ti, tf-ti)
    save_state_in(f, new_state, tf)
    return new_state

def read_simulation_file(filename: str, initial_state: SimulationState) -> Tuple[List[float], List[SimulationState]]:
    t = []
    states = []
    chunksize = len(initial_state.array) + 1
    with open(filename, "rb") as io:
        content = np.frombuffer(io.read(), dtype=np.float64)
        start, stop = 0, chunksize
        while stop <= len(content):
            section = content[start:stop]
            t.append(section[0])
            state = SimulationState(section[1:], initial_state.grid_points, initial_state.number_particles)
            states.append(state)
            start, stop = stop, stop + chunksize
    return t, states

def run_simulation(pde: OdeFunction, save_file: str, init_state: SimulationState, times: ArrayLike):
    t = list(times)
    ti = t.pop(0)
    # Scan file for existing entries
    if pathlib.Path(save_file).exists():
        read_t, read_states = read_simulation_file(save_file, init_state)
        if len(read_t) > 0:
            if ti == read_t[0] and all(np.isclose(init_state.array, read_states[0].array)):
                ti = read_t[-1]
                init_state = read_states[-1]
                t = list(filter(lambda t: t > ti, times))
            else:
                raise ValueError("Simulation file already exists but saved states differ")
    else:
        with open(save_file, "wb") as f:
            save_state_in(f, init_state, ti)
    # Loop
    states = [init_state]
    while len(t) > 0:
        tf = t.pop(0)
        print(f"Advancing state from t={ti} to t={tf}...")
        with open(save_file, "ab") as f:
            new_state = simulation_step(f, pde, init_state, ti, tf)
        states.append(new_state)
        ti = tf
        init_state = new_state
    return states