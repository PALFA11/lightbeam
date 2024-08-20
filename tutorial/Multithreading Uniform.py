import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
lightbeam_path = os.path.abspath(os.path.join(script_dir, '../src'))
print("Adding to sys.path:", lightbeam_path)

sys.path.append(lightbeam_path)
print("sys.path:", sys.path)

if os.path.exists(lightbeam_path):
    print(f"Contents of {lightbeam_path}:")
    print(os.listdir(lightbeam_path))
else:
    print(f"Directory does not exist: {lightbeam_path}")
try:
    import lightbeam
    print("Successfully imported lightbeam")
except ImportError as e:
    print(f"Failed to import lightbeam: {e}")
# Assuming the necessary imports for your custom functions and classes
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize, overlap_nonu, norm_nonu
from lightbeam import LPmodes
from config_example import *

def save_results(folder, di, dz, initial_conditions, input_light_real, input_light_imag, powers, abs_u0_final_positions, outputs_real, outputs_imag, runtimes):
    if not os.path.exists(folder):
        os.makedirs(folder)
    results = {
        "di": di,
        "dz": dz,
        "initial_conditions": initial_conditions,
        "input_light_real": input_light_real,
        "input_light_imag": input_light_imag,
        "powers": powers,
        "abs_u0_final_positions": abs_u0_final_positions,
        "outputs_real": outputs_real,
        "outputs_imag": outputs_imag,
        "runtimes": runtimes
    }
    with open(f"{folder}/results_di_{di}_dz_{dz}.json", "w") as f:
        json.dump(results, f, indent=4)

def create_input_mode_field(xg, yg, l_values, m_values, mm_rclad, wl, nclad, njack, amplitudes=None):
    # Initialize the input mode field to zero
    u0 = np.zeros_like(xg, dtype=complex)
    
    # If amplitudes are not provided, set them to 1 for all modes
    if amplitudes is None:
        amplitudes = np.ones(len(l_values))
    
    # Loop over the provided l and m values
    for i, (l, m) in enumerate(zip(l_values, m_values)):
        # Compute the LP mode field for given l, m
        lp_mode_field = LPmodes.lpfield(xg, yg, l, m, mm_rclad, wl, nclad, njack)
        
        # Normalize the mode field
        lp_mode_field_normalized = lp_mode_field / np.sqrt(np.sum(np.abs(lp_mode_field)**2))
        
        # Add the normalized mode field to u0 with the corresponding amplitude
        u0 += amplitudes[i] * lp_mode_field_normalized
    
    # Normalize the resulting input mode field
    u0 = u0 / np.sqrt(np.sum(np.abs(u0)**2))
    
    return u0

def run_simulation(di, dz, results_folder):
    wl = 1.5  # um
    njack = 1.4345
    nclad = 1.44
    ncore = 1.4522895
    mm_rjack = 76.3  # um
    mm_rclad = 32.8  # um
    taper_ratio = 10
    sm_rcore = 6.5  # um
    sm_rclad = 32.8  # um
    sm_seperation_final = 2 * 60  # um
    sm_offset = 500  # um
    sm_ex = 40000  # um
    scale_func = None
    ls = [0,1,2]
    ms = [1,2,3]
    amplitudes = np.linspace(1.5,0.5,9)
    xw_func = None
    yw_func = None
    wl0 = 1.5
    monitor_func = None
    writeto = None
    ref_val = 1
    remesh_every = 100
    dynamic_n0 = False
    fplanewidth = 0
    xpos = [0]
    ypos = [0]
    zw = sm_ex

    lant19_ipos = optics.lant19.get_19port_positions(core_spacing=sm_seperation_final / taper_ratio)
    lant19_fpos = optics.lant19.get_19port_positions(core_spacing=sm_seperation_final)
    clad_mm = optics.scaled_cyl([0, 0], mm_rclad, sm_offset, nclad, njack, 0, scale_func=scale_func, final_scale=1)
    clad_sm = optics.scaled_cyl([0, 0], sm_rclad, sm_ex, nclad, njack, sm_offset, scale_func=scale_func, final_scale=taper_ratio)
    elmnts = [clad_mm, clad_sm]

    for i in range(0, len(lant19_ipos)):
        core = optics.scaled_cyl(xy=lant19_ipos[i], r=sm_rcore / taper_ratio, z_ex=sm_ex, n=ncore, nb=nclad, z_offset=sm_offset, scale_func=scale_func, final_scale=taper_ratio)
        elmnts.append(core)

    optic = optics.OpticSys(elmnts, njack)

    print(di, dz)
    iw0 = 650
    num_PML = 100
    print(iw0, num_PML)

    mesh = RectMesh3D(iw0, iw0, zw, di, dz, num_PML, xw_func, yw_func)
    xg0, yg0 = mesh.xy.xg, mesh.xy.yg
    xg, yg = mesh.xg[num_PML:-num_PML, num_PML:-num_PML], mesh.yg[num_PML:-num_PML, num_PML:-num_PML]

    optic.set_sampling(mesh.xy)

    # Save initial conditions
    initial_conditions = {
        "ls": ls,
        "ms": ms,
        "amplitudes": amplitudes,
        "ref_val": ref_val,
        "remesh_every": remesh_every,
        "iw0":iw0,
        "num_PML" : num_PML,
        "di": di,
        "dz": dz,
        "mesh_shape": mesh.xy.shape,
        "njack": njack,
        "nclad": nclad,
        "ncore": ncore,
        "sm_rcore": sm_rcore,
        "sm_rclad": sm_rclad,
        "taper_ratio": taper_ratio,
        "sm_seperation_final": sm_seperation_final,
        "sm_offset": sm_offset,
        "sm_ex": sm_ex
    }

    # Initialize propagator
    prop = Prop3D(wl0, mesh, optic, nclad)

    u0 = create_input_mode_field(xg, yg, ls, ms, mm_rclad, wl, nclad, njack, amplitudes=amplitudes)  # (1 + 0j) * 
    print(u0.shape)
    # plt.imshow(np.abs(u0))
    # plt.show()
    input_light_real = np.real(u0).tolist()
    input_light_imag = np.imag(u0).tolist()

    # Propagation
    start_time = time.time()
    ux, u0 = prop.prop2end(u0, monitor_func=monitor_func, xyslice=None, zslice=None, writeto=writeto, ref_val=ref_val, remesh_every=remesh_every, dynamic_n0=dynamic_n0, fplanewidth=fplanewidth)
    end_time = time.time()
    runtime = end_time - start_time

    # Save outputs (real and imaginary parts separately)
    outputs_real = np.real(u0).tolist()
    outputs_imag = np.imag(u0).tolist()

    xg, yg = np.meshgrid(mesh.xy.xa, mesh.xy.ya, indexing='ij')
    w = mesh.xy.get_weights()

    # Calculate final power decomposition
    modes_after = []
    sumofmodes_after = 0
    for x, y in lant19_fpos:
        mode = norm_nonu(LPmodes.lpfield(xg - x, yg - y, 0, 1, sm_rcore, wl0, ncore, nclad), w)
        sumofmodes_after += mode
        modes_after.append(mode.tolist())

    SMFpower_after = 0
    powers = []
    for i in range(len(modes_after)):
        _p = np.power(overlap_nonu(ux, np.array(modes_after[i]), w), 2)
        powers.append(_p)
        SMFpower_after += _p

    abs_u0_final_positions = []
    for x, y in lant19_fpos:
        ix = np.argmin(np.abs(mesh.xy.xa0 - x))
        iy = np.argmin(np.abs(mesh.xy.ya0 - y))
        abs_u0_final_positions.append(np.abs(u0[ix, iy]))

    # Save the results
    save_results(results_folder, di, dz, initial_conditions, input_light_real, input_light_imag, powers, abs_u0_final_positions, outputs_real, outputs_imag, runtime)
    print(f"Simulation completed for di={di}, dz={dz}, runtime={runtime}")
    return di, dz, runtime, powers


def main():
    # Define the range of di and dz values for the grid search
    max_threads = 4  # Limit to 4 threads
    di_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    dz_values = [7.5, 6, 5.5, 5, 4.5, 4, 3.75, 3.5, 3.25, 3, 2.5]

    results_folder = "uniform"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(run_simulation, di, dz, results_folder) for di in di_values for dz in dz_values]

        for future in futures:
            di, dz, runtime, powers = future.result()
            print(f"di: {di}, dz: {dz}, runtime: {runtime}, power: {powers}")

if __name__ == "__main__":
    main()