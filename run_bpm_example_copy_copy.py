import numpy as np
import sys
sys.path.append("../src")
import os
import time
import json
import lightbeam
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize, overlap_nonu, norm_nonu
from lightbeam import LPmodes
import matplotlib.pyplot as plt
from config_example import *

def save_results(folder, di, dz, initial_conditions, input_light_real, input_light_imag, powers, abs_u0_at_positions, abs_u0_final_positions, outputs_real, outputs_imag, runtimes):
    if not os.path.exists(folder):
        os.makedirs(folder)
    results = {
        "di": di,
        "dz": dz,
        "initial_conditions": initial_conditions,
        "input_light_real": input_light_real,
        "input_light_imag": input_light_imag,
        "powers": powers,
        "abs_u0_at_positions": abs_u0_at_positions,
        "abs_u0_final_positions": abs_u0_final_positions,
        "outputs_real": outputs_real,
        "outputs_imag": outputs_imag,
        "runtimes": runtimes
    }
    with open(f"{folder}/results_di_{di}_dz_{dz}.json", "w") as f:
        json.dump(results, f, indent=4)

wl = 1.5 # um
njack = 1.4345
nclad = 1.44
ncore = 1.4522895
mm_rjack = 76.3 # um
mm_rclad = 32.8 # um
taper_ratio = 10
sm_rcore = 6.5 # um
mm_rjack = 76.3 # um
sm_rclad = 32.8 # um
# sm_seperation_final = 2*6.5
sm_seperation_final = 2*60 # um
sm_offset = 0 # um
sm_ex = 40000 # um
scale_func = None
l = 2
m = 2

xw_func = None
yw_func = None
max_remesh_iters = 10
sig_max = 1.0
wl0 = 1.55
monitor_func = None
writeto = None
ref_val = 2e-3
remesh_every = 100
dynamic_n0 = False
fplanewidth = 10
xpos = [0]
ypos = [0]
zw = sm_ex

# Define the range of di and dz values for the grid search
di_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
dz_values = [ 20, 35, 50, 65, 80, 100]
results_folder = "grid_search_results"

lant19_ipos = optics.lant19.get_19port_positions(core_spacing=sm_seperation_final/taper_ratio)
lant19_fpos = optics.lant19.get_19port_positions(core_spacing=sm_seperation_final)
clad_mm = optics.scaled_cyl([0,0],mm_rclad,sm_offset,nclad,njack,0,scale_func=scale_func,final_scale=1)
clad_sm = optics.scaled_cyl([0,0],sm_rclad,sm_ex,nclad,njack,sm_offset,scale_func=scale_func,final_scale=taper_ratio)
elmnts = [clad_mm, clad_sm]

for i in range(0,len(lant19_ipos)):
    core = optics.scaled_cyl(xy=lant19_ipos[i] ,r = sm_rcore/taper_ratio,z_ex = sm_ex,n = ncore,nb = nclad,z_offset=sm_offset,scale_func=scale_func,final_scale=taper_ratio) # fxy=lant19_fpos[i]
    elmnts.append(core)

optic = optics.OpticSys(elmnts,njack)

for di in di_values:
    for dz in dz_values:
        print(di, dz)
        iw0 = int(round(300 / di))
        num_PML = int(round(iw0 / 9))
        print(iw0, num_PML)

        mesh = RectMesh3D(iw0, iw0, zw, di, dz, num_PML, xw_func, yw_func)
        xg0, yg0 = mesh.xy.xg, mesh.xy.yg
        xg, yg = mesh.xg[num_PML:-num_PML, num_PML:-num_PML], mesh.yg[num_PML:-num_PML, num_PML:-num_PML]
        u0 = normalize(LPmodes.lpfield(xg, yg, l, m, sm_rcore, wl, nclad, njack))

        optic.set_sampling(mesh.xy)
        out = np.zeros(mesh.xy.shape)

        # Save initial conditions
        initial_conditions = {
            "l": l,
            "m": m,
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
        u0 = (1 + 0j) * normalize(LPmodes.lpfield(xg0, yg0, l, m, sm_rcore, wl, nclad, njack))

        w = mesh.xy.get_weights()

        # Save input light (real and imaginary parts separately)
        input_light_real = np.real(u0).tolist()
        input_light_imag = np.imag(u0).tolist()

        modes_before = []
        sumofmodes_before = 0
        for x, y in lant19_ipos:
            mode = norm_nonu(LPmodes.lpfield(xg0 - x, yg0 - y, l, m, sm_rcore, wl0, ncore, nclad), w)
            sumofmodes_before += mode
            modes_before.append(mode.tolist())

        SMFpower_before = 0
        for i in range(len(modes_before)):
            _p = np.power(overlap_nonu(u0, np.array(modes_before[i]), w), 2)
            SMFpower_before += _p

        abs_u0_at_positions = []
        for x, y in lant19_ipos:
            ix = np.argmin(np.abs(mesh.xy.xa0 - x))
            iy = np.argmin(np.abs(mesh.xy.ya0 - y))
            abs_u0_at_positions.append(np.abs(u0[ix, iy]))

        u0 = (1 + 0j) * normalize(LPmodes.lpfield(xg, yg, l, m, sm_rcore, wl, nclad, njack))
        print(u0.shape)

        # Propagation
        start_time = time.time()
        ux, u0 = prop.prop2end(u0, monitor_func=monitor_func, xyslice=None, zslice=None, writeto=writeto, ref_val=ref_val, remesh_every=remesh_every, dynamic_n0=dynamic_n0, fplanewidth=fplanewidth)
        end_time = time.time()
        runtime = end_time - start_time

        # Save outputs (real and imaginary parts separately)
        outputs_real = np.real(u0).tolist()
        outputs_imag = np.imag(u0).tolist()

        xg,yg = np.meshgrid(mesh.xy.xa,mesh.xy.ya,indexing='ij')
        w = mesh.xy.get_weights()

        xg0,yg0 = np.meshgrid(mesh.xy.xa0,mesh.xy.ya0,indexing='ij')
        w0 = mesh.xy.dx0*mesh.xy.dy0

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
        save_results(results_folder, di, dz, initial_conditions, input_light_real, input_light_imag, powers, abs_u0_at_positions, abs_u0_final_positions, outputs_real, outputs_imag, runtime)
        print("Saved!")

print("Grid search completed.")