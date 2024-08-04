''' example script for running the beamprop code in prop.py'''
import sys
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
import numpy as np
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize,overlap_nonu,norm_nonu
from lightbeam import LPmodes
import matplotlib.pyplot as plt
from config_example import *
import time
import json

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

wl = 1.55 # um
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
sm_offset = 500 # um
sm_ex = 40000 # um
scale_func = None
l = 2
m = 2

xw_func = None
yw_func = None
wl0 = 1.55
monitor_func = None
writeto = None
ref_val = 1e-2
remesh_every = 100
dynamic_n0 = False
fplanewidth = 0
xpos = [0]
ypos = [0]
zw = sm_ex

# Define the range of di and dz values for the grid search
di_values = [0.75]
dz_values = [50]
results_folder = "grid_search_results_testings"

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
        iw0 = 600
        num_PML = int(round(iw0 / 9))
        print(iw0, num_PML)

        mesh = RectMesh3D(iw0, iw0, zw, di, dz, num_PML, xw_func, yw_func)
        xg0, yg0 = mesh.xy.xg, mesh.xy.yg
        xg, yg = mesh.xg[num_PML:-num_PML, num_PML:-num_PML], mesh.yg[num_PML:-num_PML, num_PML:-num_PML]
        # u0 = normalize(LPmodes.lpfield(xg, yg, l, m, mm_rclad, wl, nclad, njack))

        optic.set_sampling(mesh.xy)
        
        # out = np.zeros(mesh.xy.shape)

        # optic.set_IORsq(out,sm_offset)
        # plt.imshow(out,vmin=njack*njack,vmax=ncore*ncore)
        # plt.show()

        # optic.set_IORsq(out,sm_ex)
        # plt.imshow(out,vmin=njack*njack,vmax=ncore*ncore)
        # plt.show()

        # Save initial conditions
        initial_conditions = {
            "l": l,
            "m": m,
            "ref_val": ref_val,
            "remesh_every": remesh_every,
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
        # u0 = (1 + 0j) * normalize(LPmodes.lpfield(xg0, yg0, l, m, sm_rcore, wl, nclad, njack))

        # w = mesh.xy.get_weights()

        # Save input light (real and imaginary parts separately)

        # modes_before = []
        # sumofmodes_before = 0
        # for x, y in lant19_ipos:
        #     mode = norm_nonu(LPmodes.lpfield(xg0 - x, yg0 - y, l, m, sm_rcore, wl0, ncore, nclad), w)
        #     sumofmodes_before += mode
        #     modes_before.append(mode.tolist())

        # SMFpower_before = 0
        # for i in range(len(modes_before)):
        #     _p = np.power(overlap_nonu(u0, np.array(modes_before[i]), w), 2)
        #     SMFpower_before += _p

        # abs_u0_at_positions = []
        # for x, y in lant19_ipos:
        #     ix = np.argmin(np.abs(mesh.xy.xa0 - x))
        #     iy = np.argmin(np.abs(mesh.xy.ya0 - y))
        #     abs_u0_at_positions.append(np.abs(u0[ix, iy]))

        u0 = normalize(LPmodes.lpfield(xg, yg, l, m, mm_rclad, wl, nclad, njack)) #(1 + 0j) * 
        print(u0.shape)
        input_light_real = np.real(u0).tolist()
        input_light_imag = np.imag(u0).tolist()

        # Propagation
        start_time = time.time()
        ux, u0 = prop.prop2end(u0,monitor_func=monitor_func,xyslice=None,zslice=None,writeto=writeto,ref_val=ref_val,remesh_every=remesh_every,dynamic_n0=dynamic_n0,fplanewidth=fplanewidth)
        #u0 = prop.prop2end_uniform(u0)
        end_time = time.time()
        runtime = end_time - start_time
        plt.imshow(np.abs(u0))
        plt.show()
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
        save_results(results_folder, di, dz, initial_conditions, input_light_real, input_light_imag, powers, abs_u0_final_positions, outputs_real, outputs_imag, runtime)
        print("Saved!")

print("Grid search completed.")

# Perform your analysis here
print(f"di: {di}, dz: {dz}, runtime: {runtime}, power: {powers}")
print(initial_conditions)

# Convert lists to numpy arrays for plotting
input_light_real = np.array(input_light_real)
input_light_imag = np.array(input_light_imag)
outputs_real = np.array(outputs_real)
outputs_imag = np.array(outputs_imag)
print(input_light_real)

# Plotting input light
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title(f"Input Light: di={di}, dz={dz}")
plt.imshow(np.sqrt(input_light_real**2 + input_light_imag**2))
plt.xlabel('X')
plt.ylabel('Y')

# Plotting output light
plt.subplot(1, 2, 2)
plt.title(f"Output Light: di={di}, dz={dz}")
plt.imshow(np.sqrt(outputs_real**2 + outputs_imag**2))
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()