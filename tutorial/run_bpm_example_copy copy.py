''' example script for running the beamprop code in prop.py'''
import numpy as np
import sys
sys.path.append("../src")
import lightbeam
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize,overlap_nonu,norm_nonu
from lightbeam import LPmodes
import matplotlib.pyplot as plt
from config_example import *

wl = 1.5 # um
njack = 1.4345
nclad = 1.44
ncore = 1.4522895
mm_rjack = 76.3 # um
mm_rclad = 32.8 # um
taper_ratio = 2.125
sm_rcore = 6.5/10*taper_ratio # um
mm_rjack = 76.3 # um
sm_rclad = 32.8 # um
# sm_seperation_final = 2*6.5
sm_seperation_final = 2*6*taper_ratio # um
sm_offset = 0 # um
sm_ex = 5000 # um
scale_func = None
l = 0
m = 1

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
iw0 = 600
zw = sm_ex
di = 1
dz = 10
num_PML = 100

lant19_ipos = optics.lant19.get_19port_positions(core_spacing=sm_seperation_final/taper_ratio)
lant19_fpos = optics.lant19.get_19port_positions(core_spacing=sm_seperation_final)
#clad_mm = optics.scaled_cyl([0,0],mm_rclad,sm_offset,nclad,njack,0,scale_func=scale_func,final_scale=1)
clad_sm = optics.scaled_cyl([0,0],sm_rclad,sm_ex,nclad,njack,sm_offset,scale_func=scale_func,final_scale=taper_ratio)
elmnts = [clad_sm] #[clad_mm, clad_sm]

for i in range(0,len(lant19_ipos)):
    core = optics.scaled_cyl(xy=lant19_ipos[i] ,r = sm_rcore/taper_ratio,z_ex = sm_ex,n = ncore,nb = nclad,z_offset=sm_offset,scale_func=scale_func,final_scale=taper_ratio) # fxy=lant19_fpos[i]
    elmnts.append(core)

optic = optics.OpticSys(elmnts,njack)


mesh = RectMesh3D(iw0,iw0,zw,di,dz,num_PML,xw_func,yw_func)
xg0,yg0 = mesh.xy.xg,mesh.xy.yg
xg, yg = mesh.xg[num_PML:-num_PML,num_PML:-num_PML],mesh.yg[num_PML:-num_PML,num_PML:-num_PML]
u0 = normalize(LPmodes.lpfield(xg,yg,l,m,sm_rcore,wl,nclad,njack))

optic.set_sampling(mesh.xy)
out = np.zeros(mesh.xy.shape)

fig, axs = plt.subplots(1, 1, figsize=(15, 5))
# Generate the first subplot
optic.set_IORsq(out, 0)
plt.imshow(out, vmin=njack*njack, vmax=ncore*ncore)
plt.show()
out = np.zeros(mesh.xy.shape)
optic.set_IORsq(out, sm_ex)
plt.imshow(out, vmin=njack*njack, vmax=ncore*ncore)
plt.show()
mesh.xy.max_iters = max_remesh_iters
mesh.sigma_max = sig_max

# propagator initialization (required)
prop = Prop3D(wl0,mesh,optic,nclad)
u0 = normalize(LPmodes.lpfield(xg0,yg0,l,m,sm_rcore,wl,nclad,njack))
print('launch field')
plt.imshow(np.real(u0))
plt.show()
w = mesh.xy.get_weights()

modes = []
sumofmodes = 0
for x,y in lant19_ipos:
    mode = norm_nonu(LPmodes.lpfield(xg0-x,yg0-y,0,1,sm_rcore,wl0,ncore,nclad),w) # Change this for each mode

    sumofmodes += mode
    modes.append(mode)
plt.imshow(np.abs(sumofmodes))
plt.show()
SMFpower=0
print("final field power decomposition:")
for i in range(len(modes)):
    _p = np.power(overlap_nonu(u0,modes[i],w),2)
    print("mode"+str(i)+": ", _p)
    SMFpower += _p

print("total power in SMFs: ", SMFpower)

abs_u0_at_positions = []
for x, y in lant19_ipos:
    # Find the nearest grid point index
    ix = np.argmin(np.abs(mesh.xy.xa0 - x))
    iy = np.argmin(np.abs(mesh.xy.ya0 - y))
    abs_u0_at_positions.append(np.abs(u0[ix, iy]))

print("Absolute values of u0 at lantern positions:", np.sum(abs_u0_at_positions))
u0 = normalize(LPmodes.lpfield(xg,yg,l,m,sm_rcore,wl,nclad,njack))
plt.imshow(abs(u0))
plt.show()
print(f"xg shape: {xg.shape}, yg shape: {yg.shape}")
print(mesh.xy.shape)
print(f"Initial u0 shape: {u0.shape}")
xa_in, ya_in = xg[:, 0], yg[0, :]
print(f"xa_in shape: {xa_in.shape}, ya_in shape: {ya_in.shape}")


# run the propagator (required)
ux, u0 = prop.prop2end(u0, monitor_func=monitor_func,xyslice=None,zslice=None,writeto=writeto,ref_val=ref_val,remesh_every=remesh_every,dynamic_n0=dynamic_n0,fplanewidth=fplanewidth)
#u0 = prop.prop2end_uniform(u0)
# u, u0 ^^
# compute power in output ports (optional)

u = ux
xg,yg = np.meshgrid(mesh.xy.xa,mesh.xy.ya,indexing='ij')

w = mesh.xy.get_weights()

xg0,yg0 = np.meshgrid(mesh.xy.xa0,mesh.xy.ya0,indexing='ij')
w0 = mesh.xy.dx0*mesh.xy.dy0

modes = []
sumofmodes = 0
for x,y in lant19_fpos:
    mode = norm_nonu(LPmodes.lpfield(xg-x,yg-y,0,1,sm_rcore,wl0,ncore,nclad),w) # Change this for each mode
    sumofmodes += mode
    modes.append(mode)
plt.imshow(np.abs(sumofmodes))
plt.show()
SMFpower=0
print("final field power decomposition:")
for i in range(len(modes)):
    _p = np.power(overlap_nonu(u,modes[i],w),2)
    print("mode"+str(i)+": ", _p)
    SMFpower += _p

print("total power in SMFs: ", SMFpower)

print(u0.shape)
abs_u0_at_positions = []
for x, y in lant19_fpos:
    # Find the nearest grid point index
    ix = np.argmin(np.abs(mesh.xy.xa0 - x))
    iy = np.argmin(np.abs(mesh.xy.ya0 - y))
    abs_u0_at_positions.append(np.abs(u0[ix, iy]))
print("Absolute values of u0 at lantern positions:", np.sum(abs_u0_at_positions))
for x, y in lant19_fpos:
    # Find the nearest grid point index
    ix = np.argmin(np.abs(mesh.xy.xa0 - x))
    iy = np.argmin(np.abs(mesh.xy.ya0 - y))
    print(np.abs(u0[ix, iy]))

out = np.zeros(mesh.xy.shape)
optic.set_IORsq(out, sm_ex)
plt.imshow(out, vmin=njack*njack, vmax=ncore*ncore)
plt.show()
out = np.zeros(mesh.xy.shape)
optic.set_IORsq(out, sm_ex/2)
plt.imshow(out, vmin=njack*njack, vmax=ncore*ncore)
plt.show()
# plotting (optional)
print("final field dist:")
plt.imshow(np.abs(u0)) 
plt.plot([pos[0] for pos in lant19_ipos], [pos[1] for pos in lant19_ipos], 'ro', label='Positions')
plt.show()