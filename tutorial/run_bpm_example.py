''' example script for running the beamprop code in prop.py'''
import numpy as np
from lightbeam.mesh import RectMesh3D
from lightbeam.prop import Prop3D
from lightbeam.misc import normalize,overlap_nonu,norm_nonu
from lightbeam import LPmodes
import matplotlib.pyplot as plt
from config_example import *

xw0 = 10  # Placeholder value
yw0 = 10  # Placeholder value
zw = 100  # Placeholder value
ds = 0.1  # Placeholder value
dz = 0.1  # Placeholder value
num_PML = 10  # Placeholder value
xw_func = None  # Placeholder value
yw_func = None  # Placeholder value
max_remesh_iters = 10  # Placeholder value
sig_max = 1.0  # Placeholder value
wl0 = 1.55e-6  # Placeholder value
optic = None  # Placeholder value
n0 = 1.0  # Placeholder value
  # Placeholder value
monitor_func = None  # Placeholder value
writeto = None  # Placeholder value
ref_val = 1.0  # Placeholder value
remesh_every = 10  # Placeholder value
dynamic_n0 = False  # Placeholder value
fplanewidth = 10  # Placeholder value
xpos = [0]  # Placeholder value
ypos = [0]  # Placeholder value
rcore = 1.0  # Placeholder value
scale = 1.0  # Placeholder value
ncore = 1.45  # Placeholder value
nclad = 1.44  # Placeholder value
l = 2
m = 2
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
sm_seperation_final = 2*60 # um
u0 = normalize(LPmodes.lpfield(xg,yg,l,m,mm_rclad,wl,nclad,njack))

if __name__ == "__main__":

    # mesh initialization (required)
    mesh = RectMesh3D(xw0,yw0,zw,ds,dz,num_PML,xw_func,yw_func)
    xg,yg = mesh.xy.xg,mesh.xy.yg

    mesh.xy.max_iters = max_remesh_iters
    mesh.sigma_max = sig_max

    # propagator initialization (required)
    prop = Prop3D(wl0,mesh,optic,n0)

    print('launch field')
    plt.imshow(np.real(u0))
    plt.show()

    # run the propagator (required)
    u,u0 = prop.prop2end(u0,monitor_func=monitor_func,xyslice=None,zslice=None,writeto=writeto,ref_val=ref_val,remesh_every=remesh_every,dynamic_n0=dynamic_n0,fplanewidth=fplanewidth)

    # compute power in output ports (optional)

    xg,yg = np.meshgrid(mesh.xy.xa,mesh.xy.ya,indexing='ij')

    w = mesh.xy.get_weights()

    xg0,yg0 = np.meshgrid(mesh.xy.xa0,mesh.xy.ya0,indexing='ij')
    w0 = mesh.xy.dx0*mesh.xy.dy0

    modes = []
    for x,y in zip(xpos,ypos):
        mode = norm_nonu(LPmodes.lpfield(xg-x,yg-y,0,1,rcore/scale,wl0,ncore,nclad),w)
        modes.append(mode)
    
    SMFpower=0
    print("final field power decomposition:")
    for i in range(len(modes)):
        _p = np.power(overlap_nonu(u,modes[i],w),2)
        print("mode"+str(i)+": ", _p)
        SMFpower += _p
    
    print("total power in SMFs: ", SMFpower)

    # plotting (optional)
    print("final field dist:")
    plt.imshow(np.abs(u0)[num_PML:-num_PML,num_PML:-num_PML]) 
    plt.show()
