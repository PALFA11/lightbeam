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
taper_ratio = 10
sm_rcore = 6.5 # um
mm_rjack = 76.3 # um
sm_rclad = 32.8 # um
sm_seperation_final = 2*60 # um
sm_offset = 100 # um
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
ref_val = 2e-4
remesh_every = 10
dynamic_n0 = False
fplanewidth = 10
xpos = [0]
ypos = [0]
iw0 = 600
zw = sm_ex
di = 0.75
dz = 10
num_PML = 10

lant19_ipos = optics.lant19.get_19port_positions(core_spacing=sm_seperation_final/taper_ratio)
clad_mm = optics.scaled_cyl([0,0],mm_rclad,sm_offset,nclad,njack,0,scale_func=scale_func,final_scale=1)
clad_sm = optics.scaled_cyl([0,0],sm_rclad,sm_ex,nclad,njack,sm_offset,scale_func=scale_func,final_scale=taper_ratio)
elmnts = [clad_mm, clad_sm]

for i in range(0,len(lant19_ipos)):
    core = optics.scaled_cyl(xy=lant19_ipos[i] ,r = sm_rcore/taper_ratio,z_ex = sm_ex,n = ncore,nb = nclad,z_offset=sm_offset,scale_func=scale_func,final_scale=taper_ratio) # fxy=lant19_fpos[i]
    elmnts.append(core)

optic = optics.OpticSys(elmnts,njack)


if __name__ == "__main__":

    # mesh initialization (required)
    mesh = RectMesh3D(iw0,iw0,zw,di,dz,num_PML,xw_func,yw_func)
    xg,yg = mesh.xy.xg,mesh.xy.yg
    xg, yg = mesh.xg[num_PML:-num_PML,num_PML:-num_PML],mesh.yg[num_PML:-num_PML,num_PML:-num_PML]
    u0 = normalize(LPmodes.lpfield(xg,yg,l,m,mm_rclad,wl,nclad,njack))

    mesh.xy.max_iters = max_remesh_iters
    mesh.sigma_max = sig_max

    # propagator initialization (required)
    prop = Prop3D(wl0,mesh,optic,nclad)

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
