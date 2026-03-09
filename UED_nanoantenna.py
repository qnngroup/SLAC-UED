import numpy as np
import meep as mp
import matplotlib.pyplot as plt
from nanoantenna import make_triangle

#### Cell Parameters ####
resolution = 125
pml = .2
sx  = .8
sy = .8
sz = .8
cell = mp.Vector3(sx, sy, sz)
boundary_layers=[mp.PML(pml, direction=mp.Z)]

#### Nanoantenna Geometry ###
from meep.materials import SiO2
from meep.materials import Au
from meep.materials import Au_JC_visible

print(Au.valid_freq_range)
print(SiO2.valid_freq_range)

#### Antenna ####

a = 0.22
b = .165
thickness = .02
center = mp.Vector3(0,0,thickness/2)
theta_rot = -np.pi/2
r_curvature = 0.01
antenna_mat = Au

geometry  = make_triangle(a, b, r_curvature, thickness, center, antenna_mat, theta_rot)

#### Substrate ####

geometry.append(mp.Block(
    center = mp.Vector3(0, 0, (-sz/2 + pml)/2),
    size = mp.Vector3(sx, sy, sz/2 - pml),
    material = SiO2
))

### Optical Excitation ###
lam = 1
fcen = 1/lam #(1/um)


fwhm_fs = 10                       
fwhm_um = fwhm_fs / 3.33          
sigma_um = fwhm_um / 2.355         
df = 1 / (2 * np.pi * sigma_um)     

fcen = 1 / lam
sources = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth = df, is_integrated = True), 
                     component = mp.Ex,
                     center = mp.Vector3(0,0,sz/2-pml),
                     size = mp.Vector3(sx, sy, 0))]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=boundary_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    default_material=mp.air,
    k_point = mp.Vector3(0,0,0)
)


vol_xy = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, sy, 0))  # x-y at z=thickness
vol_xz = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, 0, sz))  # x-z at y=0

Ecomps = [mp.Ex, mp.Ey, mp.Ez]

frames = {"t": [], "xy": {c: [] for c in Ecomps}, "xz": {c: [] for c in Ecomps}}

frame_dt = 2.0

def grab_frame(sim):
    frames["t"].append(sim.meep_time())
    for c in Ecomps:
        frames["xy"][c].append(sim.get_array(vol=vol_xy, component=c))
        frames["xz"][c].append(sim.get_array(vol=vol_xz, component=c))


sim.run(mp.at_every(frame_dt, grab_frame), until=100)

# Save time array
np.save("frames_t.npy", np.array(frames["t"]))

# Save field arrays
for plane in ["xy", "xz"]:
    for c in Ecomps:
        np.save(f"frames_{plane}_{c}.npy", np.array(frames[plane][c]))


cal_sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=boundary_layers,
    geometry=[],
    sources=sources,
    resolution=75,
    default_material=mp.air,
    k_point = mp.Vector3(0,0,0)
)

cal_vol_xy = mp.Volume(center=mp.Vector3(0, 0, thickness), size=mp.Vector3(sx/2, sy/2, 0))  
cal_frames = {"t": [], "xy": {c: [] for c in Ecomps}}

frame_dt = 0.2

def cal_grab_frame(cal_sim):
    cal_frames["t"].append(cal_sim.meep_time())
    for c in Ecomps:
        cal_frames["xy"][c].append(cal_sim.get_array(vol=cal_vol_xy, component=c))


cal_sim.run(mp.at_every(frame_dt, cal_grab_frame), until=100)

comp = [0,1,4]
vmax_cal = max(np.max(cal_frames["xy"][c]) for c in comp)
print(f"E0 = {vmax_cal}")


from matplotlib.animation import FuncAnimation, PillowWriter


fig, ax = plt.subplots(1,3, figsize = (15,8))
ims = []
comp = [0,1,4]

vmin = min(np.min(frames["xy"][c]) for c in comp)/vmax_cal
vmax = max(np.max(frames["xy"][c]) for c in comp)/vmax_cal
extent = [-sx/2, sx/2, -sy/2, sy/2]

def init():
    ims.append(ax[0].imshow(frames["xy"][0][0].T/vmax_cal, cmap = "RdBu", extent = extent, vmin=vmin, vmax=vmax, origin = "lower"))
    ims.append(ax[1].imshow(frames["xy"][1][0].T/vmax_cal, cmap = "RdBu", extent = extent, vmin=vmin, vmax=vmax, origin = "lower"))
    ims.append(ax[2].imshow(frames["xy"][4][0].T/vmax_cal, cmap = "RdBu", extent = extent, vmin=vmin, vmax=vmax, origin = "lower"))
    for axs in ax:
        axs.set_xlabel(r"x $(\mu m)$")
        axs.set_xlim(-.15, .15)
        axs.set_ylim(-.15, .15)
    ax[0].set_ylabel(r"y $(\mu m)$")
    fig.colorbar(ims[0], ax=ax, shrink = .5)


def update(i):
    for im,c in zip(ims,comp):
        im.set_data(frames["xy"][c][i].T/vmax_cal)
    return ims


filename = "Nanoantenna_optical.gif"
ani = FuncAnimation(fig, update, frames=len(frames["t"]), init_func=init)

ani.save(filename, writer=PillowWriter(fps=15))

plt.close(fig)
print(f"Saved animation: {filename}")
