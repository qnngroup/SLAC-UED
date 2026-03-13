import numpy as np
import meep as mp
import matplotlib.pyplot as plt
import h5py
import yaml

def make_triangle(A, B, R, T, C, mat, theta_rot=0):
    """
    Create a planar triangle antenna structure with curved tip.  It assumes isosceles shape.
    You can specify the altitude, base, center, radius of curvature, thickness, material,
    and (optionally) the rotation angle about the z-axis.

    Inputs:
      
      A --> altitude of the triangle
      B --> base of the triangle
      R --> tip radius of curvature
      T --> thickness
      C --> center of the triangle (Vector3)
      mat --> antenna material (meep material)
      theta_rot --> rotation angle (radians, around z-axis; optional)

    Outputs:

      geometry --> meep geometry list for the triangle antenna 

    """

    #We create the triangle by assuming a perfect pointed triangle with altitude A.
    #The top of this triangle is cut off to create a trapezoid.  We then add in
    #a cylinder to represent a rounded tip of ROC = R.

    # Parameters to use when setting the geometry
    theta = np.arctan(2*A/B) #Base angle of the triangle
    X = R/np.cos(theta) #Distance from the tip cylinder cylinder to the triangle peak
    Y = X - R*np.cos(theta) #Distance to cut off top of triangle to make trapezoid
    Z = R*np.sin(theta) #Half of the width of the top of the trapezoid structure

    center_offset = mp.Vector3(B/2, (A - X + R)/2, T/2)
    rot_axis = mp.Vector3(0, 0, 1)
    
    #Create the base trapezoid with 0, 0 at bottom left corner
    trapezoid_vertices = [mp.Vector3(0, 0),
                          mp.Vector3(B/2 - Z, A - Y),
                          mp.Vector3(B/2 + Z, A - Y),
                          mp.Vector3(B,0)]


    
    #Now shift to proper location:
    #  1. Shift to center at 0, 0
    #  2. Rotate by theta_rot
    #  3. Shift to center at C
    for kk in range(0, 4):
        #Shift by center offset...
        trapezoid_vertices[kk] = trapezoid_vertices[kk] - center_offset

        #Rotate the point
        trapezoid_vertices[kk] = trapezoid_vertices[kk].rotate(rot_axis, theta_rot)

        #Now shift to user center
        trapezoid_vertices[kk] = trapezoid_vertices[kk] + C


    
    #Perform same operations to set the center of curved tip cylinder...
    cylinder_center = mp.Vector3(B/2, A - X)
    cylinder_center = cylinder_center - mp.Vector3(center_offset.x, center_offset.y)
    cylinder_center = cylinder_center.rotate(rot_axis, theta_rot)
    cylinder_center = cylinder_center + C

    #Create the geometry structure for the antenna...
    geometry = []
    
    #First add in the trapezoid
    geometry.append(mp.Prism(trapezoid_vertices,
                             T,
                             material=mat))

    #Now for the rounded tip
    geometry.append(mp.Cylinder(R,
                                center=cylinder_center,
                                height=T,
                                material=mat))

    return geometry

def display_inputs(input_file):
    with open(input_file, 'r') as f:
        settings = yaml.safe_load(f)

    tcon = 1/.2998 
    # Cell Parameters
    pml = settings['pml']
    sx = settings['sx']
    sy = settings['sy']
    sz = settings['sz']
    resolution = settings['resolution']

    # Output settings
    t_final = settings['t_final']
    lam = settings['lam']
    df = settings['df']

    # Nanoantenna Geometry Settings
    thickness = settings['thickness']
    a = settings['a']
    b = settings['b']
    r_curvature = settings['r_curvature']
    print("=== Simulation Parameters ===")
    print(f"Cell size: sx={sx} um, sy={sy} um, sz={sz} um")
    print(f"PML thickness: {pml} um")
    print(f"Resolution: {resolution} pixels/um")
    print(f"Wavelength: {lam} um")
    print(f"Frequency bandwidth: {df}")
    print(f"Final time: {t_final} meep units ({t_final * tcon:.1f} fs)")
    print(f"Nanoantenna altitude: {a * 1000:.1f} nm")
    print(f"Nanoantenna base: {b * 1000:.1f} nm")
    print(f"Nanoantenna thickness: {thickness * 1000:.1f} nm")
    print(f"Radius of curvature: {r_curvature * 1000:.1f} nm")


def main_simulation(input_file):

    with open(input_file, 'r') as f:
        settings = yaml.safe_load(f)

    # Cell Parameters
    pml = settings['pml']
    sx = settings['sx']
    sy = settings['sy']
    sz = settings['sz']
    resolution = settings['resolution']

    # Output settings
    t_final = settings['t_final']
    lam = settings['lam']
    df = settings['df']

    # Nanoantenna Geometry Settings
    thickness = settings['thickness']
    a = settings['a']
    b = settings['b']
    r_curvature = settings['r_curvature']

    #### Cell Parameters ####

    cell = mp.Vector3(sx, sy, sz)
    boundary_layers=[mp.PML(pml, direction=mp.Z)]

    #### Antenna ####
    from meep.materials import Au

    center = mp.Vector3(0,0,thickness/2)
    theta_rot = -np.pi/2
    r_curvature = 0.01
    antenna_mat = Au
    geometry  = make_triangle(a, b, r_curvature, thickness, center, antenna_mat, theta_rot)

    #### Substrate ####
    from meep.materials import SiO2

    geometry.append(mp.Block(
        center = mp.Vector3(0, 0, (-sz/2 + pml)/2),
        size = mp.Vector3(sx, sy, sz/2 - pml),
        material = SiO2
    ))

    ### Optical Excitation ###

    fcen = 1/lam #(1/um)
    sources = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth = df, is_integrated = True), 
                        component = mp.Ex,
                        center = mp.Vector3(0,0,sz/2-pml),
                        size = mp.Vector3(sx, sy, 0))]

    ### Simulation ###

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=boundary_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        default_material=mp.air,
        k_point = mp.Vector3(0,0,0)
    )


    vol_xy = mp.Volume(center=mp.Vector3(0, 0, thickness/2), size=mp.Vector3(sx, sy, 0))  # x-y at z=thickness/2
    vol_xz = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, 0, sz))  # x-z at y=0

    Ecomps = [mp.Ex, mp.Ey, mp.Ez]
    Ecomp_names = ["Ex", "Ey", "Ez"]

    frame_dt = .01
    nframes = int(t_final / frame_dt) + 1

    # Pre-create HDF5 file with known dimensions
    # Get array shapes by initializing fields first
    sim.init_sim()
    sample_xy = sim.get_array(vol=vol_xy, component=mp.Ex)
    sample_xz = sim.get_array(vol=vol_xz, component=mp.Ex)
    xy_shape = sample_xy.shape
    xz_shape = sample_xz.shape

    hf = h5py.File("frames.h5", "w")
    hf.create_dataset("t", shape=(nframes,), dtype=np.float32)
    for c_name in Ecomp_names:
        hf.create_dataset(f"xy_{c_name}", shape=(nframes, *xy_shape), dtype=np.float32)
        hf.create_dataset(f"xz_{c_name}", shape=(nframes, *xz_shape), dtype=np.float32)

    frame_idx = [0]
    def grab_frame(sim):
        if frame_idx[0] >= nframes:
            return
        hf["t"][frame_idx[0]] = sim.meep_time()
        for c, c_name in zip(Ecomps, Ecomp_names):
            hf[f"xy_{c_name}"][frame_idx[0]] = sim.get_array(vol=vol_xy, component=c)
            hf[f"xz_{c_name}"][frame_idx[0]] = sim.get_array(vol=vol_xz, component=c)
        frame_idx[0] += 1

    sim.run(mp.at_every(frame_dt, grab_frame), until=t_final)
    hf.close()


    cal_sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=boundary_layers,
        geometry=[],
        sources=sources,
        resolution=75,
        default_material=mp.air,
        k_point = mp.Vector3(0,0,0)
    )

    cal_vol_xy = mp.Volume(center=mp.Vector3(0, 0, thickness/2), size=mp.Vector3(sx, sy, 0))

    # Get cal array shape and pre-create HDF5 file
    cal_sim.init_sim()
    cal_sample_xy = cal_sim.get_array(vol=cal_vol_xy, component=mp.Ex)
    cal_xy_shape = cal_sample_xy.shape

    hf_cal = h5py.File("frames_cal.h5", "w")
    hf_cal.create_dataset("t", shape=(nframes,), dtype=np.float32)
    for c_name in Ecomp_names:
        hf_cal.create_dataset(f"xy_{c_name}", shape=(nframes, *cal_xy_shape), dtype=np.float32)

    
    cal_frame_idx = [0]
    def cal_grab_frame(sim):
        if cal_frame_idx[0] >= nframes:
            return
        hf_cal["t"][cal_frame_idx[0]] = sim.meep_time()
        for c, c_name in zip(Ecomps, Ecomp_names):
            hf_cal[f"xy_{c_name}"][cal_frame_idx[0]] = cal_sim.get_array(vol=cal_vol_xy, component=c)
        cal_frame_idx[0] += 1

    cal_sim.run(mp.at_every(frame_dt, cal_grab_frame), until=t_final)
    hf_cal.close()


def transfer_function(input_file):
    import numpy as np
    import h5py
    from scipy.interpolate import interp1d
    from scipy.fft import rfft, rfftfreq
    import matplotlib.pyplot as plt

    with open(input_file, 'r') as f:
        settings = yaml.safe_load(f)

    sx = settings['sx']
    sy = settings['sy']

    tcon = 3.33564  

    def load_and_sort(filename):
        hf = h5py.File(filename, "r")
        t = hf["t"][:]
        Ex_xy = hf["xy_Ex"][:]
        hf.close()
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        Ex_xy = Ex_xy[sort_idx]
        valid = np.append(True, np.diff(t) > 0)
        return t[valid], Ex_xy[valid]

    def get_peak_trace(t, Ex_xy, sx, sy):
        nx, ny = Ex_xy.shape[1], Ex_xy.shape[2]
        x_arr = np.linspace(-sx/2, sx/2, nx)
        y_arr = np.linspace(-sy/2, sy/2, ny)
        iy = np.argmin(np.abs(y_arr - 0))
        peak_map = np.max(np.abs(Ex_xy[:, :, iy]), axis=0)
        ix = np.argmax(peak_map)
        print(f"Sampling at x={x_arr[ix]:.4f}, y={y_arr[iy]:.4f}")
        return Ex_xy[:, ix, iy]

    def interpolate_and_fft(t, Ex_trace):
        t_uniform = np.linspace(t[0], t[-1], len(t))
        Ex_uniform = interp1d(t, Ex_trace, kind='cubic')(t_uniform)
        t_fs = t_uniform * tcon
        dt_fs = t_fs[1] - t_fs[0]
        N = len(Ex_uniform)
        Ex_fft = rfft(Ex_uniform)
        freqs_fs = rfftfreq(N, d=dt_fs)
        freqs_fs = freqs_fs[1:]
        Ex_fft = Ex_fft[1:]
        return t_fs, Ex_uniform, freqs_fs, Ex_fft


    print("Loading frames.h5...")
    t_sim, Ex_xy_sim = load_and_sort("frames.h5")

    print("Loading frames_cal.h5...")
    t_cal, Ex_xy_cal = load_and_sort("frames_cal.h5")

    print("\nAntenna simulation - finding peak:")
    Ex_trace_sim = get_peak_trace(t_sim, Ex_xy_sim, sx, sy)

    print("\nCalibration simulation - sampling at center:")
    nx, ny = Ex_xy_cal.shape[1], Ex_xy_cal.shape[2]
    x_arr = np.linspace(-sx/2, sx/2, nx)
    y_arr = np.linspace(-sy/2, sy/2, ny)
    ix_cal = np.argmin(np.abs(x_arr - 0))
    iy_cal = np.argmin(np.abs(y_arr - 0))
    Ex_trace_cal = Ex_xy_cal[:, ix_cal, iy_cal]

    #### Interpolate and FFT ####
    t_fs_sim, Ex_uniform_sim, freqs_fs, Ex_fft_sim = interpolate_and_fft(t_sim, Ex_trace_sim)
    t_fs_cal, Ex_uniform_cal, freqs_fs_cal, Ex_fft_cal = interpolate_and_fft(t_cal, Ex_trace_cal)

    #### Normalize to E0 ####
    E0 = np.max(np.abs(Ex_uniform_cal))
    print(f"E0 = {E0:.4f}")

    #### Interpolate cal FFT onto sim frequency axis ####
    if len(freqs_fs) != len(freqs_fs_cal) or not np.allclose(freqs_fs, freqs_fs_cal, rtol=1e-3):
        print("Interpolating cal FFT onto sim frequency axis...")
        Ex_fft_cal_real = interp1d(freqs_fs_cal, np.real(Ex_fft_cal), kind='cubic', fill_value=0, bounds_error=False)(freqs_fs)
        Ex_fft_cal_imag = interp1d(freqs_fs_cal, np.imag(Ex_fft_cal), kind='cubic', fill_value=0, bounds_error=False)(freqs_fs)
        Ex_fft_cal_interp = Ex_fft_cal_real + 1j * Ex_fft_cal_imag
    else:
        Ex_fft_cal_interp = Ex_fft_cal

    #### Transfer Function ####
    H = Ex_fft_sim / Ex_fft_cal_interp
    H_mag = np.abs(H)
    freqs_THz = freqs_fs * 1e3
    wavelengths_um = 0.2998 / freqs_fs


    fig, ax = plt.subplots(2, 3, figsize=(18, 8))


    ax[0, 0].plot(t_fs_cal, Ex_uniform_cal / E0, color='steelblue')
    ax[0, 0].set_xlim(t_fs_cal[0], t_fs_cal[-1])
    ax[0, 0].set_xlabel("Time (fs)")
    ax[0, 0].set_ylabel("E/E$_0$")
    ax[0, 0].set_title("Incident Pulse")

    ax[1, 0].plot(t_fs_sim, Ex_uniform_sim / E0, color='darkorange')
    ax[1, 0].set_xlim(t_fs_sim[0], t_fs_sim[-1])
    ax[1, 0].set_xlabel("Time (fs)")
    ax[1, 0].set_ylabel("Ex/E$_0$")
    ax[1, 0].set_title("Antenna Tip Response")


    ax[0, 1].plot(freqs_THz, np.abs(Ex_fft_cal_interp), color='steelblue', label="Incident")
    ax[0, 1].set_xlabel("Frequency (THz)")
    ax[0, 1].set_ylabel("Incident |E(f)|", color='steelblue')
    ax[0, 1].tick_params(axis='y', labelcolor='steelblue')
    ax[0, 1].set_xlim(0, 1000)
    ax[0, 1].set_title("Spectra vs Frequency")
    ax2_top_mid = ax[0, 1].twinx()
    ax2_top_mid.plot(freqs_THz, np.abs(Ex_fft_sim), color='darkorange', label="Antenna")
    ax2_top_mid.set_ylabel("Antenna |E(f)|", color='darkorange')
    ax2_top_mid.tick_params(axis='y', labelcolor='darkorange')
    lines1 = [plt.Line2D([0], [0], color='steelblue', label='Incident'),
              plt.Line2D([0], [0], color='darkorange', label='Antenna')]
    ax[0, 1].legend(handles=lines1, loc='upper right')


    ax[1, 1].plot(freqs_THz, H_mag, color='black')
    ax[1, 1].set_xlabel("Frequency (THz)")
    ax[1, 1].set_ylabel("|H(f)|")
    ax[1, 1].set_title("Transfer Function vs Frequency")
    ax[1, 1].set_xlim(100, 600)
    ax[1, 1].set_ylim(0, 40)


    ax[0, 2].plot(wavelengths_um, np.abs(Ex_fft_cal_interp), color='steelblue', label="Incident")
    ax[0, 2].set_xlabel("Wavelength (μm)")
    ax[0, 2].set_ylabel("Incident |E(f)|", color='steelblue')
    ax[0, 2].tick_params(axis='y', labelcolor='steelblue')
    ax[0, 2].set_xlim(0.3, 3)
    ax[0, 2].invert_xaxis()
    ax[0, 2].set_title("Spectra vs Wavelength")
    ax2_top_right = ax[0, 2].twinx()
    ax2_top_right.plot(wavelengths_um, np.abs(Ex_fft_sim), color='darkorange', label="Antenna")
    ax2_top_right.set_ylabel("Antenna |E(f)|", color='darkorange')
    ax2_top_right.tick_params(axis='y', labelcolor='darkorange')
    lines2 = [plt.Line2D([0], [0], color='steelblue', label='Incident'),
              plt.Line2D([0], [0], color='darkorange', label='Antenna')]
    ax[0, 2].legend(handles=lines2, loc='upper right')

    ax[1, 2].plot(wavelengths_um, H_mag, color='black')
    ax[1, 2].set_xlabel("Wavelength (μm)")
    ax[1, 2].set_ylabel("|H(f)|")
    ax[1, 2].set_title("Transfer Function vs Wavelength")
    ax[1, 2].set_xlim(0.5, 3.0)
    ax[1, 2].set_ylim(0, 40)
    ax[1, 2].invert_xaxis()

    plt.tight_layout()
    plt.savefig("transfer_function.png", dpi=150)
    plt.show()

    np.save("H_freq.npy", H)
    np.save("freqs_fs.npy", freqs_fs)
    print("Saved H_freq.npy and freqs_fs.npy")

def visualize(input_file, stride = 5):
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    #### Parameters ####
    with open(input_file, 'r') as f:
        settings = yaml.safe_load(f)


    sx = settings['sx']
    sy = settings['sy']
    tcon = 3.33564
    comp_names = ["Ex", "Ey", "Ez"]

    #### Get E0 from cal file ####
    print("Computing E0 from frames_cal.h5...")
    hf_cal = h5py.File("frames_cal.h5", "r")
    vmax_cal = max(np.max(np.abs(hf_cal[f"xy_{c_name}"][:])) for c_name in comp_names)
    hf_cal.close()
    print(f"E0 = {vmax_cal}")

    #### Load sim frames ####
    print("Loading frames.h5...")
    hf_read = h5py.File("frames.h5", "r")
    t_arr = hf_read["t"][:]

    # Sort by time
    sort_idx = np.argsort(t_arr)
    t_arr = t_arr[sort_idx]

    # Remove duplicates
    valid = np.append(True, np.diff(t_arr) > 0)
    t_arr = t_arr[valid]
    sort_idx = sort_idx[valid]

    # Apply stride
    t_arr = t_arr[::stride]
    sort_idx = sort_idx[::stride]

    t_fs = t_arr * tcon

    #### Color scale ####
    vmin = min(np.min(hf_read[f"xy_{c}"]) for c in comp_names) / vmax_cal
    vmax = max(np.max(hf_read[f"xy_{c}"]) for c in comp_names) / vmax_cal
    extent = [-sx/2, sx/2, -sy/2, sy/2]

    #### Animation ####
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    ims = []

    def init():
        for i, c_name in enumerate(comp_names):
            im = ax[i].imshow(hf_read[f"xy_{c_name}"][sort_idx[0]].T / vmax_cal,
                            cmap="RdBu", extent=extent, vmin=vmin, vmax=vmax, origin="lower")
            ax[i].set_title(f"E{c_name[-1]}/E$_0$")
            ax[i].set_xlabel(r"x ($\mu$m)")
            ax[i].set_xlim(-.15, .15)
            ax[i].set_ylim(-.15, .15)
            ims.append(im)
        ax[0].set_ylabel(r"y ($\mu$m)")
        fig.colorbar(ims[0], ax=ax, shrink=0.5)

    def update(i):
        idx = sort_idx[i]
        for im, c_name in zip(ims, comp_names):
            im.set_data(hf_read[f"xy_{c_name}"][idx].T / vmax_cal)
        fig.suptitle(f"t = {t_fs[i]:.1f} fs")
        return ims

    print("Generating animation...")
    ani = FuncAnimation(fig, update, frames=len(t_arr), init_func=init)
    ani.save("Nanoantenna_optical.gif", writer=PillowWriter(fps=15))

    hf_read.close()
    plt.close(fig)
    print("Saved Nanoantenna_optical.gif")

def extract_field(input_file):

    import numpy as np
    import h5py
    from scipy.interpolate import interp1d

    with open(input_file, 'r') as f:
        settings = yaml.safe_load(f)

    sx = settings['sx']
    sy = settings['sy']
    tcon = 3.33564  # fs per meep unit

    def load_and_sort(filename):
        hf = h5py.File(filename, "r")
        t = hf["t"][:]
        Ex_xy = hf["xy_Ex"][:]
        hf.close()
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        Ex_xy = Ex_xy[sort_idx]
        valid = np.append(True, np.diff(t) > 0)
        return t[valid], Ex_xy[valid]

    #### Load Data ####
    t_sim, Ex_xy_sim = load_and_sort("frames.h5")
    t_cal, Ex_xy_cal = load_and_sort("frames_cal.h5")

    #### Get E0 from cal at center ####
    nx, ny = Ex_xy_cal.shape[1], Ex_xy_cal.shape[2]
    x_arr = np.linspace(-sx/2, sx/2, nx)
    y_arr = np.linspace(-sy/2, sy/2, ny)
    ix_cal = np.argmin(np.abs(x_arr - 0))
    iy_cal = np.argmin(np.abs(y_arr - 0))
    Ex_trace_cal = Ex_xy_cal[:, ix_cal, iy_cal]
    t_uniform_cal = np.linspace(t_cal[0], t_cal[-1], len(t_cal))
    Ex_uniform_cal = interp1d(t_cal, Ex_trace_cal, kind='cubic')(t_uniform_cal)
    E0 = np.max(np.abs(Ex_uniform_cal))
    print(f"E0 = {E0:.4f}")

    #### Get peak Ex trace from antenna sim ####
    nx, ny = Ex_xy_sim.shape[1], Ex_xy_sim.shape[2]
    x_arr = np.linspace(-sx/2, sx/2, nx)
    y_arr = np.linspace(-sy/2, sy/2, ny)
    iy = np.argmin(np.abs(y_arr - 0))
    peak_map = np.max(np.abs(Ex_xy_sim[:, :, iy]), axis=0)
    ix = np.argmax(peak_map)
    print(f"Sampling at x={x_arr[ix]:.4f}, y={y_arr[iy]:.4f}")
    Ex_trace_sim = Ex_xy_sim[:, ix, iy]

    #### Interpolate to uniform time grid ####
    t_uniform = np.linspace(t_sim[0], t_sim[-1], len(t_sim))
    Ex_uniform = interp1d(t_sim, Ex_trace_sim, kind='cubic')(t_uniform)
    t_fs = t_uniform * tcon

    #### Normalize and save ####
    Ex_normalized = Ex_uniform / E0

    np.save("t_fs.npy", t_fs)
    np.save("Ex_E0.npy", Ex_normalized)
    print(f"Saved t_fs.npy ({len(t_fs)} points, {t_fs[0]:.1f} to {t_fs[-1]:.1f} fs)")
    print(f"Saved Ex_E0.npy (peak = {np.max(np.abs(Ex_normalized)):.2f} E0)")

def J_FN_atomic(F, phi):
  """
  J_FN_atomic(F, phi)
   
  Description
  ------------------
  Calculates the Fowler-Nordheim emission rate as a function 
  of the work function (phi) and field strength (F) at a surface.  
  Note that the emission is in arbitrary units and is not intended 
  to provide absolute current density output.  
  
  Inputs
   -----------
  F --> electric field strength (atomic units)
  phi --> work function (energy in atomic units)
  
  Outputs
  -------------
  Current density (relative current density...i.e. to within a constant factor).
  """


  F_valid_range = np.where(F < 0)
  F_valid = F[F_valid_range]
  
  J = np.zeros(F.shape)
  J[F_valid_range] = F_valid**2*np.exp(-4*np.sqrt(2)*phi**1.5/(-3*F_valid))

  return J

def J_FN_SI(F, phi):
  """
  J_FN_SI(

  Simplified Standard Fowler-Nordheim-type equation for calculating current density.
  
  The point of this function is to provide output in SI units. As such, all inputs
  are expected in SI units as specified below.

  Likewise, the current density is returned in units of A/nm^2.
 
  Inputs
  ---------
    F -- field (V/nm)
    phi -- work function (eV)

  Outputs
  ---------
    Current density (A/nm^2)
  """
    
  # Calculates physical current density
  # https://en.wikipedia.org/wiki/Field_electron_emission#Recommended_form_for_simple_Fowler%E2%80%93Nordheim-type_calculations

  J = np.zeros(F.shape)

  #We need to invert F here as electron emission only occurs for negative fields.
  #The following formulas were defined assuming F is positive to emit the particle.
  F = -1*F

  #The FN function is only defined over regions where the field is nonzero.  We now
  #find the indices of those regions and only perform our calculations in those
  #regions.  Everywhere else will be left as 0.
  F_valid_range = np.where(F > 0)
  F_valid = F[F_valid_range]

  a_const = 1.541534e-6
  b_const = 6.83089  

  f = 1.43996453529595*F_valid/phi**2
  v_f = 1 - f + 1/6*f*np.log(abs(f))

  J[F_valid_range] = a_const/phi*F_valid**2*np.exp(-v_f*b_const*phi**(3/2)/F_valid)

  return J

def dJ_dF_FN_SI(F, phi):
  """
  dJ_dF_FN_SI(

  Derivative of Simplified Standard Fowler-Nordheim-type equation with respect to field.  
  
  This function is useful for determining the true transfer function of the emitter in the context of small-signal analysis.  
  
  The point of this function is to provide output in SI units. As such, all inputs
  are expected in SI units as specified below.
 
  Inputs
  ---------
    F -- field (V/nm)
    phi -- work function (eV)

  Outputs
  ---------
    Derivative of current density with respect to field (A/V/nm)
  """
    
  # Calculates physical current density
  # https://en.wikipedia.org/wiki/Field_electron_emission#Recommended_form_for_simple_Fowler%E2%80%93Nordheim-type_calculations
  dJ_dF = np.zeros(F.shape)

  #We need to invert F here as electron emission only occurs for negative fields.
  #The following formulas were defined assuming F is positive to emit the particle.
  F = -1*F

  #The FN function is only defined over regions where the field is nonzero.  We now
  #find the indices of those regions and only perform our calculations in those
  #regions.  Everywhere else will be left as 0.
  F_valid_range = np.where(F > 0)
  F_valid = F[F_valid_range]

  a_const = 1.541534e-6
  b_const = 6.83089  

  f = 1.43996453529595*F_valid/phi**2
  v_f = 1 - f + 1/6*f*np.log(abs(f))

  alpha = a_const/phi
  beta = v_f*b_const*phi**(3/2)

  dJ_dF[F_valid_range] = (2*a_const/phi*F_valid + a_const*v_f*b_const*phi**(1/2))*np.exp(-v_f*b_const*phi**(3/2)/F_valid)
    

  return dJ_dF



