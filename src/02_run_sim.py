# -*- coding: utf-8 -*-
"""
Subsurface flow and tranport simulation in a random binary inclusion structure
with the FEM solver of the groundwater flow equation and advection-dispersion-
equation OpenGeoSys.

This script allows to perform a 2D simulation according to the flow and transport
situation of the MADE-1 tracer tests (Columbus, Mississippi) in a 2D setting
with a binary inclusion structure of hydraulic conductivity.

Simulation are performed with the FEM solver OpenGeoSys (OGS5) for subsurface
flow and transport controlled via the API ogs5py: Initialization of the project,
writing files, simulation run and reading of simulation results. Setting-addapted
conductivity fields are generated with the script binary_inclusions.

@author: A. Zech
Licence MIT, A.Zech, 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import ogs5py
import binary_inclusions as bi

############################
### Simulation Settings  ###
############################
sim_id = 10  # ID of realization

### hydraulic properties
porosity = 0.31
storage = 0.0001
dispersivity_long = 0.01
dispersivity_trans = 0.01
keff1, keff2 = 2e-6, 2e-4

### domain dimensions and boundary conditions
domain = {"x_0": -20, "x_L": 200, "z_0": 52, "z_L": 62, "dx": 0.25, "dz": 0.05}
head_left, head_right = 63.0, 62.34
source_pump = 1.166e-5

### simulation time stepping and output time steps
time_steps = np.array([72, 500])
time_step_size = np.array([3600, 86400])
out_steps = 86400.0 * np.array([9, 49, 126, 202, 279, 370, 503])

### time dependent source/boundary conditions
rfd_tim = [0, 86400.0, 172800.0, 176400.0, 259200.0, 864000.0, 8640000.0, 86400000.0]
rfd_mass = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0]

####################################################
### Settings of block binary inclusion structure ###
####################################################

settings_binaryinclusions = dict(
    dim=2,          # dimensionality of structure
    k_bulk=1e-5,    # conductivity value of bulk
    k_incl=1e-3,    # conductivity value of inclusions
    nx=22,          # number of units in x-direction
    lx=10,          # unit length in x-direction
    nz=20,          # number of units in z-direction
    lz=0.5,         # unit length in z-direction
    nz_incl=3,      # number of inclusions (units with K_incl)
)

settings_binaryinclusions = dict(
    dim=2,                  # dimensionality of structure
    axis=0,                 # direction of blocks (x=0,z=1,y=2)
    k_bulk=[keff1, keff2],  # conductivity value of bulk
    k_incl=[keff2, keff1],  # conductivity value of inclusions
    nn=[4, 20],             # number of units per block (in x-direction)
    ll=[10, 10],            # unit length within each block (in x-direction)
    nz=20,                  # number of units in z-direction
    lz=0.5,                 # unit length in z-direction
    nn_incl=[3, 3],         # number of inclusions (units with K_incl)
)

###############################
### Setup Simulation Files  ###
###############################
print("Initiate Tranport Simulation with OpenGeoSys")

### ------------------------generate ogs base class------------------------- #
sim = ogs5py.OGS(task_root="../results/sim_{}".format(sim_id), task_id="sim_{}_".format(sim_id))

### ----------------------- write process file ------------------------------#
sim.pcs.add_block(PCS_TYPE="GROUNDWATER_FLOW")
sim.pcs.add_block(PCS_TYPE="MASS_TRANSPORT")

### ----------------------- meh & geometry---- ------------------------------#
domain.update(
    nx=int((domain["x_L"] - domain["x_0"]) / domain["dx"]),
    nz=int((domain["z_L"] - domain["z_0"]) / domain["dz"]),
)

sim.msh.generate(
    "rectangular",
    dim=2,
    mesh_origin=[domain["x_0"], domain["z_0"]],
    element_no=[domain["nx"], domain["nz"]],
    element_size=[domain["dx"], domain["dz"]],
)
sim.msh.swap_axis("y", "z")

# specify points and lines for boundary conditions and output
sim.gli.generate(
    "rectangular",
    dim=2,
    ori=(domain["x_0"], domain["z_0"]),
    size=(domain["x_L"] - domain["x_0"], domain["z_L"] - domain["z_0"]),
)
sim.gli.add_polyline(points=[0, 1], name="bottom")
sim.gli.add_polyline(points=[0, 3], name="left")
sim.gli.add_polyline(points=[2, 3], name="top")
sim.gli.add_polyline(points=[1, 2], name="right")
sim.gli.add_polyline(
    points=[[0, domain["z_0"] + 5.2, 0], [0, domain["z_0"] + 5.8, 0]],
    name="source"
)
sim.gli.swap_axis("y", "z")

### -----------------------  properties------------------------------------- #
sim.mfp.add_block(  # FLUID_PROPERTIES
    FLUID_TYPE="LIQUID",
    PCS_TYPE="HEAD",
    DENSITY=[1, 1000.0],
    VISCOSITY=[1, 0.001]
)
sim.mcp.add_block(
        NAME="CONCENTRATION1",
        MOBILE=1,
        DIFFUSION=[1, 1.0e-08]
        )
sim.msp.add_block(DENSITY=[1, 2000])

sim.mmp.add_block(
    GEOMETRY_DIMENSION=2,
    GEOMETRY_AREA="1.0",
    POROSITY="1 {}".format(porosity),
    TORTUOSITY="1 1.0",
    STORAGE="1 {}".format(storage),
    MASS_DISPERSION=[1, dispersivity_long, dispersivity_trans],
)

### --------------------heterogeneous K: binary structure ------------------ #

sim.mpd.add(name="conductivity")
sim.mpd.add_block(
    MSH_TYPE="GROUNDWATER_FLOW",
    MMP_TYPE="PERMEABILITY",
    DIS_TYPE="ELEMENT"
)

### Initialize block binary inclusion structure ###
BI = bi.Block_Binary_Inclusions(**settings_binaryinclusions)
BI.structure(bimodalKeff=True, seed=20201101 + sim_id)
BI.structure2mesh(sim.msh.centroids_flat, x0=domain["x_0"], z0=domain["z_0"])

sim.mpd.update_block(DATA=zip(range(len(BI.kk_mesh)), BI.kk_mesh))
# write the new mpd file
sim.mpd.write_file()

### set conductivity to binary inclusion structure
sim.mmp.update_block(PERMEABILITY_DISTRIBUTION=sim.mpd.file_name)

### -------- boundary, source and initial condition file --------------------- #
### flow Boundary conditions depending on bc
sim.bc.add_block(
    PCS_TYPE="GROUNDWATER_FLOW",
    PRIMARY_VARIABLE="HEAD",
    GEO_TYPE=["POLYLINE", "left"],
    DIS_TYPE=["CONSTANT", head_left],
)
sim.bc.add_block(
    PCS_TYPE="GROUNDWATER_FLOW",
    PRIMARY_VARIABLE="HEAD",
    GEO_TYPE=["POLYLINE", "right"],
    DIS_TYPE=["CONSTANT", head_right],
)
### Transport boundary conditions
sim.bc.add_block(
    PCS_TYPE="MASS_TRANSPORT",
    PRIMARY_VARIABLE="CONCENTRATION1",
    GEO_TYPE=["POLYLINE", "top"],
    DIS_TYPE=["CONSTANT", 0.0],
)
sim.bc.add_block(
    PCS_TYPE="MASS_TRANSPORT",
    PRIMARY_VARIABLE="CONCENTRATION1",
    GEO_TYPE=["POLYLINE", "bottom"],
    DIS_TYPE=["CONSTANT", 0.00],
)
sim.bc.add_block(
    PCS_TYPE="MASS_TRANSPORT",
    PRIMARY_VARIABLE="CONCENTRATION1",
    GEO_TYPE=["POLYLINE", "left"],
    DIS_TYPE=["CONSTANT", 0.00],
)
### Transport source condition
sim.st.add_block(
    PCS_TYPE="MASS_TRANSPORT",
    PRIMARY_VARIABLE="CONCENTRATION1",
    GEO_TYPE=["POLYLINE", "source"],
    DIS_TYPE=["CONSTANT_NEUMANN", 1e-08],
    TIM_TYPE="CURVE 1",
)
sim.st.add_block(
    PCS_TYPE="GROUNDWATER_FLOW",
    PRIMARY_VARIABLE="HEAD",
    GEO_TYPE=["POLYLINE", "source"],
    DIS_TYPE=["CONSTANT_NEUMANN", source_pump],
    TIM_TYPE="CURVE 1",
)
sim.rfd.add_block(CURVES=zip(rfd_tim, rfd_mass))

### initial conditions
sim.ic.add_block(
    PCS_TYPE="GROUNDWATER_FLOW",
    PRIMARY_VARIABLE="HEAD",
    GEO_TYPE="DOMAIN",
    DIS_TYPE=["CONSTANT", 0.5 * (head_right - head_left)],
)
sim.ic.add_block(
    PCS_TYPE="MASS_TRANSPORT",
    PRIMARY_VARIABLE="CONCENTRATION1",
    GEO_TYPE="DOMAIN",
    DIS_TYPE=["CONSTANT", 0],
)
### --------------- timing, output and numerics ---------------------------- #
### time file
sim.tim.add_block(
    PCS_TYPE="GROUNDWATER_FLOW",
    TIME_START=0,
    TIME_END=86400000,
    TIME_STEPS=zip(time_steps, time_step_size),
)
sim.tim.add_block(
    PCS_TYPE="MASS_TRANSPORT",
    TIME_START=0,
    TIME_END=86400000,
    TIME_STEPS=zip(time_steps, time_step_size),
)
###  output file
sim.out.add_block(
    NOD_VALUES=zip(["HEAD", "CONCENTRATION1"]),
    ELE_VALUES=zip(["VELOCITY1_X", "VELOCITY1_Y", "VELOCITY1_Z"]),
    GEO_TYPE="DOMAIN",
    DAT_TYPE="VTK",
    TIM_TYPE=zip(out_steps),
)
### num file
sim.num.add_block(  # set the parameters for the solver
    PCS_TYPE="GROUNDWATER_FLOW",
    LINEAR_SOLVER=[2, 1, 1.0e-12, 1000, 1.0, 100, 4],
    ELE_GAUSS_POINTS=3,
)
sim.num.add_block(  # set the parameters for the solver
    PCS_TYPE="MASS_TRANSPORT",
    LINEAR_SOLVER=[2, 1, 1.0e-8, 100, 1.0, 100, 4],
    ELE_GAUSS_POINTS=3,
)

################################################
### Write OGS input files and run simulation ###
################################################

print("Write Tranport Simulation files to directory: {}".format(sim.task_root))
sim.write_input()

sim.msh.export_mesh(  # export binary conductivity structure for vtk-visualization
    filepath="{}/conductivity.vtu".format(sim.task_root),
    file_format="vtk",
    cell_data_by_id={"transmissivity": BI.kk_mesh},
)

print("Run Tranport Simulation in directory: {}".format(sim.task_root))
success = sim.run_model()

####################################
### Visualize simulation results ###
####################################
if success:
    print("Simulation finished successfully.")
    it_out = -1

    out = sim.readvtk()
    times = out[""]["TIME"]
    points = out[""]["DATA"][1]["points"]
    ### Mass distribution at last output time
    mass = out[""]["DATA"][it_out]["point_data"]["CONCENTRATION1"]
    mass = np.where(mass > 0, mass, 0)

    xmesh = points[:, 0].reshape(-1, domain["nx"] + 1)
    zmesh = points[:, 2].reshape(-1, domain["nx"] + 1)
    mass_mesh = mass.reshape(-1, domain["nx"] + 1)

    print("Postprocessing: Plot of tracer plume")
    fig = plt.figure(figsize=[10, 2.5])
    ax = fig.add_subplot(1, 1, 1)
    colormap = plt.get_cmap("bone_r")

    ixmax = int(200 / domain["dx"])
    ax.contourf(
        xmesh[:, :ixmax],
        zmesh[:, :ixmax],
        mass_mesh[:, :ixmax],
        levels=20,
        cmap=colormap,
    )
    ax.text(
        0.85,
        0.08,
        "$T={:.0f}$ days".format(times[it_out] / 86400),
        bbox=dict(facecolor="w"),
        transform=ax.transAxes,
    )
    ax.text(
        0.03,
        0.88,
        "R{:.0f}".format(sim_id),
        bbox=dict(facecolor="w"),
        transform=ax.transAxes,
    )

    ax.grid(True, zorder=1)
    ax.set_xlabel("Distance from source $x$ [m]")
    ax.set_ylabel("Depth $z$ [m]")
    fig.tight_layout()

    print("Postprocessing: Save plot of tracer plume to /results.")
    plt.savefig(
        "../results/Simulation_R{:.0f}_T{:.0f}.png".format(sim_id, times[it_out] / 86400),
        dpi=300,
    )

else:
    print("Simulation did not finish successfully.")
