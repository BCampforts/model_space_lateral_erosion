#!/usr/bin/env python
# coding: utf-8

## Import Numpy and Matplotlib packages
import numpy as np
import matplotlib.pyplot as plt  # For plotting results; optional

## Import Landlab components
# Flow routing
from landlab.components import PriorityFloodFlowRouter

# SPACE model
from landlab.components import SpaceLateralEroder  # SpaceLargeScaleEroder model

## Import Landlab utilities
from landlab import RasterModelGrid  # Grid utility
from landlab import imshow_grid  # For plotting results; optional


# Set grid parameters
num_rows = 40
num_columns = 40
node_spacing = 100.0

# track sediment flux at the node adjacent to the outlet at lower-left
# node_next_to_outlet = num_columns + 1

# Instantiate model grid
mg = RasterModelGrid((num_rows, num_columns), node_spacing)
# add field ’topographic elevation’ to the grid
mg.add_zeros("node", "topographic__elevation")
# set constant random seed for consistent topographic roughness
np.random.seed(seed=5000)

# Create initial model topography:
# plane tilted towards the lower−left corner
topo = mg.node_y / 100000.0 + mg.node_x / 100000.0

# add topographic roughness
random_noise = (
    np.random.rand(len(mg.node_y)) / 1000.0
)  # impose topography values on model grid
mg["node"]["topographic__elevation"] += topo + random_noise

# add field 'soil__depth' to the grid
mg.add_zeros("node", "soil__depth")

# Set 2 m of initial soil depth at core nodes
mg.at_node["soil__depth"][mg.core_nodes] = 0  # meters

# Add field 'bedrock__elevation' to the grid
mg.add_zeros("bedrock__elevation", at="node")

# Sum 'soil__depth' and 'bedrock__elevation'
# to yield 'topographic elevation'
mg.at_node["bedrock__elevation"][:] = mg.at_node["topographic__elevation"]
mg.at_node["topographic__elevation"][:] += mg.at_node["soil__depth"]



# Close all model boundary edges
# mg.set_closed_boundaries_at_grid_edges(
#     bottom_is_closed=True, left_is_closed=True, right_is_closed=True, top_is_closed=True
# )

# Set lower-left (southwest) corner as an open boundary
# mg.set_watershed_boundary_condition_outlet_id(
#     0, mg["node"]["topographic__elevation"], -9999.0
# )


# Instantiate flow router
fr = PriorityFloodFlowRouter(mg, flow_metric="D8")

# Instantiate SPACE model with chosen parameters
sp = SpaceLateralEroder(
    mg,
    K_sed=4e-5,
    K_br=2e-5,
    F_f=0.0,
    phi=0.0,
    H_star=1.0,
    v_s=1.0,
    m_sp=0.5,
    n_sp=1.0,
    sp_crit_sed=0,
    sp_crit_br=0,
    lateral_erosion = True,
)


# Set model timestep
timestep = 100 # years

# Set elapsed time to zero
elapsed_time = 0.0  # years

# Set timestep count to zero
count = 0

# Set model run time
run_time = 2e7 # years

# Array to save sediment flux values
sed_flux = np.zeros(int(run_time // timestep))

while elapsed_time < run_time:  # time units of years

    mg.at_node['bedrock__elevation'][mg.core_nodes] += 1*1e-3*timestep 
    mg.at_node['topographic__elevation'][mg.core_nodes] += 1*1e-3*timestep 
    # Run the flow router
    fr.run_one_step()

    # Run SPACE for one time step
    sp.run_one_step(dt=timestep)

    # Save sediment flux value to array
    # sed_flux[count] = mg.at_node["sediment__flux"][node_next_to_outlet]

    # Add to value of elapsed time
    elapsed_time += timestep

    # Increase timestep count
    count += 1
    
    if np.mod(elapsed_time, 5e4)==0: 
        #%%
        # Instantiate subplot
        plot = plt.subplot()
        # Show sediment flux map
        imshow_grid(
            mg,
            "topographic__elevation",
            var_name="topographic__elevation",
            var_units=r"m",
            grid_units=("m", "m"),
            cmap="terrain",
        )
        plt.title('Time: ' + str(round(elapsed_time*1e-6,2))  + ' Myr')
        plt.show()


#%% Visualization of results

# Instantiate figure
fig = plt.figure()
# Instantiate subplot
plot = plt.subplot()
# Show sediment flux map
imshow_grid(
    mg,
    "topographic__elevation",
    plot_name="topographic__elevation",
    var_name="topographic__elevation",
    var_units=r"m",
    grid_units=("m", "m"),
    cmap="terrain",
)
#%%
# Instantiate figure
fig = plt.figure()
# Instantiate subplot
plot = plt.subplot()
# Show sediment flux map
imshow_grid(
    mg,
    "sediment__flux",
    plot_name="Sediment flux",
    var_name="Sediment flux",
    var_units=r"m$^3$/yr",
    grid_units=("m", "m"),
    cmap="terrain",
)

# Instantiate figure
fig = plt.figure()
# Instantiate subplot
plot = plt.subplot()
# Show sediment flux map
imshow_grid(
    mg,
    "drainage_area",
    plot_name="drainage__area",
    var_name="drainage__area",
    var_units=r"m$^2",
    grid_units=("m", "m"),
    cmap="terrain",
)

#%%


# Instantiate figure
fig = plt.figure()
# Instantiate subplot
plot = plt.subplot()
# Show sediment flux map
imshow_grid(
    mg,
    "volume__lateral_erosion",
    plot_name="volume lateral erosion",
    var_name="volume lateral erosion",
    var_units=r"m$^3$",
    grid_units=("m", "m"),
    cmap="terrain",
)
# In[ ]:


# Instantiate figure
fig = plt.figure()

# Instantiate subplot
sedfluxplot = plt.subplot()

# Plot data
# sedfluxplot.plot(np.arange(num_rows*num_columns), sed_flux, color="k", linewidth=3.0)

# # Add axis labels
# sedfluxplot.set_xlabel("Time [yr]")
# sedfluxplot.set_ylabel(r"Sediment flux [m$^3$/yr]")


# There is an initial increase in sediment flux from the model domain as the water reaches its equilibrium transport capacity. Over the long run, topographic gradients are reduced by the erosion of sediment, which results in lower and lower sediment fluxes from the domain over time.

# ### Click here for more <a href="https://landlab.readthedocs.io/en/latest/user_guide/tutorials.html">Landlab tutorials</a>
