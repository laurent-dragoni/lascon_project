# coding: utf-8

# In[1]:


##get_ipython().magic(u'pylab inline')


# In[2]:


import nest
import nest.topology as topo # import topology as a submodule of nest
import nest.raster_plot
import numpy as np
import pylab
import hexagonal_network as hn
import pickle

#========================================================================================== FUNCTIONS
def simulate_and_plot():
    N = (n_top * n_pyr) + (n_hc * n_bc)  # total number of neurons
    print "Total number of neurons:", N

    # for each HC, puts its neurons in a list
    # two loops because I want to keep the pyramidal and basket cells separated.
    allnodes = []
    for i in range(n_hc):
        # "nest.GetNodes(mc[0])[0]" accesses the 360 neurons (i.e., 30 neurons for each of the 12 MCs) of a HC
        allnodes.append(nest.GetNodes(mc[i])[0])
    ##    print "appended: ", nest.GetNodes(mc[i])[0]
    for i in range(n_hc):
        # "nest.GetNodes(hc[0])[0]" accesses the 24 basket cells of each HC
        allnodes.append(nest.GetNodes(hc[i])[0])
    ##    print "appended: ", nest.GetNodes(hc[i])[0]

    # just turns the array of arrays into a single array
    allnodes = np.array([item for sublist in allnodes for item in sublist])
    assert(len(allnodes) == N)

    ##allnodes = np.array(range(2,N))

    #============================

    # set up and connect spike detector
    sd = nest.Create('spike_detector')
    nest.Connect(list(allnodes), sd)

    # set random initial membrane potentials
    for id in allnodes:  # loop over all target neurons
        nest.SetStatus([id], {'V_m': Vreset + (Vth - Vreset) * np.random.rand()})

    # run simulation
    nest.Simulate(T)

    # read out recorded spikes
    spike_senders = nest.GetStatus(sd)[0]['events']['senders']
    spike_times = nest.GetStatus(sd)[0]['events']['times']

    # compute average firing rate
    rate = pylab.float32(nest.GetStatus(sd)[0]['n_events']) / T * 1e3 / N
    print("\nFiring rate = %.1f spikes/s" % (rate))

    #============================
    # plotting
    # uncomment to show plots
    ##pylab.figure(1)
    ##pylab.clf()
    ##pylab.plot(spike_times, spike_senders, 'k.', markersize=1)
    ##
    ##pylab.xlim(0, T)
    ###pylab.ylim(allnodes[0], allnodes[-1])
    ##pylab.xlabel('time (ms)')
    ##pylab.ylabel('neuron id')
    ##pylab.show()

    ## see last spike of a given neuron
    #nest.GetStatus((x,))[0]['t_spike']

    nest.raster_plot.from_device(sd)
    pylab.show()

#==========================================================================================
# In[22]:

# set up simulation
nest.ResetKernel()

# simulation parameters
T = 1000.  # simulation time (ms)
dt = 0.1  # simulation resolution (ms)

nest.SetStatus([0], {
    'resolution': dt,  # set simulation resolution
    'print_time': True  # enable printing of simulation progress
})

print "Starting..."

#========================================================================================== parameters
# In[23]:

# neuron parameters
Vth = -55.0  # spike threshold (mV)
Vreset = -80.  # reset potential (mV)
E_L = -70.0 # mV
C_m = 280.0 # pF?
g_L = 14.0 # leak conductance (pS?)
tau_w = 500.0 # ms

Delta_T = 3.0 # upstroke slope factor (mV)
b = 86.0 # pA?

# synapse parameters
JE = 0.6  # excitatory weight (mV)
g = 6.  # relative inhibitory weight (JI=-g*JE)

# delay parameters
dmin = 2.0  # minimal spike transmission delay (ms)
d_linear_a = 0.02
d_linear_c = 1.

# input parameters
Iext = 10.  # external DC input (pA)
#===============================
# derived parameters

JI = -g * JE  # inhibitory synaptic weight

#===============================

neuron_params = {'I_e': Iext,
                 'E_L': E_L,
                 'V_th': Vth,
                 'V_reset': Vreset,
                 'C_m': C_m,
                 'g_L': g_L,
                 'b': b,
                 'tau_w': tau_w,
                 'Delta_T': Delta_T
                 }

# for now, 'aeif_cond_alpha' will be used. but it could be 'aeif_cond_exp'
neuron_model_name = 'aeif_cond_alpha'

nest.SetDefaults(neuron_model_name, neuron_params)

#=============================== synapses

##
##nest.CopyModel(existing="static_synapse",
##               new="excitatory",
##               params={"weight": JE, "delay": {'distribution': 'uniform', 'low': 0.8, 'high': 2.5}})
##nest.CopyModel(existing="static_synapse",
##               new="inhibitory",
##               params={"weight": JI, "delay": {'distribution': 'uniform', 'low': 0.8, 'high': 2.5}})

##
##nest.CopyModel(existing="static_synapse",
##               new="excitatory",
##               params={"weight": JE, "delay": {"linear" :{"c":0.1,"a":0.2}}})
##nest.CopyModel(existing="static_synapse",
##               new="inhibitory",
##               params={"weight": JI, "delay": {"linear" :{"c":0.1,"a":0.2}}})

#========================================================================================== topology

'''extension of the grid'''

#ext = 2. # here we use a square layer of size (ext x ext)
ext_hc = 640. # 640 micrometers of diameter in each HC

# WARNING : the quantity called 'diameter' in the article is not the actual 
# diameter (eg the distance between two opposing vertices) of the hexagons.
# The relationship between article_diameter and true_diameter is :
# article_diameter = sqrt(3)*true_diameter/2
# Hence the following formula for the radius of HyperColumns :
radiusOfHC = ext_hc/np.sqrt(3)

radiusOfMC = 10
ext_mc_l = radiusOfMC
ext_mc_w = radiusOfMC

totalLength = 9.*ext_hc/2 # total length of the network
totalWidth = 13.*ext_hc/(2*np.sqrt(3)) # total width of the network

numberOfMCPerHC = 12
n_pyr = 30                             #pyramidal neurons in each minicolumn
n_bc = 24                              #basket cells in each hypercolumn
n_top = 192                            #number of minicolumns (12 in each of the 16 hypercolumns)
n_hc = 16                              #number of hypercolumns

# pos1 : positions of the Pyramidal Cells
# pos2 : positions of the Basket Cells
pos1, pos2 = hn.all_pyr_and_basket_cells(numberOfMCPerHC, n_pyr, n_bc, radiusOfMC, radiusOfHC)

# In[27]:

hc_pyr=[0]*(n_hc)
hc_bc=[0]*(n_hc)
mc=[0]*n_top

'''dividing the pyramical neurons in groups of 360 neurons (the number in each HC)'''
for i in range (n_hc):
    hc_pyr[i]= [pos1[j] for j in range (i*n_pyr*n_top/n_hc,(i+1)*n_pyr*n_top/n_hc)]

    
'''dividing the basket cells in groups of 24 neurons'''
for i in range (n_hc):
    hc_bc[i]= [pos2[j] for j in range (i*n_bc,(i+1)*n_bc)]

# In[29]:

layer_pyr = [0]*n_hc
for i in range(n_hc):
    layer_pyr[i] = {
        'extent': [totalLength, totalWidth],
        'positions': hc_pyr[i],
        #'elements': 'iaf_neuron'
        'elements': neuron_model_name  #adaptative exponential integrate and firing as neuron model
    }

layer_bc = [0]*n_hc
for i in range (n_hc):
    layer_bc[i] = {
        'extent': [totalLength, totalWidth],#without the 10 there is a error saying "node position outside of layer
        'positions': hc_bc[i],
        #'elements': 'iaf_neuron'
        'elements': neuron_model_name  #adaptative exponential integrate and firing as neuron model
    }

# In[31]:

mc=[0]*n_hc
hc=[0]*n_hc
for i in range (n_hc):
    mc[i]=(topo.CreateLayer(layer_pyr[i]))  #creating the layer with 360 Pyr
    hc[i]=(topo.CreateLayer(layer_bc[i]))  #creating the layer with 24 bc

# In[34]:

################################################################################################################################################


conn_pb = 0.7  #probability of connecting between pyramidal cell and basket cell 
conn_bp = 0.7  #probability of connecting between basket cell and pyramidal cell
conn_pp = 0.2  #probability of connecting between pyramidal cell and pyramidal cell 


#nest.SetDefaults("tsodyks2_synapse", {'tau_rec':5.})          #####       tau_rec = 5ms corresponds to the AMPA and GAMA receptors

conn_pb = {
    'connection_type': 'divergent',
    'mask': {'rectangular': {'lower_left': [-totalLength, -totalWidth], 'upper_right': [totalLength, totalWidth]},
             'anchor': [0., 0.]},
    'kernel': conn_pb,
    'allow_autapses': False,
##    'delays': {'distribution': 'uniform', 'low': 0.8, 'high': 2.5},
    'delays': {"linear" :{"c":d_linear_c,"a":d_linear_a}},
##    'synapse_model': 'excitatory'
    'synapse_model': 'stdp_synapse'
}

conn_bp = {
    'connection_type': 'divergent',
    'mask': {'rectangular': {'lower_left': [-totalLength, -totalWidth], 'upper_right': [totalLength, totalWidth]},
             'anchor': [0., 0.]},
    'kernel': conn_bp,
    'allow_autapses': False,
##    'delays': {'distribution': 'uniform', 'low': 0.8, 'high': 2.5},
    'delays': {"linear" :{"c":d_linear_c,"a":d_linear_a}},
##    'synapse_model': 'inhibitory'
    'synapse_model': 'static_synapse'
}

conn_pp = {
    'connection_type': 'divergent',
    'mask': {'rectangular': {'lower_left': [-totalLength, -totalWidth], 'upper_right': [totalLength, totalWidth]},
             'anchor': [0., 0.]},
    'kernel': conn_pp,
    'allow_autapses': False,
##    'delays': {'distribution': 'uniform', 'low': 0.8, 'high': 2.5},
    'delays': {"linear" :{"c":d_linear_c,"a":d_linear_a}},
##    'synapse_model': 'excitatory'
    'synapse_model': 'stdp_synapse'
}

# connection for the poisson generator layer (noiselayer). devices can only be connected with the static synapse model
conn_poisson = {
    'connection_type': 'divergent',
    'mask': {'rectangular': {'lower_left': [-totalLength, -totalWidth], 'upper_right': [totalLength, totalWidth]},
             'anchor': [0., 0.]},
    'kernel': conn_pp,
    'allow_autapses': False,
##    'delays': {'distribution': 'uniform', 'low': 0.8, 'high': 2.5},
    'delays': {"linear" :{"c":d_linear_c,"a":d_linear_a}},
##    'synapse_model': 'excitatory'
    'synapse_model': 'static_synapse'
}


# In[36]:


#connect each HC with 12 MC and vice-versa

for i in range(n_hc):
    topo.ConnectLayers(hc[i], mc[i], conn_bp)
    topo.ConnectLayers(mc[i], hc[i], conn_pb)


# In[37]:


#connect each MC with all MC:            

for i in range(n_hc):
    for j in range (n_hc):
        topo.ConnectLayers(mc[i], mc[j], conn_pp)


#%%
allCons = nest.GetConnections()

#%%

##############################################################################################################################################################

# plots the layers. generates a lot of images.
###%%
##for i in range(n_hc):
##    topo.PlotLayer(hc[i])
##
###%%
##for j in range(n_hc):
##    topo.PlotLayer(mc[j])

# In[ ]:



# In[38]:


# make a copy of the poisson_generator model and specify the rate
# (I disabled the noise layer so it would not affect the other poisson geneator I am using for the learning;
# but I'm not sure this is necessary.)
##nest.CopyModel('poisson_generator', 'my_poisson_generator', {'rate': 200000.} )
##ext=2.
##noise_layer_dict = {
##    'extent': [ext_hc, ext_hc],
##    'positions': [[0.0, 0.0]],
##    'elements': 'my_poisson_generator'
##}
##
##noiselayer = (topo.CreateLayer(noise_layer_dict))
##
##for j in range(n_hc):  #connect layer to poisson generator
##    topo.ConnectLayers(noiselayer, mc[j], conn_poisson)
##

# In[40]:

##
##layer = mc[0]
##conn_dict = conn_pp
##
##fig = topo.PlotLayer(layer)
##ctr = topo.FindCenterElement(noiselayer) # extract GID of poisson generator
##topo.PlotTargets(ctr, layer, fig=fig, mask=conn_dict['mask'], mask_color='green',
##                 src_size=2, src_color='red',
##                 tgt_size=2, tgt_color='red')
##   
### beautify
###pylab.axes().set_xticks(pylab.arange(-ext_hc*4., ext_hc*4., 0.640))
###pylab.axes().set_yticks(pylab.arange(-ext_hc*4., ext_hc*4., 0.640))
##pylab.grid(True)
##print(ctr)

#================================================================================= learning

numberOfPatterns = numberOfMCPerHC

# prepare the 12 sets of microcolumns; each set for a single pattern
pattern_neurons = [[] for i in range(numberOfPatterns)] # list of lists. e.g., pattern_neurons[0] has the neurons that encode pattern 0.

hc_range = np.arange(0,390,30)

# for each hypercolumn...
for hc_i in range(n_hc):
    # ...gets the 360 neurons (i.e., 30 neurons for each of the 12 MCs) of the HC.
    hc_neurons = nest.GetNodes(mc[hc_i])[0]

    i = 0
    # for each of the 12 patterns...
    while i < numberOfPatterns:
        # ... gets the neurons of the microcolumn corresponding to the pattern.
        pattern_neurons[i].append(hc_neurons[hc_range[i]:hc_range[i+1]])
        i = i + 1

# just in case the nodes are being appended in tuples; I want "pattern_neurons" to be a list of lists of nodes. no tuples.
for i,List in enumerate(pattern_neurons):
    pattern_neurons[i] = [item for sublist in List for item in sublist]

#=======================================

# data vector to be inputted
data = '#LASCON2018!'

# data must have 12 'patterns', one pattern for each set of microcolumns
assert(len(data) == numberOfPatterns)

# converts the characters to ASCII
inputdata = [ord(ch) for ch in data]

pgen_list = []
parrot_list = []

# for each pattern (and for each set of microcolumns):
for i, num in enumerate(inputdata):
    # create poisson generator and parameters
    pgen = nest.Create("poisson_generator")
    
    # repeat many times: 300ms of stimulation; 150ms of rest
##    pgen_start_times = [x for x in np.arange(0.0,T-450.0,450.0)]
##    pgen_stop_times = [x for x in np.arange(300.0,T,450.0)]

    # set rate to the given number
    # (I tried to set many times when the poisson generator starts and stops, but this didn't work.)
##    nest.SetStatus(pgen, {"start": pgen_start_times, "stop": pgen_stop_times, "rate": num*1000.0})
    nest.SetStatus(pgen, {"start": 0.0, "stop": 750.0, "rate": num*1000.0})

    # create parrot neuron to repeat the pgen's output to everyone
    # and connect pgen to it
    parrot = nest.Create("parrot_neuron")
    nest.Connect(pgen, parrot)

    # connect parrot neuron to the respective set of microcolumns
    nest.Connect(parrot, pattern_neurons[i])

    pgen_list.append(pgen)
    parrot_list.append(parrot)

assert(len(pgen_list) == len(parrot_list) == numberOfPatterns)

#==========================================================================================

simulate_and_plot()

#==========================================================================================

# save the connections using pickle
conns = nest.GetConnections(list(allnodes))
x = nest.GetStatus(conns)

for el in x:
    el['synapse_model'] = str(el['synapse_model'])

pickle.dump(x, open("conns.dat","w+"))


# TODO: set parameters for stdp synapse

