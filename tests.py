#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#%%
import numpy as np
import hexagonal_network as hn
import matplotlib.pyplot as pl
import pickle

#%%
print(hn.in_hexagon(0.5, 0.2))
print(hn.in_hexagon(1.1, 0.0))
print(hn.in_hexagon(1.2, 2.1, 1.0, 2.0, 0.6))
print(hn.in_hexagon(1.4, 2.4, 1.0, 2.0, 0.6))

#%%
N = 1000
Cx=1.0
Cy=2.0
R=10.0
X = np.zeros(N)
Y = np.zeros(N)

for i in range(N):
    X[i], Y[i] = hn.rand_in_circle(Cx, Cy, R)
    
pl.plot(X, Y, 'b.')
print(min(X), max(X), min(Y), max(Y))

#%%
N = 1000
Cx=1.0
Cy=2.0
R=10.0
X = np.zeros(N)
Y = np.zeros(N)

for i in range(N):
    X[i], Y[i] = hn.rand_in_hexagon(Cx, Cy, R)

pl.plot(X, Y, 'b.')
hn.plot_hexagon(Cx, Cy, R, Nedge=1000)

#%%
numberOfMC = 12
r = 0.1
Cx = 1.0
Cy = 2.0
R = 10.0
listOfMCCenters = hn.mc_centers(numberOfMC, r, Cx, Cy, R)

X = np.zeros(numberOfMC)
Y = np.zeros(numberOfMC)

for i in range(numberOfMC):
    X[i] = listOfMCCenters[i][0]
    Y[i] = listOfMCCenters[i][1]

pl.plot(X, Y, 'bo')
hn.plot_hexagon(Cx, Cy, R, Nedge=1000)

#%%
listOfAllHCCenters = hn.all_hc_centers()

#%%
numberOfMCPerHC = 30
r = 0.1
R = 10.0

listOfAllHCCenters = hn.all_hc_centers(R)
#listOfMCCenters = hn.mc_centers(numberOfMC, r, Cx, Cy, R)
listOfAllMCCenters = hn.all_mc_centers(numberOfMCPerHC, r, R)
length = len(listOfAllMCCenters)

X = np.zeros(length)
Y = np.zeros(length)

for i in range(length):
    X[i] = listOfAllMCCenters[i][0]
    Y[i] = listOfAllMCCenters[i][1]

pl.plot(X, Y, 'b.')
hn.plot_all_hexagons(R, Nedge=1000)

#%%
numberOfNeuronsPerMC = 30
numberOfMCPerHC = 12
r = 0.1
R = 1.0

listOfAllNeuronCenters = hn.all_neuron_centers(numberOfNeuronsPerMC, numberOfMCPerHC, r, R)

length = len(listOfAllNeuronCenters)

X = np.zeros(length)
Y = np.zeros(length)

for i in range(length):
    X[i] = listOfAllNeuronCenters[i][0]
    Y[i] = listOfAllNeuronCenters[i][1]

pl.plot(X, Y, 'b.')
hn.plot_all_hexagons(R, Nedge=1000)

#%%
numberOfMCPerHC=12
numberOfPyrPerMC=30
numberOfBasketPerHC=24
r=0.1
R=1.0
listOfPyrCenters, listOfBasketCenters = hn.all_pyr_and_basket_cells_in_HC(numberOfMCPerHC, numberOfPyrPerMC, numberOfBasketPerHC, r, R)

numberOfPyr = numberOfPyrPerMC*numberOfMCPerHC*16
numberOfBasket = numberOfBasketPerHC*16

Xpyr = np.zeros(numberOfPyr)
Ypyr = np.zeros(numberOfPyr)

Xbasket = np.zeros(numberOfBasket)
Ybasket = np.zeros(numberOfBasket)

for i in range(numberOfPyr):
    Xpyr[i] = listOfPyrCenters[i][0]
    Ypyr[i] = listOfPyrCenters[i][1]

for i in range(numberOfBasket):
    Xbasket[i] = listOfBasketCenters[i][0]
    Ybasket[i] = listOfBasketCenters[i][1]
    
pl.plot(Xpyr, Ypyr, 'b.')
pl.plot(Xbasket, Ybasket, 'gx')
hn.plot_all_hexagons(R, Nedge=1000)

#%%
# Write listOfPyrCenters in the file pyramidCellsPositions
with open('pyramidCellsPositions', 'wb') as pyramidFile:
    my_pickler = pickle.Pickler(pyramidFile)
    my_pickler.dump(listOfPyrCenters)

#%%
# Read positions of pyramidal cells from the file pyramidCellsPositions
with open('pyramidCellsPositions', 'rb') as pyramidFile:
    my_unpickler = pickle.Unpickler(pyramidFile)
    storedPyramidPositions = my_unpickler.load()
    
#%%
# Write listOfBasketCenters in the file basketCellsPositions
with open('basketCellsPositions', 'wb') as basketFile:
    my_pickler = pickle.Pickler(basketFile)
    my_pickler.dump(listOfBasketCenters)

#%%
# Read positions of basket cells from the file basketCellsPositions
with open('basketCellsPositions', 'rb') as basketFile:
    my_unpickler = pickle.Unpickler(basketFile)
    storedBasketPositions = my_unpickler.load()
