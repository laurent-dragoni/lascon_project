#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl


def in_hexagon(x, y, Cx=0.0, Cy=0.0, R=1.0):
    
    """
    Returns true if the point of coordinates (x,y) is in the hexagon centered
    at (Cx,Cy) and of radius R
    """
    
    V3 = np.sqrt(3)
    # (x,y) has to verify the 3 equations given by the 3 pairs of parallel edges of the hexagon
    return (np.abs(x - Cx) < V3*R/2) & (np.abs(y - x/V3 - Cy + Cx/V3) < R) & (np.abs(y + x/V3 - Cy - Cx/V3) < R)


def plot_hexagon(Cx=0.0, Cy=0.0, R=1.0, Nedge=1000, plotStyle='r-'):
    
    """
    Plots the hexagon centered at (Cx,Cy) and of radius R.
    Nedge is the number of points to plot for each edge of the hexagon.
    plotStyle : string describing the plotting style of the hexagon edges,
    default is red lines.
    """
    
    # vertical edges
    T1 = np.zeros((Nedge,2))
    T2 = np.zeros((Nedge,2))

    # first diagonal edges
    delta1 = np.zeros((Nedge,2))
    delta2 = np.zeros((Nedge,2))
    
    # second diagonal edges
    D1 = np.zeros((Nedge,2))
    D2 = np.zeros((Nedge,2))
    
    V3 = np.sqrt(3)
    
    for i in range(Nedge):
        
        T1[i,0] = Cx - V3*R/2
        T1[i,1] = Cy - R/2 + i*R/(Nedge-1)
        
        T2[i,0] = Cx + V3*R/2
        T2[i,1] = Cy - R/2 + i*R/(Nedge-1)
        
        delta1[i,0] = Cx + i*R*V3/(2*(Nedge-1))
        delta1[i,1] = delta1[i,0]/V3 + Cy - R - Cx/V3
        
        delta2[i,0] = Cx - V3*R/2 + i*R*V3/(2*(Nedge-1))
        delta2[i,1] = delta2[i,0]/V3 + Cy + R - Cx/V3
        
        D1[i,0] = Cx - V3*R/2 + i*R*V3/(2*(Nedge-1))
        D1[i,1] = -D1[i,0]/V3 + Cy - R + Cx/V3
        
        D2[i,0] = Cx + i*R*V3/(2*(Nedge-1))
        D2[i,1] = -D2[i,0]/V3 + Cy + R + Cx/V3        
    
    pl.plot(T1[:,0], T1[:,1], plotStyle)
    pl.plot(T2[:,0], T2[:,1], plotStyle)
    pl.plot(delta1[:,0], delta1[:,1], plotStyle)
    pl.plot(delta2[:,0], delta2[:,1], plotStyle)
    pl.plot(D1[:,0], D1[:,1], plotStyle)
    pl.plot(D2[:,0], D2[:,1], plotStyle)
    
def plot_all_hexagons(R=1.0, Nedge=1000, plotStyle='r-'):
    
    """
    Plots all the 16 hexagons from the article.
    Nedge is the number of points to plot for each edge of the hexagon.
    plotStyle : string describing the plotting style of the hexagon edges,
    default is red lines.
    """
    
    listOfAllHCCenters = all_hc_centers(R)
    length = len(listOfAllHCCenters)
    
    for i in range(length):
        Cxi = listOfAllHCCenters[i][0]
        Cyi = listOfAllHCCenters[i][1]
        plot_hexagon(Cxi, Cyi, R, Nedge, plotStyle)

def new_plot_all_hexagons(listOfHCCenters, R=1.0, Nedge=1000, plotStyle='r-'):
    
    """
    Plots all the hexagons of radius R from the list listOfHCCenters.
    Nedge is the number of points to plot for each edge of the hexagon.
    plotStyle : string describing the plotting style of the hexagon edges,
    default is red lines.
    """
    
    length = len(listOfHCCenters)
    
    for i in range(length):
        Cxi = listOfHCCenters[i][0]
        Cyi = listOfHCCenters[i][1]
        plot_hexagon(Cxi, Cyi, R, Nedge, plotStyle)
    
    
def rand_in_circle(Cx=0.0, Cy=0.0, R=1.0):
    
    """
    Returns a random point in the circle centered at (Cx,Cy) and of radius R.
    """
    
    R2 = R*R
    
    # (x,y) uniformly generated on a square containing the circle
    x = np.random.uniform(Cx - R, Cx + R)
    y = np.random.uniform(Cy - R, Cy + R)
    
    while ((x-Cx)*(x-Cx) + (y-Cy)*(y-Cy) >= R2):
        # as long as (x,y) is not in the circle, we generate a new one
        x = np.random.uniform(Cx - R, Cx + R)
        y = np.random.uniform(Cy - R, Cy + R)
    
    return x, y


def rand_in_hexagon(Cx=0.0, Cy=0.0, R=1.0):
    
    """
    Returns a random point in the hexagon centered at (Cx,Cy) and of radius R.
    """
    
    # (x,y) uniformly generated on a circle containing the hexagon
    x, y = rand_in_circle(Cx, Cy, R)
    
    while(not in_hexagon(x, y, Cx, Cy, R)):
        # as long as (x,y) is not in the hexagon, we generate a new one
        x, y = rand_in_circle(Cx, Cy, R)
    
    return x,y


def mc_overlap(listOfMCCenters, newx, newy, r=0.1):
    
    """
    Returns true if the new center candidate of coordinates (newx, newy)
    generates a circle which overlaps with at least one of the circles
    associated to the centers in listOfMCCenters.
    """
    
    overlap = False
    length = len(listOfMCCenters)
    
    for i in range(length):
        
        dist2 = (listOfMCCenters[i][0] - newx)**2 + (listOfMCCenters[i][1] - newy)**2
        
        if dist2 <= 4*r*r: # overlap !
            overlap = True
            break
    
    return overlap


def all_hc_centers(R=1.0):
    
    """
    Returns the list of centers of the 16 hexagons (HyperColumns) described in
    the article. The whole grid is supposed to be centered at (0,0).
    """
    
    listOfAllHCCenters = []
    d = np.sqrt(3)*R/2
    L = 9*np.sqrt(3)*R/2
    H = 13*R/2
    
    # top line hexagons centers
    listOfAllHCCenters += [(-L/2 + d, H/2 - R)]
    listOfAllHCCenters += [(-L/2 + 3*d, H/2 - R)]
    listOfAllHCCenters += [(-L/2 + 5*d, H/2 - R)]
    listOfAllHCCenters += [(-L/2 + 7*d, H/2 - R)]
    
    # seccond line hexagons centers
    listOfAllHCCenters += [(-L/2 + 2*d, H/2 - 5*R/2)]
    listOfAllHCCenters += [(-L/2 + 4*d, H/2 - 5*R/2)]
    listOfAllHCCenters += [(-L/2 + 6*d, H/2 - 5*R/2)]
    listOfAllHCCenters += [(-L/2 + 8*d, H/2 - 5*R/2)]
    
    # third line hexagons centers
    listOfAllHCCenters += [(-L/2 + d, H/2 - 4*R)]
    listOfAllHCCenters += [(-L/2 + 3*d, H/2 - 4*R)]
    listOfAllHCCenters += [(-L/2 + 5*d, H/2 - 4*R)]
    listOfAllHCCenters += [(-L/2 + 7*d, H/2 - 4*R)]
    
    # bottom line hexagons centers
    listOfAllHCCenters += [(-L/2 + 2*d, H/2 - 11*R/2)]
    listOfAllHCCenters += [(-L/2 + 4*d, H/2 - 11*R/2)]
    listOfAllHCCenters += [(-L/2 + 6*d, H/2 - 11*R/2)]
    listOfAllHCCenters += [(-L/2 + 8*d, H/2 - 11*R/2)]
    
    return listOfAllHCCenters

def new_all_hc_centers(nx, ny, R=1.0):
    
    """
    Returns the list of centers of hexagons (HyperColumns).
    The whole grid is supposed to be centered at (0,0).
    
    nx = number of hexagons on the x axis.
    ny = number of hexagons on the y axis.
    R = radius of the hexagons.
    """
    
    listOfAllHCCenters = []
    Diam = R*np.sqrt(3)
    L = (nx + 0.5)*Diam
    H = (3*ny + 1)*R/2.    
    d = Diam/2.
    
    for j in range(ny):
        currenty = H/2. - R -j*3.*R/2
        for i in range(nx):
            if j%2 == 0:
                currentx = -L/2. + d + i*Diam
                listOfAllHCCenters += [(currentx,currenty)]
            else:
                currentx = -L/2. + Diam + i*Diam
                listOfAllHCCenters += [(currentx,currenty)]
    
    return listOfAllHCCenters
    
    
def mc_centers_in_hc(numberOfMCPerHC=12, r=0.1, Cx=0.0, Cy=0.0, R=1.0):
    
    """
    Each Minicolumn is seen as a disc of radius r.
    Returns a list of numberOfMCPerHC centers of the Minicolumns belonging to the
    hexagon centered at (Cx,Cy) and of radius R.
    We also ensure that the MCs do not overlap.
    """
    
    # generating a new center in a slightly smaller hexagon, so that the 
    # circle (Minicolumn) associated is entirely in the hexagon
    newx, newy = rand_in_hexagon(Cx, Cy, R - 2*r/np.sqrt(3))
    listOfMCCenters = [(newx, newy)]
    
    for i in range(numberOfMCPerHC-1):

        newx, newy = rand_in_hexagon(Cx, Cy, R - 2*r/np.sqrt(3))
        
        while(mc_overlap(listOfMCCenters, newx, newy, r=0.1)):
            # we don't want overlaping MCs => redraw a new one
            newx, newy = rand_in_hexagon(Cx, Cy, R - 2*r/np.sqrt(3))
            
        listOfMCCenters += [(newx, newy)]
    
    return listOfMCCenters

def all_mc_centers(numberOfMCPerHC=12, r=0.1, R=1.0):
    """
    Generates all the MC centers of the model.
    numberOfMCPerHC : number of MiniColumn per HyperColumn
    r : radius of each MiniColumn
    R : radius of each HyperColumn (hexagon)
    """
    
    listOfAllMCCenters = []
    
    listOfAllHCCenters = all_hc_centers(R)
    length = len(listOfAllHCCenters)
    
    for i in range(length):
        # center of the current hexagon
        Cx, Cy = listOfAllHCCenters[i]
        # generates MC centers in this hexagon
        listOfAllMCCenters += mc_centers_in_hc(numberOfMCPerHC, r, Cx, Cy, R)
    
    return listOfAllMCCenters        
        

def pyr_centers_in_mc(numberOfPyrPerMC=30, r=0.1, Cx=0.0, Cy=0.0):
    """
    Generates numberOfPyrPerMC Pyramidal Cells centers inside the MiniColumn
    centered at (Cx,Cy) and of radius r.
    """
    
    listOfPyrCenters = []
    
    for i in range(numberOfPyrPerMC):
        x, y = rand_in_circle(Cx, Cy, r)
        while (x,y) in listOfPyrCenters:
            # we don't want neurons at exactly the same position
            x, y = rand_in_circle(Cx, Cy, r)
        listOfPyrCenters += [(x,y)]
    
    return listOfPyrCenters

def basket_centers_in_hc(numberOfBasketPerHC, listOfPyrCenters, Cx=0.0, Cy=0.0, R=1.0):
    """
    Generates numberOfBasketPerHC Basket Cells centers inside the HyperColumn
    centered at (Cx,Cy) and of radius R. We also do not want Basket Cells at
    the same coordinates as Pyramidal Cells.
    """
    
    listOfBasketCenters = []
    
    for i in range(numberOfBasketPerHC):
        x, y = rand_in_hexagon(Cx, Cy, R)
        while (x,y) in listOfPyrCenters or (x,y) in listOfBasketCenters: 
            # redraw if a Pyramidal Cell or a Basket Cell is already at 
            # this location
            x, y = rand_in_hexagon(Cx, Cy, R)
        listOfBasketCenters += [(x,y)]
    
    return listOfBasketCenters


#def all_neuron_centers(numberOfNeuronsPerMC=30, numberOfMCPerHC=12, r=0.1, R=1.0):
#    """
#    Generates all the neurons centers of the model.
#    
#    numberOfNeuronsPerMC : number of neurons per MiniColumn.
#    numberOfMCPerHC : number of MiniColumn per HyperColumn.    
#    r : radius of each MiniColumn (circle)
#    R : radius of each HyperColumn (hexagon)
#    """
#    
#    listOfAllNeuronCenters = []
#    listOfAllMCCenters = all_mc_centers(numberOfMCPerHC, r, R)
#    length = len(listOfAllMCCenters)
#    
#    for i in range(length): # for each MiniColumn MCi, generates its neurons
#        Cxi = listOfAllMCCenters[i][0]
#        Cyi = listOfAllMCCenters[i][1]
#        listOfAllNeuronCenters += pyr_centers_in_mc(numberOfNeuronsPerMC, r, Cxi, Cyi)
#    
#    return listOfAllNeuronCenters


def all_pyr_and_basket_cells(numberOfMCPerHC=12, numberOfPyrPerMC=30, numberOfBasketPerHC=24, r=0.1, R=1.0):
    """
    Generates all the Pyramidal Cells centers and all the Basket Cells centers.
    
    numberOfMCPerHC : number of MiniColumn per HyperColumn.
    numberOfPyrPerMC : number of Pyramidal Cells per MiniColumn.
    numberOfBasketPerHC : number of Basket Cells per HyperColumn.
    r : radius of each MiniColumn (circle).
    R : radius of the HyperColumn (hexagon).
    """
    
    listOfAllHCCenters = all_hc_centers(R)    
    
    listOfPyrCenters = []
    listOfBasketCenters = []
    
    # for each HyperColumn, generate its MiniColumns, Pyramidal Cells and
    # Basket Cells
    for i in range(16):
        
        listOfPyrCentersInHCi = []        
        CxHC = listOfAllHCCenters[i][0]
        CyHC = listOfAllHCCenters[i][1]        
        listOfMCCenters = mc_centers_in_hc(numberOfMCPerHC, r, CxHC, CyHC, R)
        
        # generates Pyramidal Cells in every MC of HC number i
        for j in range(numberOfMCPerHC):
            
            CxMC = listOfMCCenters[j][0]
            CyMC = listOfMCCenters[j][1]
            listOfPyrCentersInHCi = pyr_centers_in_mc(numberOfPyrPerMC, r, CxMC, CyMC)
            listOfPyrCenters += listOfPyrCentersInHCi
        
        listOfBasketCenters += basket_centers_in_hc(numberOfBasketPerHC, listOfPyrCentersInHCi, CxHC, CyHC, R)

    return listOfPyrCenters, listOfBasketCenters

def new_all_pyr_and_basket_cells(nx = 4, ny = 4, numberOfMCPerHC=12, numberOfPyrPerMC=30, numberOfBasketPerHC=24, r=0.1, R=1.0):
    """
    Generates all the Pyramidal Cells centers and all the Basket Cells centers.
    
    nx : number of HyperColumn on the x axis.
    ny : number of HyperColumn on the y axis.
    
    numberOfMCPerHC : number of MiniColumn per HyperColumn.
    numberOfPyrPerMC : number of Pyramidal Cells per MiniColumn.
    numberOfBasketPerHC : number of Basket Cells per HyperColumn.
    r : radius of each MiniColumn (circle).
    R : radius of the HyperColumn (hexagon).
    """
    
    numberOfHC = nx*ny
    listOfAllHCCenters = new_all_hc_centers(nx, ny, R)    
    
    listOfPyrCenters = []
    listOfBasketCenters = []
    
    # for each HyperColumn, generate its MiniColumns, Pyramidal Cells and
    # Basket Cells
    for i in range(numberOfHC):
        
        listOfPyrCentersInHCi = []        
        CxHC = listOfAllHCCenters[i][0]
        CyHC = listOfAllHCCenters[i][1]        
        listOfMCCenters = mc_centers_in_hc(numberOfMCPerHC, r, CxHC, CyHC, R)
        
        # generates Pyramidal Cells in every MC of HC number i
        for j in range(numberOfMCPerHC):
            
            CxMC = listOfMCCenters[j][0]
            CyMC = listOfMCCenters[j][1]
            listOfPyrCentersInHCi = pyr_centers_in_mc(numberOfPyrPerMC, r, CxMC, CyMC)
            listOfPyrCenters += listOfPyrCentersInHCi
        
        listOfBasketCenters += basket_centers_in_hc(numberOfBasketPerHC, listOfPyrCentersInHCi, CxHC, CyHC, R)

    return listOfPyrCenters, listOfBasketCenters