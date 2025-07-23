import scipy
import numpy
from scipy.integrate import quad
from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

'''A script to draw a black hole. Adapted from https://blog.devgenius.io/how-to-draw-a-blackhole-in-python-e674b831431a.'''

def draw_blackhole():
    def radial_potential(r, b):
        return r**4 - b**2*r**2 + 2*b**2*r


    def minrs(b):
        temp = np.cbrt(b**2)*((-1 + np.sqrt(1+0j-b**2/27))**(1/3)+(-1 - np.sqrt(1+0j-b**2/27))**(1/3))
        return np.real(temp).real if np.imag(temp) < 1e-10 else 0


    def d_phi_dr(r, b):
        return b / np.sqrt(radial_potential(r, b))


    def phi(rs, b):
        return quad(d_phi_dr, rs, np.inf, args=(b))

    def d_phi_dx(x, b, rt):
        return -2*b*np.abs(x) / \
            (
                (x**2+rt) * np.sqrt(-((b**2*(-2+x**2+rt)) / (x**2+rt)) + (x**2+rt)**2)
            )


    def phi(xi, b):
        rt = minrs(b)
        if xi.imag != 0:
            return np.inf
        if rt == 0 and xi <0:
            return quad(d_phi_dx, np.inf, 0, args=(b,rt))
        return quad(d_phi_dx, np.inf, xi, args=(b,rt))

    def generate_trajectory(ximin, ximax, b, steps):
        rt = minrs(b)
        coords = np.ndarray([steps,2])
        for step in range(steps):
            xi = step*(ximax - ximin)/steps + ximin
            rs = 0 if rt == 0 and xi < 0 else (xi**2 + rt)
            coords[step][0] = rs
            coords[step][1] =  phi(xi, b)[0]
        x_vals = list(map(lambda x: x[0]*numpy.cos(x[1]),coords))
        y_vals = list(map(lambda x: x[0]*numpy.sin(x[1]),coords))
        return x_vals, y_vals

    def psi(phi, theta):
        return np.arccos(np.sin(theta)*np.sin(phi))


    def varphi(phi, theta):
        return np.arctan2(np.cos(phi), np.cos(theta)*np.sin(phi))

    steps = 2000
    theta = 85*np.pi/180
    rmin = 5
    rmax = 25
    rsteps = 5

    #n = 0
    for rs in range(rmin, rmax, rsteps):
        bmax = np.sqrt(rs**3/ (rs-2))
        phivals = np.linspace(0.0001*np.pi,2*np.pi*0.9999,num=steps)
        varphivals = np.array([np.pi if phi >= np.pi else 0 for phi in phivals])+np.arccos(np.sign(np.pi-phivals)*np.cos(phivals)/np.sqrt(np.cos(phivals)**2 +np.sin(phivals)**2*np.cos(theta)**2))
        psivals1 = np.array([psi(phi, theta) for phi in phivals])
        psivals2 = np.array([[phi(np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(0,bmax,num=steps)])
        psivals3 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(3,bmax,num=steps)])
        bvals = np.array([min([min(psivals2, key=lambda x:abs(x[0]-psi)), min(psivals3, key=lambda x:abs(x[0]-psi))], key=lambda x:abs(x[0]-psi ))[1] for psi in psivals1])
        alphavals = (bvals) * np.cos(varphivals)
        betavals = -(bvals) * np.sin(varphivals)
        plt.plot(alphavals, betavals, 'b', zorder=9)


    #n=1    
    for rs in range(rmin, rmax, rsteps):
        bmax = np.sqrt(rs**3/ (rs-2))
        phivals = np.linspace(0.001*np.pi,2*np.pi*0.999,num=steps)
        varphivals = np.array([np.pi if phi >= np.pi else 0 for phi in phivals])+np.arccos(np.sign(np.pi-phivals)*np.cos(phivals)/np.sqrt(np.cos(phivals)**2 +np.sin(phivals)**2*np.cos(theta)**2))
        psivals1 = np.pi + np.array([psi(phi, theta) for phi in phivals]) # n=1 is pi degrees away from n=0
        psivals2 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        psivals3 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        bvals = np.array([min([min(psivals2, key=lambda x:abs(x[0]-psi)), min(psivals3, key=lambda x:abs(x[0]-psi))], key=lambda x:abs(x[0]-psi ))[1] for psi in psivals1])
        alphavals = (bvals) * np.cos(varphivals)
        betavals = -(bvals) * np.sin(varphivals)
        plt.plot(alphavals, betavals, 'g', zorder=8)

    #n=2    
    for rs in range(rmin, rmax, rsteps):
        bmax = np.sqrt(rs**3/ (rs-2))
        phivals = np.linspace(0.001*np.pi,2*np.pi*0.999,num=steps)
        varphivals = np.array([np.pi if phi >= np.pi else 0 for phi in phivals])+np.arccos(np.sign(np.pi-phivals)*np.cos(phivals)/np.sqrt(np.cos(phivals)**2 +np.sin(phivals)**2*np.cos(theta)**2))
        psivals1 = 2*np.pi + np.array([psi(phi, theta) for phi in phivals]) # n=2 is 2pi degrees away from n=0
        psivals2 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        psivals3 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        bvals = np.array([min([min(psivals2, key=lambda x:abs(x[0]-psi)), min(psivals3, key=lambda x:abs(x[0]-psi))], key=lambda x:abs(x[0]-psi ))[1] for psi in psivals1])
        alphavals = (bvals) * np.cos(varphivals)
        betavals = -(bvals) * np.sin(varphivals)
        plt.plot(alphavals, betavals, 'r', zorder=7)


    ax = plt.gca()   
    ax.add_patch(plt.Circle((0, 0), np.sqrt(27), color='k', zorder=8))

    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_aspect(1)
    plt.show()

def draw_xrb():
    def radial_potential(r, b):
        return r**4 - b**2*r**2 + 2*b**2*r


    def minrs(b):
        temp = np.cbrt(b**2)*((-1 + np.sqrt(1+0j-b**2/27))**(1/3)+(-1 - np.sqrt(1+0j-b**2/27))**(1/3))
        return np.real(temp).real if np.imag(temp) < 1e-10 else 0


    def d_phi_dr(r, b):
        return b / np.sqrt(radial_potential(r, b))


    def phi(rs, b):
        return quad(d_phi_dr, rs, np.inf, args=(b))

    def d_phi_dx(x, b, rt):
        return -2*b*np.abs(x) / \
            (
                (x**2+rt) * np.sqrt(-((b**2*(-2+x**2+rt)) / (x**2+rt)) + (x**2+rt)**2)
            )


    def phi(xi, b):
        rt = minrs(b)
        if xi.imag != 0:
            return np.inf
        if rt == 0 and xi <0:
            return quad(d_phi_dx, np.inf, 0, args=(b,rt))
        return quad(d_phi_dx, np.inf, xi, args=(b,rt))

    def generate_trajectory(ximin, ximax, b, steps):
        rt = minrs(b)
        coords = np.ndarray([steps,2])
        for step in range(steps):
            xi = step*(ximax - ximin)/steps + ximin
            rs = 0 if rt == 0 and xi < 0 else (xi**2 + rt)
            coords[step][0] = rs
            coords[step][1] =  phi(xi, b)[0]
        x_vals = list(map(lambda x: x[0]*numpy.cos(x[1]),coords))
        y_vals = list(map(lambda x: x[0]*numpy.sin(x[1]),coords))
        return x_vals, y_vals

    def psi(phi, theta):
        return np.arccos(np.sin(theta)*np.sin(phi))


    def varphi(phi, theta):
        return np.arctan2(np.cos(phi), np.cos(theta)*np.sin(phi))

    steps = 2000
    theta = 85*np.pi/180
    rmin = 5
    rmax = 25
    rsteps = 5

    #n = 0
    for rs in range(rmin, rmax, rsteps):
        bmax = np.sqrt(rs**3/ (rs-2))
        phivals = np.linspace(0.0001*np.pi,2*np.pi*0.9999,num=steps)
        varphivals = np.array([np.pi if phi >= np.pi else 0 for phi in phivals])+np.arccos(np.sign(np.pi-phivals)*np.cos(phivals)/np.sqrt(np.cos(phivals)**2 +np.sin(phivals)**2*np.cos(theta)**2))
        psivals1 = np.array([psi(phi, theta) for phi in phivals])
        psivals2 = np.array([[phi(np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(0,bmax,num=steps)])
        psivals3 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(3,bmax,num=steps)])
        bvals = np.array([min([min(psivals2, key=lambda x:abs(x[0]-psi)), min(psivals3, key=lambda x:abs(x[0]-psi))], key=lambda x:abs(x[0]-psi ))[1] for psi in psivals1])
        alphavals = (bvals) * np.cos(varphivals)
        betavals = -(bvals) * np.sin(varphivals)
        plt.plot(alphavals, betavals, 'b', zorder=9)


    #n=1    
    for rs in range(rmin, rmax, rsteps):
        bmax = np.sqrt(rs**3/ (rs-2))
        phivals = np.linspace(0.001*np.pi,2*np.pi*0.999,num=steps)
        varphivals = np.array([np.pi if phi >= np.pi else 0 for phi in phivals])+np.arccos(np.sign(np.pi-phivals)*np.cos(phivals)/np.sqrt(np.cos(phivals)**2 +np.sin(phivals)**2*np.cos(theta)**2))
        psivals1 = np.pi + np.array([psi(phi, theta) for phi in phivals]) # n=1 is pi degrees away from n=0
        psivals2 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        psivals3 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        bvals = np.array([min([min(psivals2, key=lambda x:abs(x[0]-psi)), min(psivals3, key=lambda x:abs(x[0]-psi))], key=lambda x:abs(x[0]-psi ))[1] for psi in psivals1])
        alphavals = (bvals) * np.cos(varphivals)
        betavals = -(bvals) * np.sin(varphivals)
        plt.plot(alphavals, betavals, 'g', zorder=8)

    #n=2    
    for rs in range(rmin, rmax, rsteps):
        bmax = np.sqrt(rs**3/ (rs-2))
        phivals = np.linspace(0.001*np.pi,2*np.pi*0.999,num=steps)
        varphivals = np.array([np.pi if phi >= np.pi else 0 for phi in phivals])+np.arccos(np.sign(np.pi-phivals)*np.cos(phivals)/np.sqrt(np.cos(phivals)**2 +np.sin(phivals)**2*np.cos(theta)**2))
        psivals1 = 2*np.pi + np.array([psi(phi, theta) for phi in phivals]) # n=2 is 2pi degrees away from n=0
        psivals2 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        psivals3 = np.array([[phi(-np.sqrt(rs-minrs(b)), b)[0],b] for b in np.linspace(5,bmax,num=steps)])
        bvals = np.array([min([min(psivals2, key=lambda x:abs(x[0]-psi)), min(psivals3, key=lambda x:abs(x[0]-psi))], key=lambda x:abs(x[0]-psi ))[1] for psi in psivals1])
        alphavals = (bvals) * np.cos(varphivals)
        betavals = -(bvals) * np.sin(varphivals)
        plt.plot(alphavals, betavals, 'r', zorder=7)


    ax = plt.gca()   
    ax.add_patch(plt.Circle((0, 0), np.sqrt(27), color='k', zorder=8))

    x_coordinate = 20
    y_coordinate = 7.5

    # Create a scatter plot with a star marker
    plt.scatter(x_coordinate, y_coordinate, s=1000, marker='*', color='gold')

    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_aspect(1)
    plt.show()