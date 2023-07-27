import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import diags
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
import time

st=time.time()

class TimeDependentRMS(object):
    def __init__(self, nt=100000, dt=0.01, NN=2**6):
        self.nt = nt
        self.dt = dt
        self.NN = NN
        self.r = np.linspace(0, 1, NN+2)
        self.dr = self.r[1] - self.r[0]

        # Other variables used in the solver

    def initialize_profiles(self, q0, q1, q2, eta_value=1e-6, nu_value=1e-6, m=1, n=1):
        # Initialize profiles for q, s, eta, and nu based on input parameters
        # ...
        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.eta = eta_value + 0*self.r
        self.nu = nu_value
        self.m = m
        self.n = n

        q = q0 + q1*self.r + q2*self.r**2
        qp = q1 + 2*q2*self.r
        qpp = 2*q2
        s = self.r*qp/q
        sp = (self.r*q*qpp + q*qp - qp**2*self.r) / q**2

        self.rr = self.r[1:self.NN]  
        self.qj = q[1:self.NN]
        self.sj = s[1:self.NN]
        self.spj = sp[1:self.NN]
        self.etaj = self.eta[1:self.NN]

        self.uu_values = []
        self.t_values = []
        self.psi_values = []
        self.phi_values = []

    def setup_matrix(self):
        # Setup matrix elements of DI - tri-diagonal and sparse matrix
        # ...
        rhojp = (1/self.dr**2 + 1/(2*self.rr*self.dr))   # rho_j[+1]
        rhoj0 = 0.0*self.rr - 2/self.dr**2             # rho_j[0]
        rhojm = (1/self.dr**2 - 1/(2*self.rr*self.dr))   # rho_j[-1]

        self.DI = diags([rhoj0 - self.m**2/self.rr**2, rhojm[1:], rhojp[:self.NN-1]], 
                        [0, -1, 1], shape=(self.NN-1, self.NN-1), dtype=complex)
        self.DI = self.DI.tocsr()

    def solve(self):
        gamma = np.zeros(self.nt)
        AA = np.zeros(self.nt)
        AR = np.zeros(self.nt)
        t = np.zeros(self.nt)

        psi = 0.0 * np.exp(-(self.rr - 0.2)**2 / 0.01)
        uu = (1.0 + 0.0j) * 0.01 * np.exp(-((self.rr - 0.3)** 2) / 0.01)
        phi = 0.0 * self.rr

        for it in range(self.nt):
            # Perform time-dependent matrix evaluation for magnetic flux function and electrostatic potential

            AA[it] = np.log(np.max(np.abs(psi)))
            AR[it] = np.log(np.max(np.abs(np.real(psi))))

            if it > 0:
                gamma[it] = (AA[it] - AA[it-1]) / self.dt

            t[it] = (it-1) * self.dt

            phi = spsolve(self.DI, uu.T)
            grad2perppsi = (self.DI @ psi.T).T
            grad2perpu = (self.DI @ uu.T).T

            RC1 = 1j * (self.n - self.m/self.qj) * phi + self.etaj * grad2perppsi
            RC2 = 1j * (self.n - self.m/self.qj) * grad2perppsi + self.nu * grad2perpu - 1j * self.m / self.rr * (self.spj / self.qj - self.sj * (self.sj - 2) / (self.rr * self.qj)) * psi

            psi = psi + RC1 * self.dt
            uu = uu + RC2 * self.dt

            
            if it==self.nt or it % (self.nt//10) == 1:

                h=plt.figure(figsize=(14,4))

                plt.subplot(1,5,1)
                plt.plot(self.rr, np.real(psi), self.rr, np.imag(psi), 'r--', linewidth=1)
                plt.title('Ψ')

                plt.subplot(1,5,2)
                plt.plot(self.rr, np.real(phi), self.rr, np.imag(phi), 'r--', linewidth=1)
                plt.ylabel('φ')

                plt.subplot(1,5,3)
                plt.plot(self.rr, np.real(uu), self.rr, np.imag(uu),'r--', linewidth=1)
                plt.xlabel('U')

                plt.subplot(1,5,4)
                plt.plot(t[:it-1], AA[:it-1], t[it-1], AR[it-1], 'r--', linewidth=1)
                plt.title('AA, t={}'.format((it-1)*self.dt))
                plt.xlabel('t')
        
                plt.subplot(1,5,5)
                plt.plot(t[1:it-1], gamma[1:it-1], 'r', [0,t[it]],[0, 0], 'g--', linewidth=1.2)
                plt.title('Growth rate= γ')
                plt.xlabel('t')
                plt.ylim([-0.012198004824249,0.012198004824249])

                plt.show(block= False)
                plt.pause(3)
                plt.clf()
            
        self.uu_values.append(uu)
        self.t_values.append(t)
        self.psi_values.append(psi)
        self.phi_values.append(phi)
        et=time.time()
        print('Simulation time =', et-st,'sec')
                

    def plot_contour(self):
        
        theta = np.arange(0, 2*np.pi, np.pi/50)
        r, theta = np.meshgrid(self.rr, theta)
        col,row = r.shape
        psi = np.tile(self.psi_values, (col, 1)) * np.exp(-1j * self.m * theta)
        phi = np.tile(self.phi_values, (col, 1)) * np.exp(-1j * self.m * theta)
        
        U = np.tile(self.uu_values, (col, 1)) * np.exp(-1j * self.m * theta)
        X = r * np.cos(theta)
        Z = r * np.sin(theta)
        
        
        h=plt.figure(figsize= (14,4))
        
        plt.subplot(1,3,1)
        plt.pcolor(X, Z, np.real(psi))
        plt.axis('equal')
        plt.title('\psi, 2D contour')
        plt.colorbar(location='right')
        plt.set_cmap('hsv')
        
        plt.subplot(1,3,2)
        plt.pcolor(X, Z, np.real(phi))
        plt.axis('equal')
        plt.title('\phi, 2D contour')
        plt.colorbar(location='right')
        plt.set_cmap('hsv')

        plt.subplot(1,3,3)
        plt.pcolor(X, Z, np.real(U))
        plt.axis('equal')
        plt.title('U, 2D contour')
        plt.colorbar(location='right')
        plt.set_cmap('hsv')


