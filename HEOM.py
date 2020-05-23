# This code is a python code for the HEOM on a dimer
# It is based on the Ishizaki paper of J. Chem. Phys. 130 234111 (2009)
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Define indexing function
def myindex(a,b,N):
  ind=int(a*N+b-a*(a-1)/2+1)
  return ind

# Main function
# Define key parameters
term=6
icm2ifs=2.99792458e-5 # cm/fs
deltat=10.0 # Timestep in fs
factor=icm2ifs*2.0*np.pi*deltat
w1=0.0 
w2=70.0 # Frequencies in cm-1
J=36.0 # Coupling in cm-1
T=150.0 # Temperature in Kelvin
kBT=T*0.6950389 # cm-1
beta=1.0/kBT
tau=50.0 # Correlation time in fs
g1=1.0/tau/factor*deltat
g2=g1 # Correlation times in cm-1
sigma=25.0 # Magnitude of fluctuations in cm-1
l1=sigma**2/kBT/2.0
l2=sigma**2/kBT/2.0 # Reorganization energy cm-1
time=5000 # Length of trajectory in fs

# Find size of HEOM Liouvillian
NN=int((term+1)*term/2)
# Define initial density matrix
rho=np.zeros((NN*4,1))
rho[1]=1.0

# Build Basis Matrices
L=np.array([[ 0.0, 0.0, -J, J],[0.0 , 0.0, J, -J],[-J, J, w1-w2, 0.0],[J,-J,0.0,w2-w1]])
phi1=np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,-1.0]])
phi2=-phi1
theta1=2*l1/beta*phi1-l1*g1*1j*np.array([[2.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
theta2=2*l2/beta*phi2-np.array([[0.0,0.0,0.0,0.0],[0.0,2.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])*l2*g2*1j
unit=np.eye(4,4)*1.0

# Build big matrix
# Diagonal part
LL=np.zeros((4*NN,4*NN))*1j
for k in range(term):
  for l in range(term):
    if (k+l+1)<term:
      ind=myindex(k,l,term)-1
      LL[4*ind:4*ind+4,4*ind:4*ind+4]=-1j*L-k*g1*unit-l*g2*unit
    # Terminator
    if (k+l+1)==term:
      ind=myindex(k,l,term)-1
      LL[4*ind:4*ind+4,4*ind:4*ind+4]=-1j*L-k*g1*unit-l*g2*unit

# Add Phi and Theta Couplings
for k in range(term+1):
  for l in range(term+1):
    if (k+l+2)<=term:
      ind=myindex(k,l,term)-1;
      if l<term-1:
        index=myindex(k,l+1,term)-1;
        LL[4*ind:4*ind+4,4*index:4*index+4]=1j*phi2
        LL[4*index:4*index+4,4*ind:4*ind+4]=(l+1)*1j*theta2
      if k<term-1:
        index=myindex(k+1,l,term)-1;
        LL[4*ind:4*ind+4,4*index:4*index+4]=1j*phi1
        LL[4*index:4*index+4,4*ind:4*ind+4]=(k+1)*1j*theta1

#print(LL)
u=expm(LL*factor)
rho0=rho
x=np.zeros(int(time/deltat))
r=np.zeros(int(time/deltat))
for t in range(int(time/deltat)):
  x[t]=t*deltat
  r[t]=float(np.real(np.matmul(np.transpose(rho),rho0)))
  rho0=np.matmul(u,rho0)

# Make plots
plt.plot(x,r)
plt.ylim((0,1)) 
plt.xlim((0,time))
plt.show()       
 
