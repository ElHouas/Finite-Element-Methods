import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pprint import pprint

#Mesh information
N_elements=5
N_nodes=N_elements+1
Lx = 1. #length of the domain
lamda = 1.
x_nodes = np.linspace(0,Lx,N_nodes)
y_nodes = np.linspace(0,0,N_nodes) #printing
dx = Lx/(N_elements)
gDl = 0.   #Dirichlet BC at 0
gDr = 0. #Dirichlet BC at 1

id_elem = np.arange(1,N_elements)
id_nodes = np.arange(0,N_nodes)

# number of nodes per element
N_loc = 2
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
fig.tight_layout(w_pad=6, h_pad=6)
ax1.plot(x_nodes,y_nodes,'-ko')
plt.axis('off')
for i in range(N_nodes):
    ax1.annotate(str(id_nodes[i]),(x_nodes[i],0.),xytext=(-1,10), textcoords='offset points')
ax2 = fig.add_subplot(122)
ax2.set_xlabel(r'$\xi$', fontsize=16)
ax2.set_title('Linear basis functions \nover the reference element', fontsize=16)
xi = np.linspace(-1,1,200)
# the two linear basis functions
ax2.plot(xi, 0.5*(1-xi),'k-', label='$\phi_0$')
ax2.plot(xi, 0.5*(1+xi),'b-', label='$\phi_1$')
ax2.legend(loc='best', fontsize=16)


## Define functions
#Elemental Matrices, define shape functions

def shape(N_loc, N_gi):
    assert (N_loc==2)
    phi = np.zeros((N_loc, N_gi))
    if(N_gi==2):
        phi[0,0] = 1.0
        phi[0,1] = 0.0
        phi[1,0] = 0.0
        phi[1,1] = 1.0
    elif(N_gi==3):
        phi[0,0] = 1.0
        phi[0,1] = 0.5
        phi[0,2] = 0.0
        phi[1,0] = 0.0
        phi[1,1] = 0.5
        phi[1,2] = 1.0
    else:
        raise Exception('N_gi value not implmemented.')
    return phi

def shape_derivatives(dx,N_loc,N_gi):
    assert(N_loc==2)
    phi_x = np.zeros((N_loc, N_gi))
    #the derivatives of our linear basis functions over the reference element
    if(N_gi==2):
        phi_x[0, 0] = -1. / 2.
        phi_x[0, 1] =  phi_x[0, 0]
        phi_x[1, 0] = -phi_x[0, 0]
        phi_x[1, 1] =  phi_x[1, 0]
    elif (N_gi == 3):
        phi_x[0, 0] = -1. / 2.
        phi_x[0, 1] =  phi_x[0, 0]
        phi_x[0, 2] =  phi_x[0, 0]
        phi_x[1, 0] = -phi_x[0, 0]
        phi_x[1, 1] =  phi_x[1, 0]
        phi_x[1, 2] =  phi_x[1, 0]
    else:
        raise Exception('N_gi value not implmemented.')

    #Jacobian contribution as seen/explained above due to the use of the chain rule
    phi_x = phi_x * (2. / dx)

    return phi_x

def quadrature(N_gi):
    weight = np.zeros(N_gi)
    if(N_gi==2): #Trapezoidal rule in 1D
        weight[0] = 0.5
        weight[1] = 0.5
    elif(N_gi==3): #Gauss Lobatto
        weight[0] = 1. / 3.
        weight[1] = 4. / 3.
        weight[2] = 1. / 3.
    else:
        raise Exception('N_gi value not implmemented.')
    return weight

def connectivity(N_loc, N_elements_CG):

    connectivity_matrix = np.zeros((N_loc, N_elements_CG), dtype=int)
    if(N_loc==2):
        for element in range (N_elements_CG):
            connectivity_matrix[0,element] = element
            connectivity_matrix[1,element] = element + 1
    else:
            raise Exception('Only linear element (Nloc=2) implemented.')
    return connectivity_matrix

#Compute the local Mass and Laplace matrices
N_gi = 3
N_loc = 2
weight = quadrature(N_gi)
phi = shape(N_loc,N_gi)
phi_x = shape_derivatives(dx,N_loc,N_gi)

MElem = np.zeros((N_loc,N_loc))
LElem = np.zeros((N_loc,N_loc))
for i_loc in range(N_loc):
    for j_loc in range(N_loc):
        for gi in range (N_gi):
            MElem[i_loc, j_loc] += weight[gi] * phi[i_loc,gi] * phi[j_loc,gi] * dx/2. #dx/2 here is the Jacobian
            LElem[i_loc, j_loc] += weight[gi] * phi_x[i_loc,gi] * phi_x[j_loc,gi] * dx/2.

pprint(MElem)
pprint(np.array([[dx/3.,dx/6.],[dx/6.,dx/3.]]))
pprint(LElem)
pprint(np.array([[1./dx,-1./dx],[-1./dx,1./dx]]))

#Assemble the global matrices M and L

def assembly(LocalMatrix, connectivity_matrix, N_elements):

    GlobalMatrix= np.zeros((N_elements+1, N_elements+1))
    for element in range(N_elements):
        for i_loc in range(N_loc):
            i_global = connectivity_matrix[i_loc, element]
            for j_loc in range(N_loc):
                j_global = connectivity_matrix[j_loc, element]
                GlobalMatrix[i_global, j_global] = GlobalMatrix[i_global, j_global] + LocalMatrix[i_loc, j_loc]

    return GlobalMatrix

connectivity_matrix = connectivity(N_loc, N_elements)
MG = assembly(MElem,connectivity_matrix, N_elements)
LG = assembly(LElem,connectivity_matrix,N_elements)
Stiff = LG+lamda*MG
pprint(Stiff)

#Check properties of global Mass and Laplacian matrix
A=np.ones(N_nodes)
A_t=np.transpose(A)
A_M=np.matmul(A_t,MG)
print(np.matmul(A_M,A))
A_L=np.matmul(A_t,LG)
print(np.matmul(A_L,A_t))

wM,vM=np.linalg.eig(MG)
wL,vL=np.linalg.eig(LG)

print(wM)
print(wL)

#Define the function or source term of the RHS

def f(x):
    return -2.-x*(1-x)

# Define the RHS

Fsource = np.zeros(N_nodes)

# Loop over the elements

for element in range (N_elements):
    # Loop over local nodes(i: test functions)
    for i_local in range(N_loc):
        i_global = connectivity_matrix[i_local, element]
        for gi in range(N_gi):
            Fsource[i_global] += weight[gi]*phi[i_loc,gi]*f(x_nodes[i_global]+gi*dx/2.) * dx/2.

approx_CG = sl.solve(MG,Fsource)

fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(111)
ax1.set_xlabel('$x$', fontsize=16)
ax1.set_ylabel('$f(x)$', fontsize=16)
ax1.set_title('Function approximation', fontsize=16)
# plot the exact function on a fine mesh
x_fine = np.linspace(0,1,100)
ax1.plot(x_fine, f(x_fine),'k-', label = 'exact')
ax1.plot(x_nodes, approx_CG,'ro-', label = 'p/w linear cts projection')
ax1.legend(loc='best', fontsize=16)
#plt.show()

def apply_bc(A, b, lbc, rbc, bc_option=0):

    if(bc_option==0): #Homogeneous Neumann
        return
    elif(bc_option==1):
        big_spring = 1.0e10
        A[0,0]   = big_spring
        b[0]     = big_spring * lbc
        A[-1,-1] = big_spring
        b[-1]    = big_spring * rbc
    else:
        raise Exception('bc option not implemented')

RHS = -Fsource
apply_bc(Stiff,RHS,gDl,gDr,1) #Dirichlet BC
uD=np.zeros(N_nodes)

uH = sl.solve(Stiff,RHS)
np.allclose(np.dot(Stiff,uH), Fsource)
print(uH)

u=uH+uD

#Analytical solution
xexact = np.linspace(0,1,1000)
uexact = xexact*(1.-xexact)
fig = plt.figure(figsize=(12,5))
#fig.tight_layout(w_pad=6, h_pad=6)
ax3 = fig.add_subplot(121)
ax3.plot(x_nodes,u,'-ob')
ax3.plot(xexact,uexact,'r')
ax3.set_xlabel('x', fontsize= 20)
ax3.set_ylabel('$u$ , $u_{exact}$', fontsize= 20)

error = np.abs(u[1:-2]-np.interp(x_nodes[1:-2],xexact,uexact))/np.interp(x_nodes[1:-2],xexact,uexact)
L2norm = np.linalg.norm(error,2)
ax4 = fig.add_subplot(122)
ax4.plot(x_nodes[1:-2],error,'-ob')
ax4.set_xlabel('x', fontsize= 20)
ax4.set_ylabel('error %', fontsize= 20)
ax4.set_xlim(0, 1)

print (L2norm)
plt.show()


# L2 norm
Nruns=100
L2norm = np.zeros(Nruns)
Elems = np.zeros(Nruns)
for i in range(Nruns):
    #The complete code
    N_elements = 1*i+5
    Elems[i] = N_elements
    N_nodes = N_elements+1
    Lx = 1.  # length of the domain
    lamda = 1.
    x_nodes = np.linspace(0, Lx, N_nodes)
    dx = Lx / (N_elements)
    gDl = 0.  # Dirichlet BC at 0
    gDr = 0.  # Dirichlet BC at 1

    # Compute the local Mass and Laplace matrices
    N_gi = 3
    N_loc = 2
    weight = quadrature(N_gi)
    phi = shape(N_loc, N_gi)
    phi_x = shape_derivatives(dx, N_loc, N_gi)

    #Compute local mass and laplace matrices, most of the variables have been
    #computed before already
    MElem = np.zeros((N_loc, N_loc))
    LElem = np.zeros((N_loc, N_loc))
    for i_loc in range(N_loc):
        for j_loc in range(N_loc):
            for gi in range(N_gi):
                MElem[i_loc, j_loc] += weight[gi] * phi[i_loc, gi] * phi[j_loc, gi] * dx / 2.  # dx/2 here is the Jacobian
                LElem[i_loc, j_loc] += weight[gi] * phi_x[i_loc, gi] * phi_x[j_loc, gi] * dx / 2.

    connectivity_matrix = connectivity(N_loc, N_elements)
    MG = assembly(MElem, connectivity_matrix, N_elements)
    LG = assembly(LElem, connectivity_matrix, N_elements)
    Stiff = LG + lamda * MG

    # Define the function or source term of the RHS
    def f(x):
        return 2.+x*(1 - x)

    #Define the RHS
    RHS = np.zeros(N_nodes)

    # Loop over the elements
    for element in range(N_elements):
        # Loop over local nodes(i: test functions)
        for i_local in range(N_loc):
            i_global = connectivity_matrix[i_local, element]
            for gi in range(N_gi):
                RHS[i_global] += weight[gi] * phi[i_loc, gi] * f(x_nodes[i_global] + gi * dx / 2.) * dx / 2.

    #Apply BCs
    apply_bc(Stiff, RHS, gDl, gDr, 1)  # Dirichlet BC
    Fsource = RHS

    #Solve the system of equations
    uH = sl.solve(Stiff, Fsource)
    np.allclose(np.dot(Stiff, uH), Fsource)

    u = uH
    xexact = np.linspace(0, 1, 1000)
    uexact = xexact*(1.-xexact)
    error = np.abs(u-np.interp(x_nodes, xexact, uexact))
    L2norm[i] = np.linalg.norm(error)


TheorErrorNormHalf = 0.01*Elems**(-0.5)
TheorErrorNormLinear = 0.01*Elems**(-1.)
fig = plt.figure(figsize=(12, 5 ))
ax5 = fig.add_subplot(121)
ax5.loglog(Elems, L2norm,'-ob',label='FEM')
ax5.loglog(Elems, TheorErrorNormHalf, '-r', label='Theoretical')
ax5.loglog(Elems, TheorErrorNormLinear, '-k', label='Theoretical')
ax5.text(1e2, 0.001, '-1/2', fontsize= 20)
ax5.text(50, 0.0003, '-1', fontsize= 20)
ax5.set_ylim(0.000001, 0.01)
ax5.set_xlabel('Number of elements', fontsize= 20)
ax5.set_ylabel('$L_2$', fontsize= 20)
ax5.legend(loc= 'upper right')
ax5.grid('on')


# Energy norm because L2 norm is not suitble in quadratic cases
Nruns=100
Energynorm = np.zeros(Nruns)
Elems = np.zeros(Nruns)
for i in range(Nruns):
    #The complete code
    N_elements = 10*(i+1)
    Elems[i] = N_elements
    N_nodes = N_elements+1
    Lx = 1.  # length of the domain
    lamda = 1.
    x_nodes = np.linspace(0, Lx, N_nodes)
    dx = Lx / (N_elements)
    gDl = 0.  # Dirichlet BC at 0
    gDr = 0.  # Dirichlet BC at 1

    # Compute the local Mass and Laplace matrices
    N_gi = 3
    N_loc = 2
    weight = quadrature(N_gi)
    phi = shape(N_loc, N_gi)
    phi_x = shape_derivatives(dx, N_loc, N_gi)

    #Compute local mass and laplace matrices, most of the variables have been
    #computed before already
    MElem = np.zeros((N_loc, N_loc))
    LElem = np.zeros((N_loc, N_loc))
    for i_loc in range(N_loc):
        for j_loc in range(N_loc):
            for gi in range(N_gi):
                MElem[i_loc, j_loc] += weight[gi] * phi[i_loc, gi] * phi[j_loc, gi] * dx / 2.  # dx/2 here is the Jacobian
                LElem[i_loc, j_loc] += weight[gi] * phi_x[i_loc, gi] * phi_x[j_loc, gi] * dx / 2.

    connectivity_matrix = connectivity(N_loc, N_elements)
    MG = assembly(MElem, connectivity_matrix, N_elements)
    LG = assembly(LElem, connectivity_matrix, N_elements)
    Stiff = LG + lamda * MG

    # Define the function or source term of the RHS
    def f(x):
        return 2.+x*(1 - x)

    #Define the RHS
    RHS = np.zeros(N_nodes)

    # Loop over the elements
    for element in range(N_elements):
        # Loop over local nodes(i: test functions)
        for i_local in range(N_loc):
            i_global = connectivity_matrix[i_local, element]
            for gi in range(N_gi):
                RHS[i_global] += weight[gi] * phi[i_loc, gi] * f(x_nodes[i_global] + gi * dx / 2.) * dx / 2.

    #Apply BCs
    apply_bc(Stiff, RHS, gDl, gDr, 1)  # Dirichlet BC
    Fsource = RHS

    #Solve the system of equations
    uH = sl.solve(Stiff, Fsource)
    np.allclose(np.dot(Stiff, uH), Fsource)

    u = uH
    xexact = np.linspace(0, 1, 1000)
    uexact = xexact*(1.-xexact)
    error = u-np.interp(x_nodes, xexact, uexact)
    Energynorm[i] = np.sqrt(np.sum(np.gradient(error, dx)**2.+lamda*error**2.)*dx)


TheorErrorNormHalf = 0.075*Elems**(-0.5)
TheorErrorNormLinear = 0.1*Elems**(-1.)
fig = plt.figure(figsize=(12, 5))
ax6 = fig.add_subplot(121)
ax6.loglog(Elems, Energynorm,'-ob',label='FEM')
ax6.loglog(Elems, TheorErrorNormHalf, '-r', label='Theoretical')
ax6.loglog(Elems, TheorErrorNormLinear, '-k', label='Theoretical')
ax6.text(1e2, 0.01, '-1/2', fontsize= 20)
ax6.text(50, 0.003, '-1', fontsize= 20)
ax6.set_ylim(0.00001, 0.1)
ax6.set_xlabel('Number of elements', fontsize= 20)
ax6.set_ylabel('$L_2$', fontsize= 20)
ax6.legend(loc= 'upper right')
ax6.grid('on')


plt.show()