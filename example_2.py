#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from findiff import FinDiff
from scalar_h import Phi3
from plot_h import *
π = np.pi

################################################################################
# This example includes the calculation of the effective potential in the      #
# unmodified FRG. This computation is significantly more computationally       #
# intensive due to the stiffness of the unmodified flow equation.              #
################################################################################


################################################################################
# Perturbative example:
#name = 'weak'                                     # name of output PDF
#m2,α,λ = (0.014, -0.09, 0.185)                    # Model parameters
#Λ = 2.0                                           # QSEA cutoff scale
#kmin_unmod = 0.015
################################################################################
# Non-perturbative example:
name = 'strong'
m2,α,λ = (1.4, -9., 18.5)                           # Model parameters
Λ = 2.5                                             # QSEA cutoff scale
kmin_unmod = 0.05

scalar = Phi3(m2,α,λ)                               # Next, create a class instance

k = np.array([Λ,0.])                                # Define k at which to evaluate the flow
                                                    # NOTE: spacing of k does NOT affect precision

φ = np.linspace(-0.25,1.25,num=100)                 # Define field values φ over which to discretize
                                                    # Spacing of φ DOES affect precision

sol = scalar.flow(φ,k)                              # Solve the flow over the defined φ,k
U = np.real(sol.U)                                  # Extract the QS effective potential from solution object


d = FinDiff(1,φ,1,acc=2)                            # Optionally, we can normalize
arg = np.argmin(d(U)[:,φ<0.3]**2,axis=1)            # the QSEA to the false vacuum,
U -= U[np.arange(arg.size),arg][:,None]             # U_k'(φ) = 0.


v_eff = np.real(scalar.V_eff(φ,Λ))                  # Calculate the one-loop effective potential
d = FinDiff(0,φ,1,acc=2)                            # and normalize to FV
arg_v = np.argmin(d(v_eff)[φ < 0.2]**2)             #
v_eff -= v_eff[arg_v]                               #

scalar.rtol,scalar.atol = (1.e-11,1.e-11)           # Calculate the effective potential for the
k_unmod = np.array([Λ,kmin_unmod])                        # unmodified FRG. We strongly recommend using
sol_unmod = scalar.flow(φ,k_unmod,                  # the BDF solver and less strict tolerances
    eqn='LPA_unmod_0T',method='BDF')                # due to the stiffness of the unmodified flow equation.
U_unmod = np.real(sol_unmod.U)                      #
U_unmod -= U_unmod[np.arange(arg.size),arg][:,None] # Normalize to FV

v_tree = np.real(scalar.V(φ))                       # Lasly, get the tree-level potential


################################################################################
# Now that we have calculated the various potentials, all that's left to do is
# plot them. An example of a script to plot the potentials is below:

# set up figure
fig,ax = plt.subplots(figsize=(3.2,2.6))

cs = ['#000000','#FF1F1F','#8F3985','#2274A5']
qsea, = ax.plot(φ,U[-1],ls='-',c=cs[0],zorder=2,lw=0.7)
one_loop, = ax.plot(φ,v_eff,c=cs[1],ls='--',zorder=4,lw=0.7,dashes=[4,4])
frg, = ax.plot(φ,U_unmod[-1],ls=':',c=cs[2],zorder=3,lw=0.9)
tree, = ax.plot(φ,v_tree,ls='-',c=cs[3],zorder=1,lw=0.7)


# legends
leg1 = plt.legend(
(qsea,one_loop,frg,tree),
(r'QSEA+FRG: $U_{k = 0}(\phi)$',
r'One-loop: $V_\mathrm{eff}(\phi)$',
r'Pure FRG: $U_{k = 0}(\phi)$',
r'Tree-level: $V(\phi)$'),
bbox_to_anchor=(0.4,0.,0.6,1.),loc='upper left',
frameon=False,handler_map={str: LegendHandler()},fontsize=6)
plt.gca().add_artist(leg1)

leg2 = plt.legend(
(r'$\Lambda_\mathrm{QSEA} = %.1f'%Λ+'$',
r"$\Lambda_\mathrm{pert.}^2 = \Lambda_\mathrm{QSEA}^2 - V''(\phi)$"),
('',''),
bbox_to_anchor=(0.4,0.,0.6,1.),loc='lower left',
frameon=False,handler_map={str: LegendHandler()},fontsize=6)

# add title
ax.set_title('Quasi-stationary effective potential')

# configure axes
ax.set_xlabel(r'$\phi$',fontsize=10,labelpad=2)
ax.set_ylabel(r'$U(\phi)$',fontsize=10,labelpad=0)

yscale = m2/1.4
ax.set_xlim(φ[0],φ[-1])
ax.set_ylim(-0.05*yscale , 0.1*yscale)

ax.axhline(0.,c='black',zorder=0,alpha=0.5)
ax.axvline(0.,c='black',zorder=0,alpha=0.5)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.05*yscale))
ax.yaxis.set_minor_locator(MultipleLocator(0.01*yscale))
ax.tick_params(which='both',reset=True,direction='in')
plt.tight_layout()

# save and show
plt.savefig('figs/example_2_'+name+'.pdf',bbox_inches = 'tight',pad_inches=0.01)
plt.show()
