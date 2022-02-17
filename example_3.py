#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from findiff import FinDiff
from scalar import Phi3
from plot_h import *
π = np.pi

################################################################################
# This is an example of evaluating the QSEA flow equations for a simple φ^3    #
# potential at zero temperature. We include parameters for both the            #
# perturbative and non-perturbative regimes; the non-perturbative parameters   #
# are the default                                                              #
################################################################################


################################################################################
# Perturbative example:
#name = 'Weak'
#m2,α,λ = (0.014, -0.-9, .185)                    # Model parameters
#Λ = 2.0                                           # QSEA cutoff scale
#scale = 2
#
# Non-perturbative example:
name = 'Strong'
m2,α,λ = (3*.14, -3*.9, 3*1.85)                           # Model parameters
scale = 1.5
Λ = 2.0                                             # QSEA cutoff scale

T = 0.1
scalar = Phi3(m2,α,λ)                               # Next, create a class instance

k = np.array([Λ,0.])                                # Define k over which to evaluate the flow
                                                    # NOTE: spacing of k does NOT affect precision

φ = np.linspace(-0.25,1.25,num=100)                 # Define field values φ over which to discretize
                                                    # Spacing of φ DOES affect precision

sol = scalar.flow(φ,k,eqn='QSEA_T_4D',T=T)          # Solve the flow over the defined φ,k
U = np.real(sol.U)                                  # Extract the QS effective potential from solution object


d = FinDiff(1,φ,1,acc=2)                            # Optionally, we can normalize
arg = np.argmin(d(U)[:,φ<0.3]**2,axis=1)            # the QSEA to the false vacuum,
U -= U[np.arange(arg.size),arg][:,None]             # U_k'(φ) = 0.

sol2 = scalar.flow(φ,k,eqn='QSEA_0T')          # Solve the flow over the defined φ,k
U0 = np.real(sol2.U)                                  # Extract the QS effective potential from solution object


arg = np.argmin(d(U0)[:,φ<0.3]**2,axis=1)            # the QSEA to the false vacuum,
U0 -= U0[np.arange(arg.size),arg][:,None]             # U_k'(φ) = 0.




v_eff = np.real(scalar.V_eff(φ,Λ,T=T))                  # Calculate the one-loop effective potential
d = FinDiff(0,φ,1,acc=2)                            # and normalize to FV
arg_v = np.argmin(d(v_eff)[φ < 0.2]**2)             #
v_eff -= v_eff[arg_v]                               #

v_tree = np.real(scalar.V(φ))                       # Lasly, get the tree-level potential

################################################################################
# Now that we have calculated the various potentials, all that's left to do is
# plot them. An example of a script to plot the potentials is below:

# set up figure
fig,ax = plt.subplots(figsize=(3.2,2.6))

cs = ['#000000','#FF1F1F','#8F3985','#2274A5']

# plot potentials
qsea_0T, = ax.plot(φ,U0[-1]*10**scale,ls=':',c=cs[2],zorder=5,lw=1.0)
qsea, = ax.plot(φ,U[-1]*10**scale,ls='-',c=cs[0],zorder=2,lw=0.7)
one_loop, = ax.plot(φ,v_eff*10**scale,c=cs[1],ls='--',zorder=4,lw=0.7,dashes=[4,4])
tree, = ax.plot(φ,v_tree*10**scale,ls='-',c=cs[3],zorder=1,lw=0.7)

# add legends
leg1 = plt.legend(
(name +' coupling regime',qsea,qsea_0T,one_loop,tree),
('',r'QSEA+FRG: $U_{k = 0}(\phi,T)$',r'QSEA+FRG: $U_{k = 0}(\phi,T=0)$',r'One-loop: $V_\mathrm{eff}(\phi)$',r'Tree-level: $V(\phi)$'),
bbox_to_anchor=(0.3,0.,0.6,1.),loc='upper left',
frameon=False,handler_map={str: LegendHandler()},fontsize=6)
plt.gca().add_artist(leg1)

leg2 = plt.legend(
(r'$\Lambda_\mathrm{QSEA} = %.1f'%Λ+', T = %.3f'%T+'$',
r"$\Lambda_\mathrm{pert.}^2 = \Lambda_\mathrm{QSEA}^2 - V''(\phi)$"),
('',''),
bbox_to_anchor=(0.3,0.,0.6,1.),loc='lower left',
frameon=False,handler_map={str: LegendHandler()},fontsize=6)

# add title
ax.set_title('Quasi-stationary effective potential')

# configure axes
ax.set_xlabel(r'$\phi$',fontsize=10,labelpad=2)
ax.set_ylabel(r'$U(\phi)\times10^{'+str(scale)+'}$',fontsize=10,labelpad=0)

ax.set_xlim(φ[0],φ[-1])
ax.set_ylim(-0.5 , 1.0)

ax.axhline(0.,c='black',zorder=0,alpha=0.5)
ax.axvline(0.,c='black',zorder=0,alpha=0.5)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.tick_params(which='both',reset=True,direction='in')
plt.tight_layout()

# save and show
plt.savefig('figs/example_3_'+name+'_T='+str(T)+'.pdf',bbox_inches = 'tight',pad_inches=0.01)
plt.savefig('figs/example_3_'+name+'_T='+str(T)+'.png',dpi=600,bbox_inches = 'tight',pad_inches=0.01)

plt.show()
