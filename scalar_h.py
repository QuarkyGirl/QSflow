import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate,optimize
from matplotlib import rc,rcParams,colors
from matplotlib.ticker import MultipleLocator
from findiff import FinDiff

π = np.pi
γ_E = np.euler_gamma

class Scalar:
    ############################################################################
    # Parent class for computing the FRG flow of the quasi-stationary effective
    # action for scalar field theories.
    #
    # For specific models (e.g. φ^3 theory) use one of the pre-written
    # subclasses; these also serve as a template for writing your own subclass
    #
    # Typical usage without using subclass:
    # V = lambda x: [your potential]
    # scalar = Scalar(V)
    # sol = scalar.flow(φ,k)
    ############################################################################

    def __init__(self, V):
        ########################################################################
        # Class constructor
        # arguments:
        #   double (n,) V( (double (n,) φ ) - function for tree-level action
        ########################################################################
        self.V = V
        self.rtol,self.atol = (100*np.finfo(np.float).eps,1.e-15)   # relative and absolute tolerances

    def dV(self,φ,δ=1.e-10,n=1):
        ########################################################################
        # generic placeholder for the nth derivative of the potential
        # returns:
        #   double (n,) --- n'th derivative of the potential w.r.t. φ
        # arguments:
        #   double (n,) φ - field value
        #   double δ ------ change in field value for derivative
        #   int n --------- order of the derivative
        ########################################################################
        if n == 1: return (self.V(φ+δ) - self.V(φ-δ))/δ
        elif n >1: return (self.dV(φ+δ,δ=δ,n=n-1) - self.dV(φ-δ,δ=δ,n=n-1))/δ
        else:
            print('Error on dV: n not INT >= 1')

    def V_CW(self,φ,Λ,Π=0):
        ########################################################################
        # Coleman-Weinberg potential: zero-temperature component of effective
        # potential.
        # returns:
        #   double (n,) ---- potential
        # arguments:
        #   double (n,) φ -- field value
        #   double (n,) m2 - field-dependent tree-level mass squared: m^2(φ) = V''(φ)
        #   double Λ ------- FRG UV cutoff
        ########################################################################
        m2_φ = self.dV(φ,n=2)
        m2 = self.dV(0.0,n=2)
        Λ = np.maximum(Λ**2-m2_φ,0)**0.5 + 0j
        return 1/(64*π**2)*(Λ**2 *(m2_φ-m2) + Λ**4* np.log((Λ**2 + m2_φ)/(Λ**2 + m2)) - m2_φ**2* np.log((Λ**2 + m2_φ)/m2_φ) + m2**2* np.log((Λ**2 + m2)/m2))

    def V_th(self,φ,T,Λ,Π=0):
        ########################################################################
        # Perturbative thermal potential
        # returns:
        #   double (n,) ---- potential
        # inputs:
        #   double (n,) φ -- field value
        #   double (n,) m2 - field-dependent mass squared m^2(φ) = V''(φ)
        #   double T ------- temperature
        ########################################################################
        #J_B,J_B_err = integrate.quad_vec(
        #    lambda x: x**2 *np.log(1 - np.exp(-((1.0 + 0.j)*(x**2 + m2/T**2))**0.5)),
        #    0, np.inf,epsrel=self.rtol)
        m2 = self.dV(φ,n=2)
        Λ = np.maximum(Λ**2-m2,0)**0.5 + 0j
        m2 += Π
        # define x = p/Λ,
        J_B,J_B_err = integrate.quad_vec(
            lambda x: x**2 *np.log(1 - np.exp(-np.sqrt((1.0 + 0.j)*((x*Λ/T)**2 + m2/T**2)))),
            self.atol,1.,epsrel=self.rtol,epsabs=self.atol)
        return T*Λ**3/(2*π**2) * J_B

    def V_eff(self,φ,Λ,T=None):
        ########################################################################
        # One loop effective potential
        # returns:
        #   double (n,) --- potential
        # arguments:
        #   double (n,) φ - field value
        #   double Λ ------ FRG UV cutoff
        #   double T ------ temperature
        ########################################################################
        if T == None or T == 0:
            return self.V(φ) + self.V_CW(φ,Λ)
        else:
            Π = 1/24. * T**2 * self.dV(φ,n=4)
            return self.V(φ) + self.V_CW(φ,Λ,Π=Π) + self.V_th(φ,T,Λ,Π=Π)

    def flow_eqn(self,k,U,φ):
        ########################################################################
        # Exact flow equation for the quasi-stationary effective action
        # returns:
        #   double (n,) --- k-derivative of effective potential ∂_k U
        # arguments:
        #   double k ------ FRG scale k
        #   double (n,) U - scale-dependent effective action at the scale k
        #   double (n,) φ - field values at which U is evaluated
        #
        #   dict options {
        #       print_k=False - if True, print k
        #   }
        ########################################################################
        if options == None: options = {}
        options.setdefault('print_k',False)

        if options['print_k']: print(k)

        d2 = FinDiff(0,φ,2,acc=2)           # define second derivative operator using finite differences
        Upp = d2(U)
        Vpp = self.dV(φ,n=2)[:,None]
        V4 = self.dV(φ,n=4)[:,None]

        kt2 = k**2 - Vpp
        cond = (kt2 > 0)
        result = k*cond/V4 * (-(kt2 + Upp) + np.sqrt((kt2 + Upp)**2 + kt2**2*V4/(16*π**2)))
        return result

    def flow_eqn_unmodified(self,k,U,φ,Λ=None,options=None):
        ########################################################################
        # Exact flow equation for the effective action in the unmodified FRG.
        # Use this flow equation to compare the (non-convex) QSEA to the
        # (convex) unmodified FRG
        # returns:
        #   double (n,) ---- k-derivative of effective potential ∂_k U
        # arguments:
        #   double k ------- FRG scale k
        #   double (n,) U -- scale-dependent effective action at the scale k
        #   double (n,) φ -- field values at which U is evaluated
        #   double Λ ------- QSEA cutoff scale that applies the momentum cutoff p^2 + V''(φ) < Λ^2
        #                    if None, no condition is applied to the flow.
        #   dict options {
        #       pcond=True -------- if True, apply momentum cutoff p^2 + V''(φ) < Λ^2 to the flow
        #       print_k=False - if True, print k
        #   }
        ########################################################################
        if options == None: options = {}
        options.setdefault('pcond',True)
        options.setdefault('print_k',False)

        if options['print_k']: print(k)

        if Λ == None or options['pcond'] == False: cond = np.ones_like(φ)
        else:
            pmax = np.maximum(Λ**2-self.dV(φ,n=2),0)**0.5 + 0j
            cond = (k**2 < pmax**2)

        d2 = FinDiff(0,φ,2,acc=2)           # define second derivative operator using finite differences
        Upp = d2(U)
        result = k**5/(32*π**2) *1/(k**2 + Upp)*cond[:,None]
        return result

    def flow(self,φ,k,eqn='LPA_0T',method='RK45',verbose=True,options=None):
        ########################################################################
        # solve flow equations using grid method, in which φ is discretized and
        # k is treated continuously using Runge-Kutta
        # return solution object:
        # {
        #   double (m,n) U -- effective potential over (k,φ)
        #   double (m,) k --- RG scale
        #   double (n,) φ --- field value
        #   double T -------- temperature
        #   double Λ -------- cutoff scale
        #   string method --- method used in scipy.solve_ivp
        # }
        # arguments:
        #   double (n,) φ --- field value
        #   double (m,) k --- evaluation points for RG scale
        #   string eqn ------ flow equation to be used; supported equations are listed below
        #   string method --- method used in scipy.solve_ivp
        #   dict options ---- options to be passed to the solver
        #   bool verbose ---- if True, provide additional messages
        #
        # supported flow equations:
        #   'LPA_0T' -------- zero-temperature flow of the QSEA in the local potential
        #                     approximation with a Heaviside function regulator.
        #   'LPA_unmod_0T' -- zero-temperature flow of the scale-dependent effective
        #                     action in the unmodified FRG in the local potential
        #                     approximation with a Heaviside function regulator.
        ########################################################################
        if verbose: print('Starting flow: eqn =',eqn)

        # define equations that can be used
        eqns = {
            'LPA_0T':       lambda ki,Ui: self.flow_eqn(ki,Ui,φ,options=options),
            'LPA_unmod_0T': lambda ki,Ui: self.flow_eqn_unmodified(ki,Ui,φ,Λ=k[0],options=options),
            }

        # choose which flow equation to use; if not valid, throw error
        if eqn in eqns: flow_eqn = eqns[eqn]
        else:
            print('ERROR at Scalar.flow: eqn =',eqn,'not a valid option.',
            'Please choose one of:', list(eqns.keys()))

        # solve flow
        sol = integrate.solve_ivp(flow_eqn, t_span=(k[0],k[-1]),t_eval=k,
            y0=self.V(φ),rtol=self.rtol,atol=self.atol,vectorized=True,method=method)
        if verbose: print(sol.message)

        # return solution object
        res = type('obj', (object,),{'U':sol.y.T,'k':sol.t,'φ':φ,'T':0,'Λ':k[0],'eqn':eqn,'method':method})
        return res

class Phi3(Scalar):
    ############################################################################
    # Subclass for φ^3 theories with tree-level potential
    # V(φ) = 1/2 m^2 φ^2 + 1/3! α φ^3 + 1/4! λ φ^4
    #
    # Typical usage:
    # scalar = Phi3(m2,α,λ)
    # sol = scalar.flow(φ,k)
    ############################################################################

    def __init__(self,m2,α,λ):
        ########################################################################
        # Class constructor
        # arguments:
        #   double m2, α, λ - parameters of the tree-level potential
        ########################################################################
        self.m2 = m2
        self.α = α
        self.λ = λ
        V = lambda φ: 1/2.*self.m2*φ**2 + self.α/6.*φ**3 + self.λ/24.*φ**4 + 0j
        Scalar.__init__(self,V)

    def dV(self,φ,n=1):
        ########################################################################
        # Model-specific n'th derivative of the tree-level potential to
        # eliminate issues with roundoff error.
        # returns:
        #   double (n,) --- n'th derivative of the potential w.r.t. φ
        # arguments:
        #   double (n,) φ - field value
        #   int n --------- order of the derivative
        ########################################################################
        if n==1: return self.m2*φ + self.α/2.*φ*2 + self.λ/6.*φ**3 + 0j
        if n==2: return self.m2 + self.α*φ + self.λ/2.*φ**2 + 0j
        if n==3: return self.α + self.λ*φ + 0j
        if n==4: return self.λ*np.ones_like(φ)
        if n>4: return np.zeros_like(φ)
        if n==0: return self.V(φ)
        if n < 0:
            print('ERROR at Phi3.dV: n < 0')
            return -1
