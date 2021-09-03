import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate,optimize
from matplotlib import rc,rcParams,colors
from matplotlib.ticker import MultipleLocator
from findiff import FinDiff

π = np.pi
γ_E = np.euler_gamma

class Scalar:
    """Parent class for computing the FRG flow of the quasi-stationary effective
    action for scalar field theories.

    For specific models (e.g. φ^3 theory) use one of the pre-written
    subclasses; these also serve as a template for writing your own subclass.

    Parameters
    ----------
    V : callable
        Tree-level potential. Calling signature ``V(φ) -> ndarray or float``.
        Here `φ` is an array_like of field values. `V` must return an ndarray
        or scalar with the same shape as `φ`.

    Attributes
    ----------
    rtol : float, default=100*`eps`
        Relative tolerance passed to ODE solver and integrator.
    atol : float, default=1e-15
        Absolute tolerance passed to ODE solver and integrator.

    """

    def __init__(self, V):
        """Class constructor.
        """
        # Create a wrapped private instance of the tree-level potential
        self._wrapped_V = V
        self.rtol,self.atol = (100*np.finfo(np.float).eps,1.e-15)

    def V(self,φ):
        """Tree-level potential

        Parameters
        ----------
        φ : ndarray or float
            Field value.

        Returns
        -------
        ndarray or float
            Tree-level potential evaluated at each `φ`.
        """
        return self._wrapped_V(φ)

    def dV(self, φ, n=1,  δ=1.e-10):
        """Generic placeholder for the nth derivative of the potential w.r.t. `φ`

        Parameters
        ----------
        φ : ndarray or float
            Field value.
        n : int, default=1
            Order of the derivative.
        δ : ndarray or float, default=1e-10
            Change in field value for derivative. If ndarray, must have same
            shape as `φ`.

        Returns
        -------
        ndarray or float
            nth derivative of the potential w.r.t. `φ`.
        """

        if n == 1: return (self.V(φ+δ) - self.V(φ-δ))/δ
        elif n >1: return (self.dV(φ+δ, δ=δ, n=n-1) - self.dV(φ-δ, δ=δ, n=n-1))/δ
        else:
            print('Error on dV: n not INT >= 1')

    def V_CW(self, φ, Λ, Π=0., QSEA=True):
        """Coleman-Weinberg potential for cutoff renormalization at one-loop.

        Parameters
        ----------
        φ : ndarray or float
            Field values at which to evaluate potential.
        Λ : float
            UV cutoff.
        Π : ndarray or float, optional
            The thermal dressing of the squared mass. Default is 0.
        QSEA : bool, default=True
            If `True`, treats `Λ` as the QSEA fluctuation cutoff, so that the
            momentum cutoff `p^2 < Λ^2 - m^2(φ)` is field-dependent. If `False`,
            treats `Λ` directly as the momentum cutoff `p^2 < Λ^2`.

        Returns
        -------
        ndarray or float
            CW potential with same shape as `φ`.
        """
        # TODO : double check dressing
        m2_φ = self.dV(φ, n=2) + Π
        m2 = self.dV(0.0, n=2) + Π
        if QSEA: Λ = np.maximum(Λ**2-m2_φ, 0)**0.5 + 0j
        return 1/(64*π**2)*(Λ**2 *(m2_φ-m2) + Λ**4* np.log((Λ**2 + m2_φ)/(Λ**2 + m2)) - m2_φ**2* np.log((Λ**2 + m2_φ)/m2_φ) + m2**2* np.log((Λ**2 + m2)/m2))

    def V_th(self, φ, Λ, T, Π=0., QSEA=True):
        """Perturbative thermal effective potential at one-loop.

        Parameters
        ----------
        φ : ndarray or float
            Field values at which to evaluate potential.
        Λ : float
            UV cutoff.
        T : float
            Temperature.
        Π : ndarray or float, optional
            The thermal dressing of the squared mass. Default is 0.
        QSEA : bool, default=True
            If `True`, treats `Λ` as the QSEA fluctuation cutoff, so that the
            momentum cutoff `p^2 < Λ^2 - m^2(φ)` is field-dependent. If `False`,
            treats `Λ` directly as the momentum cutoff `p^2 < Λ^2`.

        Returns
        -------
        ndarray or float
            Thermal potential with same shape as `φ`.
        """
        # TODO : check thermal potential for field dependent cutoff
        m2 = self.dV(φ, n=2) + Π
        if QSEA: Λ = np.maximum(Λ**2-m2, 0)**0.5 + 0j
        # define x = p/Λ,
        J_B,J_B_err = integrate.quad_vec(
            lambda x: x**2 *np.log(1 - np.exp(-np.sqrt((1.0 + 0.j)*((x*Λ/T)**2 + m2/T**2)))),
            self.atol, 1., epsrel=self.rtol, epsabs=self.atol)
        return T*Λ**3/(2*π**2) * J_B

    def V_eff(self, φ, Λ, T=None, QSEA=True):
        """One loop effective potential.

        Parameters
        ----------
        φ : ndarray or float
            Field values at which to evaluate potential.
        Λ : float
            UV cutoff.
        T : float
            Temperature.
        QSEA : bool, default=True
            If `True`, treats `Λ` as the QSEA fluctuation cutoff, so that the
            momentum cutoff `p^2 < Λ^2 - m^2(φ)` is field-dependent. If `False`,
            treats `Λ` directly as the momentum cutoff `p^2 < Λ^2`.


        Returns
        -------
        ndarray or float
            Effective potential with same shape as `φ`.
        """
        if T == None or T == 0:
            return self.V(φ) + self.V_CW(φ,Λ)
        else:
            Π = 1/24. * T**2 * self.dV(φ,n=4)
            return self.V(φ) + self.V_CW(φ,Λ,Π=Π) + self.V_th(φ,Λ,T,Π=Π)

    def flow_eqn(self, k, U, φ, **options):
        """Exact flow equation for the quasi-stationary effective action (QSEA)
        in the local potential approximation (LPA) at zero-temperature

        Parameters
        ----------
        k : float
            FRG scale
        U : ndarray shape (n,) or (n,m)
            scale-dependent effective action at the scale `k`. For explicit
            methods such as `RK45`, `U` is 1D; for implicit methods such as
            `BDF` it must be allowed to have shape (n,m), where the 0th axis
            corresponds to `φ`.
        φ : ndarray shape (n,)
            field values at which `U` is evaluated.
        **options : dict or None, optional
            Additional arguments passed to flow equation.

        Options
        -------
        print_k : bool
            If `True`, print `k` at each `solve_ivp` step

        Returns
        -------
        ndarray shape (n,) or (n,m)
            `k`-derivative of the effective potential `U`
        """
        options.setdefault('print_k', False)
        if options['print_k']: print(k)

        d2 = FinDiff(0, φ, 2, acc=2)           # define second derivative operator using finite differences
        Upp = d2(U)
        Vpp = self.dV(φ, n=2)[:,None]
        V4 = self.dV(φ, n=4)[:,None]

        kt2 = k**2 - Vpp
        cond = (kt2 > 0)
        result = k*cond/V4 * (-(kt2 + Upp) + np.sqrt((kt2 + Upp)**2 + kt2**2*V4/(16*π**2)))
        return result

    def flow_eqn_unmodified(self, k, U, φ, Λ=None, **options):
        """Exact flow equation for the effective action in the unmodified FRG
        in the local potential approximation (LPA) at zero-temperature.
        Use this flow equation to compare the (non-convex) QSEA to the
        (convex) unmodified FRG.

        Parameters
        ----------
        k : float
            FRG scale
        U : ndarray shape (n,) or (n,m)
            scale-dependent effective action at the scale `k`. For explicit
            methods such as `RK45`, `U` is 1D; for implicit methods such as
            `BDF` it must be allowed to have shape (n,m), where the 0th axis
            corresponds to `φ`.
        φ : ndarray shape (n,)
            field values at which `U` is evaluated.
        Λ : float or None, optional
            QSEA fluctuation-scale cutoff to match the perturbative and QSEA
            cutoff scheme. If `None`, no condition is applied to the flow.
            Default is `None`; however, `scalar.flow` passes `Λ=k[0]` by default.
            Cutoff is only applied if `options['pcond'] == True`.
        **options : dict or None, optional
            Additional arguments passed to flow equation.

        Options
        -------
        print_k : bool, default=False
            If `True`, print `k` at each `solve_ivp` step
        pcond : bool, default=True
            if `True` and `Lambda != None`, apply QSEA fluctuation cutoff
            `p^2 < Λ^2 - m^2(φ)` to the flow. Otherwise, a simple momentum
            cutoff `p^2 < Λ^2` is used.

        Returns
        -------
        ndarray shape (n,) or (n,m)
            `k`-derivative of the effective potential `U`
        """
        options.setdefault('pcond',True)
        options.setdefault('print_k',False)
        if options['print_k']: print(k)

        if Λ == None or options['pcond'] == False: cond = np.ones_like(φ)
        else:
            pmax = np.maximum(Λ**2-self.dV(φ,n=2),0)**0.5 + 0j
            cond = (k**2 < pmax**2)

        d2 = FinDiff(0, φ, 2, acc=2)           # define second derivative operator using finite differences
        Upp = d2(U)
        result = k**5/(32*π**2) *1/(k**2 + Upp)*cond[:,None]
        return result

    def flow(self, φ, k, eqn='LPA_0T', method='RK45', verbose=True, options=None):
        """Solve flow equations using grid method, in which φ is discretized and
        k is treated continuously using explicit or implicit Runge-Kutta methods.
        This function wraps scipy's ODE solver, scipy.integrate.solve_ivp.

        Parameters
        ----------
        φ : ndarray shape (n,)
            Field value at which to discretize the potential. The spacing of `φ`
            does affect the precision of the flow.
        k : ndarray shape (m,)
            FRG scale at which to evaluate the flow. Since the `k`-evolution is
            treated continuously, the values of `k` do not affect the precision
            of the flow, merely the points at which to evaluate the potential.
        eqn : string, default='LPA_0T'
            Name of the flow equation to be used in the flow. Currently supported
            flow equations are:

                - 'LPA_0T': zero-temperature flow of the QSEA in the local
                  potential approximation with a Heaviside function regulator.
                - 'LPA_unmod_0T': zero-temperature flow of the scale-dependent
                  effective action in the unmodified FRG in the local potential
                  approximation with a Heaviside function regulator.

        method : string, default='RK45'
            Solution method to be passed to scipy.integrate.solve_ivp. For more
            details, see the `scipy documentation
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_.
        verbose : bool, default=True
            if `True`, print additional messages on solution status.
        **options : dict or None
            Additional arguments passed to flow equation. For more details, see
            the documentation for each individual flow equation.

        Returns
        -------
        Solution object with following fields:
        U : ndarray shape (m,n)
            effective potential over `(k,φ)`.
        k : ndarray shape (m,)
            FRG scale at which the potential is evaluated. Equivalent to the
            input parameter `k` if the solver successfully reaches the end of
            the solution interval and no termination events occur.
        φ : ndarray shape (n,)
            The input parameter `φ`: field value at which the potential is
            evaluated.
        eqn : string
            The input parameter `eqn`: the flow equation used in the flow
        method : string
            The input parameter `method`: the ODE solution method used in
            `scipy.integrate.solve_ivp`
        success : bool
            Success flag from `scipy.integrate.solve_ivp`. `True` if the solver
            reached the interval end or a termination event occurred.
        message : string
            Solution message from `scipy.integrate.solve_ivp`.

        Examples
        --------
        Typical usage

        >>> V = lambda x: ...
        ... scalar = Scalar(V)
        ... sol = scalar.flow(φ,k)
        """
        if verbose: print('Starting flow: eqn =',eqn)

        # define equations that can be used
        eqns = {
            'LPA_0T':       lambda ki,Ui: self.flow_eqn(ki,Ui,φ,**options),
            'LPA_unmod_0T': lambda ki,Ui: self.flow_eqn_unmodified(ki,Ui,φ,Λ=k[0],**options),
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
        res = type('obj', (object,),{'U':sol.y.T,'k':sol.t,'φ':φ,'success':sol.success,'message':sol.message,'eqn':eqn,'method':method})
        return res

class Phi3(Scalar):
    """Subclass of Scalar for φ^3 theories with tree-level potential

    `V(φ) = 1/2 m^2 φ^2 + 1/3! α φ^3 + 1/4! λ φ^4`

    Parameters
    ----------
    m2 : float
        Tree-level mass-squared
    α : float
        Tree-level cubic coupling
    λ : float
        Tree-level quartic coupling

    Attributes
    ----------
    m2 : float
        Tree-level mass-squared
    α : float
        Tree-level cubic coupling
    λ : float
        Tree-level quartic coupling
    """

    def __init__(self, m2, α, λ):
        """Class constructor
        """
        self.m2 = m2
        self.α = α
        self.λ = λ

        Scalar.__init__(None)

    def V(self, φ):
        """Tree-level potential specific to φ^3 theories.

        Parameters
        ----------
        φ : ndarray or float
            Field value.

        Returns
        -------
        ndarray or float
            Tree-level potential evaluated at each `φ`.
        """
        return 1/2 * self.m2 * φ**2 + 1/6 * self.α * φ**3 + 1/24 * self.λ * φ**4

    def dV(self, φ, n=1):
        """Nth derivative of the potential w.r.t. `φ`

        Parameters
        ----------
        φ : ndarray or float
            Field value.
        n : int, default=1
            Order of the derivative.

        Returns
        -------
        ndarray or float
            nth derivative of the potential w.r.t. `φ`.
            """
        if n==1: return self.m2*φ + self.α/2.*φ*2 + self.λ/6.*φ**3 + 0j
        if n==2: return self.m2 + self.α*φ + self.λ/2.*φ**2 + 0j
        if n==3: return self.α + self.λ*φ + 0j
        if n==4: return self.λ*np.ones_like(φ)
        if n>4: return np.zeros_like(φ)
        if n==0: return self.V(φ)
        if n < 0:
            print('ERROR at Phi3.dV: n < 0')
            return -1
