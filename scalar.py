import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate,optimize,interpolate
from matplotlib import rc,rcParams,colors
from matplotlib.ticker import MultipleLocator
from findiff import FinDiff
import inspect
from scipy.misc import derivative

π = np.pi
γ_E = np.euler_gamma

class FlowResult(optimize.OptimizeResult):
    """This class represents the solution of the flow equation
    """
    pass

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

    def __init__(self, V, dV=None):
        """Class constructor.
        """
        # Create a wrapped private instance of the tree-level potential
        self._V = V
        self._dV = dV
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
        return self._V(φ)

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
        if self._dV == None:
            if n == 1: return (self.V(φ+δ) - self.V(φ-δ))/δ
            elif n > 1: return (self.dV(φ+δ, δ=δ, n=n-1) - self.dV(φ-δ, δ=δ, n=n-1))/δ
            else:
                print('Error on dV: n not INT >= 1')
        else:
            return self._dV(φ,n=n)

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

    def V_th(self, φ, Λ, T, dressing='PD', **options):
        """Perturbative thermal effective potential at one-loop.

        Parameters
        ----------
        φ : ndarray or float
            Field values at which to evaluate potential.
        Λ : float
            UV cutoff.
        T : float
            Temperature.
        dressing : ndarray, float, string, or None optional
            The thermal dressing of the squared mass. Default is 0.
        QSEA : bool, default=True
            If `True`, treats `Λ` as the QSEA fluctuation cutoff, so that the
            momentum cutoff `p^2 < Λ^2 - m^2(φ)` is field-dependent. If `False`,
            treats `Λ` directly as the momentum cutoff `p^2 < Λ^2`.
        method : string, default='lm'
            Method passed to scipy.optimize.root for dressings PD and FD.

        Returns
        -------
        ndarray or float
            Thermal potential with same shape as `φ`.
        """
        options.setdefault('QSEA',True)
        options.setdefault('method','lm')

        if dressing == 'PD':
            d = FinDiff(0, φ, 1, acc=2)
            d2 = FinDiff(0, φ, 2, acc=2)
            d1V = lambda δm2: derivative(lambda φ: self.V_th(φ, Λ, T, dressing=δm2), φ, n=1, dx=1.e-3,order=3)

            def f(δm2):
                res = δm2 - d(d1V(δm2))
                print(np.amax(res))
                return res

            x0 = d2(self.V_th(φ, Λ, T, dressing=0.0))
            sol = optimize.root(f,x0=x0,method=options['method'])
            δm2 = sol.x
            plt.figure()
            plt.plot(φ,δm2)
            plt.show()
            print(f(δm2))
            result = integrate.cumulative_trapezoid(d1V(δm2),φ,initial=0)
            return result
        if dressing == 'FD':
            d = FinDiff(0, φ, 1, acc=2)
            d2 = FinDiff(0, φ, 2, acc=2)
            d1V = lambda δm2: derivative(lambda φ: self.V_th(φ, Λ, T, dressing=δm2), φ, n=1, dx=1.e-3,order=3)

            def f(δm2):
                res = δm2 - d(d1V(δm2))
                print(np.amax(res))
                return res
            x0 = d2(self.V_th(φ, Λ, T, dressing=0.0))
            sol = optimize.root(f,x0=x0,method=options['method'])
            δm2 = sol.x
            plt.figure()
            plt.plot(φ,δm2)
            plt.show()
            print(f(δm2))
            result = self.V_th(φ, Λ, T, dressing=δm2)
            return result
        elif dressing == 'TFD':
            δm2 = derivative(lambda φ: self.V_th(φ, Λ, T, dressing=0.0), φ, n=2, dx=1.e-2,order=3)
            result = self.V_th(φ, Λ, T, dressing=δm2)
            return result
        elif type(dressing) == str:
            raise ValueError('dressing =' + dressing + "not a valid option. \
                Please choose one of: ('PD','TFD',`None`,`float`,`ndarray`")
        else:
            δm2 = dressing

        φ = np.atleast_1d(φ)
        m2_φ = self.dV(φ, n=2)
        m2 = self.dV(0.0, n=2)

        if options['QSEA']: Λ = np.maximum(Λ**2-m2_φ, 0)**0.5

        m2_φ += δm2

        nmax = max(np.floor(np.amax(Λ)/(2*π*T)),0)
        ωn = 2*π*T*np.arange(-nmax, nmax+1)

        pmax = np.sqrt(np.maximum(Λ[...,None]**2 - ωn**2,0))
        cond = pmax > 0

        integral = T/(12*π**2) * np.real(
            2 * pmax * (m2_φ - m2)[...,None]
            + pmax**3 * np.log((m2_φ + Λ**2)/(m2 + Λ**2) + 0j)[...,None]
            - 2 * (m2_φ[...,None] + ωn**2 + 0j)**1.5 * np.arctan(pmax/np.sqrt(m2_φ[...,None] + ωn**2 + 0j))
            + 2 * (m2 + ωn**2 + 0j)**1.5 * np.arctan(pmax/np.sqrt(m2 + ωn**2 + 0j))
        )

        result = np.sum(integral * cond, axis=-1)

        return result

    def V_eff(self, φ, Λ, T=None,**options):
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
            return self.V(φ) + self.V_CW(φ,Λ,**options)
        else:
            #Π = 1/24. * T**2 * self.dV(φ,n=4)
            return self.V(φ) + self.V_th(φ,Λ,T,**options)

    def QSEA_0T(self, k, U, φ, **options):
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
        if options['print_k']: print(k)

        d2 = FinDiff(0, φ, 2, acc=2)           # define second derivative operator using finite differences
        Upp = d2(U)
        Vpp = self.dV(φ, n=2)[:,None]
        V4 = self.dV(φ, n=4)[:,None]

        kt2 = k**2 - Vpp
        cond = (kt2 > 0)
        result = k*cond/V4 * (-(kt2 + Upp) + np.sqrt((kt2 + Upp)**2 + kt2**2*V4/(16*π**2)))
        return result

    def Litim_0T(self, k, U, φ, Λ=None, **options):
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
        if options['print_k']: print(k)

        if Λ == None or options['pcond'] == False:
            cond = np.ones_like(φ)
        else:
            pmax = np.maximum(Λ**2-self.dV(φ,n=2),0)**0.5
            cond = (k**2 < pmax**2)

        d2 = FinDiff(0, φ, 2, acc=2)           # define second derivative operator using finite differences
        Upp = d2(U)
        result = k**5/(32*π**2) *1/(k**2 + Upp)*cond[:,None]
        return result

    def QSEA_T_4D(self, k, U, φ, T, **options):
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
        T : float
            Temperature
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
        if options['print_k']: print(k)

        d2 = FinDiff(0, φ, 2, acc=2)           # define second derivative operator using finite differences
        Upp = d2(U)
        Vpp = self.dV(φ, n=2)[:,None]
        V4 = self.dV(φ, n=4)[:,None]

        kt2 = k**2 - Vpp
        cond = (kt2 > 0)

        nmax = np.floor(np.sqrt(max(np.amax(kt2),0.0))/(2*π*T))
        ωn = 2*π*T*np.arange(-nmax,nmax+1)
        Σ = np.sum(np.maximum(kt2[...,None] - ωn**2, 0)**(3/2), axis=-1)

        #Σ = np.sum([np.maximum(kt2 - ωi**2, 0)**(3/2) for ωi in ωn],axis=0)

        result = k*cond/V4 * (-(kt2 + Upp) + np.sqrt((kt2 + Upp)**2 + T*V4*Σ/(3*π**2)))
        return result


    def flow(self, φ, k, eqn='QSEA_0T', method='RK45', verbose=False, dense_output=False, **options):
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

                - 'QSEA_0T': zero-temperature flow of the QSEA in the local
                  potential approximation with a Heaviside function regulator.
                - 'Litim_0T': zero-temperature flow of the scale-dependent
                  effective action in the unmodified FRG in the local potential
                  approximation with a Heaviside function regulator.

        method : string, default='RK45'
            Solution method to be passed to scipy.integrate.solve_ivp. For more
            details, see the `scipy documentation
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_.
        verbose : bool, default=True
            if `True`, print additional messages on solution status.
        options : dict or None
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
        options.setdefault('pcond',True)
        options.setdefault('print_k',False)
        options.setdefault('Λ',k[0])
        eqns = {
            'QSEA_0T': self.QSEA_0T,
            'Litim_0T': self.Litim_0T,
            'QSEA_T_4D': self.QSEA_T_4D
            }

        # choose which flow equation to use; if not valid, throw error
        if eqn in eqns:
            eqn = eqns[eqn]
        elif type(eqn) == str:
            raise ValueError('eqn =' + eqn + 'not a valid option. \
                Please choose one of:'+ repr(list((eqns.keys()))))

        # solve flow
        sol = integrate.solve_ivp(lambda x,y: eqn(x,y,φ,**options),
                                    t_span=(k[0],k[-1]), t_eval=k, y0=self.V(φ),
                                    rtol=self.rtol, atol=self.atol,
                                    vectorized=True, method=method)

        if verbose: print(sol.message)

        U = sol.y.T
        if dense_output:
            U = tuple([interpolate.interp1d(φ, u, kind='cubic') for u in U])

        # return solution object
        return FlowResult(U=U, k=sol.t, φ=φ, success=sol.success,
                          message=sol.message, status=sol.status, eqn=eqn,
                          method=method)

class ParametricScalar(Scalar):
    """Subclass of Scalar for families of theories in which the tree-level
    potential is parameterized by a set of finite parameters `params`. This
    class serves as the parent class to specific models such as `Phi3`.

    Parameters
    ----------
    V : callable
        Tree-level potential whose signature includes dependence on `params`.
        Calling signature must be ``V(φ, ...) -> ndarray or float``, where here
        the first argument `φ` is an ndarray of field values and the subsequent
        arguments `...` are parameters controlling the shape of the potential.
    **params: dict
        Parameters controlling the shape of the potential. if not empty, the
        keys in `params` must match the names of the 2nd-onwards parameters of `V',
        and will be stored as an attribute in `ParametricScalar`. E.g., if `V` has
        signature `V(φ, a, b, c)`, then `params` must be specified as e.g.
        `a=1, b=2, c=3` and will be stored as `ParametricScalar.a`, etc.
    dV : callable or None, optional
        Nth derivative of tree-level potential. calling signature must be
        ``dV(φ, **params, n=1) -> ndarray or float``, where `φ` and `params` are
        as above and `n` is the order of the derivative. if `None`, dV defaults
        to the finite step-size fallback. Specifying `dV` is not required, but
        is recommended to eliminate roundoff error. Default is `None`.

    Attributes
    ----------
    rtol : float, default=100*`eps`
        Relative tolerance passed to ODE solver and integrator.
    atol : float, default=1e-15
        Absolute tolerance passed to ODE solver and integrator.
    """

    def __init__(self, V, dV=None, **params):
        # check that correspond to arguments of V
        args = set(inspect.getfullargspec(V).args[1:])
        if not args >= params.keys():
            raise ValueError('`params`' + repr(params - args) + 'do not match'
                             'the arguments of V.')

        # set parameters, initializing with None if not defined
        self.params = dict.fromkeys(args) | params
        Scalar.__init__(self, V, dV=dV)

    def __setattr__(self,name,value):
        self.__dict__[name] = value
        if name == 'params':
            for key,val in value.items():
                self.__dict__[key] = val
        elif 'params' in self.__dict__ and name in self.params.keys():
            self.__dict__['params'][name] = value

    def V(self, φ):
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
        return self._V(φ,**self.params)

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
        if self._dV == None:
            if n == 1: return (self.V(φ+δ) - self.V(φ-δ))/δ
            elif n >1: return (self.dV(φ+δ, δ=δ, n=n-1) - self.dV(φ-δ, δ=δ, n=n-1))/δ
            else:
                print('Error on dV: n not INT >= 1')
        else:
            return self._dV(φ,n=n,**self.params)

class Phi3(ParametricScalar):
    """Subclass of ParametricScalar for φ^3 theories with tree-level potential

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
        V = lambda φ, m2, α, λ: 1/2 * m2 * φ**2 + 1/6 * α * φ**3 + 1/24 * λ * φ**4
        def dV(φ, m2, α, λ, n=1):
            if n==1: return m2*φ + α/2.*φ*2 + λ/6.*φ**3
            if n==2: return m2 + α*φ + λ/2.*φ**2
            if n==3: return α + λ*φ
            if n==4: return λ*np.ones_like(φ)
            if n>4: return np.zeros_like(φ)
            if n==0: return V(φ)
            if n < 0:
                raise ValueError('Phi3.dV: order of the derivative n is less than 0')
        ParametricScalar.__init__(self, V, dV=dV, m2=m2, α=α, λ=λ)
