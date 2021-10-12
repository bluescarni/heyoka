using TaylorIntegration

@taylorize function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return dx
end

tspan = (0.0, 10000.0)
q0 = [1.3, 0.0]

using DiffEqBase
using OrdinaryDiffEq
prob = ODEProblem(pendulum!, q0, tspan,)

using BenchmarkTools
bV = @benchmark solve($prob, $(Vern9()), abstol=1e-15, reltol=1e-15)
bT = @benchmark solve($prob, $(TaylorMethod(19)), abstol=1e-15)
