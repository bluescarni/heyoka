using DiffEqBase
using OrdinaryDiffEq
using BenchmarkTools
using ArbNumerics

setextrabits(0)
setworkingprecision(ArbFloat, bits=256)

MPFloat = ArbFloat
#MPFloat = BigFloat

function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -9.8*sin(x[1])
    return dx
end

tspan = (MPFloat(0.0), MPFloat(.1))
q0 = [MPFloat(1.), MPFloat(0.0)]

prob = ODEProblem(pendulum!, q0, tspan)

bV = @benchmark solve($prob, $(Feagin14()), abstol=MPFloat(1.72e-77), reltol=MPFloat(1.72e-77), save_everystep=false)
