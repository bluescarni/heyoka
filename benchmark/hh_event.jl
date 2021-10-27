using DiffEqBase
using BenchmarkTools
using OrdinaryDiffEq

const tol = 1e-15

ix_vals = []

event_func(u, t, int) = u[3]

function cb_func!(int)
    push!(ix_vals, int.t)
end

function hh!(dq, q, param, t)
    vx, vy, x, y = q

    dq[1] = -x - 2. * x * y
    dq[2] = y * y - y - x * x
    dq[3] = vx
    dq[4] = vy

    return nothing
end

cb = ContinuousCallback(event_func, cb_func!, nothing)

tspan = (0.0, 2000.0)
q0 = [-0.2525875586263492, -0.2178423952983717, 0., 0.2587703282931232]

prob = ODEProblem(hh!, q0, tspan);

sol = solve(prob, Vern9(), abstol=tol, reltol=tol, save_everystep=false, callback=cb);
print(length(ix_vals))
ix_vals

bV = @benchmark solve($prob, $(Vern9()), abstol=tol, reltol=tol, save_everystep=false, callback=cb)
println(bV)

sol = solve(prob, Vern9(), abstol=tol, reltol=tol, save_everystep=false);
print(length(ix_vals))

bV = @benchmark solve($prob, $(Vern9()), abstol=tol, reltol=tol, save_everystep=false)
println(bV)
