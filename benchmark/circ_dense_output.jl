using DifferentialEquations
using LinearAlgebra
using CSV
using DelimitedFiles

function ff(du,u,p,t)
    p = u[1:2]
    v = u[3:4]
    du[1:2] = v
    du[3:4] = -p/norm(p)^3
    return u
end
u0 = [1.0,0.0,0.0,-1.0]

tspan = (0.0,1.0)
prob = ODEProblem(ff,u0,tspan)
sol = solve(prob, Vern8(), reltol=1e-12, abstol=1e-12)

c = [[cos(a),-sin(a),-sin(a),-cos(a)] for a in sol.t]

f(a) = [cos(a),-sin(a),-sin(a),-cos(a)]
times = range(0.0,1.0,1000)
err = [sol(a)-f(a) for a in times]
err_r = [ norm([x,y]) for (x,y,xd,yd) in err ]

writedlm("circ_dense.csv", err_r, " ")
