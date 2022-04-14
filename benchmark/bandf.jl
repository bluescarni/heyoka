using DiffEqBase
using OrdinaryDiffEq
using LinearAlgebra

function pendulum!(du,u,_,t)
    du[1] = u[2]
    du[2] = -9.8*sin(u[1])

   return
end

dt = 1000.

tspan = (0.0,dt)
u0 = [0.05, 0.025]
prob=ODEProblem(pendulum!,u0,tspan);

sol1=solve(prob,Vern9(),adaptive=true, reltol=1e-15, abstol=1e-15, save_everystep=false);

tspan = (dt,0.0)
prob=ODEProblem(pendulum!,last(sol1.u),tspan);

sol2=solve(prob,Vern9(),adaptive=true, reltol=1e-15, abstol=1e-15, save_everystep=false);

println(norm(last(sol2.u) - u0))
