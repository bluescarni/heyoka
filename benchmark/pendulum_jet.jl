using TaylorIntegration

H(x) = 0.5x[2]^2-cos(x[1])

@taylorize function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
end

const varorder = 8
using TaylorIntegration
#ξ = set_variables("ξ", numvars=2, order=varorder)

q0 = [1.3, 0.0]

#q0TN = q0 .+ ξ
q0TN = q0

order = 20     #the order of the Taylor expansion wrt time
abstol = 2e-16 #the absolute tolerance of the integration
using Elliptic # we use Elliptic.jl to evaluate the elliptic integral K
T = 4*Elliptic.K(sin(q0[1]/2)^2) #the libration period
t0 = 0.0        #the initial time
tmax = 6T       #the final time
integstep = T/8 #the time interval between successive evaluations of the solution vector


tv = t0:integstep:tmax # the times at which the solution will be evaluated
xv = taylorinteg(pendulum!, q0TN, tv, order, abstol)

using BenchmarkTools
bV = @benchmark taylorinteg(pendulum!, q0TN, tv, order, abstol)
println(bV)
