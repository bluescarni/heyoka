using TaylorIntegration

const μ = 0.01

@taylorize function pcr3bp!(dq, q, param, t)
    local μ = param[1]
    local onemμ = 1 - μ
    x1 = q[1]-μ
    x1sq = x1^2
    y = q[2]
    ysq = y^2
    r1_1p5 = (x1sq+ysq)^-1.5
    x2 = q[1]+onemμ
    x2sq = x2^2
    r2_1p5 = (x2sq+ysq)^-1.5
    dq[1] = q[3] + q[2]
    dq[2] = q[4] - q[1]
    dq[3] = (-((onemμ*x1)*r1_1p5) - ((μ*x2)*r2_1p5)) + q[4]
    dq[4] = (-((onemμ*y )*r1_1p5) - ((μ*y )*r2_1p5)) - q[3]
    return nothing
end

tspan = (0.0, 2000.0)
p = [μ]

q0 = [-0.8, 0.0, 0.0, -0.6276410653920694]

using DiffEqBase
prob = ODEProblem(pcr3bp!, q0, tspan, p)

using OrdinaryDiffEq

using BenchmarkTools
bV = @benchmark solve($prob, $(Vern9()), abstol=1e-15, reltol=1e-15)
bT = @benchmark solve($prob, $(TaylorMethod(19)), abstol=1e-15)
