using TaylorIntegration

@taylorize function pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return dx
end

# Initial time (t0), final time (tf) and initial condition (q0)
t0 = 0.0
tf = 10000.0
q0 = [1.3, 0.0]

# The actual integration
t1, x1 = taylorinteg(pendulum!, q0, t0, tf, 20, 2.2e-16, maxsteps=150000); # warm-up run
e1 = @elapsed taylorinteg(pendulum!, q0, t0, tf, 20, 2.2e-16, maxsteps=150000);
println(e1)

