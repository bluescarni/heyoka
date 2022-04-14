using IRKGaussLegendre
using BenchmarkTools
using LinearAlgebra

function NbodyODE!(du,u,Gm,t)
    N = length(Gm)
    du[1,:,:] .= 0
    for i in 1:N
       qi = u[2,:,i]
       Gmi = Gm[i]
       du[2,:,i] = u[1,:,i]
       for j in (i+1):N
          qj = u[2,:,j]
          Gmj = Gm[j]
          qij = qi - qj
          tmpij = sqrt(qij[1]*qij[1]+qij[2]*qij[2]+qij[3]*qij[3])
          auxij = 1 / (tmpij*tmpij*tmpij)
          du[1,:,i] -= Gmj*auxij*qij
          du[1,:,j] += Gmi*auxij*qij
       end
    end

   return
end

function NbodyEnergy(u,Gm)
    N = length(Gm)
    zerouel = zero(eltype(u))
    T = zerouel
    U = zerouel
    for i in 1:N
       qi = u[2,:,i]
       vi = u[1,:,i]
       Gmi = Gm[i]
       T += Gmi*(vi[1]*vi[1]+vi[2]*vi[2]+vi[3]*vi[3])
       for j in (i+1):N
          qj = u[2,:,j]  
          Gmj = Gm[j]
          qij = qi - qj
          U -= Gmi*Gmj/norm(qij)
       end
    end
   1/2*T + U
end

Gm = [5, 4, 3]
N=length(Gm)
q=[1,-1,0,-2,-1,0,1,3,0]
v=zeros(size(q))
q0 = reshape(q,3,:)
v0 = reshape(v,3,:)
u0 = Array{Float64}(undef,2,3,N)
u0[1,:,:] = v0
u0[2,:,:] = q0
tspan = (0.0,63.0)
prob=ODEProblem(NbodyODE!,u0,tspan,Gm);

sol1=solve(prob,IRKGL16(),adaptive=true, reltol=1e-12, abstol=1e-12, save_everystep=false);

bV = @benchmark solve($prob, $(IRKGL16()), adaptive=true, reltol=1e-12, abstol=1e-12, save_everystep=false)

println(bV)

setprecision(BigFloat, 256)
u0Big=BigFloat.(u0)
GmBig=BigFloat.(Gm)

E0=NbodyEnergy(u0Big,GmBig)
ΔE1 = map(x->NbodyEnergy(BigFloat.(x),GmBig), sol1.u)./E0.-1

println(ΔE1)
