function writeheader(s::AbstractParameters)
  open("Stats.txt","w") do f
    write(f,"iteration  time  u12  u22  u32  k  enstrophy \n")
  end
end

function stats(s::AbstractParameters,init::Integer,dt::Real)
  u2,v2,w2 = kinetic_energy(s)
  k = (u2+v2+w2)/2
  ω = enstrophy(s)
  open("Stats.txt","a+") do file 
    join(file,(init, init*dt, u2, v2, w2, k, ω, "\n"), "  ")
  end
end

function kinetic_energy(s::AbstractParameters)
  @views begin
  u2 = mean(x->x^2,s.u.rx[1:end-2,:,:])
  v2 = mean(x->x^2,s.u.ry[1:end-2,:,:])
  w2 = mean(x->x^2,s.u.rz[1:end-2,:,:])
  end
  return u2,v2,w2
end

function enstrophy(s::AbstractParameters)
  a = s.aux
  u = s.u
  rhs = s.rhs
  copy!(real(a),real(u))
  s.p*a
  curl!(rhs,a,s)
  s.p\rhs
  @. rhs.rx = rhs.rx^2 + rhs.ry^2 + rhs.rz^2
  @views ω = mean(rhs.rx[1:end-2,:,:])
  return ω
end

function writeheader(s::ScalarParameters)
  open("Stats.txt","w") do f
    write(f,"iteration  time  u12  u22  u32  k  enstrophy rho2 \n")
  end
end


function stats(s::ScalarParameters,init::Integer,dt::Real)
  u2,v2,w2 = kinetic_energy(s)
  k = (u2+v2+w2)/2
  ω = enstrophy(s)
  ρ2 = ape(s)
  open("Stats.txt","a+") do file 
    join(file,(init, init*dt, u2, v2, w2, k, ω, ρ2, "\n"), "  ")
  end
end

function ape(s::ScalarParameters)
  @views ans = mean(x->x^2,real(s.ρ)[1:end-2,:,:])
  return ans
end