function stats(s::AbstractParameters,init::Integer,dt::Real)
  u2,v2,w2 = kinetic_energy(s)
  k = (u2+v2+w2)/2
  ω = enstrophy(s)
  open("Stats.txt","a+") do file 
    write(file, string(init, "  ", init*dt, "  ", u2, "  ", v2, "  ", w2, "  ", k, "  ", ω, "\n"))
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
  @views ω = mean(x->x^2,real(rhs)[1:end-2,:,:,:])
  return ω
end