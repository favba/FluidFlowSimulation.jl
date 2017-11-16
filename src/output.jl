function writeoutput(s::AbstractParameters,init::Integer)
  copy!(rawreal(s.rhs),rawreal(s.u))
  #dealias!(s.rhs,s)
  out_transform!(s.aux,s.rhs,s)
  write("u1.$init",s.aux.rx)
  write("u2.$init",s.aux.ry)
  write("u3.$init",s.aux.rz)
end

function writeoutput(s::ScalarParameters,init::Integer)
  copy!(rawreal(s.rhs),rawreal(s.u))
  #dealias!(s.rhs,s)
  out_transform!(s.aux,s.rhs,s)
  write("u1.$init",s.aux.rx)
  write("u2.$init",s.aux.ry)
  write("u3.$init",s.aux.rz)
  copy!(rawreal(s.ρrhs),rawreal(s.ρ))
  #dealias!(s.ρrhs,s)
  s.ps\s.ρrhs
  write("rho.$init",s.ρrhs)
end