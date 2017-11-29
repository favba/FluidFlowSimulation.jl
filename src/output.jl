function writeoutput(s::AbstractParameters,init::Integer)
  mycopy!(s.aux,s.u,s)
  s.p\s.aux
  write("u1.$init",s.aux.rx)
  write("u2.$init",s.aux.ry)
  write("u3.$init",s.aux.rz)
  dealias!(s.aux,s)
end

function writeoutput(s::ScalarParameters,init::Integer)
  mycopy!(s.aux,s.u,s)
  s.p\s.aux
  write("u1.$init",s.aux.rx)
  write("u2.$init",s.aux.ry)
  write("u3.$init",s.aux.rz)
  dealias!(s.aux,s)
  copy!(rawreal(s.ρrhs),rawreal(s.ρ))
  s.ps\s.ρrhs
  write("rho.$init",s.ρrhs)
  dealias!(s.ρrhs,s)
end