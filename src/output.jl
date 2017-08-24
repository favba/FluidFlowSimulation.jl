function writeoutput(s::AbstractParameters,init::Integer)
  write("u1.$init",s.u.rx)
  write("u2.$init",s.u.ry)
  write("u3.$init",s.u.rz)
end

function writeoutput(s::ScalarParameters,init::Integer)
  write("u1.$init",s.u.rx)
  write("u2.$init",s.u.ry)
  write("u3.$init",s.u.rz)
  write("rho.$init",s.ρ)
end