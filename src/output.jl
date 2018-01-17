function writeoutput(s::AbstractSimulation,init::Integer)
  mycopy!(s.aux,s.u,s)
  back_transform!(s.aux,s.p,s)
  write("u1.$init",s.aux.rx)
  write("u2.$init",s.aux.ry)
  write("u3.$init",s.aux.rz)
  dealias!(s.aux,s)
  if isscalar(s)
    copy!(parent(real(s.ρrhs)),parent(real(s.ρ)))
    back_transform!(s.ρrhs,s.ps,s)
    write("rho.$init",s.ρrhs)
    dealias!(s.ρrhs,s)
  end
end
