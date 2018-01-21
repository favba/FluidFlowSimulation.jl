function writeoutput(s::AbstractSimulation,init::Integer)
  mycopy!(s.aux,s.u,s)
  back_transform!(s.aux,s.p,s)
  write("u1.$init",s.aux.rx)
  write("u2.$init",s.aux.ry)
  write("u3.$init",s.aux.rz)
  dealias!(s.aux,s)
  if haspassivescalar(s)
    copy!(parent(real(s.passivescalar.ρrhs)),parent(real(s.passivescalar.ρ)))
    back_transform!(s.passivescalar.ρrhs,s.passivescalar.ps,s)
    write("rho.$init",s.passivescalar.ρrhs)
    dealias!(s.passivescalar.ρrhs,s)
  end
end
