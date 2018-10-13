function writeoutput(s::AbstractSimulation,init::Integer)
    mycopy!(s.aux,s.u)
    irfft!(s.aux)
    write("u1.$init",s.aux.rr.x)
    write("u2.$init",s.aux.rr.y)
    write("u3.$init",s.aux.rr.z)
    dealias!(s.aux)
    if haspassivescalar(s)
        mycopy!(s.passivescalar.ρrhs.field.data, s.passivescalar.ρ.field.data)
        irfft!(s.passivescalar.ρrhs)
        write("scalar.$init",s.passivescalar.ρrhs)
        dealias!(s.passivescalar.ρrhs)
    end
    if hasdensity(s)
        mycopy!(s.densitystratification.ρrhs.field.data,s.densitystratification.ρ.field.data)
        irfft!(s.densitystratification.ρrhs)
        write("rho.$init",s.densitystratification.ρrhs)
        dealias!(s.densitystratification.ρrhs)
    end
end
