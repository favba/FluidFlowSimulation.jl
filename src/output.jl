function writeoutput(s::AbstractSimulation,init::Integer)
    mycopy!(s.rhs,s.u)
    setfourier!(s.rhs)
    real!(s.rhs)
    write("u1.$init",s.rhs.rr.x)
    write("u2.$init",s.rhs.rr.y)
    write("u3.$init",s.rhs.rr.z)
    #dealias!(s.rhs)
    if haspassivescalar(s)
        mycopy!(s.passivescalar.rhs.field.data, s.passivescalar.φ.field.data)
        setfourier!(s.passivescalar.rhs)
        real!(s.passivescalar.rhs)
        write("scalar.$init",s.passivescalar.rhs)
        #dealias!(s.passivescalar.ρrhs)
    end
    if hasdensity(s)
        mycopy!(s.densitystratification.rhs.field.data,s.densitystratification.ρ.field.data)
        setfourier!(s.densitystratification.rhs)
        real!(s.densitystratification.rhs)
        write("rho.$init",s.densitystratification.rhs)
        #dealias!(s.densitystratification.ρrhs)
    end
end
