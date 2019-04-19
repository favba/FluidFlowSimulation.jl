function writeoutput(s::AbstractSimulation)
    init = s.iteration[]
    #mycopy!(s.rhs,s.u)
    #setfourier!(s.rhs)
    #real!(s.rhs)
    write("u1.$init",s.u.rr.x)
    write("u2.$init",s.u.rr.y)
    write("u3.$init",s.u.rr.z)
    #dealias!(s.rhs)
    if haspassivescalar(s)
        #mycopy!(s.passivescalar.rhs.field.data, s.passivescalar.φ.field.data)
        #setfourier!(s.passivescalar.rhs)
        #real!(s.passivescalar.rhs)
        write("scalar.$init",s.passivescalar.φ.rr)
        #dealias!(s.passivescalar.ρrhs)
    end
    if hasdensity(s)
        #mycopy!(s.densitystratification.rhs.field.data,s.densitystratification.ρ.field.data)
        #setfourier!(s.densitystratification.rhs)
        #real!(s.densitystratification.rhs)
        write("rho.$init",s.densitystratification.ρ.rr)
        #dealias!(s.densitystratification.ρrhs)
    end
    if hasles(s)
        t = s.lesmodel.tau
        isrealspace(t) || real!(t)
        write("T11.$init",t.rr.xx)
        write("T12.$init",t.rr.xy)
        write("T13.$init",t.rr.xz)
        write("T22.$init",t.rr.yy)
        write("T23.$init",t.rr.yz)
        #is_FakeSmagorinsky(s) || write("pr.$init",s.lesmodel.pr)
    end
    if typeof(s.forcing) <: RfForcing
        writedlm("R.$init",s.forcing.R)
        writedlm("Eh.$init",zip(s.forcing.avgK, s.forcing.Ef))
    end
    if typeof(s.forcing) <: AForcing
        writedlm("R.$init",s.forcing.R)
    end
    if typeof(s.forcing) <: NRfForcing
        writedlm("R.$init",s.forcing.R)
        writedlm("N.$init",s.forcing.N)
        writedlm("Eh.$init",zip(s.forcing.avgK, s.forcing.Ef))
    end
end
