function (F::NRfForcing)(s::AbstractSimulation)

    cPlaneNumber = 1
    dt = get_dt(s)
    Tf = F.Tf
    ts = 10*dt
    omega = (2*π/(Tf*ts))
    alpha = F.α
    eps = Base.eps()
    _kf = F.Kf
    maxdk = F.maxDk
    avgWaveNumInShell2d = F.avgK
    nShells2d = length(avgWaveNumInShell2d)

    Ef = F.Ef
    R = F.R
    Em = F.Em
    Zf = F.Zf
    factor = F.factor
    N = F.N
    Nm1 = F.Nm1
    dtm2 = s.timestep.x.dt2[]
  
    s1 = fill!(F.forcex,0)
    s2 = fill!(F.forcey,0)

    u1 = s.u.c.x
    u2 = s.u.c.y

    # // get current spectrum for k=1 plane
    calculate_u1u2_spectrum!(Ef,s.u, cPlaneNumber)

    copyto!(Nm1,N)
    calculate_nn_spectrum!(N,s.u,s.rhs,cPlaneNumber)

    @inbounds for i=2:nShells2d
        if avgWaveNumInShell2d[i] <= _kf
            R[i] += dt*(-2*alpha*omega*(R[i] + N[i]) - (N[i]-Nm1[i])/dtm2 - omega*omega*(Ef[i] - Em[i])) 
            R[i]=max(0.0, R[i])
            factor[i] = fsqrt(R[i]/max(Ef[i],eps))*Zf[i]*dt     
        end
    end

    umag = 0.0
  #  # -------------------- Horizontal Forcing -------------------- 
    ff = 0.0
    fv = 0.0

    @inbounds for j=YRANGE
        conjFactX=1.0
        for i=XRANGE
            k = fsqrt(muladd(KX[i],KX[i],KY[j]^2))
            n = round(Int,k/maxdk) + 1
            if (1 < n <= nShells2d)
                if avgWaveNumInShell2d[n] <= _kf
                    s1[i,j,1] = u1[i,j,1]*factor[n]
                    s2[i,j,1] = u2[i,j,1]*factor[n]
                    ff += (abs2(s1[i,j,1]) + abs2(s2[i,j,1]))*conjFactX
                    fv += (proj(u1[i,j,1],s1[i,j,1]) + proj(u2[i,j,1],s2[i,j,1]))*conjFactX
                    conjFactX=2.0
                end
            end
        end
    end
    ff = ff*0.5/dt
    fv = fv/dt
    horizontalPower = ff+fv
    F.hp[] = horizontalPower
    powerFraction = 0.99

    # # -------------------- Vertical Forcing -------------------- 
    # /*
    #   Force the points with kx=ky=0 and kz=1,2,3,4,5 so that the power
    #   added is (1-powerFraction) of that added to the horizontal */
    ff = 0.0
    fv = 0.0
    umag = 0.0
    #     // get the sum of the existing velocites
    @inbounds for k = 2:6
        umag += fsqrt(abs2(u1[1,1,k]) + abs2(u2[1,1,k])) + fsqrt(abs2(u1[1,1,NZ-k+2]) + abs2(u2[1,1,NZ-k+2]))
    end

    f = dt*sqrt(2.0*horizontalPower*(1.0-powerFraction))/umag

    @inbounds for k=2:6
        s1[1,1,k] = u1[1,1,k]*f
        s2[1,1,k] = u2[1,1,k]*f
        s1[1,1,NZ-k+2] = u1[1,1,NZ-k+2]*f
        s2[1,1,NZ-k+2] = u2[1,1,NZ-k+2]*f
        ff += abs2(s1[1,1,k]) + abs2(s2[1,1,k]) + abs2(s1[1,1,NZ-k+2]) + abs2(s2[1,1,NZ-k+2])
        fv += proj(u1[1,1,k],s1[1,1,k]) + proj(u2[1,1,k],s2[1,1,k]) + proj(u1[1,1,NZ-k+2],s1[1,1,NZ-k+2]) + proj(u2[1,1,NZ-k+2],s2[1,1,NZ-k+2])
    end
    ff = ff*0.5/dt
    fv = fv/dt
    verticalPower = ff+fv
    F.vp[] = verticalPower
    return 1/dt
end

function calculate_nn_spectrum!(N,u,rhs,cplane)
    ux = u.c.x
    uy = u.c.y
    rhsx = rhs.c.x
    rhsy = rhs.c.y
    # Initialize the shells to zeros
    fill!(N,0)
    maxdk2d = max(KX[2],KY[2])
    nshells = min(NX,NY÷2)
    @inbounds for j in YRANGE
        conjFactX=1.0
        for i in XRANGE
            k = fsqrt(KX[i]^2 + KY[j]^2)
            n = round(Int, k/maxdk2d) + 1
            if n <= nshells
                magsq = proj(ux[i,j,cplane],rhsx[i,j,cplane]) + proj(uy[i,j,cplane],rhsy[i,j,cplane])
                ee = conjFactX * magsq;
                N[n]+=ee;
                conjFactX=2.0;
            end
        end
    end
end