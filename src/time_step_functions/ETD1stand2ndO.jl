

function ETD1stO!(u::AbstractArray,rhs::AbstractArray,c::AbstractArray,dt::Real)
    @mthreads for k in ZRANGE
        ETD1stO!(u,rhs,c,dt,k)
    end
end

function ETD1stO!(u::AbstractArray,rhs::AbstractArray,cc::AbstractArray,dt::Real,k::Integer)
    @inbounds for j in YRANGE
        @msimd for i in XRANGE
            c = cc[i,j,k]
            cdt = c*dt
            cdt2 = cdt*dt
            c2dt3 = cdt2*cdt
            c3dt4 = c2dt3*cdt
            test = -cdt<=1e-4
            test2 = c == -Inf

            u[i,j,k] = muladd(exp(cdt), u[i,j,k], ifelse(test2, 0.0, ifelse(test, dt + 0.5*cdt2 + c2dt3/6 + c3dt4/24, expm1(cdt)/c))*rhs[i,j,k])
        end
    end
end

function ETD2ndO!(u::AbstractArray,rhs::AbstractArray,c::AbstractArray,fm1::AbstractArray,dt::Real,dt2::Real)
    @mthreads for k in ZRANGE
        ETD2ndO!(u,rhs,c,fm1,dt,dt2,k)
    end
end

function ETD2ndO!(u::AbstractArray,rhs::AbstractArray,cc::AbstractArray,fm1::AbstractArray,dt::Real,dt2::Real,k::Integer)
    @inbounds for j in YRANGE
        @msimd for i in XRANGE
            c = cc[i,j,k]
            cp2 = c*c
            cdt = c*dt
            cdtp2 = cdt*dt
            c2dtp3 = cdtp2*cdt
            c3dtp4 = c2dtp3*cdt
            expm1cdt = expm1(cdt)
            c2dt2 = c*c*dt2
            dtp2 = dt*dt
            dtp3 = dtp2*dt
            dtp4 = dtp3*dt
            test = -cdt<=1e-4
            test2 = c == -Inf
            

            At = ifelse(test2, 0.0, ifelse(test,
            dt + cdtp2/2 + (c2dtp3)/6 + (c3dtp4)/24 + dtp2/(2dt2) + (c*dtp3)/(6*dt2) + (cp2* dtp4)/(24*dt2),
            (expm1cdt - c*(dt - dt2*expm1cdt))/c2dt2))

            Bt = ifelse(test2, 0.0, ifelse(test,
            (-(dtp2/(2dt2)) - (c*dtp3)/(6dt2) - (cp2*dtp4)/(24dt2)),
            (cdt - expm1cdt)/c2dt2))

            u[i,j,k] = muladd(exp(cdt), u[i,j,k], muladd(At, rhs[i,j,k], Bt*fm1[i,j,k]))
        end
    end
end