@par function cross!(outx::AbstractArray{T},outy::AbstractArray{T},outz::AbstractArray{T},
                     ux::AbstractArray{T},uy::AbstractArray{T},uz::AbstractArray{T},
                     vx::AbstractArray{T},vy::AbstractArray{T},vz::AbstractArray{T},
                     s::@par(AbstractSimulation)) where {T<:Complex}
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE
                outx[i,j,k] = uy[i,j,k]*vz[i,j,k] - uz[i,j,k]*vy[i,j,k]
                outy[i,j,k] = uz[i,j,k]*vx[i,j,k] - ux[i,j,k]*vz[i,j,k]
                outz[i,j,k] = ux[i,j,k]*vy[i,j,k] - uy[i,j,k]*vx[i,j,k]
            end
    end
end

@par function crossk!(outx,outy,outz,vx,vy,vz,s::@par(AbstractSimulation)) 
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE  
                outx[i,j,k] = im*(KY[j]*vz[i,j,k] - KZ[k]*vy[i,j,k])
                outy[i,j,k] = im*(KZ[k]*vx[i,j,k] - KX[i]*vz[i,j,k])
                outz[i,j,k] = im*(KX[i]*vy[i,j,k] - KY[j]*vx[i,j,k])
            end
    end  
end

function ccross!(out::VectorField,u::VectorField,v::VectorField,s::AbstractSimulation)
    cross!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,v.cx,v.cy,v.cz,s)
    return out
end

function curl!(out::VectorField,u::VectorField,s::AbstractSimulation)
    crossk!(out.cx,out.cy,out.cz,u.cx,u.cy,u.cz,s)
    return out
end

@par function grad!(out::VectorField,f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation))
    outx = out.c.x
    outy = out.c.y
    outz = out.c.z
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE
                outx[i,j,k] = f[i,j,k]*im*KX[i]
                outy[i,j,k] = f[i,j,k]*im*KY[j]
                outz[i,j,k] = f[i,j,k]*im*KZ[k]
            end
    end
    #dx!(out.cx,f,s)
    #dy!(out.cy,f,s)
    #dz!(out.cz,f,s)
    #dealias!(out,s)
end

@par function dx!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation)) 
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE
                out[i,j,k] = f[i,j,k]*im*KX[i]
            end
    end
end

@par function dy!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation)) 
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE
                out[i,j,k] = f[i,j,k]*im*KY[j]
            end
    end
end

@par function dz!(out::AbstractArray{<:Complex,3},f::AbstractArray{<:Complex,3},s::@par(AbstractSimulation)) 
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE
                out[i,j,k] = f[i,j,k]*im*KZ[k]
            end
    end
end

@par function div!(out::AbstractArray{Complex{Float64},3},ux,uy,uz,w,mdρdz::Real, s::@par(AbstractSimulation))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE
               # out[i,j,k] = mim*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k]) + mdœÅdz*w[i,j,k]
                out[i,j,k] = muladd(im,muladd(KX[i], ux[i,j,k], muladd(KY[j], uy[i,j,k], KZ[k]*uz[i,j,k])), mdρdz*w[i,j,k])
            end
    end
end

div!(out::AbstractArray{<:Complex,3},u::VectorField,s) = div!(out,u.cx,u.cy,u.cz,s)

@par function div!(out::AbstractArray{Complex{Float64},3},ux,uy,uz, s::@par(AbstractSimulation))
    @mthreads for ind in ZYRANGE
        jj,kk = Tuple(ind)
        k = ZRANGE[kk]
        j = YRANGE[jj]
            @inbounds @msimd for i in XRANGE
                #out[i,j,k] = -im*(kx[i]*ux[i,j,k] + ky[j]*uy[i,j,k] + kz[k]*uz[i,j,k])
                out[i,j,k] = im*muladd(KX[i], ux[i,j,k], muladd(KY[j], uy[i,j,k], KZ[k]*uz[i,j,k]))
            end
    end
end