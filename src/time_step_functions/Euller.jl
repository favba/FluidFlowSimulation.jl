struct Euller{Adptative} <: AbstractScalarTimeStep{Adptative,0}
    dt::Base.RefValue{Float64} 
end

function initialize!(t::Euller,rhs,vis,s)
    set_dt!(t,get_dt(s))
    return nothing
end
  
get_dt(t::Euller) = getindex(t.dt)

set_dt!(t::Euller{true},dt::Real) = setindex!(t.dt,dt)

@par function (f::Euller)(ρ::AbstractArray{<:Real,3},rhs::AbstractArray{<:Real,3},s::@par(AbstractSimulation))
    dt = get_dt(f)
    @mthreads for k in ZRANGE
        for j in YRANGE
            @inbounds @msimd for i in XRANGE
                ρ[i,j,k] = muladd(dt,rhs[i,j,k],ρ[i,j,k])
            end
        end
    end
    return nothing
end