struct Euller{Adptative,initdt} <: AbstractScalarTimeStep{Adptative,initdt,0}
    dt::Base.RefValue{Float64} 
end

function initialize!(t::Euller,rhs,s)
    set_dt!(t,get_dt(s))
    return nothing
end
  
get_dt(t::Euller{true,idt}) where {idt} = getindex(t.dt)

set_dt!(t::Euller{true,idt},dt::Real) where {idt} = setindex!(t.dt,dt)

@par function (f::Euller)(ρ::AbstractArray{<:Real,3},rhs::AbstractArray{<:Real,3},s::@par(AbstractSimulation))
    dt = get_dt(f)
    @mthreads for k in Kzr
        for y in Kyr, j in y
            @inbounds @fastmath @msimd for i in 1:(2Kxr[k][j])
                ρ[i,j,k] = muladd(dt,rhs[i,j,k],ρ[i,j,k])
            end
        end
    end
    return nothing
end