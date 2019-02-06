@inline function production_eddy_viscosity(S::SymTen,W::AntiSymTen,c::Real,Δ2::Real)
    a = S:S
    nu = ifelse(a == 0.,zero(eltype(a)),
         @fastmath c*Δ2*((square(W):S) - (square(S):S))/(2a))
    return nu
end