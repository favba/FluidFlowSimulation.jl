
@inline function Vreman_eddy_viscosity(S::SymTen,ω::Vec,c::Real,Δ2::Real)
    α = S + AntiSymTen(-0.5*ω)
    β = Δ2 * symmetric(α' ⋅ α)
    B = β.xx*(β.yy + β.zz) + β.yy*β.zz - (β.xy^2 + β.xz^2 + β.yz^2)
    B = ifelse(B<0.0,0.0,B)
    return c*fsqrt(B/(α:α))
end