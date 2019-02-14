abstract type AbstractForcing end

struct NoForcing <: AbstractForcing end

statsheader(a::NoForcing) = ""

stats(a::NoForcing,s::AbstractSimulation) = ()

msg(a::NoForcing) = "\nForcing: No forcing\n"

struct RfForcing{T<:AbstractFloat} <: AbstractForcing
    Tf::T
    α::T
    Kf::T
    maxDk::T
    avgK::Vector{T}
    Zf::Vector{T}
    Ef::Vector{T} # Velocity Field Spectrum
    Em::Vector{T} # Target Spectrum
    R::Vector{T} # Solution to ODE
  #  Zf::Vector{Float64} # Cutoff function, using as parameter
    #dRdt::Vector{Float64} # Not needed if I use Euller timestep
    factor::Vector{T} # Factor to multiply velocity Field
    forcex::PaddedArray{T,3,2,false} # Final force
    forcey::PaddedArray{T,3,2,false} # Final force
    init::Bool # Tell if the initial condition spectra should be used instead of from data
    hp::Base.RefValue{T}
    vp::Base.RefValue{T}
end

statsheader(a::RfForcing) = "hp,vp"

stats(a::RfForcing,s::AbstractSimulation) = (a.hp[],a.vp[])

msg(a::RfForcing) = "\nForcing:  Rf forcing\nTf: $(getTf(a))\nalphac: $(getalpha(a))\nKf: $(getKf(a))\n"

@inline getTf(f::RfForcing) = f.Tf

@inline getalpha(f::RfForcing) = f.α

@inline getKf(f::RfForcing) = f.Kf

@inline getmaxdk(f::RfForcing) = f.maxDk

@inline getavgk(f::RfForcing) = f.avgK

@inline getZf(f::RfForcing) = f.Zf

function initialize!(f::RfForcing,s)
    if f.init
        calculate_u1u2_spectrum!(f.Em,s.u,1)
    end
    if s.iteration[] != 0 
        copyto!(f.R, vec(readdlm("R.$(s.iteration[])",Float64)))
    end
    return nothing
end

function calculate_Em!(Em,kh)
    data = readdlm("targSpectrum.dat",skipstart=1)
    E = interpolate((data[:,1],),data[:,2],Gridded(Linear()))
    for i in eachindex(Em)
        @inbounds Em[i] = E(kh[i])
    end
    return Em
end

function forcing_model(d::AbstractDict,nx::Integer,ny::Integer,nz::Integer,ncx::Integer)

    forcingtype = NoForcing()

    if haskey(d,:forcing)
        if d[:forcing] == "rfForcing"
            TF = parse(Float64,d[:TF])
            alphac = parse(Float64,d[:alphac])
            kf = parse(Float64,d[:kf])
            nShells2D, maxdk2D, numPtsInShell2D, kh = compute_shells2D(KX,KY,ncx,ny)
            Ef = zeros(nShells2D)
            Em = zeros(nShells2D)
            R = zeros(nShells2D)
            factor = zeros(length(kh))
            forcex = PaddedArray((nx,ny,nz))
            forcey = PaddedArray((nx,ny,nz))
            Zf = calculate_Zf(kf,kh)
            if !isfile("targSpectrum.dat")
                forcingtype = RfForcing{Float64}(TF, alphac, kf, maxdk2D, kh,Zf,Ef,Em,R,factor,forcex,forcey,true,Ref(0.0),Ref(0.0))
            else
                calculate_Em!(Em,kh)
                forcingtype = RfForcing{Float64}(TF, alphac, kf, maxdk2D, kh,Zf,Ef,Em,R,factor,forcex,forcey,false,Ref(0.0),Ref(0.0))
            #todo read spectrum.dat
            end
        end
    end

    return forcingtype
end