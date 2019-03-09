abstract type AbstractForcing end

struct NoForcing <: AbstractForcing end

statsheader(a::NoForcing) = ""

stats(a::NoForcing,s::AbstractSimulation) = ()

msg(a::NoForcing) = "\nForcing: No forcing\n"

add_forcing!(u,::NoForcing) = nothing

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

function add_forcing!(u,f::RfForcing)
    add_Rf_forcing!(u.c.x,f.forcex)
    add_Rf_forcing!(u.c.y,f.forcey)
end

function add_Rf_forcing!(ui,f)
    @mthreads for j in YRANGE
        @inbounds for i in XRANGE
            ui[i,j,1] += f[i,j,1]
        end
    end

    for k in (2,3,4,5,6,NZ-4,NZ-3,NZ-2,NZ-1,NZ)
        @inbounds ui[1,1,k] += f[1,1,k]
    end
end

function initialize!(f::RfForcing,s)
    if f.init
        calculate_u1u2_spectrum!(f.Em,s.u,1)
    end
    if isfile("R.$(s.iteration[])")
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
            Zf = calculate_Zf(d,kf,kh)
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