module ReadGlobal
export readglobal, getdimsize, checkinput

if VERSION >= v"0.7-"
  using DelimitedFiles
end

function findglobal()
  filename="global"
  if !isfile(filename)
    for i = 1:10
      filename = "../" * filename
      isfile(filename) && break
    end
  end
  isfile(filename) || throw(error("Could not find global file"))
  return readdlm(filename,String,comment_char='/')
end

function readglobal()
  text = findglobal()
  dict = Dict{Symbol,String}()
  for i in 1:size(text)[1]
    dict[Symbol(text[i,1])] = text[i,2]
  end
  return dict
end

function getdimsize()
  dict = readglobal()
  nx = parse(Int,dict[:nx])
  ny = parse(Int,dict[:ny])
  nz = parse(Int,dict[:nz])
  x  = parse(Float64,dict[:xDomainSize])
  y  = parse(Float64,dict[:yDomainSize])
  z  = parse(Float64,dict[:zDomainSize])

  return nx,ny,nz,x,y,z
end

function checkinput(filename::String,nx::Int,ny::Int,nz::Int)
  sizefile = filesize(filename)
  if sizefile == (nx+2)*ny*nz*8
    dtype = Float64
    padded = true
  elseif sizefile == nx*ny*nz*8
    dtype = Float64
    padded = false
  elseif sizefile == (nx+2)*ny*nz*4
    dtype = Float32
    padded = true
  elseif sizefile == nx*ny*nz*4
    dtype = Float32
    padded = false
  else
    error("""Input file "$(filename)" not matching number of gridpoints on global file (nx:$nx,ny:$ny,nz:$nz)""")
  end
  return dtype,padded
end

end
