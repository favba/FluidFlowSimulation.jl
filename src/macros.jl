macro def(name, definition) # thanks to http://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/
    return esc(quote
        macro $name()
            esc($(Expr(:quote, definition)))
        end
    end)
end

macro condthreads(condition,loop)
  return esc(quote
    if $condition
      Threads.@threads $loop
    else
      $loop
    end
  end)
end

sim_par = (:Nx,:Ny,:Nz,:Lcs,:Lcv,:Nrx,:Lrs,:Lrv,:Tr,:Integrator)

macro par(t::Symbol)
  return esc(Expr(:curly,t,sim_par...))
end

macro par(ex::Expr)
  if ex.head == :function
    if ex.args[1].head == :where
      append!(ex.args[1].args,sim_par)
    elseif ex.args[1].head == :call
      ex.args[1] = Expr(:where, ex.args[1], sim_par...) 
    end
  end
  return esc(ex)
end