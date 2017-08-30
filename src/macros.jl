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