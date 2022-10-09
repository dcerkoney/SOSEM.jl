module IntegerCompositions

using Combinatorics:
    partitions, permutations, WithReplacementCombinations, with_replacement_combinations
using IterTools: Iterators.popfirst!, Iterators.Stateful

# TODO: profile against a greedy algorithm for weak compositions
export weak_integer_compositions, rpadded_weak_integer_compositions

struct WeakIntegerCompositions{T}
    it::WithReplacementCombinations{T}
end

Base.eltype(::Type{WeakIntegerCompositions{T}}) where {T} = Vector{eltype(T)}
Base.length(c::WeakIntegerCompositions) = length(c.it)

"""
Generate all weak compositions of an integer `n` with size `t`.
"""
function weak_integer_compositions(n::Integer, t::Integer)
    return WeakIntegerCompositions(WithReplacementCombinations(1:t, n))
end

function Base.iterate(c::WeakIntegerCompositions, s=Stateful(c.it))
    # Advance stateful version of member iterator
    isempty(s) && return
    # Reinterpret size n multicombinations (combinations with replacement) of [1:t]
    # as size n weak integer compositions of t by histogramming indices
    weak_comp = zeros(Int, length(c.it.a))
    # for i in this_multicombination
    for i in popfirst!(s)
        weak_comp[i] += 1
    end
    return (weak_comp, s)
end

# Same as above, but with optional right-aligned zero-padding
struct RPaddedWeakIntegerCompositions{T}
    pad::Integer
    it::WithReplacementCombinations{T}
end

Base.eltype(::Type{RPaddedWeakIntegerCompositions{T}}) where {T} = Vector{eltype(T)}
Base.length(c::RPaddedWeakIntegerCompositions) = length(c.it)

"""
Generate all weak compositions of an integer `n` with size `t` right-padded with `pad` zeros
"""
function rpadded_weak_integer_compositions(n::Integer, t::Integer; pad::Integer=0)
    return RPaddedWeakIntegerCompositions(pad, WithReplacementCombinations(1:t, n))
end

function Base.iterate(c::RPaddedWeakIntegerCompositions, s=Stateful(c.it))
    # Advance stateful version of member iterator
    isempty(s) && return
    # Reinterpret size n multicombinations (combinations with replacement) of [1:t] as
    # right-padded size n weak integer compositions of t by histogramming indices
    weak_comp = zeros(Int, length(c.it.a) + c.pad)
    # for i in this_multicombination
    for i in popfirst!(s)
        weak_comp[i] += 1
    end
    return (weak_comp, s)
end

end  # module IntegerCompositions