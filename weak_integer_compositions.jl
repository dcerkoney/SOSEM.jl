module IntegerCompositions
export weak_integer_compositions, rpadded_weak_integer_compositions

using Combinatorics:
    partitions,
    permutations,
    WithReplacementCombinations,
    with_replacement_combinations
using IterTools:
    Iterators.popfirst!,
    Iterators.Stateful

# TODO: profile against a greedy algorithm for weak compositions

######################
# New implementation #
######################

# NOTE: slightly slower and ~20% more allocs this way, but probably
#       more Julian, and doesn't modify Combinatorics source code

struct WeakIntegerCompositions{T}
    it::WithReplacementCombinations{T}
end

Base.eltype(::Type{WeakIntegerCompositions{T}}) where {T} = Vector{eltype(T)}
Base.length(c::WeakIntegerCompositions) = length(c.it)

"""
Generate all weak compositions of an integer `n` with size `t`.
"""
weak_integer_compositions(n::Integer, t::Integer) = WeakIntegerCompositions(
    WithReplacementCombinations(1:t, n)
)

function Base.iterate(c::WeakIntegerCompositions, s=Stateful(c.it))
    # Advance stateful version of member iterator
    isempty(s) && return
    # Reinterpret size n multicombinations (combinations with replacement) of [1:t]
    # as size n weak integer compositions of t by counting instances of each index
    weak_comp = zeros(Int, length(c.it.a))
    # for i in this_multicombination
    for i in popfirst!(s)
        weak_comp[i] += 1
    end
    (weak_comp, s)
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
rpadded_weak_integer_compositions(n::Integer, t::Integer; pad::Integer=0) = RPaddedWeakIntegerCompositions(
    pad, WithReplacementCombinations(1:t, n)
)

function Base.iterate(c::RPaddedWeakIntegerCompositions, s=Stateful(c.it))
    # Advance stateful version of member iterator
    isempty(s) && return
    # Reinterpret size n multicombinations (combinations with replacement) of [1:t] as
    # right-padded size n weak integer compositions of t by counting instances of each index
    weak_comp = zeros(Int, length(c.it.a) + c.pad)
    # for i in this_multicombination
    for i in popfirst!(s)
        weak_comp[i] += 1
    end
    (weak_comp, s)
end

###########################
# Original implementation #
###########################

# NOTE: This is modified Combinatorics.jl source code! Duplicate of WithReplacementCombinations
#       iterator with additional post-processing step to reinterpret as weak compositions

struct WeakIntegerCompositionsV2{T}
    a::T
    t::Int
end

Base.eltype(::Type{WeakIntegerCompositionsV2{T}}) where {T} = Vector{eltype(T)}
Base.length(c::WeakIntegerCompositionsV2) = binomial(length(c.a) + c.t - 1, c.t)

"""
Generate all weak compositions of an integer `n` with size `t`.
"""
weak_integer_compositions_v2(n::Integer, t::Integer) = WeakIntegerCompositionsV2(1:t, n)

function Base.iterate(c::WeakIntegerCompositionsV2, s=[1 for i in 1:c.t])
    (!isempty(s) && s[1] > length(c.a) || c.t < 0) && return
    n = length(c.a)
    t = c.t
    comb = [c.a[si] for si in s]
    if t > 0
        s = copy(s)
        changed = false
        for i in t:-1:1
            if s[i] < n
                s[i] += 1
                for j in (i+1):t
                    s[j] = s[i]
                end
                changed = true
                break
            end
        end
        !changed && (s[1] = n + 1)
    else
        s = [n + 1]
    end
    # Reinterpret size t multicombinations (combinations with replacement) of [1:n]
    # as size n weak integer compositions of t by counting instances of each index
    weak_comp = zeros(Int, n)
    for i in comb
        weak_comp[i] += 1
    end
    (weak_comp, s)
end

########################
# Kun's implementation #
########################

# NOTE: This is function `Parquet.orderedPartition`, specialized to weak
#       compositions (lowerbound = 0) and without debug assertions

"""
Generate all weak compositions of an integer `n` with size `t`.
"""
function weak_integer_compositions_kun(t, n)
    unorderedPartition = collect(partitions(t + n, n))
    orderedPartition = Vector{Vector{Int}}([])
    for p in unorderedPartition
        p = p .- 1
        append!(orderedPartition, Set(permutations(p)))
    end
    return orderedPartition
end

end # module WeakCompositions