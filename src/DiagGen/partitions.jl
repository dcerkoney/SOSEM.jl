"""
Hard-coded counterterm partitions in the form (nloop, nμ, nλ).
"""
function partition(order::Int)
    #! format: off
    # normal order, G order, W order
    par = [
        # order 0
        (0, 0, 0),
        # order 1
        (1, 0, 0),
        #order 2
        (2, 0, 0), (1, 1, 0), (1, 0, 1),
        #order 3
        (3, 0, 0), (2, 1, 0), (2, 0, 1),
        (1, 1, 1), (1, 2, 0), (1, 0, 2),
        #order 4
        (4, 0, 0), (3, 1, 0), (3, 0, 1), (2, 1, 1), (2, 2, 0),
        (2, 0, 2), (1, 3, 0), (1, 0, 3), (1, 2, 1), (1, 1, 2),
        #order 5
        (5, 0, 0), (4, 1, 0), (4, 0, 1), (3, 2, 0), (3, 1, 1), (3, 0, 2), (2, 3, 0),
        (2, 2, 1), (2, 1, 2), (2, 0, 3), (1, 4, 0), (1, 3, 1), (1, 2, 2), (1, 1, 3), (1, 0, 4),
    ]
    #! format: on
    return sort([p for p in par if p[1] + p[2] + p[3] <= order])
end

"""
Get all (μ and/or λ) counterterm partitions (n1, n2, n3) satisfying the following constraints:

    (1) n1 ≥ n_lowest
    (2) n_min <= n1 + n2 + n3 <= n_max

By convention, we interpret: n1 = n_loop, n2 = n_∂μ, n3 = n_∂λ 
(normal order, G order, W order), where n_loop ≥ 2 is the total loop number.
If `renorm_mu` is false, then n_ct_mu = 0. 
Similarly, if `renorm_lambda` is false, then n_ct_lambda = 0. 
`n_lowest` is the lowest valid loop order for the given observable.

By default, generates partitions with interaction counterterms only.
"""
function counterterm_partitions(
    n_min::Int,
    n_max::Int;
    n_lowest::Int,
    renorm_mu=true,
    renorm_lambda=true,
)
    partitions = Vector{PartitionType}()
    if n_max < n_min
        return partitions
    end
    # No μ or λ counterterms
    if !renorm_mu && !renorm_lambda
        partitions = [(n1, 0, 0) for n1 in n_min:n_max]
        # Generate only λ counterterms
    elseif !renorm_mu
        partitions = [
            (n1, 0, n3) for
            (n1, n3) in counterterm_single_split(n_max, n_lowest) if n1 + n3 ≥ n_min
        ]
        # Generate only μ counterterms
    elseif !renorm_lambda
        partitions = [
            (n1, n2, 0) for
            (n1, n2) in counterterm_single_split(n_max, n_lowest) if n1 + n2 ≥ n_min
        ]
        # Generate both μ and λ counterterms
    else
        partitions =
            [p for p in partition(n_max) if p[1] ≥ n_lowest && p[1] + p[2] + p[3] ≥ n_min]
    end
    return partitions
end

"""
Get all (μ and/or λ) counterterm partitions (n1, n2, n3) satisfying the constraint
s.min_order ≤ n1 + n2 + n3 ≤ s.max_order for the given SOSEM measurement settings. 
If `renorm_mu` is false, then n_ct_mu = 0. 
"""
function counterterm_partitions(s::Settings; renorm_mu=true, renorm_lambda=true)
    return counterterm_partitions(
        s.min_order,
        s.max_order;
        n_lowest=_get_lowest_loop_order(s.observable),  # Lowest loop order depends on the observable
        renorm_mu=renorm_mu,
        renorm_lambda=renorm_lambda,
    )
end

# """
# Get all counterterm partitions (n1, n2, n3) at fixed order n = n1 + n2 + n3
# for a SOSEM measurement specified by the given settings. 
# If `renorm_mu` is false, then n_ct_mu = 0. 
# """
# function counterterm_partitions_fixed_order(
#     s::Settings;
#     renorm_mu=false,
#     renorm_lambda=true,
# )
#     return counterterm_partitions(
#         s.min_order,
#         s.max_order;
#         n_lowest=_get_lowest_loop_order(s.observable),
#         renorm_mu=renorm_mu,
#         renorm_lambda=renorm_lambda,
#     )
# end

"""
Generate weak compositions of size 2 of an integer n
(i.e., the cycle (n, 0), (n-1, 1), ..., (0, n)), where
(n_order, n_ct_lambda) = (i, j) with an additional constraint
that max_order ≥ 2 (the minimum order for SOSEM observables).
`n_lowest` is the lowest valid loop order for the given observable.
"""
function counterterm_single_split(n::Int, n_lowest::Int=2)
    if n < n_lowest
        return Tuple{Int,Int}[]
    end
    splits = Tuple{Int,Int}[]
    max    = n_lowest
    n1     = n_lowest
    n2     = 0
    while max <= n
        push!(splits, (n1, n2))
        n1 += 1
        n2 -= 1
        if n2 < 0
            max += 1
            n1 = n_lowest
            n2 = max - n_lowest
        end
    end
    return sort(splits)
end
