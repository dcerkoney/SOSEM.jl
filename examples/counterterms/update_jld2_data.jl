using CodecZlib
using ElectronLiquid
using JLD2

# abstract type GridBoundType end
# struct ClosedBound <: GridBoundType end
# const CLOSEDBOUND = ClosedBound()

# BoundType(::Type{<:CompositeGrids.SimpleG.Arbitrary{T,BT}}) where {T,BT} = BT()
# BoundType(::Type{<:CompositeGrids.SimpleG.Uniform{T,BT}}) where {T,BT} = BT()
# function BoundType(
#     ::Type{<:Union{CompositeGrids.SimpleG.BaryCheb,CompositeGrids.SimpleG.GaussLegendre}},
# )
#     return OPENBOUND
# end
# BoundType(::Type{<:CompositeGrids.SimpleG.Log}) = CLOSEDBOUND

# function JLD2.rconvert(
#     ::Type{CompositeGrids.SimpleG.Uniform{T}},
#     nt::NamedTuple,
# ) where {T<:AbstractFloat}
#     println("Trying to convert Uniform")
#     return CompositeGrids.SimpleG.Uniform{T,CompositeGrids.SimpleG.ClosedBound}(
#         nt.bound,
#         nt.size,
#         nt.grid,
#         nt.weight,
#     )
# end

# function JLD2.rconvert(
#     ::Type{CompositeGrids.SimpleG.Arbitrary{T}},
#     nt::NamedTuple,
# ) where {T<:AbstractFloat}
#     println("Trying to convert Arbitrary")
#     return CompositeGrids.SimpleG.Arbitrary{T,CompositeGrids.SimpleG.ClosedBound}(
#         nt.bound,
#         nt.size,
#         nt.grid,
#         nt.weight,
#     )
# end

# # Specify the type of qgrid and τgrid explicitly, otherwise, there will be a type stability issue with interactionDynamic and interactionStatic
# const GridType = CompositeGrids.CompositeG.Composite{
#     Float64,
#     CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
#     CompositeGrids.CompositeG.Composite{
#         Float64,
#         CompositeGrids.SimpleG.Log{Float64},
#         CompositeGrids.SimpleG.Uniform{Float64,CompositeGrids.SimpleG.ClosedBound},
#     },
# }

# const GridTypeOld = CompositeGrids.CompositeG.Composite{
#     Float64,
#     CompositeGrids.SimpleG.Arbitrary{Float64},
#     CompositeGrids.CompositeG.Composite{
#         Float64,
#         CompositeGrids.SimpleG.Log{Float64},
#         CompositeGrids.SimpleG.Uniform{Float64},
#     },
# }

# @with_kw mutable struct ParaMC
#     ### fundamental parameters
#     beta::Float64
#     rs::Float64
#     order::Int = 2
#     Fs::Float64 = -0.0
#     Fa::Float64 = -0.0
#     # δFs = []

#     mass2::Float64 = 1e-6
#     massratio::Float64 = 1.0

#     dim::Int = 3
#     spin::Int = 2
#     isFock::Bool = false
#     isDynamic::Bool = false

#     ### MC parameters #######
#     # seed::Int = abs(rand(Int)) % 1000000
#     # steps::Int = 1e6

#     #### derived parameters ###########
#     basic::Parameter.Para = Parameter.rydbergUnit(1.0 / beta, rs, dim; Λs=mass2, spin=spin)

#     kF::Float64 = basic.kF
#     EF::Float64 = basic.EF
#     β::Float64 = basic.β
#     maxK::Float64 = 6 * basic.kF
#     me::Float64 = basic.me
#     ϵ0::Float64 = basic.ϵ0
#     e0::Float64 = basic.e0
#     μ::Float64 = basic.μ
#     NF::Float64 = basic.NF
#     NFstar::Float64 = basic.NF * massratio
#     qTF::Float64 = basic.qTF

#     fs::Float64 = Fs / NFstar
#     fa::Float64 = Fa / NFstar

#     ##########   effective interaction and counterterm ###############
#     qgrid::GridType =
#         CompositeGrid.LogDensedGrid(:uniform, [0.0, maxK], [0.0, 2kF], 16, 0.01 * kF, 8)
#     τgrid::GridType =
#         CompositeGrid.LogDensedGrid(:uniform, [0.0, β], [0.0, β], 16, β * 1e-4, 8)

#     # ######### only need to be initialized for MC simulation ###########################
#     initialized::Bool = false
#     dW0::Matrix{Float64} = Matrix{Float64}(undef, length(qgrid), length(τgrid))
#     dW0_f::Matrix{Float64} = Matrix{Float64}(undef, length(qgrid), length(τgrid))
#     cRs::Vector{Matrix{Float64}} = []
#     cRs_f::Vector{Matrix{Float64}} = []

#     # dW0::Matrix{Float64} = KOdynamic_T(basic, qgrid, τgrid, mass2, massratio, fs, fa)
#     # cRs::Vector{Matrix{Float64}} = [counterKO_T(basic, qgrid, τgrid, o, mass2, massratio, fs, fa) for o in 1:order]

#     additional = Any[]
# end

# Converts JLD2 data from old to new ParaMC format on load by adding the `initialized` field (see: https://juliaio.github.io/JLD2.jl/stable/advanced/)
# NOTE: Requires the type name `ElectronLiquid.UEG.ParaMC` to be explicitly specified
function JLD2.rconvert(::Type{ElectronLiquid.UEG.ParaMC}, nt::NamedTuple)
    return ElectronLiquid.UEG.ParaMC(; nt..., initialized=false)
end

"""
    function update_jld2_data(filename; save=false, log=true)

Removes redundant entries of type ParaMC from a JLD2 archive and upgrades key formats (field `order` now present in paraid).
For example, the data entry `(a, p::ParaMC, b, c)` will be replaced by `(a, b, c)`, and its key will be updated to include
the substring `order_\$(p.order)`.

# Arguments
- `filename`: the name of the JLD2 archive to be updated
- `save`: whether to write the updated archive to file `filename`
- `log`: whether to write log info to file `update_jld2_data.log`
- `is_mass_ratio`: whether the JLD2 archive contains processed data for mass ratios. If `true`, the suffix `_idk=<idk>` is present in the keys.
"""
function update_jld2_data(filename; save=false, log=false, is_mass_ratio=false)
    io = IOBuffer()
    println.([io, stdout], "\nChecking for redundant ParaMC entries in $filename...\n")

    # Load the JLD2 archive. 
    # NOTE: A typemap must be specified to update outdated ParaMC types stored in the data to match the new struct definition.
    data = load(
        filename;
        typemap=Dict(
            "ElectronLiquid.UEG.ParaMC" => JLD2.Upgrade(ElectronLiquid.UEG.ParaMC),
        ),
    )
    updated_data = Dict()

    # Convert JLD2 data entries like (a, p::ParaMC, b, c, ...) to (a, b, c, ...)
    has_updates = false
    for k in keys(data)
        new_key = k
        try
            # First, check that we can parse this key as a ParaMC object
            local srep, idk
            if is_mass_ratio
                # remove suffix `_idk=<idk>` to extract ParaMC string representation
                srep, idk = split(k, "_idk=")  
                # NOTE: typeof(srep) == typeof(idk) == SubString
                UEG.ParaMC(string(srep))
            else
                UEG.ParaMC(k)
            end
            # Add field `order` to keys and remove all ParaMC instances
            if data[k] isa Tuple
                idx_para = findfirst(v -> v isa ParaMC, data[k])
                if isnothing(idx_para)
                    println.([io, stdout], "No ParaMC in data entry with key $k...\n")
                else
                    has_updates = true
                    # Get updated key name including field `order`
                    new_key = UEG.short(data[k][idx_para])
                    if is_mass_ratio
                        new_key *= "_idk=$idk"
                    end
                    if save
                        println.(
                            [io, stdout],
                            "Removing ParaMC in data entry with key $k...",
                        )
                        println.([io, stdout], "Updating key name to $new_key...\n")
                        # Update this entry
                        data[k] = Tuple(v for (i, v) in enumerate(data[k]) if i != idx_para)
                    else
                        println.(
                            [io, stdout],
                            "Will remove ParaMC in data entry with key $k...",
                        )
                        println.([io, stdout], "Will update key name to $new_key...\n")
                    end
                end
            end
        catch
            println.(
                [io, stdout],
                "Failed to parse key $k as a ParaMC object, skipping it...\n",
            )
        end
        updated_data[new_key] = data[k]
    end
    if has_updates == false
        println.([io, stdout], "No changes needed to $filename.\n")
        println.([io, stdout], "Done!\n")
        return
    end

    if save
        # Backup the JLD2 archive
        suffix = 0
        backup_name = "$(filename).bak"
        if isfile(backup_name)
            while isfile(backup_name)
                suffix += 1
                backup_name = "$(filename).bak$(suffix)"
            end
        end
        println.([io, stdout], "Creating a backup at $(backup_name)...\n")
        cp(filename, backup_name)

        # Save the updated JLD2 archive
        println.([io, stdout], "Saving data...\n")
        jldopen(filename, "w"; compress=true) do f
            for (k, v) in updated_data
                f[k] = v
            end
        end
    end
    println.([io, stdout], "Done!\n")

    if log
        # Save the log info to file
        open("update_jld2_data.log", "a+") do f
            return write(f, String(take!(io)))
        end
    end
    return
end
