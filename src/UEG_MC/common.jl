"""Convert MCIntegration results for partitions {P} to a Dict of measurements."""
function restodict(res, partitions::Vector{PartitionType})
    N = length(partitions) == 1 ? ndims(res.mean) : ndims(res.mean[1])
    T = length(partitions) == 1 ? eltype(res.mean) : eltype(res.mean[1])
    # N = max(1, N)
    if N == 0
        S = Array{Measurement{T},1}
    else
        S = Array{Measurement{T},N}
    end
    println(N, " ", T, " ", S)
    data = MeasType{S}()
    println(data)
    if length(partitions) == 1
        if N == 0
            data[partitions[1]] = [measurement(res.mean, res.stdev)]
        else
            data[partitions[1]] = measurement.(res.mean, res.stdev)
        end
    else
        for (i, p) in enumerate(partitions)
            if N == 0
                data[p] = [measurement(res.mean[i], res.stdev[i])]
            else
                data[p] = measurement.(res.mean[i], res.stdev[i])
            end
        end
    end
    return data
end

function restodict(res_list::Vector{Result}, partitions_list::Vector{Vector{PartitionType}})
    @assert length(res_list) == length(partitions_list)
    o1 = partitions_list[1]
    N = length(partitions) == 1 ? ndims(res_list[o1].mean) : ndims(res_list[o1].mean[1])
    T = length(partitions) == 1 ? eltype(res_list[o1].mean) : eltype(res_list[o1].mean[1])
    S = Array{Measurement{T},N}
    # println(N, " ", T, " ", S)
    data = MeasType{S}()
    for o in eachindex(partitions_list)
        partitions = partitions_list[o]
        if length(partitions) == 1
            data[partitions[1]] = measurement.(res_list[o].mean, res_list[o].stdev)
        else
            for (i, p) in enumerate(partitions)
                data[p] = measurement.(res_list[o].mean[i], res_list[o].stdev[i])
            end
        end
    end
    return data
end

"""
Merge interaction order and the main order
(normal_order, G_order, W_order) --> (normal+W_order, G_order)
"""
function mergeInteraction(data)
    # nothing to merge
    if (data isa Dict && all(x -> length(x) == 3, keys(data))) == false
        return data
    end
    T = valtype(data)
    res = MergedMeasType{T}()
    for (p, val) in data
        @assert length(p) == 3
        mp = (p[1] + p[3], p[2])
        if haskey(res, mp)
            res[mp] += val
        else
            res[mp] = val
        end
    end
    return res
end

"""
Aggregate measurement partitions up to order N for nmin ≤ N ≤ nmax.
Assumes the input data has been merged by interaction order and no
chemical potential renormalization is being performed.
"""
function aggregate_orders(data::MergedMeasType; nmax, nmin=2)
    # merged data is a Dict of interaction-merged partitions P; sort by keys
    for n in nmin:nmax
        total_data[n] = zero(data[(n, 0)])
        println(n)
        for (p, meas) in data
            if sum(p) <= n
                println("adding partition $p to $n-order aggregate")
                total_data[n] += meas
            end
        end
    end
end

"""
Aggregate measurement partitions up to order N for nmin ≤ N ≤ nmax.
Assumes the input data has been merged by interaction order and reexpanded in μ.
"""
function aggregate_orders(data::RenormMeasType{T}) where {T}
    return TotalMeasType{T}(zip(keys(data), accumulate(+, values(data))))
end

"""Load non-SOSEM JLD2 data from multiple calculations of varying orders."""
function load_fixed_order_jld2_data(
    filenames,
    loadparams::Vector{UEG.ParaMC};
    # merge_orders=true,  # merge data by orders where applicable?
    has_kgrid=true,     # Obs(K, ...)
    has_tgrid=false,    # Obs(..., T)
)
    # TODO: Refactor---what is the cleanest way to merge the data? 
    #       Should we compose the MCIntegration.Result structs?
    @assert length(filenames) == length(loadparams)

    # ParaMC fields for which we require equality between the different JLD2 data
    required_param = [:rs, :beta, :mass2, :isFock, :isDynamic]

    # Load JDL2 data from different order calculations
    data_list = []
    for (i, f) in enumerate(filenames)
        data = jldopen("$f.jld2", "a+") do f
            key = "$(UEG.short(loadparams[i]))"
            return f[key]
        end
        # Validate required parameters for all data
        valid = true
        # NOTE: All JLD2 measurement data have this prefix
        _, _param, _ = data
        for field in required_param
            data_param = getfield(_param, field)
            plot_param = getfield(loadparams[i], field)
            if data_param != plot_param
                @warn (
                    "Skipping file '$f': MC params " * "differ from current plot params:"
                ) maxlog = 1
                @warn (" • Data param: $data_param" * "\n • Plot param: $plot_param")
                valid = false
                break
            end
        end
        valid == false && continue  # skip invalid data
        push!(data_list, data)
    end
    return data_list
end

"""Load non-SOSEM JLD2 data from multiple calculations of varying orders."""
function load_fixed_order_jld2_data(
    filename,
    plotparam::UEG.ParaMC;
    has_kgrid=true,     # Obs(K, ...)
    has_tgrid=false,    # Obs(..., T)
)
    return load_fixed_order_jld2_data(
        [filename],
        [plotparam];
        has_kgrid=has_kgrid,
        has_tgrid=has_tgrid,
    )
end

"""Load non-SOSEM JLD2 data from multiple calculations of varying orders."""
function merge_jld2_data(
    data_list;
    has_kgrid=true,     # Obs(K, ...)
    has_tgrid=false,    # Obs(..., T)
)
    # Load JDL2 data from different fixed-order calculations
    orders_list = []
    params = UEG.ParaMC[]
    kgrids = []
    tgrids = []
    partitions_list = Vector{PartitionType}[]
    res_list = Result[]
    for (i, data) in enumerate(data_list)
        # Unpack data depending on observable type
        if has_kgrid && has_tgrid
            _orders, _param, _grid1, _grid2, _partitions, _res = data
        elseif has_kgrid || has_tgrid
            _orders, _param, _grid1, _partitions, _res = data
        else
            _orders, _param, _partitions, _res = data
        end
        # Merge data with equivalent MC params & kgrids
        @todo
        # Add current data to lists
        push!(orders_list, _orders)
        push!(params, _param)
        if has_kgrid && has_tgrid
            push!(kgrids, _grid1)
            push!(tgrids, _grid2)
        elseif has_kgrid
            push!(kgrids, _grid1)
        elseif has_tgrid
            push!(tgrids, _grid1)
        end
        push!(partitions_list, _partitions)
        push!(res_list, _res)
    end
    # Return results
    if has_kgrid && has_tgrid
        return (orders_list, params, kgrids, tgrids, partitions_list, res_list)
    elseif has_kgrid
        return (orders_list, params, kgrids, partitions_list, res_list)
    elseif has_tgrid
        return (orders_list, params, tgrids, partitions_list, res_list)
    else
        return (orders_list, params, partitions_list, res_list)
    end
end

"""Load SOSEM JLD2 data from multiple fixed orders."""
function load_fixed_order_sosem_jld2_data(
    filenames,
    loadsettings::Settings,
    loadparams::Vector{UEG.ParaMC},
)
    # TODO: Refactor---what is the cleanest way to merge the data? 
    #       Should we compose the MCIntegration.Result structs?
    @assert length(filenames) == length(loadparams)

    # ParaMC fields for which we require equality between the different JLD2 data
    required_param = [:rs, :beta, :mass2, :isFock, :isDynamic]

    # Settings fields for which we require equality between the different JLD2 data
    required_settings = [:filter, :interaction, :expand_bare_interactions]

    # Merge JDL2 data from different fixed-order calculations
    local settings, param, kgrid
    res_list = Vector{Result}()
    partitions_list = Vector{Vector{PartitionType}}()
    for (i, f) in enumerate(filenames)
        _settings, _param, _kgrid, _partitions, _res = jldopen("$f.jld2", "a+") do f
            key = "$(UEG.short(loadparams[i]))"
            return f[key]
        end
        # NOTE: should this requirement be relaxed?
        if i == 1
            kgrid = _kgrid
        else
            @assert kgrid ≈ _kgrid
        end
        # Validate the data
        valid = true
        # Make sure that required settings are the same for all data
        for field in required_settings
            data_setting = getfield(_settings, field)
            plot_setting = getfield(loadsettings, field)
            if data_setting != plot_setting
                @warn (
                    "Skipping file '$f': DiagGen settings differ from current plot settings:"
                ) maxlog = 1
                @warn (
                    " • Data setting: $data_setting" * "\n • Plot setting: $plot_setting"
                )
                valid = false
                break
            end
        end
        # Skip the current file if it is invalid
        valid == false && continue
        # Make sure that required parameters are the same for all data
        for field in required_param
            data_param = getfield(_param, field)
            plot_param = getfield(loadparams[i], field)
            if data_param != plot_param
                @warn (
                    "Skipping file '$f': MC params " * "differ from current plot params:"
                ) maxlog = 1
                @warn (" • Data param: $data_param" * "\n • Plot param: $plot_param")
                valid = false
                break
            end
        end
        # Skip the current file if it is invalid
        valid == false && continue
        # Pick param & settings from max order
        if i == length(filenames)
            param = _param
            settings = _settings
        end
        # Add the current results and partitions to the lists
        push!(res_list, _res)
        push!(partitions_list, _partitions)
    end
    return (settings, params, kgrid, partitions_list, res_list)
end

"""Load SOSEM JLD2 data from a single fixed order."""
function load_fixed_order_sosem_jld2_data(
    filename,
    loadsettings::Settings,
    plotparam::UEG.ParaMC,
)
    return load_fixed_order_sosem_jld2_data([filename], loadsettings, [plotparam])
end