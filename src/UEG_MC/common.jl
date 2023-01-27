"""Convert MCIntegration results for partitions {P} to a Dict of measurements."""
function restodict(res, partitions::Vector{PartitionType})
    data = Dict()
    if length(partitions) == 1
        data[partitions[1]] = measurement.(res.mean, res.stdev)
    else
        for (i, p) in enumerate(partitions)
            data[p] = measurement.(res.mean[i], res.stdev[i])
        end
    end
    return data
end

function restodict(res_list::Vector{Result}, partitions_list::Vector{Vector{PartitionType}})
    @assert length(res_list) == length(partitions_list)
    data = Dict()
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

"""Load JLD2 data from multiple fixed orders."""
function load_fixed_order_data_jld2(
    filenames,
    plotsettings::Settings,
    plotparams::Vector{UEG.ParaMC},
)
    # TODO: Refactor---what is the cleanest way to merge the data? 
    #       Should we compose the MCIntegration.Result structs?
    @assert length(filenames) == length(plotparams)

    # ParaMC fields for which we require equality between the different JLD2 data
    required_param = [:rs, :beta, :mass2, :isDynamic]

    # Settings fields for which we require equality between the different JLD2 data
    required_settings = [:filter, :interaction, :expand_bare_interactions]

    # Merge JDL2 data from different fixed-order calculations
    local settings, params, kgrid
    res_list = Vector{Result}()
    partitions_list = Vector{Vector{PartitionType}}()
    for (i, f) in enumerate(filenames)
        _settings, _params, _kgrid, _partitions, _res = jldopen("$f.jld2", "a+") do f
            key = "$(UEG.short(plotparams[i]))"
            return f[key]
        end
        # TODO: relax requirement that kgrid is the same for all data
        if i == 1
            kgrid = _kgrid
        else
            @assert kgrid ≈ _kgrid
        end
        # Make sure that required settings are the same for all data
        for field in required_settings
            data_setting = getfield(_settings, field)
            plot_setting = getfield(plotsettings, field)
            if data_setting != plot_setting
                @warn (
                    "Skipping file '$f': DiagGen settings" *
                    " differ from current plot settings:"
                ) maxlog = 1
                @warn (
                    " • Data setting: $data_setting" * "\n • Plot setting: $plot_setting"
                )
            end
        end
        # Make sure that required parameters are the same for all data
        for field in required_param
            data_param = getfield(_params, field)
            plot_param = getfield(plotparams[i], field)
            if data_param != plot_param
                @warn ("Skipping file '$f': MC params differ from current plot params:") maxlog =
                    1
                @warn (" • Data param: $data_param" * "\n • Plot param: $plot_param")
            end
        end
        # Pick param & settings from max order
        if i == length(filenames)
            params = _params
            settings = _settings
        end
        # Add the current results and partitions to the lists
        push!(res_list, _res)
        push!(partitions_list, _partitions)
    end
    return (settings, params, kgrid, partitions_list, res_list)
end

"""Load JLD2 data from a single fixed order."""
function load_fixed_order_data_jld2(filename, plotsettings::Settings, plotparam::UEG.ParaMC)
    return load_fixed_order_data_jld2([filename], plotsettings, [plotparam])
end

"""
Aggregate the measurements for C⁽¹ᶜ⁾ up to order N for nmin ≤ N ≤ nmax.
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
Aggregate the measurements for C⁽¹ᶜ⁾ up to order N for nmin ≤ N ≤ nmax.
Assumes the input data has been merged by interaction order and
reexpanded in μ.
"""
function aggregate_orders(data::RenormMeasType)
    return TotalMeasType(zip(keys(data), accumulate(+, values(data))))
end
function aggregate_orders(renorm_data::Vector{Vector{Measurement}}; nmax, nmin=2)
    # TODO: deprecate this method (chemicalpotential_renormalization_sosem return type updated)
    @todo
    # merged data is an ordered vector of data at each order nmin ≤ n ≤ nmax
    imin = nmin - 1
    imax = nmax - 1
    return TotalMeasType(
        zip(nmin:nmax, accumulate(+, renorm_data[i] for i in collect(imin:imax))),
    )
end
