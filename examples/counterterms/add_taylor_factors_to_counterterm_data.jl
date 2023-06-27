using CodecZlib
using ElectronLiquid
using JLD2
using SOSEM

"""
Adds the Taylor factors 1 / (n! m!) to the counterterm data in
the 'examples/counterterms/data' directory. This is necessary
because the Taylor factors were not included in the data files
when they were originally saved, and are now pre-processed in
function `ElectronLiquid.Sigma.diagram`. 

For simplicity, for all SOSEM data (i.e., directory 'results/data'), 
we simply keep the convention of post-processing the Taylor factors.
"""
function main()
    # Change to counterterm directory
    if haskey(ENV, "SOSEM_CEPH")
        cd("$(ENV["SOSEM_CEPH"])/examples/counterterms/data")
    elseif haskey(ENV, "SOSEM_HOME")
        cd("$(ENV["SOSEM_HOME"])/examples/counterterms/data")
    end

    # Log changes to file 'add_taylor_factors_to_counterterm_data.txt'
    open("add_taylor_factors_to_counterterm_data.txt", "a+") do f
        println.([f, stdout])
        # Load all JLD2 data in 'examples/counterterms/data' directory
        filelist2 = readdir()
        backup_dir2 = "before_taylor_factors/"
        if isdir(backup_dir2) == false
            mkdir(backup_dir2)
        end
        for filename in filelist2
            @assert filename isa String
            if startswith(filename, "data_") && endswith(filename, ".jld2")
                print.([f, stdout], "Updating file '$filename'...")

                # If this data already includes Taylor factors, skip it.
                # Otherwise, tag the data as not including Taylor factors.
                skip = false
                jldopen(filename, "a+"; compress=true) do f_old
                    if (
                        haskey(f_old, "has_taylor_factors") &&
                        f_old["has_taylor_factors"] == true
                    )
                        println.([f, stdout], "contains Taylor factors, skipping file.")
                        skip = true
                    else
                        f_old["has_taylor_factors"] = false
                    end
                    return
                end
                if skip
                    continue
                end

                # Backup the data
                print.([f, stdout], "creating backup...")
                cp(filename, filename * ".bak")
                mv(filename, backup_dir2 * filename * ".bak")

                # Add the taylor factors to data & tag
                print.([f, stdout], "adding factors...")
                jldopen(filename * ".bak", "r"; compress=true) do f_old
                    jldopen(filename, "w"; compress=true) do f_new
                        for key in keys(f_old)
                            if (f_old[key] isa Tuple) == false
                                continue
                            end
                            d_new = [f_old[key]...]
                            for (i, data) in enumerate(d_new)
                                if data isa UEG_MC.MeasType
                                    # Add updated data
                                    tmp = typeof(data)()
                                    for (k, v) in data
                                        tmp[k] = v / (factorial(k[2]) * factorial(k[3]))
                                    end
                                    # Save the updated data
                                    d_new[i] = tmp
                                end
                            end
                            f_new[key] = Tuple(d_new)
                        end
                        # Add a tag indicating that the data includes the Taylor factors
                        f_new["has_taylor_factors"] = true
                        return
                    end
                end
                println.([f, stdout], "done.")
            end
        end
    end
end

main()

# # Load all JLD2 data in 'results/data' directory
# filelist1 = readdir("results/data")
# backup_dir1 = "results/data/before_taylor_factors/"
# mkdir(backup_dir1)
# for filename in filelist1
#     @assert filename isa String
#     if endswith(filename, ".jld2")
#         print.([f, stdout], "Updating file '$filename'...")
#         # First, make a backup of the file
#         cp(filename, filename * ".bak")
#         mv(filename, backup_dir1 * filename * ".bak")

#         # Then, load the data, add the taylor factors, and resave the data
#         f_old = load(filename; compress=true)
#         for (key, data) in f_old
#             @assert data isa Dict

#             # Add a tag indicating that the original data does not include the Taylor factors
#             data["has_taylor_factors"] = false

#             # TODO: Only do this step when we find an entry in data that looks like a MeasType!
#             @todo
#             data_with_taylor_factors = typeof(data)()
#             # Include the Taylor factors 1 / (n! m!) in the data
#             # jldopen(filename, "w"; compress=true) do data_with_taylor_factors
#             #     for (k, v) in data
#             #         if k isa MeasType
#             #             data_with_taylor_factors[k] = v / (factorial(k[2]) * factorial(k[3]))
#             #         end
#             #         # Add a tag indicating that the data includes the Taylor factors
#             #         data_with_taylor_factors["has_taylor_factors"] = true
#             #     end
#             # end
#         end
#     end
#     # Save the data
#     save(filename, data)
#     println.([f, stdout], "done.")
# end
