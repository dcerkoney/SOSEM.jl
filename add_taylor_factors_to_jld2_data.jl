using CodecZlib
using ElectronLiquid
using JLD2
using SOSEM

# Adds the Taylor factors 1 / (n! m!) to the JLD2 data files in
# the 'results/data' and 'examples/counterterms/data' directories.
# This is necessary because the Taylor factors were not included in the
# data files when they were originally saved. This function should only
# be run once, and only on the original data files. It should not be run
# on the data files that have already been updated by this function.
function main()
    # Log changes to file 'add_taylor_factors_to_jld2_data.txt'
    open("add_taylor_factors_to_jld2_data.txt", "r") do f

        # Load all JLD2 data in 'results/data' directory
        filelist1 = readdir("results/data")
        backup_dir1 = "results/data/before_taylor_factors/"
        mkdir(backup_dir1)
        for filename in filelist1
            @assert filename isa String
            if endswith(filename, ".jld2")
                print.([f, stdout], "Updating file '$filename'...")
                # First, make a backup of the file
                cp(filename, filename * ".bak")
                mv(filename, backup_dir1 * filename * ".bak")

                # Then, load the data, add the taylor factors, and resave the data
                jld2_archive = load(filename; compress=true)
                for (key, data) in jld2_archive
                    @assert data isa Dict

                    # Add a tag indicating that the original data does not include the Taylor factors
                    data["has_taylor_factors"] = false

                    # TODO: Only do this step when we find an entry in data that looks like a MeasType!
                    @todo
                    data_with_taylor_factors = typeof(data)()
                    # Include the Taylor factors 1 / (n! m!) in the data
                    # jldopen(filename, "w"; compress=true) do data_with_taylor_factors
                    #     for (k, v) in data
                    #         if k isa MeasType
                    #             data_with_taylor_factors[k] = v / (factorial(k[2]) * factorial(k[3]))
                    #         end
                    #         # Add a tag indicating that the data includes the Taylor factors
                    #         data_with_taylor_factors["has_taylor_factors"] = true
                    #     end
                    # end
                end
            end
            # Save the data
            save(filename, data)
            println.([f, stdout], "done.")
        end

        # Load all JLD2 data in 'examples/counterterms/data' directory
        filelist2 = readdir("examples/counterterms/data")
        backup_dir2 = "examples/counterterms/data/before_taylor_factors/"
        mkdir(backup_dir2)
        for filename in filelist2
            @assert filename isa String
            if endswith(filename, ".jld2")
                print.([f, stdout], "Updating file '$filename'...")
                # First, make a backup of the file
                cp(filename, filename * ".bak")
                mv(filename, backup_dir2 * filename * ".bak")

                # Then, load the data, add the taylor factors, and resave the data
                jld2_archive = load(filename; compress=true)
                for (key, data) in jld2_archive
                    @assert data isa Dict

                    # Add a tag indicating that the original data does not include the Taylor factors
                    data["has_taylor_factors"] = false

                    # TODO: Only do this step when we find an entry in data that looks like a MeasType!
                    @todo
                    data_with_taylor_factors = typeof(data)()
                    # Include the Taylor factors 1 / (n! m!) in the data
                    # jldopen(filename, "w"; compress=true) do data_with_taylor_factors
                    #     for (k, v) in data
                    #         if k isa MeasType
                    #             data_with_taylor_factors[k] = v / (factorial(k[2]) * factorial(k[3]))
                    #         end
                    #     end
                    # end
                end
            end
            # Save the data
            save(filename, data)
            println.([f, stdout], "done.")
        end
    end
end

main()