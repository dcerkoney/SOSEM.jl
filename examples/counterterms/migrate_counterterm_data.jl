using CodecZlib
using ElectronLiquid
using JLD2
using SOSEM

# Union of all UEG MC measurement types
const UEG_MC_MeasType =
    Union{UEG_MC.MeasType,UEG_MC.MergedMeasType,UEG_MC.RenormMeasType,UEG_MC.TotalMeasType}

function migrate_counterterm_data(
    filename_to_migrate,
    filename;
    save=false,
    is_mass_ratio=false,
    raw_data=true,
)
    # The counterterm data is either a z/μ, sigma(k), or mass ratio measurement.
    local target_ngrid
    if is_mass_ratio
        # For mass ratio measurements, only migrate data with
        # ngrid = [0] and Taylor factors to the new archive.
        target_ngrid = [0]
    else
        # For z/μ and sigma(k) measurements, only migrate data with 
        # ngrid = [-1, 0] and Taylor factors to the new archive.
        target_ngrid = [-1, 0]
    end

    # Load data
    println("Loading archives...\n")
    data_to_migrate = load(filename_to_migrate)
    data = load(filename)
    if save
        println("Migrating data from $filename_to_migrate to $filename...\n")
    else
        println("Will migrate data from $filename_to_migrate to $filename...\n")
    end
    is_missing_factors = (
        haskey(data_to_migrate, "has_taylor_factors") == false ||
        data_to_migrate["has_taylor_factors"] == false
    )
    # NOTE: Only raw data has the requirement that d["has_taylor_factors"] == true.
    if is_missing_factors && raw_data
        println("No migratable data found in $(filename_to_migrate)!")
        return
    end

    # Migrate data
    for k in keys(data_to_migrate)
        k == "has_taylor_factors" && continue
        ngrid_found = data_to_migrate[k][2]
        if ngrid_found == target_ngrid
            if k ∉ keys(data)
                # Add missing data
                print("Found a missing key...")
                if save
                    println("adding data with key $k to archive...\n")
                    data[k] = data_to_migrate[k]
                else
                    println("will add data with key $k to archive...\n")
                end
            elseif k ∈ keys(data)
                # Look for upgradable data (skip if the max order is smaller than existing)
                para_to_migrate = data_to_migrate[k][1]
                para = data[k][1]
                if para_to_migrate.order < para.order
                    continue
                end
                order = para.order

                # Sum total measurement error in old archive
                meas_to_migrate = data_to_migrate[k][end]
                total_old = zero(first(values(meas_to_migrate)))
                if meas_to_migrate isa UEG_MC_MeasType
                    # raw measurement, sum over partitions
                    for (P, v) in meas_to_migrate
                        if sum(P) ≤ order
                            total_old += v
                        end
                    end
                elseif meas_to_migrate isa AbstractVector
                    # processed data, sum over orders
                    total_old = sum(meas_to_migrate[1:order])
                end
                total_old = sum(total_old)
                total_err_old = abs(total_old).err

                # Sum total measurement error in new archive
                meas = data[k][end]
                total_new = zero(first(values(meas)))
                if meas isa UEG_MC_MeasType
                    # raw measurement, sum over partitions
                    for (P, v) in meas
                        if sum(P) ≤ order
                            total_new += v
                        end
                    end
                elseif meas isa AbstractVector
                    # processed data, sum over orders
                    total_new = sum(meas[1:order])
                end
                total_new = sum(total_new)
                total_err_new = abs(total_new).err

                # Upgrade data if total measurement error is smaller
                if total_err_old < total_err_new
                    if save
                        println("Upgrading data with key $k in archive...\n")
                        data[k] = data_to_migrate[k]
                    else
                        println("Will upgrade data with key $k in archive...\n")
                    end
                end
            end
        end
    end

    # Save data
    if save
        # Backup the archive
        suffix = 0
        backup_name = "$(filename).bak"
        if isfile(backup_name)
            while isfile(backup_name)
                suffix += 1
                backup_name = "$(filename).bak$(suffix)"
            end
        end
        cp(filename, backup_name)
        # Update the archive
        println("Saving data...\n")
        jldopen(filename, "w"; compress=true) do f
            for (k, v) in data
                f[k] = v
            end
        end
    end
    println("Done!")
    return
end
