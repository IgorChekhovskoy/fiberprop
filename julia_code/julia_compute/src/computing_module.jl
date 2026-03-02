module ComputingJuliaModule

using Pkg
Pkg.activate(".")

function get_missing_packages(package_names)
    deps = Pkg.dependencies()
    results = Vector{String}()

    for pkg_name in package_names
        found = false
        for (uuid, pkg) in deps
            if pkg.name == pkg_name
                found = true
                break
            end
        end
        if !found
            push!(results, pkg_name)
        end
    end
    return results
end

necessary_packages = ["FFTW", "LinearAlgebra", "Random", "ProgressMeter", "PythonCall"]
missing_packages = get_missing_packages(necessary_packages)
if !isempty(missing_packages)
    Pkg.add(missing_packages)
end

include("general_code.jl")

end
