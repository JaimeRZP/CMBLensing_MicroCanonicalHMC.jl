
struct HDF5Chain
    file
    dataset
    buffer
    chunk::Int
    n::Ref{Int}
end

"""
    $(TYPEDSIGNATURES)
    $(FUNCTIONNAME)(...) do chain_file
        ...
    end

Create a disk-backed chain object that you can `push!` samples to. The
do-block version calls `close` automatically when the block exits.
"""
function HDF5Chain(file_name::String, sample_length, num_steps, T, chunk)
    file = h5open(file_name, "w")
    dataset = create_dataset(file, "samples", datatype(Float64), (sample_length, num_steps))
    dataset[:,:] = NaN
    buffer = Array{T}(undef, sample_length, chunk)
    return HDF5Chain(file, dataset, buffer, chunk, 1)
end

write_chain(func::Base.Callable, file_name::Nothing, args...; kwargs...) = func(nothing)
function write_chain(func::Base.Callable, file_name::String, args...; kwargs...)
    chain_file = HDF5Chain(file_name, args...; kwargs...)
    try
        return func(chain_file)
    finally
        close(chain_file)
    end
end

function Base.push!(chain::HDF5Chain, sample::AbstractVector)
    (;dataset, buffer, n, chunk) = chain
    buffer[:, (n[]-1) % chunk + 1] = adapt(Array, sample)
    if n[] % chunk == 0
        dataset[:, n[]-chunk+1:n[]] = buffer
    end
    n[] += 1
    return chain
end

function Base.close(chain::HDF5Chain)
    (;dataset, buffer, n, chunk) = chain
    if n[] % chunk != 1
        dataset[:, n[]-n[] % chunk+1:n[]-1] = buffer[:, 1:n[] % chunk-1]
    end
    close(chain.file)
end

load_chain(file_name::String) = h5read(file_name, "samples")