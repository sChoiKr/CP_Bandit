# Rank–parameter geometry: CPD dictionaries at each rank.
using TensorKitchen, Random, DelimitedFiles, Printf, LinearAlgebra

const BASE = joinpath(@__DIR__, "results", "rank_geometry")
const RANKS = (2, 3, 4, 5, 6, 8)
const NTRIALS = 150
const NTRAIN = 400
const STARTS = 3
const MAXITER = 240

function read_input(path)
    bytes = read(path)
    length(bytes) == 8 * NTRAIN * NTRIALS * 4 || error("incorrect binary tensor length: $path")
    return reshape(copy(reinterpret(Float64, bytes)), NTRAIN, NTRIALS, 4)
end

function run_fit(X, r, seed, iters)
    Random.seed!(seed)
    return cpd(X, r; solver=:als, init=:random, maxiter=iters, tol=1e-7, verbose=false)
end

function run()
    lo = parse(Int, ENV["RW_ENV_START"])
    hi = parse(Int, ENV["RW_ENV_END"])
    0 <= lo < hi <= 20 || error("invalid environment shard")
    # Exclude first-call JIT compilation from timing.
    toy = rand(MersenneTwister(22), 12, 16, 4)
    run_fit(toy, 2, 12121, 2)
    println("JULIA_WARMUP_COMPLETE")
    flush(stdout)
    for env in lo:(hi - 1)
        d = joinpath(BASE, @sprintf("env_%02d", env))
        X = read_input(joinpath(d, "train.f64"))
        open(joinpath(d, "cpd_starts.csv"), "w") do io
            println(io, "env_index,rank,start,relative_error,iterations,converged,seconds")
            for r in RANKS
                best_error = Inf
                best_result = nothing
                for start in 0:(STARTS - 1)
                    seed = 100_001 + 991*env + 101*r + 7*start
                    t0 = time_ns()
                    result = run_fit(X, r, seed, MAXITER)
                    seconds = (time_ns() - t0) / 1e9
                    e = Float64(rel_error(result))
                    @printf(io, "%d,%d,%d,%.16g,%d,%s,%.9g\n",
                            env, r, start, e, iterations(result),
                            string(converged(result)), seconds)
                    flush(io)
                    if isfinite(e) && e < best_error
                        best_error = e
                        best_result = result
                    end
                end
                isnothing(best_result) && error("all starts failed: env=$env rank=$r")
                prefix = joinpath(d, "cp_rank_$(r)")
                writedlm(prefix * "_weights.csv", weights(best_result), ',')
                U = factors(best_result)
                for m in 1:3
                    writedlm(prefix * "_factor$(m).csv", U[m], ',')
                end
                @printf("FIT env=%d rank=%d best_err=%.9g\n",env,r,best_error)
                flush(stdout)
            end
        end
    end
end
run()
