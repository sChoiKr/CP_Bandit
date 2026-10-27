using TensorKitchen, LinearAlgebra, Random, DelimitedFiles, Printf
const ROOT=joinpath(@__DIR__,"results","full")
function readtensor(path,n,t)
    bytes=read(path);length(bytes)==n*t*4*8 || error("binary tensor has wrong length")
    reshape(copy(reinterpret(Float64,bytes)),n,t,4)
end
function fit_one(X,R,method,seed,maxiter)
    Random.seed!(seed)
    if method=="TensorKitchen"
        return cpd(X,R;solver=:als,init=:random,maxiter=maxiter,tol=1e-7,verbose=false)
    end
    return nncpd(X,R;solver=:als,init=:random,geometry=:canonical,nn_update=:hals,
                 maxiter=maxiter,tol=1e-7,verbose=false)
end
function main()
    lo=parse(Int,ENV["RW_ENV_START"]);hi=parse(Int,ENV["RW_ENV_END"])
    ranks=[3,5,8];starts=3;maxiter=240;T=150
    # Crucial: execute a small fit before measurement so initial package
    # compilation is not attributed to a benchmarked configuration.
    toy=rand(MersenneTwister(101),12,15,4)
    fit_one(toy,3,"TensorKitchen",901,2)
    fit_one(toy,3,"TensorKitchen_NN",902,2)
    println("WARMUP_DONE";flush(stdout))
    for env in lo:hi-1
        for R in ranks
            folder=joinpath(ROOT,@sprintf("env_%02d",env),"rank_$(R)")
            mkpath(folder)
            open(joinpath(folder,"tensorkitchen_fits.csv"),"w") do io
                println(io,"task,method,start,train_rel_error,iterations,fit_seconds,converged")
                for task in ("phen","cont")
                    n=task=="phen" ? 320 : 400
                    X=readtensor(joinpath(folder,task*"_train.f64"),n,T)
                    for method in ("TensorKitchen","TensorKitchen_NN")
                        for start in 0:starts-1
                            seed=(task=="phen" ? 31001+R*991 : 12701+R*911)+199*start
                            t0=time_ns()
                            res=fit_one(X,R,method,seed,maxiter)
                            secs=(time_ns()-t0)/1e9
                            base=joinpath(folder,task*"_"*method*"_s"*string(start))
                            writedlm(base*"_weights.csv",weights(res),',')
                            U=factors(res)
                            for mode in 1:3
                                writedlm(base*"_factor"*string(mode)*".csv",U[mode],',')
                            end
                            @printf(io,"%s,%s,%d,%.14g,%d,%.8f,%s\n",
                                task,method,start,rel_error(res),iterations(res),secs,string(converged(res)))
                            flush(io)
                            @printf("DONE env=%d rank=%d task=%s method=%s start=%d train_error=%.6f seconds=%.3f\n",
                                env,R,task,method,start,rel_error(res),secs)
                            flush(stdout)
                        end
                    end
                end
            end
        end
    end
end
main()
