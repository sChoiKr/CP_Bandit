using TensorKitchen, LinearAlgebra, DelimitedFiles, Printf, Random
const ROOT=joinpath(@__DIR__,"results","matched")
function readtensor(path,n,t)
    b=read(path)
    length(b)==n*t*4*8 || error("binary dimensions: $path")
    return reshape(copy(reinterpret(Float64,b)),n,t,4)
end
function start_factors(folder,task,rank,start,nn)
    k=nn ? "nn" : "cp"
    U=[Matrix{Float64}(readdlm(joinpath(folder,
        "$(task)_init_$(k)_s$(start)_factor$(i).csv"),',',Float64)) for i=1:3]
    @assert all(size(u,2)==rank for u in U)
    @assert all(u -> all(isfinite, u), U)
    return U
end
function fit(X,R,U,nn,maxiter)
    return TensorKitchen.fit_cp_als(X,R;
        init_factors=(ones(Float64,R),U),
        nonnegative=nn,nn_update=(nn ? :hals : :ls),
        normalization=:none,
        maxiter=maxiter,tol=0.0,miniter=0,
        verbose=false,return_stats=true)
end
function main()
    lo=parse(Int,ENV["RW_ENV_START"]);hi=parse(Int,ENV["RW_ENV_END"])
    # Warm up both solver families with explicit common factors.
    small=rand(MersenneTwister(1),10,12,4)
    for nn in (false,true)
        U=[rand(MersenneTwister(i),d,3).+0.1 for (i,d) in enumerate(size(small))]
        fit(small,3,U,nn,2)
    end
    println("WARMUP_COMPLETED");flush(stdout)
    for env in lo:hi-1, R in (3,5,8)
        folder=joinpath(ROOT,@sprintf("env_%02d",env),"rank_$(R)")
        open(joinpath(folder,"tensorkitchen_fits.csv"),"w") do io
            println(io,"task,method,start,train_rel_error,iterations,fit_seconds,converged")
            for (task,n) in (("phen",320),("cont",400))
                X=readtensor(joinpath(folder,task*"_train.f64"),n,150)
                for (method,nn) in (("TensorKitchen",false),("TensorKitchen_NN",true))
                    for start in 0:2
                        U=start_factors(folder,task,R,start,nn)
                        t0=time_ns()
                        result=fit(X,R,U,nn,240)
                        sec=(time_ns()-t0)*1.0e-9
                        prefix=joinpath(folder,"$(task)_$(method)_s$(start)")
                        writedlm(prefix*"_weights.csv",result.weights,',')
                        for mode in 1:3
                            writedlm(prefix*"_factor$(mode).csv",result.factors[mode],',')
                        end
                        @printf(io,"%s,%s,%d,%.14g,%d,%.9f,%s\n",
                            task,method,start,result.rel_error,result.iterations,sec,result.converged)
                        flush(io)
                        @printf("FITTED env=%d rank=%d %s %s start=%d error=%.8f seconds=%.3f\n",
                            env,R,task,method,start,result.rel_error,sec)
                        flush(stdout)
                    end
                end
            end
        end
        @printf("JULIA_COMPLETE env=%d rank=%d\n",env,R);flush(stdout)
    end
end
main()
