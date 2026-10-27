using TensorKitchen, LinearAlgebra, Random, DelimitedFiles, Printf
const ROOT=joinpath(@__DIR__,"results",get(ENV,"BENCH_MODE","smoke"))
function loadX(path,n,t)
    x=reinterpret(Float64,read(path)); length(x)==n*t*4 || error("bad binary shape")
    reshape(copy(x),n,t,4)
end
function main()
    meta=read(joinpath(ROOT,"settings.json"),String)
    # Pass shapes through environment to avoid adding JSON dependency.
    T=parse(Int,ENV["BENCH_T"]);R=parse(Int,ENV["BENCH_R"])
    starts=parse(Int,ENV["BENCH_STARTS"]);iter=parse(Int,ENV["BENCH_MAXITER"])
    sizes=(("phen",parse(Int,ENV["BENCH_PHEN_TRAIN"])),("cont",parse(Int,ENV["BENCH_CONT_TRAIN"])))
    mkpath(ROOT)
    open(joinpath(ROOT,"tensorkitchen_fits.csv"),"w") do io
        println(io,"task,method,start,train_rel_error,iterations,fit_seconds,converged")
        for (task,n) in sizes
            X=loadX(joinpath(ROOT,task*"_train.f64"),n,T)
            for (method,nonneg) in (("TensorKitchen",false),("TensorKitchen_NN",true))
                for s in 0:starts-1
                    seed=(task=="phen" ? 31001+R*991 : 12701+R*911)+199*s
                    Random.seed!(seed)
                    t0=time_ns()
                    res=nonneg ?
                        nncpd(X,R;solver=:als,init=:random,geometry=:canonical,nn_update=:hals,maxiter=iter,tol=1e-7,verbose=false) :
                        cpd(X,R;solver=:als,init=:random,maxiter=iter,tol=1e-7,verbose=false)
                    secs=(time_ns()-t0)/1e9
                    w=weights(res);U=factors(res)
                    base=joinpath(ROOT,task*"_"*method*"_s"*string(s))
                    writedlm(base*"_weights.csv",w,',')
                    for k in 1:3
                        writedlm(base*"_factor"*string(k)*".csv",U[k],',')
                    end
                    @printf(io,"%s,%s,%d,%.14g,%d,%.8f,%s\n",task,method,s,rel_error(res),iterations(res),secs,string(converged(res)))
                    flush(io)
                    @printf("%s %s start=%d error=%.6f iters=%d sec=%.2f\n",task,method,s,rel_error(res),iterations(res),secs)
                end
            end
        end
    end
end
main()
