using BanditCPPipeline

function argument_value(args, flag, default)
    index = findfirst(==(flag), args)
    index === nothing && return default
    index < length(args) || error("$flag requires a value")
    return args[index + 1]
end

function main(args)
    smoke = "--smoke" in args
    oracle = "--oracle" in args
    output_dir = argument_value(
        args,
        "--output",
        joinpath(dirname(@__DIR__), "results", smoke ? "smoke" : (oracle ? "pilot_oracle" : "pilot_observable")),
    )

    simulation = smoke ?
        SimulationConfig(
            n_agents = 40,
            n_trials = 40,
            reversal_block = 10,
            parameter_seed = 20260823,
            environment_seed = 20260824,
            choice_seed = 20260825,
        ) :
        SimulationConfig()
    experiment = ExperimentConfig(
        simulation = simulation,
        n_train = smoke ? 30 : 400,
        ranks = smoke ? [2, 3] : [2, 3, 4, 5, 6],
        starts = smoke ? 2 : 10,
        cpd_maxiter = smoke ? 50 : 500,
        oracle = oracle,
    )

    println("condition: ", oracle ? "oracle (observable + Q)" : "observable only")
    println("agents/trials: ", simulation.n_agents, "/", simulation.n_trials)
    result = run_pipeline(experiment)
    write_results(result, output_dir)
    println("results: ", abspath(output_dir))
end

main(ARGS)
