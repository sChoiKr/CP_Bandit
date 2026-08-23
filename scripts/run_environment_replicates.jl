#!/usr/bin/env julia

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
    default_seeds = smoke ? "20260824,20260825" : "20260824,20260825,20260826,20260827,20260828"
    seeds = parse.(Int, split(argument_value(args, "--seeds", default_seeds), ','))
    output_dir = argument_value(
        args,
        "--output",
        joinpath(dirname(@__DIR__), "results", smoke ? "replicate_smoke" : "environment_replicates"),
    )

    simulation = smoke ?
        SimulationConfig(
            n_agents = 40,
            n_trials = 40,
            reversal_block = 10,
            parameter_seed = 20260823,
            environment_seed = first(seeds),
            choice_seed = 20260825,
        ) :
        SimulationConfig(environment_seed = first(seeds))
    experiment = ExperimentConfig(
        simulation = simulation,
        n_train = smoke ? 30 : 400,
        ranks = smoke ? [2] : [2, 3, 4, 5, 6],
        starts = smoke ? 1 : 10,
        cpd_maxiter = smoke ? 50 : 300,
        oracle = oracle,
    )

    println("condition: ", oracle ? "oracle (observable + Q)" : "observable only")
    println("environment seeds: ", join(seeds, ", "))
    result = run_environment_replicates(experiment, seeds)
    write_replicate_results(result, output_dir)
    println("results: ", abspath(output_dir))
end

main(ARGS)
