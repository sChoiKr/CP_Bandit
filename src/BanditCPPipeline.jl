module BanditCPPipeline

using DelimitedFiles
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics
using TensorKitchen

export SimulationConfig,
    ExperimentConfig,
    BanditDataset,
    simulate_bandits,
    validate_dataset,
    observable_tensor,
    oracle_tensor,
    split_agents,
    flatten_agents,
    project_agents,
    projection_diagnostics,
    run_pipeline,
    write_results,
    with_environment_seed,
    summarize_metric_rows,
    run_environment_replicates,
    write_replicate_results

include("simulation.jl")
include("models.jl")
include("cp_analysis.jl")
include("pipeline.jl")
include("replicates.jl")

end
