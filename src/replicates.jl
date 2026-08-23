"""Copy a simulation configuration while changing only environment randomness."""
function with_environment_seed(cfg::SimulationConfig, environment_seed::Int)
    return SimulationConfig(
        n_agents = cfg.n_agents,
        n_trials = cfg.n_trials,
        alpha_range = cfg.alpha_range,
        beta_range = cfg.beta_range,
        reward_probabilities = cfg.reward_probabilities,
        reversal_block = cfg.reversal_block,
        initial_q = cfg.initial_q,
        parameter_seed = cfg.parameter_seed,
        environment_seed = environment_seed,
        choice_seed = cfg.choice_seed,
    )
end

function _with_simulation(cfg::ExperimentConfig, simulation::SimulationConfig)
    return ExperimentConfig(
        simulation = simulation,
        n_train = cfg.n_train,
        ranks = copy(cfg.ranks),
        starts = cfg.starts,
        cpd_maxiter = cfg.cpd_maxiter,
        cpd_tolerance = cfg.cpd_tolerance,
        cpd_solver = cfg.cpd_solver,
        ridge_penalty = cfg.ridge_penalty,
        projection_penalties = copy(cfg.projection_penalties),
        split_seed = cfg.split_seed,
        cpd_seed = cfg.cpd_seed,
        oracle = cfg.oracle,
    )
end

function _t_critical_95(degrees_of_freedom::Int)
    degrees_of_freedom >= 1 || return NaN
    critical_values = (
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
    )
    return degrees_of_freedom <= length(critical_values) ?
           critical_values[degrees_of_freedom] : 1.96
end

function _uncertainty(values)
    numeric = Float64.(values)
    n = length(numeric)
    center = mean(numeric)
    spread = n > 1 ? std(numeric) : NaN
    standard_error = n > 1 ? spread / sqrt(n) : NaN
    critical = _t_critical_95(n - 1)
    half_width = critical * standard_error
    return (
        mean = center,
        sd = spread,
        se = standard_error,
        ci95_critical = critical,
        ci95_low = center - half_width,
        ci95_high = center + half_width,
    )
end

"""Aggregate held-out metrics over environment seeds by rank, method, and target."""
function summarize_metric_rows(rows::AbstractVector)
    isempty(rows) && throw(ArgumentError("cannot summarize empty metric rows"))
    groups = Dict{Tuple{Int,String,String},Vector{NamedTuple}}()
    for row in rows
        key = (row.rank, String(row.method), String(row.target))
        push!(get!(groups, key, NamedTuple[]), row)
    end

    summary = NamedTuple[]
    for (key, group) in groups
        r2 = _uncertainty(getproperty.(group, :r2))
        mae = _uncertainty(getproperty.(group, :mae))
        rmse = _uncertainty(getproperty.(group, :rmse))
        push!(summary, (
            rank = key[1],
            method = key[2],
            target = key[3],
            n_environment_seeds = length(group),
            mean_r2 = r2.mean,
            sd_r2 = r2.sd,
            se_r2 = r2.se,
            ci95_critical = r2.ci95_critical,
            ci95_low_r2 = r2.ci95_low,
            ci95_high_r2 = r2.ci95_high,
            mean_mae = mae.mean,
            sd_mae = mae.sd,
            mean_rmse = rmse.mean,
            sd_rmse = rmse.sd,
        ))
    end
    sort!(summary; by = row -> (row.rank, row.method, row.target))
    return summary
end

function _summarize_projection_sensitivity(rows::AbstractVector)
    groups = Dict{Tuple{Int,String,Float64},Vector{NamedTuple}}()
    for row in rows
        key = (row.rank, String(row.target), Float64(row.relative_penalty))
        push!(get!(groups, key, NamedTuple[]), row)
    end
    summary = NamedTuple[]
    for (key, group) in groups
        r2 = _uncertainty(getproperty.(group, :r2))
        push!(summary, (
            rank = key[1],
            target = key[2],
            relative_penalty = key[3],
            n_environment_seeds = length(group),
            mean_r2 = r2.mean,
            sd_r2 = r2.sd,
            se_r2 = r2.se,
            ci95_critical = r2.ci95_critical,
            ci95_low_r2 = r2.ci95_low,
            ci95_high_r2 = r2.ci95_high,
            mean_condition_number = mean(getproperty.(group, :condition_number)),
        ))
    end
    sort!(summary; by = row -> (row.rank, row.target, row.relative_penalty))
    return summary
end

"""
Run paired environment replications. Parameter, choice, split, and CPD seeds
remain fixed; only the potential-reward realization changes.
"""
function run_environment_replicates(cfg::ExperimentConfig, environment_seeds)
    seeds = collect(Int, environment_seeds)
    length(seeds) >= 2 || throw(ArgumentError("at least two environment seeds are required"))
    length(unique(seeds)) == length(seeds) ||
        throw(ArgumentError("environment seeds must be unique"))

    runs = Dict{Int,Any}()
    metric_rows = NamedTuple[]
    sensitivity_rows = NamedTuple[]
    for environment_seed in seeds
        println("environment seed: ", environment_seed)
        simulation = with_environment_seed(cfg.simulation, environment_seed)
        result = run_pipeline(_with_simulation(cfg, simulation))
        runs[environment_seed] = result
        append!(metric_rows, [merge((environment_seed = environment_seed,), row) for row in result.metrics])
        append!(
            sensitivity_rows,
            [merge((environment_seed = environment_seed,), row) for row in result.projection_sensitivity],
        )
    end

    return (
        config = cfg,
        environment_seeds = seeds,
        runs,
        metrics = metric_rows,
        metric_summary = summarize_metric_rows(metric_rows),
        projection_sensitivity = sensitivity_rows,
        projection_sensitivity_summary = _summarize_projection_sensitivity(sensitivity_rows),
    )
end

function write_replicate_results(result, output_dir::AbstractString)
    mkpath(output_dir)
    for environment_seed in result.environment_seeds
        write_results(
            result.runs[environment_seed],
            joinpath(output_dir, "environment_seed_$(environment_seed)"),
        )
    end
    _write_namedtuple_csv(joinpath(output_dir, "replicate_metrics.csv"), result.metrics)
    _write_namedtuple_csv(joinpath(output_dir, "metric_summary.csv"), result.metric_summary)
    _write_namedtuple_csv(
        joinpath(output_dir, "replicate_projection_sensitivity.csv"),
        result.projection_sensitivity,
    )
    _write_namedtuple_csv(
        joinpath(output_dir, "projection_sensitivity_summary.csv"),
        result.projection_sensitivity_summary,
    )
    return output_dir
end
