Base.@kwdef struct ExperimentConfig
    simulation::SimulationConfig = SimulationConfig()
    n_train::Int = 400
    ranks::Vector{Int} = [2, 3, 4, 5, 6]
    starts::Int = 10
    cpd_maxiter::Int = 300
    cpd_tolerance::Float64 = 1e-6
    cpd_solver::Symbol = :als
    ridge_penalty::Float64 = 1.0
    projection_penalties::Vector{Float64} = [0.0, 1e-8, 1e-6, 1e-4, 1e-2]
    split_seed::Int = 20260824
    cpd_seed::Int = 20260826
    oracle::Bool = false
end

function _metric_row(rank, method, target, metric, dimension, original_dimension, recon)
    return (
        rank = rank,
        method = method,
        target = target,
        r2 = metric.r2,
        mae = metric.mae,
        rmse = metric.rmse,
        representation_dimension = dimension,
        original_dimension = original_dimension,
        compression_ratio = original_dimension / dimension,
        reconstruction_error = recon,
    )
end

function _target_scores(train_Z, test_Z, data, train_idx, test_idx, penalty)
    alpha_metric = fit_and_score(
        train_Z,
        test_Z,
        data.alpha[train_idx],
        data.alpha[test_idx];
        penalty,
    )
    beta_metric = fit_and_score(
        train_Z,
        test_Z,
        data.beta[train_idx],
        data.beta[test_idx];
        penalty,
    )
    return alpha_metric, beta_metric
end

"""Run simulation, inductive CPD recovery, PCA/raw baselines, and diagnostics."""
function run_pipeline(cfg::ExperimentConfig = ExperimentConfig())
    N = cfg.simulation.n_agents
    1 <= cfg.n_train < N || throw(ArgumentError("n_train must be less than n_agents"))
    isempty(cfg.ranks) && throw(ArgumentError("at least one rank is required"))
    all(>=(0.0), cfg.projection_penalties) ||
        throw(ArgumentError("projection penalties must be nonnegative"))
    0.0 in cfg.projection_penalties ||
        throw(ArgumentError("projection penalties must include the unregularized value 0.0"))

    data = simulate_bandits(cfg.simulation)
    tensor = cfg.oracle ? oracle_tensor(data) : observable_tensor(data)
    train_idx, test_idx = split_agents(N, cfg.n_train; seed = cfg.split_seed)
    X_train = tensor[train_idx, :, :]
    X_test = tensor[test_idx, :, :]
    flat_train = flatten_agents(X_train)
    flat_test = flatten_agents(X_test)
    original_dimension = size(flat_train, 2)

    raw_alpha, raw_beta = _target_scores(
        flat_train,
        flat_test,
        data,
        train_idx,
        test_idx,
        cfg.ridge_penalty,
    )

    metric_rows = NamedTuple[]
    stability_rows = NamedTuple[]
    identifiability_rows = NamedTuple[]
    projection_rows = NamedTuple[]
    projection_sensitivity_rows = NamedTuple[]
    best_fits = Dict{Int,CPFit}()

    for rank in cfg.ranks
        @printf("rank %d: TensorKitchen CPD (%d starts)\n", rank, cfg.starts)
        fits = fit_cpd_multistart(
            X_train,
            rank;
            starts = cfg.starts,
            maxiter = cfg.cpd_maxiter,
            tol = cfg.cpd_tolerance,
            solver = cfg.cpd_solver,
            seed = cfg.cpd_seed,
        )
        best = first(fits)
        best_fits[rank] = best
        train_cp = training_coordinates(best)
        test_cp = project_agents(X_test, best.factors[2], best.factors[3])
        cp_alpha, cp_beta = _target_scores(
            train_cp,
            test_cp,
            data,
            train_idx,
            test_idx,
            cfg.ridge_penalty,
        )

        pca = fit_pca(flat_train, rank)
        train_pca = transform(pca, flat_train)
        test_pca = transform(pca, flat_test)
        pca_alpha, pca_beta = _target_scores(
            train_pca,
            test_pca,
            data,
            train_idx,
            test_idx,
            cfg.ridge_penalty,
        )

        for (method, alpha_metric, beta_metric, dimension, reconstruction) in [
            ("CPD", cp_alpha, cp_beta, rank, best.rel_error),
            ("PCA", pca_alpha, pca_beta, rank, NaN),
            ("Raw", raw_alpha, raw_beta, original_dimension, NaN),
        ]
            push!(metric_rows, _metric_row(rank, method, "alpha", alpha_metric, dimension, original_dimension, reconstruction))
            push!(metric_rows, _metric_row(rank, method, "beta", beta_metric, dimension, original_dimension, reconstruction))
        end

        stability = stability_summary(fits)
        push!(stability_rows, merge((rank = rank,), stability))
        identifiability = identifiability_summary(best)
        push!(identifiability_rows, merge((rank = rank,), identifiability))
        projection = projection_diagnostics(best.factors[2], best.factors[3])
        push!(projection_rows, merge((rank = rank,), projection))
        for relative_penalty in cfg.projection_penalties
            penalized_test_cp = project_agents(
                X_test,
                best.factors[2],
                best.factors[3];
                relative_penalty,
            )
            penalty_alpha, penalty_beta = _target_scores(
                train_cp,
                penalized_test_cp,
                data,
                train_idx,
                test_idx,
                cfg.ridge_penalty,
            )
            for (target, metric) in (("alpha", penalty_alpha), ("beta", penalty_beta))
                push!(projection_sensitivity_rows, (
                    rank = rank,
                    target = target,
                    relative_penalty = relative_penalty,
                    r2 = metric.r2,
                    mae = metric.mae,
                    rmse = metric.rmse,
                    condition_number = projection.condition_number,
                ))
            end
        end
    end

    return (
        config = cfg,
        data,
        train_indices = train_idx,
        test_indices = test_idx,
        metrics = metric_rows,
        stability = stability_rows,
        identifiability = identifiability_rows,
        projection = projection_rows,
        projection_sensitivity = projection_sensitivity_rows,
        best_fits,
    )
end

function _csv_value(value)
    value isa Bool && return value ? "true" : "false"
    value isa AbstractFloat && return @sprintf("%.10g", value)
    return string(value)
end

function _write_namedtuple_csv(path::AbstractString, rows::AbstractVector)
    isempty(rows) && throw(ArgumentError("cannot write empty CSV"))
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(string.(columns), ','))
        for row in rows
            println(io, join((_csv_value(getproperty(row, column)) for column in columns), ','))
        end
    end
end

"""Write compact CSV tables plus a Julia-serialized reproducibility artifact."""
function write_results(result, output_dir::AbstractString)
    mkpath(output_dir)
    _write_namedtuple_csv(joinpath(output_dir, "rank_metrics.csv"), result.metrics)
    _write_namedtuple_csv(joinpath(output_dir, "stability.csv"), result.stability)
    _write_namedtuple_csv(joinpath(output_dir, "identifiability.csv"), result.identifiability)
    _write_namedtuple_csv(joinpath(output_dir, "projection_conditioning.csv"), result.projection)
    _write_namedtuple_csv(
        joinpath(output_dir, "projection_sensitivity.csv"),
        result.projection_sensitivity,
    )
    open(joinpath(output_dir, "run.jls"), "w") do io
        serialize(io, result)
    end
    return output_dir
end
