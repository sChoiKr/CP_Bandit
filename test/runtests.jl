using BanditCPPipeline
using LinearAlgebra
using Test

@testset "shared-environment Q-learning simulation" begin
    cfg = SimulationConfig(
        n_agents = 12,
        n_trials = 20,
        reversal_block = 5,
        parameter_seed = 11,
        environment_seed = 12,
        choice_seed = 13,
    )
    first_run = simulate_bandits(cfg)
    second_run = simulate_bandits(cfg)

    @test validate_dataset(first_run)
    @test first_run.X_obs == second_run.X_obs
    @test first_run.alpha == second_run.alpha
    @test size(observable_tensor(first_run)) == (12, 20, 4)
    @test size(oracle_tensor(first_run)) == (12, 20, 6)
    @test all(first_run.alpha .>= 0.05)
    @test all(first_run.alpha .<= 0.95)
    @test all(first_run.beta .>= 0.5)
    @test all(first_run.beta .<= 8.0)

    changed_environment = simulate_bandits(SimulationConfig(
        n_agents = 12,
        n_trials = 20,
        reversal_block = 5,
        parameter_seed = 11,
        environment_seed = 99,
        choice_seed = 13,
    ))
    @test changed_environment.alpha == first_run.alpha
    @test changed_environment.beta == first_run.beta
    @test changed_environment.potential_rewards != first_run.potential_rewards

    changed_parameters = simulate_bandits(SimulationConfig(
        n_agents = 12,
        n_trials = 20,
        reversal_block = 5,
        parameter_seed = 98,
        environment_seed = 12,
        choice_seed = 13,
    ))
    @test changed_parameters.alpha != first_run.alpha
    @test changed_parameters.potential_rewards == first_run.potential_rewards

    for i in axes(first_run.choices, 1), t in axes(first_run.choices, 2)
        arm = first_run.choices[i, t]
        @test first_run.obtained_rewards[i, t] == first_run.potential_rewards[t, arm]
    end
end

@testset "split and flatten contracts" begin
    train, test = split_agents(20, 15; seed = 3)
    @test length(train) == 15
    @test length(test) == 5
    @test isempty(intersect(train, test))
    @test sort(vcat(train, test)) == 1:20

    X = reshape(collect(1.0:48.0), 4, 3, 4)
    flat = flatten_agents(X)
    @test size(flat) == (4, 12)
    @test flat[2, :] == vec(Array(@view X[2, :, :]))
end

@testset "environment replicate summaries" begin
    base = SimulationConfig(parameter_seed = 1, environment_seed = 2, choice_seed = 3)
    changed = with_environment_seed(base, 99)
    @test changed.parameter_seed == base.parameter_seed
    @test changed.choice_seed == base.choice_seed
    @test changed.environment_seed == 99

    rows = [
        (environment_seed = 1, rank = 2, method = "CPD", target = "alpha", r2 = 0.4, mae = 0.2, rmse = 0.3),
        (environment_seed = 2, rank = 2, method = "CPD", target = "alpha", r2 = 0.6, mae = 0.4, rmse = 0.5),
    ]
    summary = only(summarize_metric_rows(rows))
    @test summary.n_environment_seeds == 2
    @test summary.mean_r2 ≈ 0.5
    @test summary.mean_mae ≈ 0.3
    @test summary.sd_r2 ≈ sqrt(0.02)
    @test summary.ci95_low_r2 < summary.mean_r2 < summary.ci95_high_r2
end

@testset "out-of-sample CP coordinate convention" begin
    T, F, R, N = 7, 4, 3, 5
    B = randn(T, R)
    C = randn(F, R)
    Z = randn(N, R)
    X = zeros(N, T, F)
    for i in 1:N, r in 1:R
        X[i, :, :] .+= Z[i, r] .* (B[:, r] * C[:, r]')
    end
    recovered = project_agents(X, B, C)
    @test recovered ≈ Z atol = 1e-9 rtol = 1e-9
    diagnostics = projection_diagnostics(B, C)
    @test diagnostics.basis_rows == T * F
    @test diagnostics.basis_columns == R
    @test diagnostics.basis_rank == R
    @test isfinite(diagnostics.condition_number)
    regularized = project_agents(X, B, C; relative_penalty = 1e-4)
    @test size(regularized) == size(Z)
    @test all(isfinite, regularized)
    @test_throws ArgumentError project_agents(X, B, C; relative_penalty = -1.0)
end

@testset "end-to-end smoke pipeline" begin
    cfg = ExperimentConfig(
        simulation = SimulationConfig(
            n_agents = 24,
            n_trials = 20,
            reversal_block = 5,
            parameter_seed = 19,
            environment_seed = 20,
            choice_seed = 21,
        ),
        n_train = 18,
        ranks = [2],
        starts = 1,
        cpd_maxiter = 5,
    )
    result = run_pipeline(cfg)
    @test length(result.metrics) == 6
    @test length(result.stability) == 1
    @test length(result.identifiability) == 1
    @test length(result.projection) == 1
    @test length(result.projection_sensitivity) == 10
    @test haskey(result.best_fits, 2)
    @test all(row -> isfinite(row.mae) && isfinite(row.rmse), result.metrics)

    mktempdir() do directory
        write_results(result, directory)
        @test isfile(joinpath(directory, "rank_metrics.csv"))
        @test isfile(joinpath(directory, "stability.csv"))
        @test isfile(joinpath(directory, "identifiability.csv"))
        @test isfile(joinpath(directory, "projection_conditioning.csv"))
        @test isfile(joinpath(directory, "projection_sensitivity.csv"))
        @test isfile(joinpath(directory, "run.jls"))
    end
end
