struct RidgeModel
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
    coefficients::Vector{Float64}
    response_mean::Float64
end

function fit_ridge(X::AbstractMatrix, y::AbstractVector; penalty::Float64 = 1.0)
    penalty >= 0 || throw(ArgumentError("ridge penalty must be nonnegative"))
    size(X, 1) == length(y) || throw(DimensionMismatch("X/y row mismatch"))
    μ = vec(mean(X; dims = 1))
    σ = vec(std(X; dims = 1, corrected = false))
    σ[σ .<= sqrt(eps(Float64))] .= 1.0
    Z = (Float64.(X) .- μ') ./ σ'
    yc = Float64.(y) .- mean(y)
    n, p = size(Z)

    coefficients = if p <= n
        (Z' * Z + penalty * I) \ (Z' * yc)
    else
        Z' * ((Z * Z' + penalty * I) \ yc)
    end
    return RidgeModel(μ, σ, coefficients, mean(y))
end

function predict(model::RidgeModel, X::AbstractMatrix)
    Z = (Float64.(X) .- model.feature_mean') ./ model.feature_scale'
    return model.response_mean .+ Z * model.coefficients
end

function regression_metrics(y::AbstractVector, yhat::AbstractVector)
    residual = Float64.(y) .- Float64.(yhat)
    sse = sum(abs2, residual)
    centered = Float64.(y) .- mean(y)
    sst = sum(abs2, centered)
    return (
        r2 = sst > 0 ? 1 - sse / sst : NaN,
        mae = mean(abs, residual),
        rmse = sqrt(mean(abs2, residual)),
    )
end

function fit_and_score(
    train_X::AbstractMatrix,
    test_X::AbstractMatrix,
    train_y::AbstractVector,
    test_y::AbstractVector;
    penalty::Float64 = 1.0,
)
    model = fit_ridge(train_X, train_y; penalty)
    return regression_metrics(test_y, predict(model, test_X))
end

"""
Select one ridge penalty per response using training-only K-fold CV.

The fold SVD is shared across all candidate penalties and response columns,
which keeps the high-dimensional Raw baseline computationally practical.
"""
function fit_ridge_targets_cv(
    train_X::AbstractMatrix,
    test_X::AbstractMatrix,
    train_Y::AbstractMatrix,
    test_Y::AbstractMatrix;
    penalties::AbstractVector{<:Real},
    folds::Int = 5,
    seed::Int = 20260827,
)
    n, p = size(train_X)
    size(train_Y, 1) == n || throw(DimensionMismatch("train X/Y row mismatch"))
    size(test_X, 1) == size(test_Y, 1) ||
        throw(DimensionMismatch("test X/Y row mismatch"))
    size(train_Y, 2) == size(test_Y, 2) ||
        throw(DimensionMismatch("train/test response count mismatch"))
    2 <= folds <= n || throw(ArgumentError("folds must lie in 2:n"))
    isempty(penalties) && throw(ArgumentError("at least one ridge penalty is required"))
    all(penalty -> penalty >= 0, penalties) ||
        throw(ArgumentError("ridge penalties must be nonnegative"))

    candidates = sort(unique(Float64.(penalties)))
    n_targets = size(train_Y, 2)
    squared_errors = zeros(Float64, length(candidates), n_targets)
    order = randperm(MersenneTwister(seed), n)
    fold_id = zeros(Int, n)
    for (position, row) in enumerate(order)
        fold_id[row] = mod1(position, folds)
    end

    for fold in 1:folds
        validation = findall(==(fold), fold_id)
        fitting = findall(!=(fold), fold_id)
        X_fit = Float64.(train_X[fitting, :])
        X_validation = Float64.(train_X[validation, :])
        Y_fit = Float64.(train_Y[fitting, :])
        Y_validation = Float64.(train_Y[validation, :])

        feature_mean = vec(mean(X_fit; dims = 1))
        feature_scale = vec(std(X_fit; dims = 1, corrected = false))
        feature_scale[feature_scale .<= sqrt(eps(Float64))] .= 1.0
        Z_fit = (X_fit .- feature_mean') ./ feature_scale'
        Z_validation = (X_validation .- feature_mean') ./ feature_scale'
        response_mean = vec(mean(Y_fit; dims = 1))
        Y_centered = Y_fit .- response_mean'

        decomposition = svd(Z_fit; full = false)
        projected_response = decomposition.U' * Y_centered
        for (penalty_index, penalty) in enumerate(candidates)
            shrinkage = decomposition.S ./ (decomposition.S .^ 2 .+ penalty)
            coefficients = decomposition.V * (shrinkage .* projected_response)
            predictions = response_mean' .+ Z_validation * coefficients
            squared_errors[penalty_index, :] .+=
                vec(sum(abs2, Y_validation .- predictions; dims = 1))
        end
    end

    selected_indices = [argmin(@view squared_errors[:, target]) for target in 1:n_targets]
    selected_penalties = candidates[selected_indices]
    cv_mse = [squared_errors[selected_indices[target], target] / n for target in 1:n_targets]
    models = [
        fit_ridge(train_X, @view(train_Y[:, target]); penalty = selected_penalties[target]) for
        target in 1:n_targets
    ]
    metrics = [
        regression_metrics(
            @view(test_Y[:, target]),
            predict(models[target], test_X),
        ) for target in 1:n_targets
    ]
    return (;
        models,
        metrics,
        selected_penalties,
        cv_mse,
        candidates,
        folds,
    )
end

struct PCAModel
    mean::Vector{Float64}
    loadings::Matrix{Float64}
end

function fit_pca(X::AbstractMatrix, rank::Int)
    1 <= rank <= min(size(X)...) || throw(ArgumentError("invalid PCA rank"))
    μ = vec(mean(X; dims = 1))
    centered = Float64.(X) .- μ'
    decomposition = svd(centered; full = false)
    return PCAModel(μ, decomposition.V[:, 1:rank])
end

transform(model::PCAModel, X::AbstractMatrix) =
    (Float64.(X) .- model.mean') * model.loadings
