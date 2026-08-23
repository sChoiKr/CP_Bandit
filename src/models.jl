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
