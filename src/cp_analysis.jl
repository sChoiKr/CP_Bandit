struct CPFit
    weights::Vector{Float64}
    factors::Vector{Matrix{Float64}}
    rel_error::Float64
    converged::Bool
    iterations::Int
    seed::Int
end

function fit_cpd_multistart(
    X::AbstractArray{<:AbstractFloat,3},
    rank::Int;
    starts::Int = 10,
    maxiter::Int = 300,
    tol::Float64 = 1e-6,
    solver::Symbol = :als,
    seed::Int = 20260823,
)
    starts >= 1 || throw(ArgumentError("starts must be positive"))
    fits = CPFit[]
    for start in 1:starts
        start_seed = seed + 10_000 * rank + start
        Random.seed!(start_seed)
        try
            result = TensorKitchen.cpd(
                Float64.(X),
                rank;
                init = :random,
                solver,
                maxiter,
                tol,
                verbose = false,
            )
            push!(
                fits,
                CPFit(
                    collect(Float64, TensorKitchen.weights(result)),
                    [Matrix{Float64}(factor) for factor in TensorKitchen.factors(result)],
                    Float64(TensorKitchen.rel_error(result)),
                    TensorKitchen.converged(result),
                    TensorKitchen.iterations(result),
                    start_seed,
                ),
            )
        catch error
            @warn "CPD start failed" rank start_seed exception = (error, catch_backtrace())
        end
    end
    isempty(fits) && error("all CPD starts failed for rank $rank")
    sort!(fits; by = fit -> fit.rel_error)
    return fits
end

"""
Training coordinates absorb CP weights: `Z = A * Diagonal(lambda)`.
This fixes the coordinate convention used by out-of-sample projection.
"""
training_coordinates(fit::CPFit) = fit.factors[1] .* fit.weights'

function khatri_rao_basis(B::AbstractMatrix, C::AbstractMatrix)
    size(B, 2) == size(C, 2) || throw(DimensionMismatch("factor rank mismatch"))
    return reduce(hcat, (kron(C[:, r], B[:, r]) for r in axes(B, 2)))
end

"""Numerical conditioning of the fixed out-of-sample CP basis `C ⊙ B`."""
function projection_diagnostics(B::AbstractMatrix, C::AbstractMatrix)
    basis = khatri_rao_basis(B, C)
    singular_values = svdvals(basis)
    largest = maximum(singular_values)
    smallest = minimum(singular_values)
    rank = count(>(sqrt(eps(Float64)) * largest), singular_values)
    condition = smallest > 0 ? largest / smallest : Inf
    return (
        basis_rows = size(basis, 1),
        basis_columns = size(basis, 2),
        basis_rank = rank,
        largest_singular_value = largest,
        smallest_singular_value = smallest,
        condition_number = condition,
    )
end

"""
Project unseen agent slices with fixed temporal/feature factors.

Because `vec(B[:,r] * C[:,r]') = kron(C[:,r], B[:,r])`, the returned
coordinates solve `(C ⊙ B) z ≈ vec(X_agent)` in least squares.
"""
function project_agents(
    X::AbstractArray{<:Real,3},
    B::AbstractMatrix,
    C::AbstractMatrix;
    relative_penalty::Float64 = 0.0,
)
    relative_penalty >= 0 || throw(ArgumentError("relative_penalty must be nonnegative"))
    size(X, 2) == size(B, 1) || throw(DimensionMismatch("trial mode mismatch"))
    size(X, 3) == size(C, 1) || throw(DimensionMismatch("feature mode mismatch"))
    basis = khatri_rao_basis(B, C)
    decomposition = svd(basis; full = false)
    largest_squared = max(first(decomposition.S)^2, eps(Float64))
    absolute_penalty = relative_penalty * largest_squared
    threshold = sqrt(eps(Float64)) * first(decomposition.S)
    inverse_weights = if iszero(relative_penalty)
        [value > threshold ? inv(value) : 0.0 for value in decomposition.S]
    else
        decomposition.S ./ (decomposition.S .^ 2 .+ absolute_penalty)
    end
    return flatten_agents(X) *
           decomposition.U *
           Diagonal(inverse_weights) *
           decomposition.Vt
end

@inline function _column_cosine(A, i, B, j)
    denominator = norm(@view A[:, i]) * norm(@view B[:, j])
    denominator > 0 || return 0.0
    return abs(dot(@view(A[:, i]), @view(B[:, j])) / denominator)
end

function _component_similarity(reference::CPFit, candidate::CPFit)
    rank = length(reference.weights)
    similarity = Matrix{Float64}(undef, rank, rank)
    for i in 1:rank, j in 1:rank
        product = 1.0
        for mode in 1:3
            product *= clamp(_column_cosine(reference.factors[mode], i, candidate.factors[mode], j), 0.0, 1.0)
        end
        similarity[i, j] = cbrt(product)
    end
    return similarity
end

"""Exact maximum-weight assignment by bitmask DP; intended for pilot ranks <= 10."""
function _maximum_assignment(scores::AbstractMatrix)
    n, m = size(scores)
    n == m || throw(DimensionMismatch("assignment matrix must be square"))
    n <= 20 || throw(ArgumentError("assignment helper supports rank <= 20"))
    states = 1 << n
    dp = fill(-Inf, states)
    previous = fill(-1, states)
    chosen = fill(0, states)
    dp[1] = 0.0
    for mask in 0:(states - 2)
        row = count_ones(mask) + 1
        row > n && continue
        base = dp[mask + 1]
        isfinite(base) || continue
        for column in 1:n
            bit = 1 << (column - 1)
            mask & bit == 0 || continue
            next_mask = mask | bit
            value = base + scores[row, column]
            if value > dp[next_mask + 1]
                dp[next_mask + 1] = value
                previous[next_mask + 1] = mask
                chosen[next_mask + 1] = column
            end
        end
    end
    assignment = Vector{Int}(undef, n)
    mask = states - 1
    for row in n:-1:1
        assignment[row] = chosen[mask + 1]
        mask = previous[mask + 1]
    end
    return assignment
end

function stability_summary(fits::Vector{CPFit})
    errors = getfield.(fits, :rel_error)
    reference = first(fits)
    congruences = Float64[]
    for candidate in fits[2:end]
        scores = _component_similarity(reference, candidate)
        assignment = _maximum_assignment(scores)
        append!(congruences, [scores[r, assignment[r]] for r in axes(scores, 1)])
    end
    isempty(congruences) && push!(congruences, 1.0)
    return (
        error_min = minimum(errors),
        error_max = maximum(errors),
        error_spread = maximum(errors) - minimum(errors),
        mean_congruence = mean(congruences),
        min_congruence = minimum(congruences),
        successful_starts = length(fits),
    )
end

function numerical_rank(M::AbstractMatrix; relative_tolerance::Float64 = 1e-8)
    values = svdvals(M)
    isempty(values) && return 0
    threshold = relative_tolerance * maximum(values)
    return count(>(threshold), values)
end

function _minimum_pairwise_sine(C::AbstractMatrix)
    size(C, 2) < 2 && return 1.0
    minimum_sine = 1.0
    for i in 1:(size(C, 2) - 1), j in (i + 1):size(C, 2)
        cosine = clamp(_column_cosine(C, i, C, j), 0.0, 1.0)
        minimum_sine = min(minimum_sine, sqrt(max(0.0, 1 - cosine^2)))
    end
    return minimum_sine
end

function identifiability_summary(fit::CPFit; tolerance::Float64 = 1e-8)
    A, B, C = fit.factors
    rank = length(fit.weights)
    rank_A = numerical_rank(A; relative_tolerance = tolerance)
    rank_B = numerical_rank(B; relative_tolerance = tolerance)
    min_sine = _minimum_pairwise_sine(C)
    kC_lower_bound = rank >= 2 && min_sine > tolerance ? 2 : 1
    nonzero_weights = all(abs.(fit.weights) .> tolerance * maximum(abs, fit.weights))
    kruskal_sufficient =
        rank_A == rank && rank_B == rank && kC_lower_bound >= 2 && nonzero_weights
    return (
        rank_A,
        rank_B,
        kC_lower_bound,
        min_pairwise_sine = min_sine,
        nonzero_weights,
        kruskal_sufficient,
    )
end
