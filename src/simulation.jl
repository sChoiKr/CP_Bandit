Base.@kwdef struct SimulationConfig
    n_agents::Int = 500
    n_trials::Int = 150
    alpha_range::Tuple{Float64,Float64} = (0.05, 0.95)
    beta_range::Tuple{Float64,Float64} = (0.5, 8.0)
    reward_probabilities::Tuple{Float64,Float64} = (0.8, 0.2)
    reversal_block::Int = 30
    initial_q::Float64 = 0.5
    parameter_seed::Int = 20260823
    environment_seed::Int = 20260824
    choice_seed::Int = 20260825
end

"""All simulated quantities. Only `X_obs` is passed to the main CPD condition."""
struct BanditDataset
    X_obs::Array{Float64,3}
    choices::Matrix{Int8}
    obtained_rewards::Matrix{Float64}
    potential_rewards::Matrix{Float64}
    reward_probabilities::Matrix{Float64}
    q_before_choice::Array{Float64,3}
    alpha::Vector{Float64}
    beta::Vector{Float64}
end

function _validate_config(cfg::SimulationConfig)
    cfg.n_agents > 1 || throw(ArgumentError("n_agents must be > 1"))
    cfg.n_trials > 1 || throw(ArgumentError("n_trials must be > 1"))
    0.0 < cfg.alpha_range[1] < cfg.alpha_range[2] <= 1.0 ||
        throw(ArgumentError("alpha_range must lie in (0, 1]"))
    0.0 < cfg.beta_range[1] < cfg.beta_range[2] ||
        throw(ArgumentError("beta_range must be positive and ordered"))
    all(probability -> 0.0 <= probability <= 1.0, cfg.reward_probabilities) ||
        throw(ArgumentError("reward probabilities must lie in [0, 1]"))
    cfg.reversal_block > 0 || throw(ArgumentError("reversal_block must be positive"))
    return cfg
end

@inline function _logistic(x::Float64)
    if x >= 0
        z = exp(-x)
        return 1 / (1 + z)
    end
    z = exp(x)
    return z / (1 + z)
end

"""
    simulate_bandits(cfg)

Simulate continuous `(alpha, beta)` ground truth under a shared, pre-generated
two-arm potential-reward schedule. Reward probabilities reverse every
`reversal_block` trials so that learning rate has an observable dynamical role.

The observable tensor has features
`(choose arm 1, choose arm 2, reward on arm 1, reward on arm 2)`.
"""
function simulate_bandits(cfg::SimulationConfig = SimulationConfig())
    _validate_config(cfg)
    N, T = cfg.n_agents, cfg.n_trials

    parameter_rng = MersenneTwister(cfg.parameter_seed)
    environment_rng = MersenneTwister(cfg.environment_seed)
    choice_rng = MersenneTwister(cfg.choice_seed)

    alpha_width = cfg.alpha_range[2] - cfg.alpha_range[1]
    beta_width = cfg.beta_range[2] - cfg.beta_range[1]
    alpha = cfg.alpha_range[1] .+ alpha_width .* rand(parameter_rng, N)
    beta = cfg.beta_range[1] .+ beta_width .* rand(parameter_rng, N)

    probabilities = Matrix{Float64}(undef, T, 2)
    high, low = cfg.reward_probabilities
    for t in 1:T
        first_arm_is_high = iseven((t - 1) ÷ cfg.reversal_block)
        probabilities[t, :] .= first_arm_is_high ? (high, low) : (low, high)
    end

    # One potential-outcome schedule is shared by every agent.
    potential = Float64.(rand(environment_rng, T, 2) .< probabilities)
    choices = Matrix{Int8}(undef, N, T)
    rewards = Matrix{Float64}(undef, N, T)
    q_history = Array{Float64}(undef, N, T, 2)
    X = zeros(Float64, N, T, 4)

    for i in 1:N
        q = [cfg.initial_q, cfg.initial_q]
        for t in 1:T
            q_history[i, t, :] .= q
            p_arm1 = _logistic(beta[i] * (q[1] - q[2]))
            arm = rand(choice_rng) < p_arm1 ? 1 : 2
            reward = potential[t, arm]

            choices[i, t] = Int8(arm)
            rewards[i, t] = reward
            X[i, t, arm] = 1.0
            X[i, t, 2 + arm] = reward

            q[arm] += alpha[i] * (reward - q[arm])
        end
    end

    data = BanditDataset(
        X,
        choices,
        rewards,
        potential,
        probabilities,
        q_history,
        alpha,
        beta,
    )
    validate_dataset(data)
    return data
end

observable_tensor(data::BanditDataset) = data.X_obs

"""Reference-only tensor. It must never be used as the main observable condition."""
function oracle_tensor(data::BanditDataset)
    return cat(data.X_obs, data.q_before_choice; dims = 3)
end

function validate_dataset(data::BanditDataset)
    X = data.X_obs
    N, T, F = size(X)
    F == 4 || throw(ArgumentError("observable tensor must have exactly four features"))
    size(data.choices) == (N, T) || throw(DimensionMismatch("choices shape mismatch"))
    size(data.potential_rewards) == (T, 2) ||
        throw(DimensionMismatch("potential reward shape mismatch"))
    all(isone, dropdims(sum(X[:, :, 1:2]; dims = 3); dims = 3)) ||
        throw(ArgumentError("choice indicators must be one-hot"))

    for i in 1:N, t in 1:T
        arm = Int(data.choices[i, t])
        reward = data.obtained_rewards[i, t]
        reward == data.potential_rewards[t, arm] ||
            throw(ArgumentError("agent reward does not match shared potential schedule"))
        X[i, t, arm] == 1.0 || throw(ArgumentError("choice feature mismatch"))
        X[i, t, 2 + arm] == reward || throw(ArgumentError("reward feature mismatch"))
        X[i, t, 2 + (3 - arm)] == 0.0 ||
            throw(ArgumentError("unchosen-arm reward must be masked"))
    end
    return true
end

"""Deterministic agent split, independent of simulation RNG streams."""
function split_agents(n_agents::Int, n_train::Int; seed::Int = 20260824)
    1 <= n_train < n_agents || throw(ArgumentError("n_train must lie in 1:(N-1)"))
    order = randperm(MersenneTwister(seed), n_agents)
    return sort(order[1:n_train]), sort(order[(n_train + 1):end])
end

"""Flatten each `trial x feature` slice using Julia column-major `vec` order."""
function flatten_agents(X::AbstractArray{<:Real,3})
    N, T, F = size(X)
    M = Matrix{Float64}(undef, N, T * F)
    for i in 1:N
        M[i, :] .= vec(Array(@view X[i, :, :]))
    end
    return M
end
