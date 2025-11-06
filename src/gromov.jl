# Gromov-Wasserstein solver
# Code by zsteve in https://github.com/JuliaOptimalTransport/OptimalTransport.jl/tree/gromov

abstract type EntropicGromovWasserstein end

struct EntropicGromovWassersteinSinkhorn <: EntropicGromovWasserstein
    alg_step::Sinkhorn
end

"""
    entropic_gromov_wasserstein(
        μ, ν, Cμ, Cν, ε, alg=EntropicGromovWassersteinSinkhorn(SinkhornGibbs());
        atol = nothing, rtol = nothing, check_convergence = 10, maxiter = 1_000, kwargs...
    )

Computes the transport map for the entropically regularized Gromov-Wasserstein optimal transport problem with source and target
marginals `μ` and `ν` and corresponding cost matrices `Cμ` and `Cν`. That is, we seek `γ` a local minimizer of
```math
    \\inf_{\\gamma \\in \\Pi(\\mu, \\nu)} \\sum_{i, j, i', j'} |C^{(\\mu)}_{i,i'} - C^{(\\nu)}_{j,j'}|^2 \\gamma_{i,j} \\gamma_{i',j'} + \\varepsilon \\Omega(\\gamma),
```
where ``\\Omega(\\gamma)`` is the entropic regularization term, see e.g. [`sinkhorn`](@ref).

This function employs the iterative method described in (Section 10.6.4, [^PC19]), which solves a series of Sinkhorn iteration sub-problems to arrive at a solution. Note that the Gromov-Wasserstein problem is non-convex owing to the cross-terms in the
objective function, and thus in general one is guaranteed to arrive at a local optimum.

Every `check_convergence` steps, the current iteration of `γ` is compared with `γ_prev` (the previous iteration from `check_convergence` ago).
The quantity ``\\| \\gamma - \\gamma_\\text{prev} \\|_1`` is compared against `atol` and `rtol`.

[^PC19]: Peyré, G. and Cuturi, M., 2019. Computational optimal transport: With applications to data science. Foundations and Trends® in Machine Learning, 11(5-6), pp.355-607.

See also: [`sinkhorn`](@ref)
"""
function entropic_gromov_wasserstein(
        μ::AbstractVector,
        ν::AbstractVector,
        Cμ::AbstractMatrix,
        Cν::AbstractMatrix,
        ε::Real,
        alg::EntropicGromovWasserstein = EntropicGromovWassersteinSinkhorn(SinkhornGibbs());
        atol = nothing,
        rtol = nothing,
        check_convergence = 10,
        maxiter::Int = 1_000,
        kwargs...
)
    T = float(Base.promote_eltype(μ, one(eltype(Cμ)) / ε, eltype(Cν)))

    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    return _entropic_gromov_wasserstein!(
        μ, ν, Cμ, Cν, ε, _init_storage(μ, ν, Cμ, Cν, ε, alg.alg_step)...;
        atol = _atol,
        rtol = _rtol,
        check_convergence = check_convergence,
        maxiter = maxiter
    )
end

function _init_storage(μ, ν, Cμ, Cν, ε, alg; kwargs...)
    T = float(Base.promote_eltype(μ, one(eltype(Cμ)) / ε, eltype(Cν)))
    C = similar(Cμ, T, size(μ, 1), size(ν, 1))
    tmp = similar(C)
    plan = similar(C)
    plan_prev = similar(C)
    solver = build_solver(μ, ν, C, ε, alg; kwargs...)
    return C, tmp, plan, plan_prev, solver
end

function _entropic_gromov_wasserstein!(
        μ::AbstractVector,
        ν::AbstractVector,
        Cμ::AbstractMatrix,
        Cν::AbstractMatrix,
        ε::Real,
        C,
        tmp,
        plan,
        plan_prev,
        solver;
        atol = 1e-9,
        rtol = 0.0,
        check_convergence = 10,
        maxiter::Int = 1_000)
    @. plan = μ * ν'
    plan_prev .= plan
    norm_plan = sum(μ) * sum(ν)

    # potentially slow: POT uses decomposition assumption on loss
    function get_new_cost!(C, plan, tmp, Cμ, Cν)
        A_batched_mul_B!(tmp, Cμ, plan)
        lmul!(-4, tmp)
        return A_batched_mul_B!(C, tmp, Cν)
        # seems to be a missing factor of 4 (or something like that...) compared to the POT implementation?
        # added the factor of 4 here to ensure reproducibility for the same value of ε.
        # https://github.com/PythonOT/POT/blob/9412f0ad1c0003e659b7d779bf8b6728e0e5e60f/ot/gromov.py#L247
    end

    get_new_cost!(C, plan, tmp, Cμ, Cν)
    to_check_step = check_convergence

    isconverged = false
    for iter in 1:maxiter
        reset_cache!(solver, C, ε)
        # perform Sinkhorn algorithm
        solve!(solver)
        # compute optimal transport plan
        sinkhorn_plan!(plan, solver)

        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence
            err = _fast_norm(plan, plan_prev)
            isconverged = err ≤ max(atol, rtol * norm_plan)
            if isconverged
                break
            end
            plan_prev .= plan
        end
        get_new_cost!(C, plan, tmp, Cμ, Cν)
    end

    return plan
end

"""
    entropic_gromov_barycenters(
        N, Cs, ps, p, lambdas, ε, alg=EntropicGromovWassersteinSinkhorn(SinkhornGibbs());
        atol = nothing, rtol = nothing, check_convergence = 10, maxiter = 1_000,
        init_C = nothing, random_state = nothing, kwargs...
    )

Compute the Gromov-Wasserstein barycenter of S measured similarity matrices.

The function solves the optimization problem:
```math
C^* = \\arg\\min_{C \\in \\mathbb{R}^{N \\times N}} \\sum_s \\lambda_s \\mathrm{GW}(C, C_s, p, p_s)
```

# Arguments
- `N::Int`: Size of the targeted barycenter
- `Cs::Vector{<:AbstractMatrix}`: Vector of S cost matrices, each of size (ns, ns)
- `ps::Union{Nothing, Vector{<:AbstractVector}}`: Vector of S probability distributions. If `nothing`, uniform distributions are used.
- `p::Union{Nothing, AbstractVector}`: Weights in the barycenter space of size N. If `nothing`, uniform distribution is used.
- `lambdas::Union{Nothing, AbstractVector}`: Weights for each input space. If `nothing`, uniform weights are used.
- `ε::Real`: Entropic regularization parameter
- `alg`: Algorithm for the inner Gromov-Wasserstein problems (default: SinkhornGibbs)

# Keyword Arguments
- `atol`: Absolute tolerance for convergence
- `rtol`: Relative tolerance for convergence
- `check_convergence::Int`: Check convergence every this many iterations (default: 10)
- `maxiter::Int`: Maximum number of iterations (default: 1000)
- `init_C::Union{Nothing, AbstractMatrix}`: Initial barycenter structure. If `nothing`, random initialization is used.
- `random_state::Union{Nothing, Int}`: Random seed for initialization
- `kwargs...`: Additional arguments passed to the inner Sinkhorn solver

# Returns
- `C`: The barycenter cost matrix of size (N, N)

# References
[^PCS16]: Gabriel Peyré, Marco Cuturi, and Justin Solomon. "Gromov-Wasserstein averaging of kernel and distance matrices." International Conference on Machine Learning (ICML). 2016.

# Example
```julia
using OptimalTransport, Distances

# Create sample cost matrices
C1 = pairwise(SqEuclidean(), rand(10, 2), dims=1)
C2 = pairwise(SqEuclidean(), rand(12, 2), dims=1)
C3 = pairwise(SqEuclidean(), rand(15, 2), dims=1)
Cs = [C1, C2, C3]

# Compute barycenter
N = 8
ε = 0.1
C_bar = entropic_gromov_barycenters(N, Cs, nothing, nothing, nothing, ε)
```

See also: [`entropic_gromov_wasserstein`](@ref)
"""
function entropic_gromov_barycenters(
        N::Int,
        Cs::Vector{<:AbstractMatrix};
        ps::Vector{<:AbstractVector} = [ones(size(C, 1)) ./ size(C, 1) for C in Cs],
        p::AbstractVector = ones(N) ./ N,
        lambdas::AbstractVector = ones(length(Cs)) ./ length(Cs),
        ε::Real = 1e-1,
        alg::EntropicGromovWasserstein = EntropicGromovWassersteinSinkhorn(SinkhornGibbs()),
        atol = 1e-9,
        rtol = 0.0,
        check_convergence::Int = 10,
        maxiter::Int = 1_000,
        caches_provided = nothing,
        kwargs...
)
    S = length(Cs)

    # Validate inputs
    length(ps) == S ||
        throw(ArgumentError("Length of ps must equal number of cost matrices"))
    length(lambdas) == S ||
        throw(ArgumentError("Length of lambdas must equal number of cost matrices"))
    sum(lambdas) ≈ 1.0 || throw(ArgumentError("Lambdas must sum to 1"))

    # Type promotion
    T = float(Base.promote_eltype(p, one(eltype(Cs[1])) / ε))

    C = rand(T, N, N)

    #define cache for each plan
    if !isnothing(caches_provided)
        caches = caches_provided
    else
        caches = [_init_storage(p, ps[s], C, Cs[s], ε, alg.alg_step) for s in 1:S]
    end

    # cache for barycenter update
    max_size_s = maximum(size.(Cs, 1))
    cache_update_barycenter = similar(C, T, size(C, 1), max_size_s)

    # Initialize transport plans
    T_plans = Vector{Matrix{T}}(undef, S)

    # Initialize previous C for convergence check
    C_prev = similar(C)
    C_prev .= C

    to_check_step = check_convergence
    isconverged = false
    norm_plan = sum(p) * sum(p)

    for iter in 1:maxiter
        # Compute transport plans from barycenter to each input
        for s in 1:S
            T_plans[s] = _entropic_gromov_wasserstein!(
                p, ps[s], C, Cs[s], ε, caches[s]...; atol = 1e-4,
                rtol = 1e-4, maxiter = maxiter, kwargs...
            )
        end
        update_barycenter!(C, T_plans, Cs, lambdas, p, cache_update_barycenter)
        # Check convergence

        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            to_check_step = check_convergence
            err = _fast_norm(C, C_prev)
            isconverged = err ≤ max(atol, rtol * norm_plan)

            if isconverged || iter == maxiter
                @debug "GW barycenter with $(alg) ($iter/$maxiter): converged with error $err"
                break
            end

            C_prev .= C
        end
    end

    return C
end

# use l2 norm to match POT
function _fast_norm(x, y)
    s::eltype(x) = 0
    @simd for i in eachindex(x, y)
        @inbounds s += abs2(x[i] - y[i])
    end
    return sqrt(s)
end

"""
    update_square_loss_barycenter!(C,T, Cs, lambdas, p, cache)

Update the barycenter structure matrix C for square loss.

# Arguments
- `T`: vector of transport plans
- `Cs`: vector of cost matrices
- `lambdas`: weights for each input
- `p`: barycenter weights
- `cache`: intermediate storage for C update
"""
function update_barycenter!(
        C::AbstractMatrix, Ts::Vector, Cs::Vector, lambdas::Vector, p::AbstractVector, cache
)
    N = length(p)
    S = length(Cs)

    # For square loss, the barycenter update is:
    # C = sum_s lambda_s * (T_s * C_s * T_s^T) / (p * p^T)

    # first iteration to properly reset C
    cache_1 = view(cache, :, 1:size(Cs[1], 1))
    LinearAlgebra.BLAS.gemm!('N', 'N', 1.0, Ts[1], Cs[1], 0.0, cache_1)
    LinearAlgebra.BLAS.gemm!('N', 'T', lambdas[1], cache_1, Ts[1], 0.0, C)

    @inbounds for s in 2:S
        # T_s * C_s * T_s^T using cache[s] as intermediate storage
        cache_s = view(cache, :, 1:size(Cs[s], 1))
        LinearAlgebra.BLAS.gemm!('N', 'N', 1.0, Ts[s], Cs[s], 0.0, cache_s)
        LinearAlgebra.BLAS.gemm!('N', 'T', lambdas[s], cache_s, Ts[s], 1.0, C)
    end

    # Normalize by outer product of barycenter weights
    # Avoid division by zero
    @inbounds for j in 1:N  # Outer loop over columns for column-major access
        for i in 1:N  # Inner loop over rows
            pp_ij = p[i] * p[j]
            if pp_ij > eps(eltype(C))
                C[i, j] = C[i, j] / pp_ij
            end
        end
    end

    return C
end
