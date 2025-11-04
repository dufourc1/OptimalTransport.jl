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
    alg::EntropicGromovWasserstein=EntropicGromovWassersteinSinkhorn(SinkhornGibbs());
    atol=nothing,
    rtol=nothing,
    check_convergence=10,
    maxiter::Int=1_000,
    kwargs...,
)
    T = float(Base.promote_eltype(μ, one(eltype(Cμ)) / ε, eltype(Cν)))
    C = similar(Cμ, T, size(μ, 1), size(ν, 1))
    tmp = similar(C)
    plan = similar(C)
    @. plan = μ * ν'
    plan_prev = similar(C)
    plan_prev .= plan
    norm_plan = sum(plan)

    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

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
        # perform Sinkhorn algorithm
        solver = build_solver(μ, ν, C, ε, alg.alg_step; kwargs...)
        solve!(solver)
        # compute optimal transport plan
        plan = sinkhorn_plan(solver)

        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            # reset counter
            to_check_step = check_convergence
            plan_prev .-= plan
            isconverged = sum(abs, plan_prev) < max(_atol, _rtol * norm_plan)
            if isconverged
                @debug "Gromov Wasserstein with $(solver.alg) ($iter/$maxiter): converged"
                break
            end
            plan_prev .= plan
        end
        get_new_cost!(C, plan, tmp, Cμ, Cν)
    end

    return plan
end

# Helper functions for Gromov-Wasserstein computations

"""
    init_matrix_square_loss(Cμ, Cν, μ, ν)

Initialize matrices for square loss in Gromov-Wasserstein computations.
Returns (constC, hCμ, hCν) where:
- constC contains constant terms
- hCμ and hCν are transformations of Cμ and Cν
"""
function init_matrix_square_loss(
    Cμ::AbstractMatrix, Cν::AbstractMatrix, μ::AbstractVector, ν::AbstractVector
)
    # For square loss: L(a,b) = (a - b)^2 = a^2 + b^2 - 2ab
    # So f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = b

    # Compute constant term
    constC = dot(μ, Cμ .^ 2 * μ) + dot(ν, Cν .^ 2 * ν)

    # hCμ and hCν are the matrices themselves for square loss
    hCμ = Cμ
    hCν = Cν

    return constC, hCμ, hCν
end

"""
    gwggrad(constC, hCμ, hCν, γ)

Compute the gradient of the Gromov-Wasserstein objective for square loss.
"""
function gwggrad(constC::Real, hCμ::AbstractMatrix, hCν::AbstractMatrix, γ::AbstractMatrix)
    # Gradient for square loss: 2 * [C1^2 μ 1^T + 1 ν^T C2^2 - 2 C1 T C2^T]
    # Simplified: 2 * [-C1 T C2^T + C1^2 μ 1^T + 1 ν^T C2^2]
    # Further simplified to match POT: 2 * [-C1 T C2^T] + const_terms
    # But the constant terms vanish in the Sinkhorn projection, so:
    return -2 * (hCμ * γ * hCν')
end

"""
    gwloss(constC, hCμ, hCν, γ)

Compute the Gromov-Wasserstein loss for square loss.
"""
function gwloss(constC::Real, hCμ::AbstractMatrix, hCν::AbstractMatrix, γ::AbstractMatrix)
    # Loss = sum_{i,j,k,l} L(Cμ_{i,k}, Cν_{j,l}) γ_{i,j} γ_{k,l}
    # For square loss = sum (Cμ_{i,k} - Cν_{j,l})^2 γ_{i,j} γ_{k,l}
    # = constC - 2 * <Cμ γ Cν^T, γ>
    return constC + sum(gwggrad(constC, hCμ, hCν, γ) .* γ)
end

"""
    update_square_loss_barycenter(T, Cs, lambdas, p)

Update the barycenter structure matrix C for square loss.

# Arguments
- `T`: vector of transport plans
- `Cs`: vector of cost matrices
- `lambdas`: weights for each input
- `p`: barycenter weights
"""
function update_square_loss_barycenter(
    T::Vector, Cs::Vector, lambdas::Vector, p::AbstractVector
)
    N = length(p)
    S = length(Cs)

    # Initialize the barycenter
    C = zeros(eltype(Cs[1]), N, N)

    # For square loss, the barycenter update is:
    # C = sum_s lambda_s * (T_s * C_s * T_s^T) / (p * p^T)
    for s in 1:S
        # T_s * C_s * T_s^T
        tmp = T[s] * Cs[s] * T[s]'
        C .+= lambdas[s] .* tmp
    end

    # Normalize by outer product of barycenter weights
    # Avoid division by zero
    pp = p * p'
    @inbounds for j in 1:N  # Outer loop over columns for column-major access
        for i in 1:N  # Inner loop over rows
            if pp[i, j] > 1e-16
                C[i, j] /= pp[i, j]
            end
        end
    end

    return C
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
    Cs::Vector{<:AbstractMatrix},
    ps::Union{Nothing,Vector{<:AbstractVector}},
    p::Union{Nothing,AbstractVector},
    lambdas::Union{Nothing,AbstractVector},
    ε::Real,
    alg::EntropicGromovWasserstein=EntropicGromovWassersteinSinkhorn(SinkhornGibbs());
    atol=nothing,
    rtol=nothing,
    check_convergence::Int=10,
    maxiter::Int=1_000,
    init_C::Union{Nothing,AbstractMatrix}=nothing,
    kwargs...,
)
    S = length(Cs)

    # Handle default values for ps
    if ps === nothing
        ps = [ones(size(C, 1)) ./ size(C, 1) for C in Cs]
    end

    # Handle default values for p
    if p === nothing
        p = ones(N) / N
    end

    # Handle default values for lambdas
    if lambdas === nothing
        lambdas = ones(S) / S
    end

    # Validate inputs
    length(ps) == S ||
        throw(ArgumentError("Length of ps must equal number of cost matrices"))
    length(lambdas) == S ||
        throw(ArgumentError("Length of lambdas must equal number of cost matrices"))
    sum(lambdas) ≈ 1.0 || throw(ArgumentError("Lambdas must sum to 1"))

    # Type promotion
    T = float(Base.promote_eltype(p, one(eltype(Cs[1])) / ε))

    # Initialize barycenter structure C
    if init_C === nothing
        # Random initialization
        xalea = randn(N, 2)
        # Use broadcasting for efficient distance computation
        C = zeros(T, N, N)
        @inbounds for j in 1:N  # Outer loop over columns for column-major access
            for i in 1:N  # Inner loop over rows
                C[i, j] = sum(abs2, xalea[i, :] .- xalea[j, :])
            end
        end
        C ./= maximum(C)
    else
        C = convert(Matrix{T}, init_C)
    end

    # Set tolerances
    _atol = atol === nothing ? 0 : atol
    _rtol = rtol === nothing ? (_atol > zero(_atol) ? zero(T) : sqrt(eps(T))) : rtol

    # Initialize transport plans
    T_plans = Vector{Matrix{T}}(undef, S)

    # Initialize previous C for convergence check
    C_prev = similar(C)
    C_prev .= C

    to_check_step = check_convergence
    isconverged = false

    for iter in 1:maxiter
        # Compute transport plans from barycenter to each input
        for s in 1:S
            T_plans[s] = entropic_gromov_wasserstein(
                p, ps[s], C, Cs[s], ε, alg; atol=1e-4, rtol=1e-4, maxiter=maxiter, kwargs...
            )
        end

        # Update barycenter structure
        C = update_square_loss_barycenter(T_plans, Cs, lambdas, p)

        # Check convergence
        to_check_step -= 1
        if to_check_step == 0 || iter == maxiter
            to_check_step = check_convergence

            err = sum(abs, C .- C_prev) / sum(abs, C)
            isconverged = err < max(_atol, _rtol)

            if isconverged
                @debug "Gromov-Wasserstein barycenter ($iter/$maxiter): converged with error $err"
                break
            end

            C_prev .= C
        end
    end

    return C
end
