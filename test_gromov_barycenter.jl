using OptimalTransport
using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(42)

println("Testing entropic_gromov_barycenters implementation...")

# Create simple test cost matrices
function create_distance_matrix(points)
    n = size(points, 1)
    C = zeros(n, n)
    for i in 1:n
        for j in 1:n
            C[i, j] = sum((points[i, :] .- points[j, :]) .^ 2)
        end
    end
    return C
end

# Generate sample data
n1, n2, n3 = 10, 12, 8
points1 = randn(n1, 2)
points2 = randn(n2, 2)
points3 = randn(n3, 2)

# Create cost matrices
C1 = create_distance_matrix(points1)
C2 = create_distance_matrix(points2)
C3 = create_distance_matrix(points3)

Cs = [C1, C2, C3]

println("\nInput matrices:")
println("  C1: $(size(C1))")
println("  C2: $(size(C2))")
println("  C3: $(size(C3))")

# Test with default parameters
N = 6
ε = 0.1

println("\nComputing Gromov-Wasserstein barycenter...")
println("  Target size: $N")
println("  Epsilon: $ε")

try
    C_bar = entropic_gromov_barycenters(N, Cs, nothing, nothing, nothing, ε; maxiter = 50)

    println("\nSuccess!")
    println("  Barycenter size: $(size(C_bar))")
    println("  Barycenter is symmetric: $(isapprox(C_bar, C_bar', rtol=1e-6))")
    println("  Barycenter diagonal is zero: $(all(abs.(diag(C_bar)) .< 1e-10))")
    println("  Barycenter range: [$(minimum(C_bar)), $(maximum(C_bar))]")

    # Test with custom weights
    println("\nTesting with custom weights...")
    lambdas = [0.5, 0.3, 0.2]
    C_bar2 = entropic_gromov_barycenters(N, Cs, nothing, nothing, lambdas, ε; maxiter = 50)
    println("  Success with custom lambdas!")

    # Test with custom initialization
    println("\nTesting with custom initialization...")
    init_C = create_distance_matrix(randn(N, 2))
    C_bar3 = entropic_gromov_barycenters(
        N, Cs, nothing, nothing, nothing, ε; init_C = init_C, maxiter = 50)
    println("  Success with custom initialization!")

    println("\n✓ All tests passed!")

catch e
    println("\n✗ Error occurred:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
