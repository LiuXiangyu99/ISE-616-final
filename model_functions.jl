using JuMP, MosekTools

# \alpha + Tab -> α



function build_DP_simple(X, y, G::DROGraphTemplate, gammas, epsilon)
    N, d = size(X)

    model = Model(MosekTools.Optimizer)
    set_optimizer_attributes(model, "MSK_IPAR_NUM_THREADS" => 1)

    @variable(model, λ >= 0)
    @variable(model, r[1:N] >= 0)
    @variable(model, β[1:d])
    @variable(model, β0)
    @variable(model, μ[i=1:N, v=1:G.num_nodes])

    @objective(model, Min, λ * epsilon + sum(r) / N)

    src = G.source
    sink = G.layer_nodes[end][end]   

    # margin
    for i in 1:N
        @constraint(model,
            y[i] * (dot(β, X[i, :]) + β0) >= -μ[i, src] + μ[i, sink]
        )
    end

    # μ_t - μ_s ≥ w_ie
    for i in 1:N
        for e in G.arcs
            
            w_ie = 0.0
            @constraint(model, μ[i, e.t] - μ[i, e.s] >= w_ie)
        end
    end

    # |γ_j^{-1} β_j| ≤ λ
    for j in 1:d
        γ_inv = 1.0 / gammas[j]
        @constraint(model,  γ_inv * β[j] <= λ)
        @constraint(model, -λ <= γ_inv * β[j])
    end

    return model
end
