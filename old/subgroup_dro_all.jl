###############################
# Imports                     #
###############################

using JuMP
using MathOptInterface
const MOI = MathOptInterface

###############################
# Data structures for the DAG #
###############################

"""
Represents a single directed arc in the DAG.

Fields:
- move : :stay or :flip (you can extend later)
- feat : index of the feature (1..T) for this transition
- s    : source node id
- t    : target node id
"""
struct Arc
    move::Symbol  # :stay or :flip
    feat::Int
    s::Int
    t::Int
end

"""
Template of the DAG used in subgroup DR logistic regression.

Fields:
- num_nodes   : total number of nodes (k=0..T, no sink)
- source      : id of the unique source node (layer 0, distance 0)
- layer_values: per-layer vector of "raw distances" (e.g. #flips)
- layer_nodes : per-layer vector of node ids, aligned with layer_values
- arcs        : all arcs between layers (stay/flip)
"""
struct DROGraphTemplate
    num_nodes::Int
    source::Int
    layer_values::Vector{Vector{Float64}}
    layer_nodes::Vector{Vector{Int}}
    arcs::Vector{Arc}
end

#############################################
# Step 1: build distance "layers" (0..T)    #
#############################################

"""
    build_distance_layers(gammas::Vector{Float64}) -> Vector{Vector{Float64}}

Given gammas[1..T], build T+1 layers of cumulative "raw distances".

- layers[1]   = {0.0}  (no features processed)
- layers[k+1] = all possible sums of gammas[1..k].
"""
function build_distance_layers(gammas::Vector{Float64})
    T = length(gammas)
    layers = Vector{Vector{Float64}}(undef, T + 1)
    layers[1] = [0.0]  # layer 0 distance

    for k in 1:T
        prev = layers[k]
        step = gammas[k]

        shifted = prev .+ step
        new_layer = unique(vcat(prev, shifted))
        sort!(new_layer)
        layers[k + 1] = new_layer
    end

    return layers
end

###################################################
# Step 2: assign global node IDs to each distance #
###################################################

"""
    assign_node_ids(layers) -> (layer_values, layer_nodes, num_nodes)

Assign global node IDs to each distance value in each layer.
"""
function assign_node_ids(layers::Vector{Vector{Float64}})
    num_layers   = length(layers)
    layer_values = Vector{Vector{Float64}}(undef, num_layers)
    layer_nodes  = Vector{Vector{Int}}(undef, num_layers)

    next_id = 1

    for l in 1:num_layers
        vals = layers[l]
        ids  = collect(next_id:(next_id + length(vals) - 1))
        layer_values[l] = vals
        layer_nodes[l]  = ids
        next_id += length(vals)
    end

    num_nodes = next_id - 1
    return layer_values, layer_nodes, num_nodes
end

##########################################################
# Helper: find the index of a value (with tolerance)     #
##########################################################

"""
    find_value_index(vals, value; atol=1e-8) -> Int

Return index i with vals[i] ≈ value, or 0 if none found.
"""
function find_value_index(vals::Vector{Float64}, value::Float64; atol = 1e-8)
    for (i, v) in enumerate(vals)
        if isapprox(v, value; atol = atol, rtol = 0.0)
            return i
        end
    end
    return 0
end

######################################################
# Step 3: build arcs between successive feature      #
#         layers for the *binary* case               #
######################################################

"""
    build_arcs_binary(layers, layer_nodes, gammas) -> Vector{Arc}

For each feature k, connect layer (k-1) to layer k with:

- :stay arcs: distance unchanged
- :flip arcs: distance increases by gammas[k]
"""
function build_arcs_binary(
    layers::Vector{Vector{Float64}},
    layer_nodes::Vector{Vector{Int}},
    gammas::Vector{Float64}
)
    T = length(gammas)
    arcs = Arc[]

    for feat in 1:T
        prev_vals = layers[feat]
        prev_ids  = layer_nodes[feat]

        curr_vals = layers[feat + 1]
        curr_ids  = layer_nodes[feat + 1]

        step = gammas[feat]

        for (h_idx, h_val) in enumerate(curr_vals)
            t_id = curr_ids[h_idx]

            # stay: h_prev = h_val
            idx_prev_stay = find_value_index(prev_vals, h_val)
            if idx_prev_stay != 0
                s_id = prev_ids[idx_prev_stay]
                push!(arcs, Arc(:stay, feat, s_id, t_id))
            end

            # flip: h_prev = h_val - step
            h_prev = h_val - step
            idx_prev_flip = find_value_index(prev_vals, h_prev)
            if idx_prev_flip != 0
                s_id = prev_ids[idx_prev_flip]
                push!(arcs, Arc(:flip, feat, s_id, t_id))
            end
        end
    end

    return arcs
end

######################################################
# Step 4: build the full DAG template                #
######################################################

"""
    build_graph_template(gammas) -> DROGraphTemplate

Build DAG template for Hamming distance on binary z_rest.

Typically: gammas = ones(m_z).
"""
function build_graph_template(gammas::Vector{Float64})
    layers = build_distance_layers(gammas)
    layer_values, layer_nodes, num_nodes = assign_node_ids(layers)
    arcs = build_arcs_binary(layer_values, layer_nodes, gammas)
    source = layer_nodes[1][1]  # unique node at layer 0

    return DROGraphTemplate(
        num_nodes,
        source,
        layer_values,
        layer_nodes,
        arcs,
    )
end

##########################################################
# Nonlinear sink-arc helper                              #
##########################################################

"""
Add the nonlinear constraint for sink arcs:

    μ[i, sink] - μ[i, u] ≥ -log(-1 + exp(r_i + λ * d_weighted))
"""
function add_terminal_arc_nl!(
    model::Model,
    μ,
    i::Int,
    u::Int,
    sink_id::Int,
    r_i,
    λ,
    d_weighted::Float64
)
    @NLconstraint(
        model,
        μ[i, sink_id] - μ[i, u] >= -log(-1 + exp(r_i + λ * d_weighted))
    )
    return nothing
end

##########################################################
# Main model builder                                     #
##########################################################

"""
    build_subgroup_dro_graph_model(...)

Build the subgroup DR logistic regression model using the
graph-based formulation and a DROGraphTemplate.

Types are kept loose on purpose to avoid include-time errors;
we convert inside the function.
"""
function build_subgroup_dro_graph_model(
    X,
    Zrest,
    y,
    group;
    w_group,
    gamma_group,
    ε,
    template,
    optimizer
)

    # ------------------
    # Basic dimensions
    # ------------------
    N, n_x   = size(X)
    N2, m_z  = size(Zrest)
    N3       = length(y)
    N4       = length(group)

    @assert N == N2 == N3 == N4 "X, Zrest, y, group must have matching first dimension (N)."

    # Convert to concrete arrays (for safe indexing / types)
    Xmat = Array{Float64}(X)
    Zmat = Array{Int}(round.(Int, Zrest))   # force 0/1 -> Int
    yvec = Array{Int}(round.(Int, y))
    gvec = Array{Int}(round.(Int, group))

    G = maximum(gvec)
    @assert length(w_group)     >= G "w_group length must cover all groups."
    @assert length(gamma_group) >= G "gamma_group length must cover all groups."

    # ------------------
    # Graph template basics
    # ------------------
    num_core_nodes = template.num_nodes
    source_id      = template.source
    sink_id        = num_core_nodes + 1      # extra sink node
    total_nodes    = num_core_nodes + 1

    @assert length(template.layer_values) == m_z + 1 "Template must have m_z+1 layers (0..m_z)."

    # ------------------
    # Build model
    # ------------------
    model = Model(optimizer)

    # Decision variables
    @variable(model, β0)
    @variable(model, βx[1:G, 1:n_x])
    @variable(model, βz[1:m_z])
    @variable(model, λ >= 0)
    @variable(model, r[1:N])
    @variable(model, μ[1:N, 1:total_nodes])

    # Objective
    @objective(model, Min, λ * ε + (1.0 / N) * sum(r[i] for i in 1:N))

    # 1) Margin constraints
    for i in 1:N
        gi = gvec[i]
        @constraint(
            model,
            yvec[i] * (dot(βx[gi, :], Xmat[i, :]) + β0) >=
            -μ[i, source_id] + μ[i, sink_id]
        )
    end

    # 2) Feature-arc constraints
    arcs = template.arcs
    for i in 1:N
        yi = yvec[i]
        for arc in arcs
            s  = arc.s
            t  = arc.t
            k  = arc.feat
            mv = arc.move

            z0 = Zmat[i, k]
            @assert (z0 == 0 || z0 == 1) "Zrest must be 0/1 binary."

            z_arc = (mv == :stay) ? z0 : (1 - z0)

            @constraint(
                model,
                μ[i, t] - μ[i, s] >= - yi * βz[k] * z_arc
            )
        end
    end

    # 3) Sink-arc constraints
    last_layer_values = template.layer_values[end]
    last_layer_nodes  = template.layer_nodes[end]
    @assert length(last_layer_values) == length(last_layer_nodes)

    for i in 1:N
        gi = gvec[i]
        γg = gamma_group[gi]
        for idx in eachindex(last_layer_nodes)
            u      = last_layer_nodes[idx]
            d_raw  = last_layer_values[idx]
            d_w    = γg * d_raw
            add_terminal_arc_nl!(model, μ, i, u, sink_id, r[i], λ, d_w)
        end
    end

    # 4) Group-wise norm constraints on β_x (ℓ2 example)
    for g in 1:G
        @NLconstraint(
            model,
            sum(βx[g, j]^2 for j in 1:n_x) <= (λ * w_group[g])^2
        )
    end

    return model
end

println("loaded subgroup_dro_all.jl")
@show @isdefined(build_subgroup_dro_graph_model)
