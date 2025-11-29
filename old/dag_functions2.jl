###############################
# Data structures for the DAG #
###############################


"""
Represents a single directed arc in the DAG.

Fields:
- s    : global ID of the source node
- t    : global ID of the target node
- feat : index of the feature (1..T) that this arc corresponds to
         (for transitions between layer k-1 and layer k).
         If you later decide to add terminal arcs, you can set feat = 0.
- move : describes whether this arc keeps or increases the Hamming distance:
         :stay  -> do not flip this feature (distance unchanged)
         :flip  -> flip this feature (distance increases by 1)
"""

struct Arc
    move::Symbol  # :stay or :flip (you can add :terminal later if you want)
    feat::Int
    s::Int
    t::Int
end


"""
Template of the DAG used in subgroup DR logistic regression.

This template is:
- independent of the dataset
- independent of the subgroup g
- only depends on the number of binary features T (= length(gammas))

Fields:
- num_nodes   : total number of nodes in the DAG
- source      : global node ID of the unique source node
                (this is the node in layer 0 with distance 0)
- layer_values: for each layer k, a vector of all possible "raw distances"
                (here, raw distance = number of flipped features so far,
                 or more generally, the cumulative sum of gammas[1:k]),
                mainly useful for debugging or later distance computations
- layer_nodes : for each layer k, a vector of global node IDs,
                aligned with layer_values[k]
- arcs        : a list of all arcs between layers,
                each arc is reused for all samples and all groups
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

Given a vector gammas[1..T], build T+1 layers of cumulative "raw distances".

Interpretation:
- T is the number of binary features z_rest_1, ..., z_rest_T.
- For the subgroup metric we typically set gammas[k] = 1.0 for all k,
  so that the "raw distance" is exactly the number of flipped coordinates.
- Later, for sample i in group g_i, the *true* Wasserstein distance will be:
      d_i = gamma_group[g_i] * raw_distance

Layers:
- layers[1] = layer 0  = { 0.0 }
  (no features processed yet, distance is 0)
- layers[k+1] = all possible cumulative sums of gammas[1..k].
  These are the distances after processing the first k features,
  where each feature can be either "stay" or "flip".
"""

function build_distance_layers(gammas::Vector{Float64})
    T = length(gammas)

    # Allocate a vector of length T+1, each entry will be a Vector{Float64}
    layers = Vector{Vector{Float64}}(undef, T + 1)

    # Layer 0: distance is always 0.0
    layers[1] = [0.0]

    # For each feature k = 1..T, build the next layer
    for k in 1:T
        prev = layers[k]           # distances after processing first k-1 features
        step = gammas[k]           # distance increment if we flip feature k

        # If we flip feature k, we add step to each previous distance
        shifted = prev .+ step

        # Combine "stay" (prev) and "flip" (shifted)
        new_layer = vcat(prev, shifted)

        # Remove duplicates (e.g., if step = 0, or by coincidence)
        new_layer = unique(new_layer)

        # Sort in ascending order for consistent indexing
        sort!(new_layer)

        # Store as layer k (i.e., "after processing k features")
        layers[k + 1] = new_layer
    end

    return layers
end


###################################################
# Step 2: assign global node IDs to each distance #
###################################################

"""
    assign_node_ids(layers::Vector{Vector{Float64}})
        -> (layer_values, layer_nodes, num_nodes)

Given the distance layers, assign a unique global integer node ID
to each distance value in each layer.

- Input:
    layers[k] = vector of distance values in layer k-1 (0-based in math,
                  but we store it as index k in Julia).
  So layers[1] corresponds to k=0, layers[end] to k=T.

- Output:
    layer_values[k] = same as layers[k], stored explicitly
    layer_nodes[k]  = vector of node IDs aligned with layer_values[k]
    num_nodes       = total number of nodes over all layers
"""
function assign_node_ids(layers::Vector{Vector{Float64}})
    num_layers   = length(layers)     # T+1
    layer_values = Vector{Vector{Float64}}(undef, num_layers)
    layer_nodes  = Vector{Vector{Int}}(undef, num_layers)

    next_id = 1  # next available global node ID

    for l in 1:num_layers
        vals = layers[l]                          # distance values in this layer
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
    find_value_index(vals::Vector{Float64}, value::Float64; atol=1e-8) -> Int

Find the index i such that vals[i] ≈ value, using absolute tolerance atol.
If no such index exists, return 0.

We use this when we know that "value" should be present in the previous layer
(e.g. h or h - gamma_k), but due to floating-point arithmetic we do not want
to rely on exact equality.
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

Construct arcs for the case where each gammas[k] corresponds to a single
binary feature (z_rest_k in {0,1}), and flipping feature k increases the
distance by gammas[k].

For each feature k = 1..T, we connect layer k-1 to layer k:

- layers[k]   contains all cumulative distances after using the first k-1 gammas
- layers[k+1] contains all cumulative distances after using the first k gammas

Between layer k-1 and layer k we add two types of arcs:

1. stay:
     distance does not change
     h_prev = h
   meaning that the previous layer also contains the same distance h.

2. flip:
     distance increases by gammas[k]
     h_prev = h - gammas[k]
   meaning that the previous layer contains (h - gammas[k]).

For each valid transition, we add one directed arc (s -> t) with:
    s = node ID in layer k-1,
    t = node ID in layer k,
    feat = k,
    move = :stay or :flip.
"""
function build_arcs_binary(
    layers::Vector{Vector{Float64}},
    layer_nodes::Vector{Vector{Int}},
    gammas::Vector{Float64}
)
    T = length(gammas)
    arcs = Arc[]

    # For each feature k, connect layer (k-1) to layer k
    # Note: in our storage, layer index = k means "after using k-1 features".
    # So "previous" layer = layers[k], "current" layer = layers[k+1].
    for feat in 1:T
        prev_vals = layers[feat]          # distances after first (feat-1) features
        prev_ids  = layer_nodes[feat]

        curr_vals = layers[feat + 1]      # distances after first feat features
        curr_ids  = layer_nodes[feat + 1]

        step = gammas[feat]

        # For each distance value in the current layer, determine where it came from
        for (h_idx, h_val) in enumerate(curr_vals)
            t_id = curr_ids[h_idx]

            # 1) stay: previous layer contains the same distance h_val
            idx_prev_stay = find_value_index(prev_vals, h_val)
            if idx_prev_stay != 0
                s_id = prev_ids[idx_prev_stay]
                push!(arcs, Arc(:stay, feat, s_id, t_id))
            end

            # 2) flip: previous layer contains (h_val - step)
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
    build_graph_template(gammas::Vector{Float64}) -> DROGraphTemplate

Build a DAG template for the subgroup Wasserstein DRO with binary
categorical features.

- Input:
    gammas[1..T] : per-feature increments of the *raw* distance.
                   In the pure Hamming case, you typically set:
                       gammas[k] = 1.0 for all k.
                   For subgroup DR, you will later multiply the raw distance
                   by a group-specific gamma_group[g_i] when constructing
                   the JuMP model.

- Output:
    DROGraphTemplate that contains:
      * num_nodes   : total number of nodes over all layers
      * source      : node ID of the unique source node in layer 0
      * layer_values: list of per-layer distance values
      * layer_nodes : list of per-layer node IDs
      * arcs        : all arcs between successive layers (stay/flip)
"""
function build_graph_template(gammas::Vector{Float64})
    # 1. Build cumulative-distance layers
    layers = build_distance_layers(gammas)

    # 2. Assign global node IDs to each distance in each layer
    layer_values, layer_nodes, num_nodes = assign_node_ids(layers)

    # 3. Construct arcs between successive layers (binary case)
    arcs = build_arcs_binary(layer_values, layer_nodes, gammas)

    # 4. The source node is the unique node in layer 0 (distance 0)
    source = layer_nodes[1][1]

    # (No sink stored here. In the subgroup graph-based dual,
    #  you usually add a "terminal node" separately in the JuMP model
    #  and connect all nodes in the last layer to that terminal node.
    #  This allows you to attach group-specific gamma_group[g]
    #  only at the terminal (distance) stage.)

    return DROGraphTemplate(
        num_nodes,
        source,
        layer_values,
        layer_nodes,
        arcs,
    )
end
