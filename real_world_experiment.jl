using JuMP
using Ipopt   
using LinearAlgebra
using MathOptInterface
const MOI = MathOptInterface
using Random



############################
# Encoding information type
############################

"""
    ZGEncodingInfo

Stores the information needed to convert between
original (integer-coded) categorical/group features
and their one-hot encodings.

Fields:
- k_z::Vector{Int}     : number of categories for each categorical component (length m)
- z_start::Vector{Int} : start index (column) of each component in Z_onehot (length m)
- num_g::Int           : total number of groups
"""

struct ZGEncodingInfo
    k_z::Vector{Int}
    z_start::Vector{Int}
    num_g::Int
end



#############################################
# Original (Z, group) → Reduced (Z_enc, G_enc)
#############################################

"""
    encode_zg_reduced(Z::AbstractMatrix{<:Integer},
                      group::AbstractVector{<:Integer})

Encode categorical features Z and group indexes into a reduced dummy
representation:

- Each categorical component l with k_l levels is mapped to (k_l - 1)
  dummy variables. The last category k_l is the baseline (all zeros).
- The group variable with num_g levels is mapped to (num_g - 1)
  dummy variables. The last group num_g is the baseline (all zeros).

Input:
- Z     : N × m matrix, each column l stores an integer in {1, ..., k_l}
- group : length-N vector, each entry in {1, ..., num_g}

Output:
- Z_enc : N × sum_l (k_l - 1) matrix (reduced dummies for Z)
- G_enc : N × (num_g - 1) matrix (reduced dummies for group)
- info  : ZGEncodingInfo, stores the encoding scheme for later decoding
"""
function encode_zg_reduced(Z::AbstractMatrix{<:Integer},
                           group::AbstractVector{<:Integer})
    N, m = size(Z)
    @assert length(group) == N "group must have length N"

    # k_z[l] = number of categories for component l (assumed {1, ..., k_l})
    k_z = [maximum(Z[:, l]) for l in 1:m]

    # Column start indices in the reduced dummy matrix Z_enc.
    # Component l uses (k_z[l] - 1) columns (baseline = category k_z[l]).
    z_start = Vector{Int}(undef, m)
    col = 1
    for l in 1:m
        z_start[l] = col
        d_l = max(k_z[l] - 1, 0)  # number of dummies for component l
        col += d_l
    end
    total_z = col - 1   # total number of columns in Z_enc

    # Allocate Z_enc: N × total_z
    Z_enc = zeros(Int8, N, total_z)

    for i in 1:N
        for l in 1:m
            val = Z[i, l]
            @assert 1 ≤ val ≤ k_z[l] "Category out of range in column $l (row $i)"

            k_l = k_z[l]
            d_l = max(k_l - 1, 0)

            # If d_l == 0, there is effectively a single category (no column needed).
            if d_l > 0 && val < k_l
                col0 = z_start[l]              # start index for component l
                Z_enc[i, col0 + val - 1] = 1   # categories 1..(k_l-1)
            end
            # If val == k_l, baseline category -> all zeros for this component
        end
    end

    # Encode group using (num_g - 1) dummies; last group is baseline.
    num_g = maximum(group)
    d_g   = max(num_g - 1, 0)
    G_enc = zeros(Int8, N, d_g)

    if d_g > 0
        for i in 1:N
            gi = group[i]
            @assert 1 ≤ gi ≤ num_g "Group index out of range at row $i"

            if gi < num_g
                # Non-baseline group in {1, ..., num_g - 1}
                G_enc[i, gi] = 1
            else
                # Baseline group num_g -> all zeros
            end
        end
    end

    info = ZGEncodingInfo(k_z, z_start, num_g)
    return Z_enc, G_enc, info
end



#############################################
# Reduced (Z_enc, G_enc) → Original (Z, group)
#############################################

"""
    decode_zg_reduced(Z_enc::AbstractMatrix{<:Integer},
                      G_enc::AbstractMatrix{<:Integer},
                      info::ZGEncodingInfo)

Decode reduced dummy encoded categorical and group features back to
their original integer-coded form.

For each categorical component l with k_l levels:
- If the corresponding block has all zeros, we recover category k_l
  (the baseline).
- Otherwise, the position of the 1 determines the category in {1, ..., k_l - 1}.

For the group variable with num_g levels:
- If the row in G_enc is all zeros, we recover group num_g (baseline).
- Otherwise, the position of the 1 determines the group in {1, ..., num_g - 1}.

Input:
- Z_enc : N × sum_l (k_l - 1) matrix (reduced dummies for Z)
- G_enc : N × (num_g - 1) matrix (reduced dummies for group)
- info  : ZGEncodingInfo produced earlier by `encode_zg_reduced`

Output:
- Z     : N × m matrix, each entry in {1, ..., k_l}
- group : length-N vector, each entry in {1, ..., num_g}
"""
function decode_zg_reduced(Z_enc::AbstractMatrix{<:Integer},
                           G_enc::AbstractMatrix{<:Integer},
                           info::ZGEncodingInfo)
    N, total_z = size(Z_enc)
    m = length(info.k_z)

    # Check group encoding consistency
    num_g = info.num_g
    d_g   = max(num_g - 1, 0)
    @assert size(G_enc, 1) == N "Z_enc and G_enc must have same number of rows"
    @assert size(G_enc, 2) == d_g "G_enc columns must be num_g - 1"

    # Recover categorical matrix Z (N × m)
    Z = zeros(Int, N, m)
    for l in 1:m
        k_l = info.k_z[l]
        d_l = max(k_l - 1, 0)
        s   = info.z_start[l]
        e   = s + d_l - 1

        if d_l == 0
            # Only one category exists for this component; always category 1.
            for i in 1:N
                Z[i, l] = 1
            end
            continue
        end

        @assert e ≤ total_z "Z_enc has too few columns for component $l"

        # View the reduced dummy block for component l
        sub = @view Z_enc[:, s:e]  # N × d_l
        for i in 1:N
            idx = findfirst(==(1), sub[i, :])
            if idx === nothing
                # All zeros: baseline category k_l
                Z[i, l] = k_l
            else
                # Non-baseline category in {1, ..., k_l - 1}
                Z[i, l] = idx
            end
        end
    end

    # Recover group vector (length N) from reduced dummies
    group = Vector{Int}(undef, N)
    if d_g == 0
        # Only one group; always group 1
        for i in 1:N
            group[i] = 1
        end
    else
        for i in 1:N
            idx = findfirst(==(1), G_enc[i, :])
            if idx === nothing
                # Baseline group num_g
                group[i] = num_g
            else
                # Non-baseline group in {1, ..., num_g - 1}
                group[i] = idx
            end
        end
    end

    return Z, group
end

###############################
# Graph structure for sample-level DAGs
# compatible with Theorem 2 (graph-based formulation)
###############################

# -----------------------------
# Arc kind (structure only)
# -----------------------------

"""
    ArcKind

Abstract type for different kinds of arcs in the DAG.
We distinguish between:
  - `CatArc` : categorical feature transitions
  - `TermArc`: terminal transitions to (m+1, 0) with a chosen group g
"""
abstract type ArcKind end

"""
    CatArc

Categorical arc corresponding to choosing category `c` for component `k`.

Fields:
- k      : which categorical component (1..m)
- c      : chosen category in {1, ..., k_z[k]}
- d_prev : previous accumulated categorical distance at state (k-1, d_prev)
- d      : new accumulated distance at state (k, d)
"""

struct CatArc <: ArcKind
    k::Int
    c::Int
    d_prev::Float64
    d::Float64
end

"""
    TermArc

Terminal arc from (m, d) to (m+1, 0) with a chosen destination group g.

Fields:
- d : accumulated categorical distance at state (m, d)
- g : destination group in {1, ..., num_g}
"""
struct TermArc <: ArcKind
    d::Float64
    g::Int
end

# -----------------------------
# Arc and per-sample DAG
# -----------------------------

"""
    Arc

Directed edge in the DAG for a single sample i.

Fields:
- src  : index of the source node in `nodes`
- dst  : index of the destination node in `nodes`
- kind : arc type (CatArc or TermArc), which encodes all information
         needed to build w^i(e; β, λ, r_i, ...) later
"""
struct Arc
    src::Int
    dst::Int
    kind::ArcKind
end






"""
    SampleDAG

Graph structure associated with a single sample i, compatible with
Theorem 2 (graph-based formulation).

Fields:
- sample_index : index i of the sample this DAG corresponds to
- nodes        : vector of DP states (k, d), plus the terminal node (m+1, 0.0)
                 each node is a Tuple{Int,Float64}
- arcs         : list of directed edges with their arc kind (no numeric weight)
- source       : index of the source node (corresponds to state (0, 0.0))
- sink         : index of the terminal node (corresponds to state (m+1, 0.0))
"""
struct SampleDAG
    sample_index::Int
    nodes::Vector{Tuple{Int,Float64}}
    arcs::Vector{Arc}
    source::Int
    sink::Int
end




# -----------------------------
# Categorical encoding info
# -----------------------------
"""
    CatEncodingInfo

Encoding information for categorical features (structure level only).

Fields:
- k_z::Vector{Int} : number of categories per component (length m).
                     For component k, categories are {1, ..., k_z[k]}.
- num_g::Int       : total number of groups (used for terminal arcs).
"""
struct CatEncodingInfo
    k_z::Vector{Int}
    num_g::Int
end


# -----------------------------
# Build DAG structure for one sample i
# -----------------------------
"""
    build_sample_dag_structure(i, info, delta, z_i) -> SampleDAG

Build the DAG structure G^i = (V^i, A^i) for a fixed sample i,
compatible with Theorem 2 (graph-based formulation), using our setup.

This function ONLY builds:
  - the nodes (states (k, d) plus the terminal (m+1, 0)),
  - the arcs with their structural information (CatArc or TermArc).

It does NOT compute numeric weights w^i(e). Those should be constructed
later as expressions of (β, λ, r_i, y_i, g_i, B_{g_i}, C_{g_i}, ...).

Arguments:
- i      : sample index
- info   : CatEncodingInfo (k_z, num_g)
- delta  : length-m vector of δ_k in
           d(z, zᶦ) = Σ_k δ_k * 1[z_k ≠ z_kᶦ]
- z_i    : length-m vector of original categories for sample i,
           each z_i[k] ∈ {1, ..., k_z[k]}

Output:
- SampleDAG describing the structure of G^i
"""
function build_sample_dag_structure(
    i::Int,
    info::CatEncodingInfo,
    delta::AbstractVector{<:Real},
    z_i::AbstractVector{<:Integer}
)::SampleDAG
    k_z   = info.k_z
    num_g = info.num_g
    m     = length(k_z)

    # -------------------------
    # 0. Sanity checks
    # -------------------------
    @assert length(delta) == m "delta must have length m"
    @assert length(z_i) == m "z_i must have length m"

    # -------------------------
    # 1. Enumerate states (k, d) with per-layer dedup
    # -------------------------
    nodes = Vector{Tuple{Int,Float64}}()
    node_index = Dict{Tuple{Int,Float64},Int}()

    # Source state (0, 0.0)
    push!(nodes, (0, 0.0))
    node_index[(0, 0.0)] = 1
    source_idx = 1

    arcs = Vector{Arc}()

    # current_layer holds all unique states (k-1, d_prev)
    current_layer = [(0, 0.0)]

    for k in 1:m
        next_layer = Tuple{Int,Float64}[]
        seen_next = Set{Tuple{Int,Float64}}()

        k_l   = k_z[k]
        δ_k   = float(delta[k])
        z_i_k = z_i[k]

        for (k_prev, d_prev) in current_layer
            @assert k_prev == k - 1

            # Enumerate all categories c ∈ {1, ..., k_l}
            for c in 1:k_l
                mismatch = (c != z_i_k)
                d = d_prev + (mismatch ? δ_k : 0.0)
                state = (k, d)

                # Add state to global node list if new
                if !haskey(node_index, state)
                    push!(nodes, state)
                    node_index[state] = length(nodes)
                end

                # Ensure each (k, d) appears at most once in next_layer
                if !(state in seen_next)
                    push!(next_layer, state)
                    push!(seen_next, state)
                end

                # Add categorical arc from (k-1, d_prev) to (k, d)
                src = node_index[(k-1, d_prev)]
                dst = node_index[state]
                kind = CatArc(k, c, d_prev, d)
                push!(arcs, Arc(src, dst, kind))
            end
        end

        # Move to next layer (already deduplicated)
        current_layer = next_layer
    end

    # States with k = m are the final DP layer S_1^i
    # current_layer is already deduplicated, but we keep the name for clarity
    S1_states = current_layer

    # -------------------------
    # 2. Add terminal node (m+1, 0.0) and terminal arcs
    # -------------------------
    terminal_state = (m+1, 0.0)
    push!(nodes, terminal_state)
    node_index[terminal_state] = length(nodes)
    sink_idx = node_index[terminal_state]

    # For each (m, d) in S_1^i, and for each group g, add a TermArc
    for (k_state, d) in S1_states
        @assert k_state == m
        src = node_index[(m, d)]
        for g in 1:num_g
            kind = TermArc(d, g)
            push!(arcs, Arc(src, sink_idx, kind))
        end
    end

    # -------------------------
    # 3. Return SampleDAG
    # -------------------------
    return SampleDAG(i, nodes, arcs, source_idx, sink_idx)
end

using JuMP

"""
    build_group_dro_graph_model(
        X, Z, group, y,
        encinfo,
        delta,
        A_group,
        B_group, C_group,
        gamma_x,
        ε,
        optimizer
    ) -> (model, meta)

Build our group-dependent, graph-based DRO logistic regression model,
using the same reduced encoding convention as `ZGEncodingInfo` /
`encode_zg_reduced`.

Arguments
---------
- X        :: N × n_x matrix of continuous features.
- Z        :: N × m   matrix of original categorical features.
              Entry Z[i, k] ∈ {1, ..., k_z[k]} (no one-hot).

- group    :: length-N vector of group indices g_i ∈ {1,...,num_g}.
- y        :: length-N vector of labels in {-1, +1}.

- encinfo  :: ZGEncodingInfo
              (k_z, z_start, num_g) describing reduced encoding blocks
              for the categorical features and groups.

- delta    :: length-m vector δ_k used in
              d_cat(z, z^i) = Σ_k δ_k * 1[z_k ≠ z_k^i].

- A_group  :: length-num_g vector A_g
              continuous-part metric weights in
              A_g Σ_j γ_j |x_j - x_j^i|.

- B_group  :: length-num_g vector B_g
              categorical-part metric weight.

- C_group  :: length-num_g vector C_g
              group-change penalty.

- gamma_x  :: length n_x vector γ_j for continuous features.
              Continuous dual constraint will be
                  |β_{xj}| ≤ λ * A_min * γ_j,
              where A_min = minimum(A_group).

- ε        :: Wasserstein radius ε.

- optimizer: optimizer constructor for JuMP, e.g.
              optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)

Returns
-------
- model :: JuMP.Model
- meta  :: NamedTuple: (encinfo=encinfo, dags=dags, n_nodes=n_nodes)
"""
function build_group_dro_graph_model(
    X::AbstractMatrix{<:Real},
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
    y::AbstractVector{<:Integer},
    encinfo::ZGEncodingInfo,
    delta::AbstractVector{<:Real},
    A_group::AbstractVector{<:Real},
    B_group::AbstractVector{<:Real},
    C_group::AbstractVector{<:Real},
    gamma_x::AbstractVector{<:Real},
    ε::Real,
    optimizer,
)
    # -------------------------
    # 0. Dimensions & sanity checks
    # -------------------------
    N, n_x = size(X)
    N_Z, m = size(Z)
    @assert N_Z == N "X and Z must have the same number of rows (samples)."
    @assert length(group) == N "group must have length N."
    @assert length(y) == N "y must have length N."
    @assert length(delta) == m "delta must have length m."
    @assert length(gamma_x) == n_x "gamma_x must have length n_x."

    k_z     = encinfo.k_z
    z_start = encinfo.z_start
    num_g   = encinfo.num_g

    @assert length(k_z) == m "encinfo.k_z must have length m."
    @assert length(z_start) == m "encinfo.z_start must have length m."
    @assert length(A_group) == num_g "A_group length must equal num_g."
    @assert length(B_group) == num_g "B_group length must equal num_g."
    @assert length(C_group) == num_g "C_group length must equal num_g."

    # Total length of β_z under reduced encoding:
    # last block starts at z_start[m], length (k_z[m] - 1)
    # so p_z = z_start[m] + (k_z[m] - 1) - 1
    p_z = z_start[end] + (k_z[end] - 1) - 1

    # Small positive constant for log(exp(inner) - 1) domain
    # We will enforce: r[i] + λ * inner_coeff ≥ η
    η = 1e-8

    # -------------------------
    # 1. Build per-sample DAGs (discrete part only)
    # -------------------------
    info = CatEncodingInfo(k_z, num_g)

    dags      = Vector{SampleDAG}(undef, N)
    n_nodes   = Vector{Int}(undef, N)
    max_nodes = 0

    for i in 1:N
        # Z[i, :] is an AbstractVector{<:Integer}
        dags[i] = build_sample_dag_structure(i, info, delta, Z[i, :])
        n_nodes[i] = length(dags[i].nodes)
        max_nodes = max(max_nodes, n_nodes[i])
    end

    # -------------------------
    # 2. Create JuMP model & decision variables
    # -------------------------
    model = Model(optimizer)

    # Wasserstein dual variable λ ≥ 0
    @variable(model, λ >= 0.0)

    # Per-sample slack variables r_i ∈ ℝ (free),
    # domain of log(exp(...)-1) will be enforced via extra constraints
    @variable(model, r[1:N])

    # Logistic regression parameters
    @variable(model, β0)                # intercept
    @variable(model, β_x[1:n_x])        # continuous coefficients
    @variable(model, β_z[1:p_z])        # categorical coefficients (reduced)
    @variable(model, β_grp[1:(num_g-1)])# group coefficients (reduced)

    # Graph dual variables μ_i_v for each sample i and each node v
    @variable(model, μ[1:N, 1:max_nodes])

    # -------------------------
    # 3. Objective: λ ε + (1/N) Σ r_i
    # -------------------------
    @objective(model, Min, λ * ε + (1.0 / N) * sum(r[i] for i in 1:N))

    # -------------------------
    # 4. Continuous part dual constraints with A_group
    #
    # Metric for x uses A_g:
    #   d_x(x^i, x; g) = A_g Σ_j γ_j |x_j - x_j^i|
    # Dual boundedness ⇒ for each j:
    #   |β_{xj}| ≤ λ * γ_j * min_g A_g
    # We enforce:
    #   -λ * A_min * γ_j ≤ β_x[j] ≤ λ * A_min * γ_j
    # -------------------------
    A_min = minimum(A_group)

    for j in 1:n_x
        @constraint(model,  β_x[j] <=  λ * A_min * gamma_x[j])
        @constraint(model, -β_x[j] <=  λ * A_min * gamma_x[j])
    end

    # -------------------------
    # 5. Outer logistic inequality:
    #    y^i (β_x^T x^i + β0) ≥ -μ_i(0,0) + μ_i(m+1,0)
    # -------------------------
    for i in 1:N
        dag = dags[i]
        s = dag.source
        t = dag.sink

        # Left-hand side: y_i * (β0 + β_x^T x^i)
        lhs = y[i] * (β0 + sum(β_x[j] * X[i, j] for j in 1:n_x))

        # Right-hand side: - μ_i(source) + μ_i(sink)
        rhs = - μ[i, s] + μ[i, t]

        @constraint(model, lhs >= rhs)
    end

    # -------------------------
    # 6. Edge constraints: μ_t(e) - μ_s(e) ≥ w^i(e)
    #
    # - For CatArc(k,c,...):
    #     w^i(e) = - y^i β_{z_k}^T z_k(c)
    #   Reduced encoding aligned with ZGEncodingInfo:
    #     if c < k_z[k], β_{z_k}^T z_k(c) = β_z[z_start[k] + c - 1]
    #     if c = k_z[k], baseline ⇒ 0.
    #
    # - For TermArc(d,g):
    #     w^i(e) =
    #       - y^i β_grp^T φ_g(g)
    #       - log( exp( r_i + λ (B_{g_i} d + C_{g_i} 1[g ≠ g_i]) ) - 1 )
    #
    #   with group reduced encoding:
    #     if g < num_g: β_grp^T φ_g(g) = β_grp[g]
    #     if g = num_g: baseline ⇒ 0.
    #
    #   Additionally, domain constraints:
    #     r_i + λ (B_{g_i} d + C_{g_i} 1[g ≠ g_i]) ≥ η
    #   ensure that log(exp(inner) - 1) is well-defined.
    # -------------------------
    for i in 1:N
        dag  = dags[i]
        y_i  = y[i]
        g_i  = group[i]

        for arc in dag.arcs
            src = arc.src
            dst = arc.dst

            if arc.kind isa CatArc
                kind = arc.kind::CatArc
                k = kind.k
                c = kind.c
                k_l = k_z[k]

                # Categorical part:
                # w_cat^i(e) = - y_i β_{z_k}^T z_k(c)
                if c < k_l
                    idx = z_start[k] + (c - 1)
                    w_expr = - y_i * β_z[idx]
                else
                    w_expr = 0.0
                end

                @constraint(model, μ[i, dst] - μ[i, src] >= w_expr)

            elseif arc.kind isa TermArc
                kind     = arc.kind::TermArc
                d_val    = kind.d
                g_choice = kind.g

                # Metric coefficient for categorical + group part:
                B_gi = float(B_group[g_i])
                C_gi = float(C_group[g_i])
                cross = (g_choice != g_i) ? C_gi : 0.0
                inner_coeff = B_gi * d_val + cross

                # --- Domain constraint: inner = r[i] + λ * inner_coeff ≥ η ---
                @NLconstraint(model, r[i] + λ * inner_coeff >= η)

                # --- w^i(e) constraint with nonlinear log/exp ---
                if g_choice < num_g
                    # group linear term: - y_i * β_grp[g_choice]
                    @NLconstraint(model,
                        μ[i, dst] - μ[i, src] >=
                        - y_i * β_grp[g_choice] -
                        log(exp(r[i] + λ * inner_coeff) - 1)
                    )
                else
                    # baseline group: no β_grp contribution
                    @NLconstraint(model,
                        μ[i, dst] - μ[i, src] >=
                        - log(exp(r[i] + λ * inner_coeff) - 1)
                    )
                end
            else
                error("Unknown arc kind in DAG.")
            end
        end
    end

    # -------------------------
    # 7. (Optional) Give Ipopt a safe starting point inside the domain
    # -------------------------
    set_start_value(λ, 1.0)
    for i in 1:N
        set_start_value(r[i], 1.0)
    end

    # -------------------------
    # 8. Return model + metadata
    # -------------------------
    meta = (
        encinfo = encinfo,
        dags    = dags,
        n_nodes = n_nodes,

        β0      = β0,
        β_x     = β_x,
        β_z     = β_z,
        β_grp   = β_grp,
        λ       = λ,
        r       = r,
        μ       = μ,
    )

    return model, meta
end


function build_group_dro_graph_model_swap(
    X::AbstractMatrix{<:Real},
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
    y::AbstractVector{<:Integer},
    encinfo::ZGEncodingInfo,
    delta::AbstractVector{<:Real},
    A_group::AbstractVector{<:Real},
    B_group::AbstractVector{<:Real},
    C_group::AbstractVector{<:Real},
    gamma_x::AbstractVector{<:Real},
    ε::Real,
    optimizer,
)
    # -------------------------
    # 0. Dimensions & sanity checks
    # -------------------------
    N, n_x = size(X)
    N_Z, m = size(Z)
    @assert N_Z == N "X and Z must have the same number of rows (samples)."
    @assert length(group) == N "group must have length N."
    @assert length(y) == N "y must have length N."
    @assert length(delta) == m "delta must have length m."
    @assert length(gamma_x) == n_x "gamma_x must have length n_x."

    k_z     = encinfo.k_z
    z_start = encinfo.z_start
    num_g   = encinfo.num_g

    @assert length(k_z) == m "encinfo.k_z must have length m."
    @assert length(z_start) == m "encinfo.z_start must have length m."
    @assert length(A_group) == num_g "A_group length must equal num_g."
    @assert length(B_group) == num_g "B_group length must equal num_g."
    @assert length(C_group) == num_g "C_group length must equal num_g."

    # Total length of β_z under reduced encoding:
    # last block starts at z_start[m], length (k_z[m] - 1)
    # so p_z = z_start[m] + (k_z[m] - 1) - 1
    p_z = z_start[end] + (k_z[end] - 1) - 1

    # Small positive constant for log(exp(inner) - 1) domain
    # We will enforce: r[i] + λ * inner_coeff ≥ η
    η = 1e-8

    # -------------------------
    # 1. Build per-sample DAGs (discrete part only)
    # -------------------------
    info = CatEncodingInfo(k_z, num_g)

    dags      = Vector{SampleDAG}(undef, N)
    n_nodes   = Vector{Int}(undef, N)
    max_nodes = 0

    for i in 1:N
        # Z[i, :] is an AbstractVector{<:Integer}
        dags[i] = build_sample_dag_structure(i, info, delta, Z[i, :])
        n_nodes[i] = length(dags[i].nodes)
        max_nodes = max(max_nodes, n_nodes[i])
    end

    # -------------------------
    # 2. Create JuMP model & decision variables
    # -------------------------
    model = Model(optimizer)

    # Wasserstein dual variable λ ≥ 0
    @variable(model, λ >= 0.0)

    # Per-sample slack variables r_i ∈ ℝ (free),
    # domain of log(exp(...)-1) will be enforced via extra constraints
    @variable(model, r[1:N])

    # Logistic regression parameters
    @variable(model, β0)                 # intercept
    @variable(model, β_x[1:n_x])         # continuous coefficients
    @variable(model, β_z[1:p_z])         # categorical coefficients (reduced)
    @variable(model, β_grp[1:(num_g-1)]) # group coefficients (reduced)

    # Graph dual variables μ_i_v for each sample i and each node v
    @variable(model, μ[1:N, 1:max_nodes])

    # -------------------------
    # 3. Objective: λ ε + (1/N) Σ r_i
    # -------------------------
    @objective(model, Min, λ * ε + (1.0 / N) * sum(r[i] for i in 1:N))

    # -------------------------
    # 4. Continuous part dual constraints with A_group
    #
    # Metric for x uses A_g:
    #   d_x(x^i, x; g) = A_g Σ_j γ_j |x_j - x_j^i|
    # Dual boundedness ⇒ for each j:
    #   |β_{xj}| ≤ λ * γ_j * min_g A_g
    # We enforce:
    #   -λ * A_min * γ_j ≤ β_x[j] ≤ λ * A_min * γ_j
    # -------------------------
    A_min = minimum(A_group)

    for j in 1:n_x
        @constraint(model,  β_x[j] <=  λ * A_min * gamma_x[j])
        @constraint(model, -β_x[j] <=  λ * A_min * gamma_x[j])
    end

    # -------------------------
    # 5. Outer logistic inequality:
    #    y^i (β_x^T x^i + β0) ≥ -μ_i(0,0) + μ_i(m+1,0)
    # -------------------------
    for i in 1:N
        dag = dags[i]
        s = dag.source
        t = dag.sink

        # Left-hand side: y_i * (β0 + β_x^T x^i)
        lhs = y[i] * (β0 + sum(β_x[j] * X[i, j] for j in 1:n_x))

        # Right-hand side: - μ_i(source) + μ_i(sink)
        rhs = - μ[i, s] + μ[i, t]

        @constraint(model, lhs >= rhs)
    end

    # -------------------------
    # 6. Edge constraints: μ_t(e) - μ_s(e) ≥ w^i(e)
    #
    # - For CatArc(k,c,...):
    #     w^i(e) = - y^i β_{z_k}^T z_k(c)
    #   Reduced encoding aligned with ZGEncodingInfo:
    #     if c < k_z[k], β_{z_k}^T z_k(c) = β_z[z_start[k] + c - 1]
    #     if c = k_z[k], baseline ⇒ 0.
    #
    # - For TermArc(d,g):
    #     w^i(e) =
    #       - y^i β_grp^T φ_g(g)
    #       - log( exp( r_i + λ * inner_coeff ) - 1 )
    #
    #   IMPORTANT (swapped-argument metric):
    #     inner_coeff uses TARGET group g_choice (not source g_i):
    #       inner_coeff = B_{g_choice} * d + C_{g_choice} * 1[g_choice ≠ g_i]
    #
    #   with group reduced encoding:
    #     if g < num_g: β_grp^T φ_g(g) = β_grp[g]
    #     if g = num_g: baseline ⇒ 0.
    #
    #   Additionally, domain constraints:
    #     r_i + λ * inner_coeff ≥ η
    #   ensure that log(exp(inner) - 1) is well-defined.
    # -------------------------
    for i in 1:N
        dag  = dags[i]
        y_i  = y[i]
        g_i  = group[i]

        for arc in dag.arcs
            src = arc.src
            dst = arc.dst

            if arc.kind isa CatArc
                kind = arc.kind::CatArc
                k = kind.k
                c = kind.c
                k_l = k_z[k]

                # Categorical part:
                # w_cat^i(e) = - y_i β_{z_k}^T z_k(c)
                if c < k_l
                    idx = z_start[k] + (c - 1)
                    w_expr = - y_i * β_z[idx]
                else
                    w_expr = 0.0
                end

                @constraint(model, μ[i, dst] - μ[i, src] >= w_expr)

            elseif arc.kind isa TermArc
                kind     = arc.kind::TermArc
                d_val    = kind.d
                g_choice = kind.g

                # Metric coefficient for categorical + group part (SWAPPED ARGUMENTS):
                # Use target group g_choice instead of source group g_i.
                B_tgt = float(B_group[g_choice])
                C_tgt = float(C_group[g_choice])
                cross = (g_choice != g_i) ? C_tgt : 0.0
                inner_coeff = B_tgt * d_val + cross

                # --- Domain constraint: inner = r[i] + λ * inner_coeff ≥ η ---
                @NLconstraint(model, r[i] + λ * inner_coeff >= η)

                # --- w^i(e) constraint with nonlinear log/exp ---
                if g_choice < num_g
                    # group linear term: - y_i * β_grp[g_choice]
                    @NLconstraint(model,
                        μ[i, dst] - μ[i, src] >=
                        - y_i * β_grp[g_choice] -
                        log(exp(r[i] + λ * inner_coeff) - 1)
                    )
                else
                    # baseline group: no β_grp contribution
                    @NLconstraint(model,
                        μ[i, dst] - μ[i, src] >=
                        - log(exp(r[i] + λ * inner_coeff) - 1)
                    )
                end
            else
                error("Unknown arc kind in DAG.")
            end
        end
    end

    # -------------------------
    # 7. (Optional) Give Ipopt a safe starting point inside the domain
    # -------------------------
    set_start_value(λ, 1.0)
    for i in 1:N
        set_start_value(r[i], 1.0)
    end

    # -------------------------
    # 8. Return model + metadata
    # -------------------------
    meta = (
        encinfo = encinfo,
        dags    = dags,
        n_nodes = n_nodes,

        β0      = β0,
        β_x     = β_x,
        β_z     = β_z,
        β_grp   = β_grp,
        λ       = λ,
        r       = r,
        μ       = μ,
    )

    return model, meta
end


using JuMP

"""
    build_group_dro_graph_model(
        X, Z, group, y,
        encinfo,
        delta,
        A_group,
        B_group, C_group,
        gamma_x,
        ε,
        optimizer
    ) -> (model, meta)

Build our group-dependent, graph-based DRO logistic regression model,
using the same reduced encoding convention as `ZGEncodingInfo` /
`encode_zg_reduced`.

Arguments
---------
- X        :: N × n_x matrix of continuous features.
- Z        :: N × m   matrix of original categorical features.
              Entry Z[i, k] ∈ {1, ..., k_z[k]} (no one-hot).

- group    :: length-N vector of group indices g_i ∈ {1,...,num_g}.
- y        :: length-N vector of labels in {-1, +1}.

- encinfo  :: ZGEncodingInfo
              (k_z, z_start, num_g) describing reduced encoding blocks
              for the categorical features and groups.

- delta    :: length-m vector δ_k used in
              d_cat(z, z^i) = Σ_k δ_k * 1[z_k ≠ z_k^i].

- A_group  :: length-num_g vector A_g
              continuous-part metric weights in
              A_g Σ_j γ_j |x_j - x_j^i|.

- B_group  :: length-num_g vector B_g
              categorical-part metric weight.

- C_group  :: length-num_g vector C_g
              group-change penalty.

- gamma_x  :: length n_x vector γ_j for continuous features.
              Continuous dual constraint will be
                  |β_{xj}| ≤ λ * A_min * γ_j,
              where A_min = minimum(A_group).

- ε        :: Wasserstein radius ε.

- optimizer: optimizer constructor for JuMP, e.g.
              optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)

Returns
-------
- model :: JuMP.Model
- meta  :: NamedTuple: (encinfo=encinfo, dags=dags, n_nodes=n_nodes)
"""
function build_group_dro_graph_model(
    X::AbstractMatrix{<:Real},
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
    y::AbstractVector{<:Integer},
    encinfo::ZGEncodingInfo,
    delta::AbstractVector{<:Real},
    A_group::AbstractVector{<:Real},
    B_group::AbstractVector{<:Real},
    C_group::AbstractVector{<:Real},
    gamma_x::AbstractVector{<:Real},
    ε::Real,
    optimizer,
)
    # -------------------------
    # 0. Dimensions & sanity checks
    # -------------------------
    N, n_x = size(X)
    N_Z, m = size(Z)
    @assert N_Z == N "X and Z must have the same number of rows (samples)."
    @assert length(group) == N "group must have length N."
    @assert length(y) == N "y must have length N."
    @assert length(delta) == m "delta must have length m."
    @assert length(gamma_x) == n_x "gamma_x must have length n_x."

    k_z     = encinfo.k_z
    z_start = encinfo.z_start
    num_g   = encinfo.num_g

    @assert length(k_z) == m "encinfo.k_z must have length m."
    @assert length(z_start) == m "encinfo.z_start must have length m."
    @assert length(A_group) == num_g "A_group length must equal num_g."
    @assert length(B_group) == num_g "B_group length must equal num_g."
    @assert length(C_group) == num_g "C_group length must equal num_g."

    # Total length of β_z under reduced encoding:
    # last block starts at z_start[m], length (k_z[m] - 1)
    # so p_z = z_start[m] + (k_z[m] - 1) - 1
    p_z = z_start[end] + (k_z[end] - 1) - 1

    # Small positive constant for log(exp(inner) - 1) domain
    # We will enforce: r[i] + λ * inner_coeff ≥ η
    η = 1e-8

    # -------------------------
    # 1. Build per-sample DAGs (discrete part only)
    # -------------------------
    info = CatEncodingInfo(k_z, num_g)

    dags      = Vector{SampleDAG}(undef, N)
    n_nodes   = Vector{Int}(undef, N)
    max_nodes = 0

    for i in 1:N
        # Z[i, :] is an AbstractVector{<:Integer}
        dags[i] = build_sample_dag_structure(i, info, delta, Z[i, :])
        n_nodes[i] = length(dags[i].nodes)
        max_nodes = max(max_nodes, n_nodes[i])
    end

    # -------------------------
    # 2. Create JuMP model & decision variables
    # -------------------------
    model = Model(optimizer)

    # Wasserstein dual variable λ ≥ 0
    @variable(model, λ >= 0.0)

    # Per-sample slack variables r_i ∈ ℝ (free),
    # domain of log(exp(...)-1) will be enforced via extra constraints
    @variable(model, r[1:N])

    # Logistic regression parameters
    @variable(model, β0)                # intercept
    @variable(model, β_x[1:n_x])        # continuous coefficients
    @variable(model, β_z[1:p_z])        # categorical coefficients (reduced)
    @variable(model, β_grp[1:(num_g-1)])# group coefficients (reduced)

    # Graph dual variables μ_i_v for each sample i and each node v
    @variable(model, μ[1:N, 1:max_nodes])

    # -------------------------
    # 3. Objective: λ ε + (1/N) Σ r_i
    # -------------------------
    @objective(model, Min, λ * ε + (1.0 / N) * sum(r[i] for i in 1:N))

    # -------------------------
    # 4. Continuous part dual constraints with A_group
    #
    # Metric for x uses A_g:
    #   d_x(x^i, x; g) = A_g Σ_j γ_j |x_j - x_j^i|
    # Dual boundedness ⇒ for each j:
    #   |β_{xj}| ≤ λ * γ_j * min_g A_g
    # We enforce:
    #   -λ * A_min * γ_j ≤ β_x[j] ≤ λ * A_min * γ_j
    # -------------------------
    A_min = minimum(A_group)

    for j in 1:n_x
        @constraint(model,  β_x[j] <=  λ * A_min * gamma_x[j])
        @constraint(model, -β_x[j] <=  λ * A_min * gamma_x[j])
    end

    # -------------------------
    # 5. Outer logistic inequality:
    #    y^i (β_x^T x^i + β0) ≥ -μ_i(0,0) + μ_i(m+1,0)
    # -------------------------
    for i in 1:N
        dag = dags[i]
        s = dag.source
        t = dag.sink

        # Left-hand side: y_i * (β0 + β_x^T x^i)
        lhs = y[i] * (β0 + sum(β_x[j] * X[i, j] for j in 1:n_x))

        # Right-hand side: - μ_i(source) + μ_i(sink)
        rhs = - μ[i, s] + μ[i, t]

        @constraint(model, lhs >= rhs)
    end

    # -------------------------
    # 6. Edge constraints: μ_t(e) - μ_s(e) ≥ w^i(e)
    #
    # - For CatArc(k,c,...):
    #     w^i(e) = - y^i β_{z_k}^T z_k(c)
    #   Reduced encoding aligned with ZGEncodingInfo:
    #     if c < k_z[k], β_{z_k}^T z_k(c) = β_z[z_start[k] + c - 1]
    #     if c = k_z[k], baseline ⇒ 0.
    #
    # - For TermArc(d,g):
    #     w^i(e) =
    #       - y^i β_grp^T φ_g(g)
    #       - log( exp( r_i + λ (B_{g_i} d + C_{g_i} 1[g ≠ g_i]) ) - 1 )
    #
    #   with group reduced encoding:
    #     if g < num_g: β_grp^T φ_g(g) = β_grp[g]
    #     if g = num_g: baseline ⇒ 0.
    #
    #   Additionally, domain constraints:
    #     r_i + λ (B_{g_i} d + C_{g_i} 1[g ≠ g_i]) ≥ η
    #   ensure that log(exp(inner) - 1) is well-defined.
    # -------------------------
    for i in 1:N
        dag  = dags[i]
        y_i  = y[i]
        g_i  = group[i]

        for arc in dag.arcs
            src = arc.src
            dst = arc.dst

            if arc.kind isa CatArc
                kind = arc.kind::CatArc
                k = kind.k
                c = kind.c
                k_l = k_z[k]

                # Categorical part:
                # w_cat^i(e) = - y_i β_{z_k}^T z_k(c)
                if c < k_l
                    idx = z_start[k] + (c - 1)
                    w_expr = - y_i * β_z[idx]
                else
                    w_expr = 0.0
                end

                @constraint(model, μ[i, dst] - μ[i, src] >= w_expr)

            elseif arc.kind isa TermArc
                kind     = arc.kind::TermArc
                d_val    = kind.d
                g_choice = kind.g

                # Metric coefficient for categorical + group part:
                # B_gi = float(B_group[g_i])
                # C_gi = float(C_group[g_i])
                # cross = (g_choice != g_i) ? C_gi : 0.0
                # inner_coeff = B_gi * d_val + cross

                # Metric coefficient for categorical + group part (SWAPPED ARGUMENTS):
                # Use target group g_choice instead of source group g_i.
                B_tgt = float(B_group[g_choice])
                C_tgt = float(C_group[g_choice])
                cross = (g_choice != g_i) ? C_tgt : 0.0
                inner_coeff = B_tgt * d_val + cross


                # --- Domain constraint: inner = r[i] + λ * inner_coeff ≥ η ---
                @NLconstraint(model, r[i] + λ * inner_coeff >= η)

                # --- w^i(e) constraint with nonlinear log/exp ---
                if g_choice < num_g
                    # group linear term: - y_i * β_grp[g_choice]
                    @NLconstraint(model,
                        μ[i, dst] - μ[i, src] >=
                        - y_i * β_grp[g_choice] -
                        log(exp(r[i] + λ * inner_coeff) - 1)
                    )
                else
                    # baseline group: no β_grp contribution
                    @NLconstraint(model,
                        μ[i, dst] - μ[i, src] >=
                        - log(exp(r[i] + λ * inner_coeff) - 1)
                    )
                end
            else
                error("Unknown arc kind in DAG.")
            end
        end
    end

    # -------------------------
    # 7. (Optional) Give Ipopt a safe starting point inside the domain
    # -------------------------
    set_start_value(λ, 1.0)
    for i in 1:N
        set_start_value(r[i], 1.0)
    end

    # -------------------------
    # 8. Return model + metadata
    # -------------------------
    meta = (
        encinfo = encinfo,
        dags    = dags,
        n_nodes = n_nodes,

        β0      = β0,
        β_x     = β_x,
        β_z     = β_z,
        β_grp   = β_grp,
        λ       = λ,
        r       = r,
        μ       = μ,
    )

    return model, meta
end


########################################################
# 1. Parameter containers
########################################################
"""
DROParams

Stores all hyperparameters that define the subgroup
Wasserstein ground metric and DRO radius:

"""
struct DROParams
    delta::Vector{Float64}      # length m, δ_ℓ for categorical features
    A_group::Vector{Float64}    # length num_g, A_g for continuous part
    B_group::Vector{Float64}    # length num_g, B_g for categorical part
    C_group::Vector{Float64}    # length num_g, C_g for group jumps
    gamma_x::Vector{Float64}    # length n_x, γ_j for continuous features
    epsilon::Float64            # Wasserstein radius ε
end


"""
LogitParams

Hyperparameters for standard (non-robust) logistic regression.
"""
struct LogitParams
    lambda_l2::Float64          # L2 regularization coefficient
end


"""
PerturbParams

Extra knobs for generating perturbed test sets.

The actual "cost" of moving features comes from DROParams
(via A_g * γ_j and B_g * δ_ℓ). Here we only store:

- mode : scenario label
- w    : half-width for continuous noise U(-w, w)
"""
struct PerturbParams
    mode::Symbol
    w::Float64
end

########################################################
# 2. Simple helper: DRO → Perturb
########################################################

"""
    dro_to_perturb_params(dro; w=0.4, mode=:generic)

Create a PerturbParams object from a DROParams, choosing only
the non-metric perturbation hyperparameters (mode, w).

The metric itself (A_group, B_group, gamma_x, delta) is used
directly inside the perturbation routine, not stored here.
"""
# function dro_to_perturb_params(
#     dro::DROParams;
#     kappa::Float64 = 1.0,
#     mode::Symbol = :generic,
# )
#     # Rough typical unit-cost across continuous, categorical, and group moves.
#     typ_x = mean(dro.A_group) * mean(dro.gamma_x)
#     typ_z = mean(dro.B_group) * mean(dro.delta)
#     typ_g = mean(dro.C_group)

#     typ_cost = mean([max(typ_x, 1e-6), max(typ_z, 1e-6), max(typ_g, 1e-6)])
#     w = kappa * dro.epsilon / typ_cost

#     return PerturbParams(mode, w)
# end

function dro_to_perturb_params(
    dro::DROParams;
    kappa::Float64 = 1.0,
    mode::Symbol = :generic,
)
    # Rough typical unit-cost across continuous, categorical, and group moves.
    typ_x = mean(dro.A_group) * mean(dro.gamma_x)
    typ_z = mean(dro.B_group) * mean(dro.delta)
    typ_g = mean(dro.C_group)

    typ_cost = mean([max(typ_x, 1e-6), max(typ_z, 1e-6), max(typ_g, 1e-6)])

    # IMPORTANT: normalize by the number of "channels" that can consume budget.
    # n_x continuous dims + m categorical dims + 1 group-jump channel.
    n_x = length(dro.gamma_x)
    m   = length(dro.delta)
    dim_factor = max(n_x + m + 1, 1)

    w = kappa * dro.epsilon / (typ_cost * dim_factor)

    return PerturbParams(mode, w)
end



########################################################
# 3. Model fitting (DRO vs Logistic) + shared prediction
########################################################

using JuMP
using Ipopt

"""
    fit_dro_model(X, Z, group, y, encinfo, params; optimizer)

Fit the subgroup-Wasserstein DRO logistic regression using
`build_group_dro_graph_model` and return coefficients for prediction.

Inputs
------
- X       : N×n_x continuous features
- Z       : N×m integer-coded categorical features
- group   : length-N group indices in {1,…,num_g}
- y       : length-N labels in {-1,+1}
- encinfo : ZGEncodingInfo (reduced encoding info)
- params  : DROParams
- optimizer : JuMP optimizer constructor (default Ipopt)

Returns
-------
NamedTuple (β0, β_x, β_z, β_grp, encinfo)
"""
function fit_dro_model(
    X::AbstractMatrix,
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
    y::AbstractVector{<:Integer},
    encinfo::ZGEncodingInfo,
    params::DROParams;
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
)
    model, meta = build_group_dro_graph_model(
        X,
        Z,
        group,
        y,
        encinfo,
        params.delta,
        params.A_group,
        params.B_group,
        params.C_group,
        params.gamma_x,
        params.epsilon,
        optimizer,
    )

    optimize!(model)

    β0̂   = value(meta.β0)
    βx̂   = value.(meta.β_x)
    βẑ   = value.(meta.β_z)
    βgrp̂ = value.(meta.β_grp)

    return (
        β0    = β0̂,
        β_x   = βx̂,
        β_z   = βẑ,
        β_grp = βgrp̂,
        encinfo = encinfo,
    )
end


"""
    fit_logistic_model(Xtr, Zenc_tr, Genc_tr, ytr, encinfo, params; optimizer)

Fit standard logistic regression with L2 regularization, using the
same reduced encoding as the DRO model:

    f_β(x,z,g) = β₀ + β_xᵀ x + β_zᵀ φ_z(z) + β_grpᵀ φ_g(g)

and

    min_β (1/N) Σ log(1 + exp(-y_i f_β(x_i,z_i,g_i)))
          + (λ/2)(‖β_x‖² + ‖β_z‖² + ‖β_grp‖²),

with λ = params.lambda_l2.
"""
function fit_logistic_model(
    Xtr::AbstractMatrix,
    Zenc_tr::AbstractMatrix,
    Genc_tr::AbstractMatrix,
    ytr::AbstractVector{<:Integer},
    encinfo::ZGEncodingInfo,
    params::LogitParams;
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
)
    N, n_x  = size(Xtr)
    N2, p_z = size(Zenc_tr)
    N3, p_g = size(Genc_tr)

    @assert N2 == N "Zenc_tr must have same rows as Xtr"
    @assert N3 == N "Genc_tr must have same rows as Xtr"

    λ = params.lambda_l2

    model = Model(optimizer)

    @variable(model, β0)
    @variable(model, β_x[1:n_x])
    @variable(model, β_z[1:p_z])
    @variable(model, β_grp[1:p_g])

    # Logistic loss + L2 penalty (no penalty on intercept)
    @NLobjective(model, Min,
        (1.0 / N) * sum(
            log(1 + exp(-ytr[i] * (
                β0
                + sum(β_x[j] * Xtr[i, j] for j in 1:n_x)
                + sum(β_z[k] * Zenc_tr[i, k] for k in 1:p_z)
                + sum(β_grp[h] * Genc_tr[i, h] for h in 1:p_g)
            )))
            for i in 1:N
        )
        + (λ / 2.0) * (
            sum(β_x[j]^2 for j in 1:n_x) +
            sum(β_z[k]^2 for k in 1:p_z) +
            sum(β_grp[h]^2 for h in 1:p_g)
        )
    )

    optimize!(model)

    return (
        β0    = value(β0),
        β_x   = value.(β_x),
        β_z   = value.(β_z),
        β_grp = value.(β_grp),
        encinfo = encinfo,
    )
end


########################################################
# Shared prediction functions
########################################################

# Logistic link
σ(t) = 1.0 / (1.0 + exp(-t))

"""
    predict_scores(β, X, Z, group)

Compute linear scores f_β(x,z,g) for each row, using the
encoding described by β.encinfo.
"""
function predict_scores(
    β,
    X::AbstractMatrix,
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
)
    N, n_x = size(X)
    k_z     = β.encinfo.k_z
    z_start = β.encinfo.z_start
    num_g   = β.encinfo.num_g

    m = length(k_z)
    @assert size(Z, 2) == m
    @assert length(group) == N

    scores = zeros(Float64, N)

    for i in 1:N
        s = β.β0 + dot(β.β_x, view(X, i, :))

        # categorical part
        for ℓ in 1:m
            val = Z[i, ℓ]
            k_l = k_z[ℓ]
            if val < k_l
                idx = z_start[ℓ] + (val - 1)
                s += β.β_z[idx]
            end
        end

        # group part
        g_i = group[i]
        if g_i < num_g
            s += β.β_grp[g_i]
        end

        scores[i] = s
    end

    return scores
end

"""
    predict_proba(β, X, Z, group)

Return P(y=+1 | x,z,g) = σ(f_β(x,z,g)) for each row.
"""
predict_proba(β, X, Z, group) = σ.(predict_scores(β, X, Z, group))

"""
    predict_label(β, X, Z, group)

Return hard labels in {-1,+1} using sign(f_β).
"""
function predict_label(
    β,
    X::AbstractMatrix,
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
)
    scores = predict_scores(β, X, Z, group)
    yhat   = Vector{Int}(undef, length(scores))
    @inbounds for i in eachindex(scores)
        yhat[i] = scores[i] >= 0 ? 1 : -1
    end
    return yhat
end


########################################################
# 4. Group-aware test-set perturbation driven by metric
########################################################

using Random
using Distributions   # for Laplace

"""
    perturb_testset(X, Z, group, y, encinfo, dro, pert; rng)

Generate one perturbed test set (X̃, Z̃, g̃, ỹ) from
(X, Z, group, y), using the subgroup Wasserstein metric:

d(ξ^i, ξ) =
  A_g Σ_j γ_j |x_j - x_j^i|
+ B_g Σ_ℓ δ_ℓ 1[z_ℓ ≠ z_ℓ^i]
+ C_g 1[g ≠ g^i].

We use these products A_g·γ_j, B_g·δ_ℓ, C_g to control the
amount of noise:

Continuous features:
  For sample i and feature j (group g_i):
    cost_x = A_{g_i} * γ_j
    scale  = w / cost_x
    Δ_raw  ~ Laplace(0, scale)
    Δ      = clamp(Δ_raw, -w, w)
    X̃[i,j] = X[i,j] + Δ

Categorical features:
  For sample i and feature ℓ (group g_i):
    cost_z = B_{g_i} * δ_ℓ
    q      = 1 - exp(- w / cost_z)   ∈ (0,1)
    With prob 1-q: keep z_ℓ^i
    With prob q  : change uniformly to one of the other levels.

Group index:
  For sample i:
    cost_g = C_{g_i}
    qg     = 1 - exp(- w / cost_g)   ∈ (0,1)
    With prob 1-qg: keep g_i
    With prob qg  : change to a different group, uniformly.

Inputs:
- X, Z, group, y : test data
- encinfo        : ZGEncodingInfo (k_z, num_g)
- dro            : DROParams (delta, A_group, B_group, C_group, gamma_x, epsilon)
- pert           : PerturbParams (mode, w)
- rng            : random number generator

Returns:
- X̃, Z̃, g̃, ỹ : perturbed copies (same shapes as inputs)
"""
function perturb_testset(
    X::AbstractMatrix,
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
    y::AbstractVector,
    encinfo::ZGEncodingInfo,
    dro::DROParams,
    pert::PerturbParams;
    rng = Random.default_rng(),
)
    N, n_x = size(X)
    _, m   = size(Z)
    k_z    = encinfo.k_z
    num_g  = encinfo.num_g

    @assert length(dro.gamma_x) == n_x
    @assert length(dro.delta)   == m
    @assert length(dro.A_group) == num_g
    @assert length(dro.B_group) == num_g
    @assert length(dro.C_group) == num_g

    X̃ = copy(X)
    Z̃ = copy(Z)
    g̃ = copy(group)
    ỹ = copy(y)

    w = pert.w
    @assert w > 0 "pert.w must be positive"

    for i in 1:N
        g_i = group[i]
        @assert 1 ≤ g_i ≤ num_g

        A_gi = dro.A_group[g_i]
        B_gi = dro.B_group[g_i]
        C_gi = max(dro.C_group[g_i], 1e-6)  # avoid zero

        # -------------------------
        # 1. Continuous features
        # -------------------------
        for j in 1:n_x
            cost_x = max(A_gi * dro.gamma_x[j], 1e-6)
            scale  = w / cost_x
            Δ_max = w / cost_x
            Δ = rand(rng, Laplace(0.0, scale))
            # while abs(Δ) > Δ_max
            #     Δ = rand(rng, Laplace(0.0, scale))
            # end
            X̃[i,j] = X[i,j] + Δ
        end

        # -------------------------
        # 2. Categorical features
        # -------------------------
        for ℓ in 1:m
            k_l   = k_z[ℓ]
            cost_z = max(B_gi * dro.delta[ℓ], 1e-6)
            # higher cost_z → smaller q
            q = 1.0 - exp(- w / cost_z)
            q = clamp(q, 0.0, 0.99)

            if rand(rng) < q
                old = Z[i, ℓ]
                # choose any other category uniformly
                choices = Vector{Int}(undef, k_l - 1)
                idx = 1
                for c in 1:k_l
                    if c != old
                        choices[idx] = c
                        idx += 1
                    end
                end
                Z̃[i, ℓ] = rand(rng, choices)
            end
        end

        # -------------------------
        # 3. Group index
        # -------------------------
        # If C_g very large, qg is very small (hard to move group)
        qg = 1.0 - exp(- w / C_gi)
        qg = clamp(qg, 0.0, 0.99)

        if rand(rng) < qg
            choices_g = Vector{Int}(undef, num_g - 1)
            idxg = 1
            for gg in 1:num_g
                if gg != g_i
                    choices_g[idxg] = gg
                    idxg += 1
                end
            end
            g̃[i] = rand(rng, choices_g)
        end
    end

    return X̃, Z̃, g̃, ỹ
end



########################################################
# 5. Evaluation metrics: AUC and ACE
########################################################

using Statistics

"""
    auc_binary(scores, y)

Binary AUC (ROC area).

- `scores`: model scores (logits or probabilities).
- `y`: true labels, positives are `y > 0`.

Implements the classic rank-based formula (Mann–Whitney U):

    AUC = (∑ rank(pos) − P(P+1)/2) / (P·N),

where P = #positives, N = #negatives.
Returns 0.5 if all labels are the same.
"""
function auc_binary(scores::AbstractVector{<:Real},
                    y::AbstractVector{<:Real})
    @assert length(scores) == length(y)
    n = length(scores)

    # map labels to 0/1
    y01 = Vector{Int}(undef, n)
    for i in 1:n
        y01[i] = y[i] > 0 ? 1 : 0
    end

    P = sum(y01)
    N = n - P
    (P == 0 || N == 0) && return 0.5

    # ranks of scores (ascending)
    idx   = sortperm(scores)
    ranks = similar(scores, Float64)
    for (r, i) in enumerate(idx)
        ranks[i] = r
    end

    # sum of ranks for positives
    sum_r_pos = 0.0
    for i in 1:n
        if y01[i] == 1
            sum_r_pos += ranks[i]
        end
    end

    auc = (sum_r_pos - P * (P + 1) / 2) / (P * N)
    return auc
end


"""
    ace_binary(probs, y; B = 10)

Adaptive Calibration Error (ACE) for binary classification.

- `probs`: predicted P(y=1) ∈ [0,1].
- `y`: true labels, positives are `y > 0`.
- `B`: number of equal-mass bins (default 10).

Procedure:
1. Sort by `probs`.
2. Split into B bins with (almost) equal size.
3. For each bin:
       | mean(probs) − mean(labels) |
4. ACE = average of these B values.

Lower ACE = better calibration.
"""
function ace_binary(probs::AbstractVector{<:Real},
                    y::AbstractVector{<:Real};
                    B::Int = 10)
    @assert length(probs) == length(y)
    n = length(probs)
    n == 0 && return 0.0

    # map labels to 0/1
    y01 = Vector{Float64}(undef, n)
    for i in 1:n
        y01[i] = y[i] > 0 ? 1.0 : 0.0
    end

    B = min(B, n)

    # sort by predicted probability
    idx = sortperm(probs)

    base  = div(n, B)
    extra = n % B

    start   = 1
    err_sum = 0.0
    for b in 1:B
        sz   = base + (b <= extra ? 1 : 0)
        stop = start + sz - 1

        inds = idx[start:stop]
        mean_conf = mean(@view probs[inds])
        mean_acc  = mean(@view y01[inds])

        err_sum += abs(mean_acc - mean_conf)
        start = stop + 1
    end

    return err_sum / B
end


"""
    default_metrics(probs, y)

Convenience wrapper:

- `auc`: ROC AUC (higher is better)
- `ace`: Adaptive Calibration Error (lower is better)
"""
default_metrics(probs::AbstractVector{<:Real},
                y::AbstractVector{<:Real}) = (
    auc = auc_binary(probs, y),
    ace = ace_binary(probs, y; B = 10),
)

########################################################
# 6. High-level experiment driver (UNPAIRED)
########################################################

"""
    run_experiment(
        X, Z, group, y;
        train_ratio = 0.7,
        method = :dro,
        dro_params = nothing,
        logit_params = nothing,
        pert_params::PerturbParams,
        n_splits = 1,
        n_pert_per_split = 100,
        metric_fun = default_metrics,
        rng = Random.default_rng(),
        optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
    )

Run a robustness experiment for one model type:

- `method = :dro`       → subgroup DRO logistic regression
- `method = :logistic`  → standard L2 logistic regression

For each split s = 1,…,n_splits:

1. Randomly split data into train / test (ratio = `train_ratio`).
2. Fit the chosen model on the train set.
3. Generate `n_pert_per_split` independent perturbed test sets
   with `perturb_testset`, using the *same* DRO metric (`dro_params`)
   but possibly different model (`method`).
4. On each perturbed test set, compute metrics via `metric_fun`.
5. Aggregate metrics over perturbations into:
   - average performance (mean AUC / ACE),
   - worst-case performance (min AUC / max ACE).

Note (important):
- Even for `method = :logistic`, `dro_params` is still required: it
  defines the ground metric that drives the test-set perturbations.
- DRO vs logistic are compared under the same *shift model*.

Returns:
- `Vector{NamedTuple}`, one per split, with fields:

    (split, auc_avg, auc_min, ace_avg, ace_max)
"""
function run_experiment(
    X::AbstractMatrix,
    Z::AbstractMatrix{<:Integer},
    group::AbstractVector{<:Integer},
    y::AbstractVector,
    ;
    train_ratio::Float64 = 0.7,
    method::Symbol = :dro,
    dro_params::Union{DROParams,Nothing} = nothing,
    logit_params::Union{LogitParams,Nothing} = nothing,
    pert_params::PerturbParams,
    n_splits::Int = 1,
    n_pert_per_split::Int = 100,
    metric_fun = default_metrics,
    rng = Random.default_rng(),
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
)
    N = size(X, 1)
    @assert 0.0 < train_ratio < 1.0 "train_ratio must be in (0,1)"
    @assert size(Z, 1) == N
    @assert length(group) == N
    @assert length(y) == N
    @assert dro_params !== nothing "dro_params is required (also for :logistic, to drive perturbations)"

    dro = dro_params::DROParams

    # Encode Z and group once, so DRO and logistic share the same encoding.
    Z_enc_all, G_enc_all, encinfo = encode_zg_reduced(Z, group)

    results = NamedTuple[]

    for s in 1:n_splits
        # 1. Random train/test split
        idx      = randperm(rng, N)
        n_train  = max(1, min(N - 1, round(Int, train_ratio * N)))
        train_idx = idx[1:n_train]
        test_idx  = idx[n_train+1:end]

        Xtr = X[train_idx, :]
        Ztr = Z[train_idx, :]
        gtr = group[train_idx]
        ytr = y[train_idx]

        Xte = X[test_idx, :]
        Zte = Z[test_idx, :]
        gte = group[test_idx]
        yte = y[test_idx]

        Zenc_tr = Z_enc_all[train_idx, :]
        Genc_tr = G_enc_all[train_idx, :]

        # 2. Fit model
        β =
            if method == :dro
                fit_dro_model(
                    Xtr, Ztr, gtr, ytr,
                    encinfo, dro;
                    optimizer = optimizer,
                )
            elseif method == :logistic
                @assert logit_params !== nothing "logit_params must be provided for method = :logistic"
                fit_logistic_model(
                    Xtr, Zenc_tr, Genc_tr, ytr,
                    encinfo, logit_params;
                    optimizer = optimizer,
                )
            else
                error("Method $(method) not implemented. Use :dro or :logistic.")
            end

        # 3. Monte Carlo over perturbed test sets
        metric_vals = Vector{NamedTuple}(undef, n_pert_per_split)

        for r in 1:n_pert_per_split
            if r == 1 || r % 10 == 0 || r == n_pert_per_split
                println("[run_experiment] Split $(s)/$(n_splits), perturb $(r)/$(n_pert_per_split)")
            end
            X̃, Z̃, g̃, ỹ = perturb_testset(
                Xte, Zte, gte, yte,
                encinfo, dro, pert_params;
                rng = rng,
            )
            probs = predict_proba(β, X̃, Z̃, g̃)
            metric_vals[r] = metric_fun(probs, ỹ)
        end

        # 4. Aggregate to average / worst-case
        aucs = [mv.auc for mv in metric_vals]
        aces = [mv.ace for mv in metric_vals]

        push!(results, (
            split   = s,
            auc_avg = mean(aucs),
            auc_min = minimum(aucs),
            ace_avg = mean(aces),
            ace_max = maximum(aces),
        ))
    end

    return results
end



using CSV
using DataFrames
using Statistics

"""
    load_churn_dataset(path; standardize=true)

Load the Bank Customer Churn dataset and convert it to:
- X :: Matrix{Float64}         (continuous features)
- Z :: Matrix{Int}             (integer-coded categorical features, excluding group)
- g :: Vector{Int}             (group index derived from country)
- y :: Vector{Int}             (labels in {-1,+1})
- meta :: NamedTuple           (column info + level mappings + standardization stats)

Conventions:
- group g comes from `country` (France/Germany/Spain), mapped to 1..num_g.
- categorical Z includes: gender, products_number, credit_card, active_member.
- continuous X includes: credit_score, age, tenure, balance, estimated_salary.
"""
function load_churn_dataset(path::AbstractString; standardize::Bool = true)
    df = CSV.read(path, DataFrame)

    # -------------------------
    # 1) Define columns
    # -------------------------
    group_col = :country
    y_col     = :churn

    x_cols = [:credit_score, :age, :tenure, :balance, :estimated_salary]
    z_cols = [:gender, :products_number, :credit_card, :active_member]

    N = nrow(df)

    # -------------------------
    # 2) Build group g from country (SOURCE group concept used elsewhere)
    # -------------------------
    # Fix an explicit order so experiments are reproducible.
    group_levels = ["France", "Germany", "Spain"]
    group_map = Dict(level => i for (i, level) in enumerate(group_levels))

    g = Vector{Int}(undef, N)
    for i in 1:N
        c = string(df[i, group_col])
        @assert haskey(group_map, c) "Unknown country level: $c"
        g[i] = group_map[c]
    end

    # -------------------------
    # 3) Build y in {-1,+1}
    # -------------------------
    y = Vector{Int}(undef, N)
    for i in 1:N
        y[i] = (df[i, y_col] == 1) ? 1 : -1
    end

    # -------------------------
    # 4) Continuous matrix X
    # -------------------------
    X = Matrix{Float64}(undef, N, length(x_cols))
    for (j, col) in enumerate(x_cols)
        X[:, j] = Float64.(df[:, col])
    end

    x_mean = zeros(Float64, size(X, 2))
    x_std  = ones(Float64,  size(X, 2))

    if standardize
        for j in 1:size(X, 2)
            μ = mean(@view X[:, j])
            σ = std(@view X[:, j])
            σ = (σ <= 1e-12) ? 1.0 : σ
            X[:, j] .= (X[:, j] .- μ) ./ σ
            x_mean[j] = μ
            x_std[j]  = σ
        end
    end

    # -------------------------
    # 5) Categorical matrix Z (integer-coded 1..k per column)
    # -------------------------
    Z = Matrix{Int}(undef, N, length(z_cols))
    z_levels = Vector{Vector{String}}(undef, length(z_cols))
    z_maps   = Vector{Dict{String,Int}}(undef, length(z_cols))

    for (ℓ, col) in enumerate(z_cols)
        # Collect levels in a stable order (sorted string)
        lvls = sort!(unique(string.(df[:, col])))
        z_levels[ℓ] = lvls
        mp = Dict(lvl => i for (i, lvl) in enumerate(lvls))
        z_maps[ℓ] = mp

        for i in 1:N
            s = string(df[i, col]) 
            Z[i, ℓ] = mp[s]
        end
    end

    meta = (
        x_cols = x_cols,
        z_cols = z_cols,
        group_col = group_col,
        y_col = y_col,
        group_levels = group_levels,
        group_map = group_map,
        z_levels = z_levels,
        z_maps = z_maps,
        standardize = standardize,
        x_mean = x_mean,
        x_std  = x_std,
    )

    return X, Z, g, y, meta
end

using Statistics  
"""
    summarize_experiment(res_vec)

Given the vector of per-split results returned by `run_experiment`,
compute simple means across splits.

Each element of `res_vec` is expected to have fields:
    (split, auc_avg, auc_min, ace_avg, ace_max)

Returns a NamedTuple:
    (auc_avg_mean, auc_min_mean, ace_avg_mean, ace_max_mean)
"""
function summarize_experiment(res_vec::Vector{<:NamedTuple})
    auc_avg_mean = mean(r.auc_avg for r in res_vec)
    auc_min_mean = mean(r.auc_min for r in res_vec)
    ace_avg_mean = mean(r.ace_avg for r in res_vec)
    ace_max_mean = mean(r.ace_max for r in res_vec)
    return (
        auc_avg_mean = auc_avg_mean,
        auc_min_mean = auc_min_mean,
        ace_avg_mean = ace_avg_mean,
        ace_max_mean = ace_max_mean,
    )
end


############################################################
# Churn: scenario-based DRO + perturb parameters (U / V / R)
############################################################

"""
    build_churn_scenario_params(scenario, theta;
                                n_x, m, num_g, kappa = 1.0)

Build (dro_params, pert_params) for the Bank Churn dataset
under one of three subgroup metric scenarios:

- :U (Uniform):   A_g = B_g = C_g = 1         for all groups.
- :V (Vulnerable): (A,B,C) = (0.5,0.5,0.5) for group 1,
                   (1,1,1)               for group 2,
                   (2,2,2)               for group 3.
- :R (Reversed):   same pattern as :V but swap group 1 and 3.

Here we keep the data-generating side (γ_x, δ) simple:

- γ_x[j] = 1     for all continuous features (already standardized).
- δ[ℓ]  = 1     for all categorical features.
- ε = -log(theta).

The perturbation object only carries (mode, w ) , where w is derived from (epsilon, kappa).
(A_g · γ_x, B_g · δ, C_g) controls how large the actual shifts are
inside `perturb_testset`.
"""
function build_churn_scenario_params(
    scenario::Symbol,
    theta::Float64;
    n_x::Int,
    m::Int,
    num_g::Int,
    kappa::Float64 = 1.0,
)
    @assert num_g == 3 "This helper assumes 3 groups (France / Germany / Spain)."

    # Base feature scales
    gamma_x = ones(Float64, n_x)
    delta   = ones(Float64, m)

    # Scenario-specific A_g, B_g, C_g
    A_group = zeros(Float64, num_g)
    B_group = zeros(Float64, num_g)
    C_group = zeros(Float64, num_g)

    if scenario == :U
        # All groups equally easy to move
        A_group .= 1.0
        B_group .= 1.0
        C_group .= 1.0

    elseif scenario == :V
        # Group 1 most vulnerable, group 3 most robust
        A_group .= [0.5, 1.0, 2.0]
        B_group .= [0.5, 1.0, 2.0]
        C_group .= [0.5, 1.0, 2.0]

    elseif scenario == :R
        # Reverse: group 3 most vulnerable, group 1 most robust
        A_group .= [2.0, 1.0, 0.5]
        B_group .= [2.0, 1.0, 0.5]
        C_group .= [2.0, 1.0, 0.5]

    else
        error("Unknown scenario = $scenario. Use :U, :V or :R.")
    end

    # DRO radius
    epsilon = -log(theta)

    dro_params = DROParams(
        delta,
        A_group,
        B_group,
        C_group,
        gamma_x,
        epsilon,
    )

    # Perturbation knob w: global noise budget
    pert_params = dro_to_perturb_params(dro_params; kappa = kappa, mode = scenario)

    return dro_params, pert_params
end

############################################################
# Churn experiment driver (U / V / R, 1000-sample subset)
############################################################

using Random
using Statistics
using Ipopt

"""
    run_churn_scenarios_uvr(
        path;
        scenarios = [:U, :V, :R],
        thetas = [0.5, 0.75, 0.9],
        train_ratio = 0.7,
        n_splits = 5,
        n_pert_per_split = 100,
        lambda_l2 = 1e-2,
        max_samples = 1000,
        kappa = 1.0,
        seed = 2025,
        optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
    )

Apply the same U / V / R subgroup-metric pipeline to the
Bank Customer Churn dataset, using only a random subset of
at most `max_samples` points for speed.

For each scenario ∈ {U,V,R} and θ ∈ {0.5,0.75,0.9}:

1. Build (dro_params, pert_params) via `build_churn_scenario_params`.
2. Run `run_experiment` with method = :dro.
3. Run `run_experiment` with method = :logistic (same perturbation law).
4. Aggregate per-split summaries into means over splits.

Returns a vector of NamedTuples with fields:
    (scenario, theta, method, auc_avg_mean, auc_min_mean,
     ace_avg_mean, ace_max_mean)
"""
function run_churn_scenarios_uvr(
    path::AbstractString;
    scenarios = [:U, :V, :R],
    thetas = [0.5, 0.75, 0.9],
    train_ratio::Float64 = 0.7,
    n_splits::Int = 5,
    n_pert_per_split::Int = 100,
    lambda_l2::Float64 = 1e-2,
    max_samples::Int = 1000,
    kappa::Float64 = 1.0,
    seed::Int = 2025,
    optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
)
    rng = MersenneTwister(seed)

    # 1) Load and subsample dataset
    X_full, Z_full, g_full, y_full, meta = load_churn_dataset(path; standardize = true)
    N_total = size(X_full, 1)
    N_use   = min(max_samples, N_total)

    idx_sub = randperm(rng, N_total)[1:N_use]
    X = X_full[idx_sub, :]
    Z = Z_full[idx_sub, :]
    g = g_full[idx_sub]
    y = y_full[idx_sub]

    n_x  = size(X, 2)
    m    = size(Z, 2)
    num_g = maximum(g)

    logit_params = LogitParams(lambda_l2)

    results = NamedTuple[]

    for scen in scenarios
        #println(scen)
        for θ in thetas
            println("Scenario = ", scen, ", theta = ", θ, ", kappa = ", kappa)
            # 2) Build DRO + perturbation parameters for this scenario
            dro_params, pert_params = build_churn_scenario_params(
                scen, θ;
                n_x = n_x,
                m = m,
                num_g = num_g,
                kappa = kappa,
            )

            # 3) DRO model
            res_dro = run_experiment(
                X, Z, g, y;
                train_ratio = train_ratio,
                method = :dro,
                dro_params = dro_params,
                logit_params = logit_params,
                pert_params = pert_params,
                n_splits = n_splits,
                n_pert_per_split = n_pert_per_split,
                rng = rng,
                optimizer = optimizer,
            )

            # 4) Logistic baseline
            res_logit = run_experiment(
                X, Z, g, y;
                train_ratio = train_ratio,
                method = :logistic,
                dro_params = dro_params,
                logit_params = logit_params,
                pert_params = pert_params,
                n_splits = n_splits,
                n_pert_per_split = n_pert_per_split,
                rng = rng,
                optimizer = optimizer,
            )

            # 5) Average the per-split summaries
            for (method, res) in zip((:dro, :logistic), (res_dro, res_logit))
                auc_avg_mean = mean(r.auc_avg for r in res)
                auc_min_mean = mean(r.auc_min for r in res)
                ace_avg_mean = mean(r.ace_avg for r in res)
                ace_max_mean = mean(r.ace_max for r in res)

                push!(results, (
                    scenario     = scen,
                    theta        = θ,
                    method       = method,
                    auc_avg_mean = auc_avg_mean,
                    auc_min_mean = auc_min_mean,
                    ace_avg_mean = ace_avg_mean,
                    ace_max_mean = ace_max_mean,
                ))
            end
        end
    end

    return results
end

path = "Bank Customer Churn Prediction.csv"

results_churn_uvr = run_churn_scenarios_uvr(
    path;
    scenarios = [:U, :V, :R],
    thetas = [0.5, 0.75, 0.9],
    train_ratio = 0.7,
    n_splits = 10,
    n_pert_per_split = 40,
    lambda_l2 = 1e-2,
    max_samples = 1000,
    kappa = 1.0,
    seed = 2025,
)


using Serialization
mkpath("out")
serialize(joinpath("out","results_churn_uvr.jls"), results_churn_uvr)


