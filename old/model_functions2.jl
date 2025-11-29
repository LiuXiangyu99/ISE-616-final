using JuMP
using Ipopt   
using MathOptInterface
const MOI = MathOptInterface

# We will NOT import a specific optimizer here.
# When you call build_subgroup_dro_graph_model, pass something like:
#   using Ipopt
#   model = build_subgroup_dro_graph_model(...; optimizer = Ipopt.Optimizer)
#
# The model uses @NLconstraint (log / exp), so you need an NLP-capable solver.

"""
    add_terminal_arc_nl!(
        model, μ, i, u, sink_id,
        r_i, λ, d_weighted
    )

Add the nonlinear constraint corresponding to the "sink arc" in the graph-based
formulation (Theorem 2, arcs from (m, d) to (m+1, 0)):

    μ[i, sink] - μ[i, u] ≥ -log(-1 + exp(r_i + λ * d_weighted))

Here:
- `i`       : datapoint index
- `u`       : node index in the last categorical layer (state (m, d))
- `sink_id` : global node id of the sink (state (m+1, 0))
- `r_i`     : scalar variable r[i]
- `λ`       : scalar variable lambda
- `d_weighted` : γ_{g_i} * d_raw, i.e., group-weighted categorical distance

We implement it as a single nonlinear inequality using JuMP's @NLconstraint.
"""
function add_terminal_arc_nl!(
    model::Model,
    μ::Array{VariableRef,2},
    i::Int,
    u::Int,
    sink_id::Int,
    r_i::VariableRef,
    λ::VariableRef,
    d_weighted::Float64
)
    @NLconstraint(
        model,
        μ[i, sink_id] - μ[i, u] >= -log(-1 + exp(r_i + λ * d_weighted))
    )
    return nothing
end


"""
    build_subgroup_dro_graph_model(
        X, Zrest, y, group;
        w_group, gamma_group, ε,
        template::DROGraphTemplate,
        optimizer
    ) -> JuMP.Model

Build the subgroup distributionally robust logistic regression model using the
graph-based formulation (Theorem 2) and your pre-built DAG template.

Arguments
---------
- `X`           :: N × n numeric feature matrix (Float64 or any Real)
- `Zrest`       :: N × m_rest binary matrix (0/1), the "rest" categorical features
- `y`           :: length-N vector with labels in {+1, -1}
- `group`       :: length-N vector of subgroup indices in {1, ..., G}
- `w_group`     :: length-G vector of continuous-part weights w_g
- `gamma_group` :: length-G vector of categorical-part weights γ_g
- `ε`           :: Wasserstein ball radius epsilon
- `template`    :: DROGraphTemplate (built once, shared across datapoints)
- `optimizer`   :: a JuMP optimizer constructor, e.g. Ipopt.Optimizer

Model
-----
Minimize:
    λ * ε + (1/N) * Σ_i r_i

Subject to:
1. "Margin" constraints (Theorem 2, first line):

    y_i (β_x^{(g_i)}⋅x_i + β0) ≥ - μ_i(source) + μ_i(sink)

2. Graph dual constraints for feature arcs:

    μ_i(t(e)) - μ_i(s(e)) ≥ w_i(e)
    where w_i(e) = - y_i * (β_z[k] * z_value_on_this_arc)

   For binary z_rest, z_value_on_this_arc is either:
     - original z_{i,k}  (stay), or
     - 1 - z_{i,k}       (flip)

3. Graph dual constraints for sink arcs:

    μ_i(sink) - μ_i(u) ≥ -log(-1 + exp(r_i + λ * γ_{g_i} * d_raw))

   Implemented via @NLconstraint in add_terminal_arc_nl!.

4. Group-wise norm constraints on β_x (continuous part of the metric):

    ‖β_x^{(g)}‖_2 ≤ λ * w_g,   for each group g

   (If your ground metric uses another norm, you can modify this accordingly.)
"""

function build_subgroup_dro_graph_model(
    X::AbstractMatrix{<:Real},
    Zrest::AbstractMatrix{<:Integer},
    y::AbstractVector{<:Integer},
    group::AbstractVector{<:Integer};
    w_group::AbstractVector{<:Real},
    gamma_group::AbstractVector{<:Real},
    ε::Real,
    template::DROGraphTemplate,
    optimizer
)::Model

    # ------------------
    # Basic dimensions
    # ------------------
    N, n_x   = size(X)
    N2, m_z  = size(Zrest)
    N3       = length(y)
    N4       = length(group)

    @assert N == N2 == N3 == N4 "X, Zrest, y, group must have matching first dimension (N)."

    G = maximum(group)
    @assert length(w_group)     >= G "w_group length must cover all groups."
    @assert length(gamma_group) >= G "gamma_group length must cover all groups."

    # Convert to concrete arrays (for safe indexing / types)
    Xmat = Array{Float64}(X)
    Zmat = Array{Int}(Zrest)
    yvec = Array{Int}(y)
    gvec = Array{Int}(group)

    # ------------------
    # Graph template basics
    # ------------------
    num_core_nodes = template.num_nodes              # nodes for k=0..m layers (no sink)
    source_id      = template.source                 # id of state (0,0)
    sink_id        = num_core_nodes + 1              # we create one extra node as sink
    total_nodes    = num_core_nodes + 1

    # Sanity: last layer should correspond to k = m_z
    @assert length(template.layer_values) == m_z + 1 "Template must have m_z+1 layers (0..m_z)."

    # ------------------
    # Build model
    # ------------------
    model = Model(optimizer)

    # Decision variables
    # ------------------
    # Intercept
    @variable(model, β0)

    # β_x is group-specific: β_x[g, j] for group g and numeric feature j
    @variable(model, βx[1:G, 1:n_x])

    # β_z is shared across groups: one coefficient per binary z_rest coordinate
    @variable(model, βz[1:m_z])

    # λ and r_i
    @variable(model, λ >= 0)
    @variable(model, r[1:N])

    # μ_i(v) for each datapoint i and each node v in the graph (including sink)
    @variable(model, μ[1:N, 1:total_nodes])

    # ------------------
    # Objective
    # ------------------
    @objective(model, Min, λ * ε + (1.0 / N) * sum(r[i] for i in 1:N))

    # ------------------
    # 1) Margin constraints:
    #    y_i (β_x^{(g_i)}⋅x_i + β0) ≥ -μ_i(source) + μ_i(sink)
    # ------------------
    for i in 1:N
        gi = gvec[i]
        @constraint(
            model,
            yvec[i] * (dot(βx[gi, :], Xmat[i, :]) + β0) >=
            -μ[i, source_id] + μ[i, sink_id]
        )
    end

    # ------------------
    # 2) Feature-arc constraints:
    #    μ_i(t(e)) - μ_i(s(e)) ≥ -y_i * βz[k] * z_value_on_arc
    #
    #    where z_value_on_arc:
    #       if move == :stay : z_{i,k}
    #       if move == :flip : 1 - z_{i,k}
    # ------------------
    arcs = template.arcs

    for i in 1:N
        yi = yvec[i]
        for arc in arcs
            s  = arc.s
            t  = arc.t
            k  = arc.feat
            mv = arc.move

            # Base value of this binary coordinate for datapoint i
            z0 = Zmat[i, k]
            @assert (z0 == 0 || z0 == 1) "Zrest must be 0/1 binary."

            # Value of z_k on this arc (stay: same; flip: 1 - z0)
            z_arc = (mv == :stay) ? z0 : (1 - z0)

            # Arc weight: - y_i * βz[k] * z_arc
            @constraint(
                model,
                μ[i, t] - μ[i, s] >= - yi * βz[k] * z_arc
            )
        end
    end

    # ------------------
    # 3) Sink-arc constraints:
    #    For each last-layer node u (state (m_z, d_raw)):
    #
    #      μ_i(sink) - μ_i(u) ≥ -log(-1 + exp(r_i + λ * γ_{g_i} * d_raw))
    #
    #    We read d_raw from template.layer_values[end].
    #    Then multiply by γ_{g_i} to get the weighted distance in the metric.
    # ------------------
    last_layer_values = template.layer_values[end]   # vector of all possible d_raw
    last_layer_nodes  = template.layer_nodes[end]    # corresponding node ids

    @assert length(last_layer_values) == length(last_layer_nodes)

    for i in 1:N
        gi = gvec[i]
        γg = gamma_group[gi]
        for idx in eachindex(last_layer_nodes)
            u      = last_layer_nodes[idx]
            d_raw  = last_layer_values[idx]
            d_w    = γg * d_raw   # group-weighted categorical distance
            add_terminal_arc_nl!(model, μ, i, u, sink_id, r[i], λ, d_w)
        end
    end

    # ------------------
    # 4) Group-wise norm constraints on β_x:
    #    ‖β_x^{(g)}‖_2 ≤ λ * w_g
    #
    #    If your ground metric uses a different norm / dual norm,
    #    you can replace this with the appropriate constraint.
    # ------------------
    for g in 1:G
        @NLconstraint(
            model,
            sum(βx[g, j]^2 for j in 1:n_x) <= (λ * w_group[g])^2
        )
    end
    return model
end



# ===== Toy data =====
# 4 samples, 2 numeric features
X = [
    1.0   0.5;
    0.9   0.7;
   -1.0  -0.5;
   -0.8  -0.3
]

# 4 samples, 2 binary categorical features (z_rest)
Zrest = [
    0  0;
    0  1;
    1  0;
    1  1
]

# Labels in {+1, -1}
y = [ 1,  1, -1, -1]

# Subgroup index for each sample
group = [1, 1, 2, 2]   

# Group weights in the ground metric
w_group     = [1.0, 1.0]        # continuous part weight w_g
gamma_group = [0.3, 0.7]        # categorical part weight γ_g

# Wasserstein radius
ε = 0.5

# ===== Build DAG template (for Hamming on z_rest) =====
m_z = size(Zrest, 2)                   # number of binary categorical features
gammas = ones(Float64, m_z)            # each flip adds 1 unit of raw distance
template = build_graph_template(gammas)


model = build_subgroup_dro_graph_model(
    X, Zrest, y, group;
    w_group      = w_group,
    gamma_group  = gamma_group,
    ε            = ε,
    template     = template,
    optimizer    = Ipopt.Optimizer,
)
