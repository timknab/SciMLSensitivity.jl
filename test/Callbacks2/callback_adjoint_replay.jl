using DifferentiationInterface
using ForwardDiff
using LinearAlgebra
using Mooncake
using OrdinaryDiffEq
using SciMLBase
using SciMLSensitivity
using Test

const ADJ_REPLAY_TSPAN = (0.0, 3.0)
const ADJ_REPLAY_EVENT_TIMES = [0.5, 1.0, 1.75, 2.25]
const ADJ_REPLAY_OBS_TIMES = [0.25, 0.75, 1.25, 2.0, 2.75]
const ADJ_REPLAY_ABSTOL = 1.0e-10
const ADJ_REPLAY_RELTOL = 1.0e-10
const ADJ_REPLAY_SIGMA = 0.15
const ADJ_REPLAY_NOISE = [0.03, -0.015, 0.02, -0.025, 0.01]
const ADJ_REPLAY_X_DATA = log.([0.38, 1.15, 0.72])
const ADJ_REPLAY_X0 = log.([0.43, 1.05, 0.85])

function adj_replay_rhs!(du, u, p, t)
    du[1] = -p[1] * u[1] + p[3]
    return nothing
end

function adj_replay_problem(x::AbstractVector)
    k = exp(x[1])
    u0 = [exp(x[2])]
    scale = exp(x[3])
    p = [k, scale, zero(k + scale)]
    return ODEProblem(adj_replay_rhs!, u0, ADJ_REPLAY_TSPAN, p)
end

function adj_replay_action!(integrator, idx::Integer)
    if idx == 1
        integrator.u[1] += 0.35 * integrator.p[2]
    elseif idx == 2
        integrator.p[3] += 0.18 * integrator.p[2]
    elseif idx == 3
        integrator.p[2] = 1.12 * integrator.p[2]
    elseif idx == 4
        integrator.p[3] -= 0.18 * integrator.p[2]
    else
        error("unknown event index $idx")
    end
    return nothing
end

function adj_replay_vector_callback()
    function condition(out, u, t, integrator)
        @inbounds for i in eachindex(ADJ_REPLAY_EVENT_TIMES)
            out[i] = t - ADJ_REPLAY_EVENT_TIMES[i]
        end
        return nothing
    end
    function affect!(integrator, event_mask)
        if event_mask isa Integer
            adj_replay_action!(integrator, Int(event_mask))
            return nothing
        end
        @inbounds for i in eachindex(ADJ_REPLAY_EVENT_TIMES)
            iszero(event_mask[i]) || adj_replay_action!(integrator, i)
        end
        return nothing
    end
    return VectorContinuousCallback(
        condition, affect!, length(ADJ_REPLAY_EVENT_TIMES);
        save_positions = (false, false),
        interp_points = 0
    )
end

function adj_replay_discrete_callback()
    function matching_event_indices(t)
        return findall(
            tt -> isapprox(t, tt; rtol = 0.0, atol = 1.0e-12),
            ADJ_REPLAY_EVENT_TIMES
        )
    end
    condition(u, t, integrator) = !isempty(matching_event_indices(t))
    function affect!(integrator)
        for i in matching_event_indices(integrator.t)
            adj_replay_action!(integrator, i)
        end
        return nothing
    end
    return DiscreteCallback(condition, affect!; save_positions = (false, false))
end

function adj_replay_callback(kind::Symbol)
    kind === :vector && return adj_replay_vector_callback(), (;)
    kind === :discrete &&
        return adj_replay_discrete_callback(), (; tstops = ADJ_REPLAY_EVENT_TIMES)
    error("unknown callback kind $kind")
end

function adj_replay_predictions(sol)
    preds = similar(ADJ_REPLAY_OBS_TIMES, eltype(sol.u[1]))
    for (i, t_obs) in pairs(ADJ_REPLAY_OBS_TIMES)
        idx = findfirst(
            t -> isapprox(t, t_obs; rtol = 0.0, atol = 1.0e-9),
            sol.t
        )
        idx === nothing && error("solution did not save observation time $t_obs")
        preds[i] = sol.u[idx][1]
    end
    return preds
end

function adj_replay_solve(x::AbstractVector, callback_kind::Symbol; sensealg = nothing)
    cb, cb_kwargs = adj_replay_callback(callback_kind)
    kwargs = sensealg === nothing ? (;) : (; sensealg)
    sol = solve(
        adj_replay_problem(x), Tsit5();
        callback = cb,
        saveat = ADJ_REPLAY_OBS_TIMES,
        save_start = false,
        save_end = false,
        save_everystep = false,
        dense = false,
        abstol = ADJ_REPLAY_ABSTOL,
        reltol = ADJ_REPLAY_RELTOL,
        cb_kwargs...,
        kwargs...
    )
    SciMLBase.successful_retcode(sol) ||
        error("solve failed with retcode $(sol.retcode)")
    return adj_replay_predictions(sol)
end

function adj_replay_observations(callback_kind::Symbol)
    return adj_replay_solve(ADJ_REPLAY_X_DATA, callback_kind) .+ ADJ_REPLAY_NOISE
end

function adj_replay_objective(
        x::AbstractVector, callback_kind::Symbol; sensealg = nothing)
    pred = adj_replay_solve(x, callback_kind; sensealg)
    resid = (pred .- adj_replay_observations(callback_kind)) ./ ADJ_REPLAY_SIGMA
    return -0.5 * sum(abs2, resid) - 0.025 * sum(abs2, x)
end

function adj_replay_value_gradient(f, x)
    value, grad = DifferentiationInterface.value_and_gradient(
        f, DifferentiationInterface.AutoMooncake(), x)
    return value, collect(grad)
end

adj_replay_rel_l2(a, b) = norm(a .- b) / max(norm(b), eps(Float64))

function adj_replay_sensealgs()
    sensealgs = [
        (
            "InterpolatingAdjoint MooncakeVJP",
            InterpolatingAdjoint(;
                autojacvec = SciMLSensitivity.MooncakeVJP(),
                checkpointing = false
            )
        ),
        (
            "checkpointed InterpolatingAdjoint MooncakeVJP",
            InterpolatingAdjoint(;
                autojacvec = SciMLSensitivity.MooncakeVJP(),
                checkpointing = true
            )
        ),
        (
            "QuadratureAdjoint MooncakeVJP",
            QuadratureAdjoint(;
                autojacvec = SciMLSensitivity.MooncakeVJP(),
                abstol = ADJ_REPLAY_ABSTOL,
                reltol = ADJ_REPLAY_RELTOL
            )
        )
    ]

    if VERSION < v"1.12"
        append!(
            sensealgs,
            [
                (
                    "InterpolatingAdjoint EnzymeVJP",
                    InterpolatingAdjoint(;
                        autojacvec = SciMLSensitivity.EnzymeVJP(),
                        checkpointing = false
                    )
                ),
                (
                    "QuadratureAdjoint EnzymeVJP",
                    QuadratureAdjoint(;
                        autojacvec = SciMLSensitivity.EnzymeVJP(),
                        abstol = ADJ_REPLAY_ABSTOL,
                        reltol = ADJ_REPLAY_RELTOL
                    )
                )
            ]
        )
    end

    return sensealgs
end

@testset "callback-aware adjoint replay" begin
    for callback_kind in (:discrete, :vector)
        @testset "$callback_kind callback" begin
            ref_f = x -> adj_replay_objective(x, callback_kind)
            ref_value = ref_f(ADJ_REPLAY_X0)
            ref_grad = ForwardDiff.gradient(ref_f, ADJ_REPLAY_X0)

            for (name, sensealg) in adj_replay_sensealgs()
                @testset "$name" begin
                    f = x -> adj_replay_objective(x, callback_kind; sensealg)
                    value, grad = adj_replay_value_gradient(f, ADJ_REPLAY_X0)

                    @test value ≈ ref_value atol = 1.0e-9 rtol = 1.0e-9
                    @test adj_replay_rel_l2(grad, ref_grad) ≤ 1.0e-8
                end
            end
        end
    end
end
