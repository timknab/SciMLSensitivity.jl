struct ODEQuadratureAdjointSensitivityFunction{
        C <: AdjointDiffCache,
        Alg <: QuadratureAdjoint,
        uType, SType,
        fType <: AbstractDiffEqFunction,
    } <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    f::fType
end

function ODEQuadratureAdjointSensitivityFunction(
        g, sensealg, discrete, sol, dgdu, dgdp,
        alg
    )
    p = sol.prob.p
    _use_full_p = hasproperty(sensealg, :diff_tunables) &&
        sensealg.diff_tunables isa Val{false} &&
        isscimlstructure(p) && !(p isa AbstractArray)
    diffcache,
        y = adjointdiffcache(
        g, sensealg, discrete, sol, dgdu, dgdp, sol.prob.f, alg;
        quad = true, use_full_p = _use_full_p
    )
    return ODEQuadratureAdjointSensitivityFunction(
        diffcache, sensealg, discrete,
        y, sol, sol.prob.f
    )
end

# u = λ'
function (S::ODEQuadratureAdjointSensitivityFunction)(du, u, p, t)
    (; sol, discrete) = S
    f = sol.prob.f

    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    vecjacobian!(dλ, y, λ, p, t, S)
    dλ .*= -one(eltype(λ))

    discrete || accumulate_cost!(dλ, y, p, t, S)
    return nothing
end

function (S::ODEQuadratureAdjointSensitivityFunction)(u, p, t)
    (; sol, discrete) = S
    f = sol.prob.f

    λ, grad, y, dgrad, dy = split_states(u, t, S)

    dy, dλ, dgrad = vecjacobian(y, λ, p, t, S; dgrad, dy)
    dλ *= (-one(eltype(λ)))

    if !discrete
        dλ, dgrad = accumulate_cost(dλ, y, p, t, S, dgrad)
    end
    return dλ
end

function split_states(du, u, t, S::ODEQuadratureAdjointSensitivityFunction; update = true)
    (; y, sol) = S

    if update
        if t isa ForwardDiff.Dual && eltype(y) <: AbstractFloat
            y = sol(t, continuity = :right)
        else
            sol(y, t, continuity = :right)
        end
    end

    λ = u
    dλ = du

    return λ, nothing, y, dλ, nothing, nothing
end

function split_states(u, t, S::ODEQuadratureAdjointSensitivityFunction; update = true)
    (; y, sol) = S

    if update
        y = sol(t, continuity = :right)
    end

    λ = u

    return λ, nothing, y, nothing, nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(
        sol, sensealg::QuadratureAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false); no_start = false,
        callback = CallbackSet()
    ) where {
        DG1, DG2, DG3, DG4, G,
        RetCB,
    }
    dgdu_discrete === nothing && dgdu_continuous === nothing && g === nothing &&
        error("Either `dgdu_discrete`, `dgdu_continuous`, or `g` must be specified.")
    t !== nothing && dgdu_discrete === nothing && dgdp_discrete === nothing &&
        error("It looks like you're using the direct `adjoint_sensitivities` interface
               with a discrete cost function but no specified `dgdu_discrete` or `dgdp_discrete`.
               Please use the higher level `solve` interface or specify these two contributions.")

    (; p, u0, tspan) = sol.prob

    ## Force recompile mode until vjps are specialized to handle this!!!
    f = if sol.prob.f isa ODEFunction &&
            sol.prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper
        ODEFunction{isinplace(sol.prob), true}(unwrapped_f(sol.prob.f))
    else
        sol.prob.f
    end

    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == ReturnCode.Terminated
            tspan = (tspan[1], last(current_time(sol)))
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (
        t !== nothing &&
            (
            dgdu_continuous === nothing && dgdp_continuous === nothing ||
                g !== nothing
        )
    )

    if ArrayInterface.ismutable(u0)
        len = length(u0)
        λ = similar(u0, len)
        λ .= false
    else
        λ = zero(u0)
    end
    sense = ODEQuadratureAdjointSensitivityFunction(
        g, sensealg, discrete, sol,
        dgdu_continuous, dgdp_continuous, alg
    )

    init_cb = (discrete || dgdu_discrete !== nothing) # && tspan[1] == t[end]
    z0 = vec(zero(λ))
    cb, rcb,
        _ = generate_callbacks(
        sense, dgdu_discrete, dgdp_discrete,
        λ, t, tspan[2],
        callback, init_cb, terminated, no_start
    )

    jac_prototype = sol.prob.f.jac_prototype
    adjoint_jac_prototype = !sense.discrete || jac_prototype === nothing ? nothing :
        copy(jac_prototype')

    # When the user provides an analytical Jacobian for the forward problem,
    # construct an adjoint Jacobian that evaluates -(df/du)^T at the
    # interpolated forward solution. This avoids finite-difference Jacobian
    # computation in the adjoint solver.
    adjoint_jac = if SciMLBase.has_jac(sol.prob.f) && jac_prototype !== nothing
        _fwd_jac_cache = copy(jac_prototype)
        _fwd_jac_fn = sol.prob.f.jac
        _fwd_sol = sol
        if jac_prototype isa SparseArrays.AbstractSparseMatrixCSC
            function (J_adj, _λ, _p, _t)
                _y = _fwd_sol(_t, continuity = :right)
                _fwd_jac_fn(_fwd_jac_cache, _y, _p, _t)
                SparseArrays.ftranspose!(J_adj, _fwd_jac_cache, -)
                return nothing
            end
        else
            function (J_adj, _λ, _p, _t)
                _y = _fwd_sol(_t, continuity = :right)
                _fwd_jac_fn(_fwd_jac_cache, _y, _p, _t)
                transpose!(J_adj, _fwd_jac_cache)
                lmul!(-1, J_adj)
                return nothing
            end
        end
    else
        nothing
    end

    original_mm = sol.prob.f.mass_matrix
    if original_mm === I || original_mm === (I, I)
        odefun = ODEFunction{ArrayInterface.ismutable(z0), true}(
            sense,
            jac_prototype = adjoint_jac_prototype,
            jac = adjoint_jac
        )
    else
        odefun = ODEFunction{ArrayInterface.ismutable(z0), true}(
            sense,
            mass_matrix = sol.prob.f.mass_matrix',
            jac_prototype = adjoint_jac_prototype,
            jac = adjoint_jac
        )
    end
    if RetCB
        return ODEProblem(odefun, z0, tspan, p, callback = cb), rcb
    else
        return ODEProblem(odefun, z0, tspan, p, callback = cb)
    end
end

struct AdjointSensitivityIntegrand{
        pType, uType, lType, rateType, S, AS, PF, PJC, PJT, DGP,
        G, tType, rType,
    }
    sol::S
    adj_sol::AS
    p::pType
    y::uType
    λ::lType
    pf::PF
    f_cache::rateType
    pJ::PJT
    paramjac_config::PJC
    sensealg::QuadratureAdjoint
    dgdp_cache::DGP
    dgdp::G
    tunables::tType
    repack::rType
end

function AdjointSensitivityIntegrand(sol, adj_sol, sensealg, dgdp = nothing)
    prob = sol.prob
    adj_prob = adj_sol.prob
    (; f, tspan) = prob
    p = parameter_values(prob)
    u0 = state_values(prob)

    _use_full_p = hasproperty(sensealg, :diff_tunables) &&
        sensealg.diff_tunables isa Val{false} &&
        isscimlstructure(p) && !(p isa AbstractArray)
    if _use_full_p
        tunables, repack = p, identity
    elseif isscimlstructure(p) && !(p isa AbstractArray)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = p, identity
    end

    numparams = length(tunables)
    y = zero(state_values(prob))
    λ = zero(state_values(adj_prob))
    # we need to alias `y`
    f_cache = zero(y)
    isautojacvec = get_jacvec(sensealg)

    unwrappedf = unwrapped_f(f)

    dgdp_cache = dgdp === nothing ? nothing : zero(tunables)

    if sensealg.autojacvec isa ReverseDiffVJP
        tape = if DiffEqBase.isinplace(prob)
            ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                du1 = similar(tunables, size(u))
                du1 .= false
                unwrappedf(du1, u, repack(tunables), first(t))
                return vec(du1)
            end
        else
            ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                vec(unwrappedf(u, repack(tunables), first(t)))
            end
        end
        if compile_tape(sensealg.autojacvec)
            paramjac_config = ReverseDiff.compile(tape)
        else
            paramjac_config = tape
        end
        pf = nothing
        pJ = nothing
    elseif sensealg.autojacvec isa EnzymeVJP
        pf = SciMLBase.isinplace(sol.prob.f) ? SciMLBase.Void(unwrappedf) : unwrappedf
        paramjac_config = zero(y), zero(y), Enzyme.make_zero(pf)
        pJ = nothing
    elseif sensealg.autojacvec isa MooncakeVJP
        _pf = get_pf(sensealg.autojacvec, prob, f)
        # Wrap pf to accept flat tunables and repack to full parameter struct
        _needs_repack = isscimlstructure(p) && !(p isa AbstractArray)
        pf = if _needs_repack
            let _pf = _pf, repack = repack
                (out, u, _tunables, t) -> _pf(out, u, repack(_tunables), t)
            end
        else
            _pf
        end
        paramjac_config = get_paramjac_config(
            MooncakeLoaded(), sensealg.autojacvec, pf, tunables, f, y, tspan[2]
        )
        pJ = nothing
    elseif sensealg.autojacvec isa ReactantVJP
        pf = get_pf(sensealg.autojacvec, prob, f)
        paramjac_config = get_paramjac_config(
            ReactantLoaded(), sensealg.autojacvec, pf, tunables, f, y, tspan[2]
        )
        pJ = nothing
    elseif isautojacvec # Zygote
        paramjac_config = nothing
        pf = nothing
        pJ = nothing
    else
        _needs_repack = isscimlstructure(p) && !(p isa AbstractArray)
        _pjac_f = _needs_repack ?
            (du, u, p, t) -> unwrappedf(du, u, repack(p), t) :
            unwrappedf
        pf = SciMLBase.ParamJacobianWrapper(_pjac_f, tspan[1], y)
        pJ = similar(u0, length(u0), numparams)
        pJ .= false
        paramjac_config = build_param_jac_config(sensealg, pf, y, tunables)
    end
    return AdjointSensitivityIntegrand(
        sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp, tunables, repack
    )
end

function gclosure4(f, du, u, p, t)
    Base.copyto!(du, f(u, p, t))
    return nothing
end

# `λ ⋅ f(y, repack(tunables), t)`, differentiated by Enzyme with respect to
# `tunables` only (everything else held constant). Top-level so the out-of-place
# immutable-state parameter vjp can call `Enzyme.gradient` without a closure.
_enzyme_vecpjac_dot(f, repack, y, tunables, t, λ) = dot(vec(f(y, repack(tunables), t)), vec(λ))

# out = λ df(u, p, t)/dp at u=y, p=p, t=t
function vec_pjac!(out, λ, y, t, S::AdjointSensitivityIntegrand)
    (; pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, adj_sol, tunables, repack) = S
    _odef = sol.prob.f
    f = unwrapped_f(_odef)

    # Priority: vjp_p > paramjac > autojacvec dispatch.
    if SciMLBase.has_vjp_p(_odef)
        _odef.vjp_p(out, λ, y, p, t)
        return out
    elseif SciMLBase.has_paramjac(_odef)
        _pJ = pJ
        if _pJ === nothing
            _pJ = similar(y, length(y), length(out))
        end
        _odef.paramjac(_pJ, y, p, t)
        # Use _pJ' * λ instead of out' = λ' * _pJ to avoid GPU scalar indexing
        mul!(vec(out), _pJ', λ)
        return out
    end

    isautojacvec = get_jacvec(sensealg)
    # y is aliased
    if !isautojacvec
        if SciMLBase.has_paramjac(f)
            f.paramjac(pJ, y, p, t) # Calculate the parameter Jacobian into pJ
        else
            pf.t = t
            pf.u = y
            jacobian!(pJ, pf, tunables, f_cache, sensealg, paramjac_config)
        end
        # Use pJ' * λ instead of out' = λ' * pJ to avoid GPU scalar indexing
        mul!(vec(out), pJ', λ)
    elseif sensealg.autojacvec isa ReverseDiffVJP
        tape = paramjac_config
        tu, tp, tt = ReverseDiff.input_hook(tape)
        output = ReverseDiff.output_hook(tape)
        ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
        ReverseDiff.unseed!(tp)
        ReverseDiff.unseed!(tt)
        ReverseDiff.value!(tu, y)
        ReverseDiff.value!(tp, tunables)
        ReverseDiff.value!(tt, [t])
        ReverseDiff.forward_pass!(tape)
        ReverseDiff.increment_deriv!(output, λ)
        ReverseDiff.reverse_pass!(tape)
        copyto!(vec(out), ReverseDiff.deriv(tp))
    elseif sensealg.autojacvec isa ZygoteVJP
        if SciMLBase.isinplace(sol.prob.f)
            _dy, back = Zygote.pullback(tunables) do tunables
                du_buf = Zygote.Buffer(y)
                f(du_buf, y, repack(tunables), t)
                vec(copy(du_buf))
            end
        else
            _dy, back = Zygote.pullback(tunables) do tunables
                vec(f(y, repack(tunables), t))
            end
        end
        tmp = back(λ)
        if tmp[1] === nothing
            recursive_copyto!(out, 0)
        else
            recursive_copyto!(out, tmp[1])
        end
    elseif sensealg.autojacvec isa MooncakeVJP
        _, _, p_grad = mooncake_run_ad(paramjac_config, y, p, t, λ)
        out .= p_grad
    elseif sensealg.autojacvec isa ReactantVJP
        reactant_run_ad!(nothing, out, nothing, paramjac_config, y, p, t, λ)
    elseif sensealg.autojacvec isa EnzymeVJP
        if !ArrayInterface.ismutable(y)
            # Out-of-place immutable state (e.g. `SVector`): the buffered
            # `Duplicated` path below needs mutable temporaries built from
            # `zero(y)`, so instead form the parameter vjp `(∂f/∂p)^T λ` as the
            # gradient of `λ ⋅ f(y, repack(tunables), t)` with respect to the
            # tunables, holding `f`, `repack`, `y`, `t`, `λ` constant.
            if out isa typeof(tunables) && ArrayInterface.ismutable(out)
                # Write the gradient directly into `out` instead of allocating a
                # fresh gradient (and result tuple) on every quadrature node.
                Enzyme.remake_zero!(out)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Const(_enzyme_vecpjac_dot),
                    Enzyme.Active, Enzyme.Const(f), Enzyme.Const(repack),
                    Enzyme.Const(y), Enzyme.Duplicated(tunables, out),
                    Enzyme.Const(t), Enzyme.Const(λ)
                )
            else
                res = Enzyme.gradient(
                    sensealg.autojacvec.mode, _enzyme_vecpjac_dot, Enzyme.Const(f),
                    Enzyme.Const(repack), Enzyme.Const(y), tunables,
                    Enzyme.Const(t), Enzyme.Const(λ)
                )
                recursive_copyto!(out, res[4])
            end
        else
            tmp3, tmp4, tmp6 = paramjac_config
            vtmp4 = vec(tmp4)

            Enzyme.remake_zero!(out)
            Enzyme.remake_zero!(tmp3)
            vtmp4 .= λ

            _shadow_enzyme = nothing
            if !(p isa AbstractArray)
                _shadow_enzyme = repack(out)
                dup = Enzyme.Duplicated(p, _shadow_enzyme)
            else
                dup = Enzyme.Duplicated(p, out)
            end

            if SciMLBase.isinplace(sol.prob.f)
                Enzyme.remake_zero!(tmp6)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode,
                    Enzyme.Duplicated(SciMLBase.Void(f), tmp6), Enzyme.Const,
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), dup, Enzyme.Const(t)
                )
            else
                tmp6 = Enzyme.make_zero(f)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Const(gclosure4), Enzyme.Const,
                    Enzyme.Duplicated(f, tmp6),
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), dup, Enzyme.Const(t)
                )
            end

            if _shadow_enzyme !== nothing && _shadow_enzyme !== out
                _use_full_p_enzyme = hasproperty(sensealg, :diff_tunables) &&
                    sensealg.diff_tunables isa Val{false}
                if !_use_full_p_enzyme && isscimlstructure(_shadow_enzyme)
                    grad_tunables, _, _ = canonicalize(Tunable(), _shadow_enzyme)
                else
                    grad_tunables = _shadow_enzyme
                end
                copyto!(out, grad_tunables)
            end
        end
    end

    # TODO: Add tracker?
    return out
end

function (S::AdjointSensitivityIntegrand)(out, t)
    (; y, λ, pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, adj_sol) = S
    if ArrayInterface.ismutable(y)
        sol(y, t)
        adj_sol(λ, t)
    else
        y = sol(t)
        λ = adj_sol(t)
    end
    vec_pjac!(out, λ, y, t, S)

    if S.dgdp !== nothing
        S.dgdp(dgdp_cache, y, p, t)
        out .+= dgdp_cache
    end
    return out'
end

function (S::AdjointSensitivityIntegrand)(t)
    out = similar(S.tunables)
    out .= false
    return S(out, t)
end

function _adjoint_sensitivities(
        sol, sensealg::QuadratureAdjoint, alg; t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing, no_start = false,
        abstol = sensealg.abstol, reltol = sensealg.reltol,
        callback = CallbackSet(),
        kwargs...
    )
    forward_sol = event_replay_solution(sol, callback, alg, abstol, reltol)
    adj_prob,
        rcb = ODEAdjointProblem(
        forward_sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
        dgdu_continuous, dgdp_continuous, g, Val(true);
        callback, no_start
    )
    adj_sol = solve(
        adj_prob, alg; abstol, reltol,
        save_everystep = true, save_start = true, kwargs...
    )

    p = forward_sol.prob.p
    if p === nothing || p === SciMLBase.NullParameters()
        return state_values(adj_sol)[end], nothing
    else
        integrand = AdjointSensitivityIntegrand(
            forward_sol, adj_sol, sensealg, dgdp_continuous
        )
        if t === nothing
            res,
                err = quadgk(
                integrand, forward_sol.prob.tspan[1], forward_sol.prob.tspan[2],
                atol = abstol, rtol = reltol
            )
        else
            res = zero(integrand.tunables)'
            event_times = callback_event_times(callback, eltype(t))
            sensitivity_times = isempty(event_times) ? t :
                                sort!(unique!(vcat(collect(t), event_times)))

            # handle discrete dgdp contributions
            if dgdp_discrete !== nothing
                (; y) = integrand
                dgdp_cache = copy(res)
                dgdp_discrete(dgdp_cache, y, p, t[end], length(t))
                res .+= dgdp_cache
            end

            if callback !== nothing
                dλ = similar(integrand.λ)
                dλ .*= false
                dgrad = similar(res)
                dgrad .*= false
            end

            # correction for end interval.
            if sensitivity_times[end] != forward_sol.prob.tspan[2] &&
                    forward_sol.retcode !== ReturnCode.Terminated
                res .+= quadgk(
                    integrand, sensitivity_times[end], forward_sol.prob.tspan[end],
                    atol = abstol, rtol = reltol
                )[1]
            end

            if forward_sol.retcode === ReturnCode.Terminated
                loss_idx = searchsortedlast(t, sensitivity_times[end])
                loss_idx = loss_idx >= firstindex(t) &&
                    t[loss_idx] == sensitivity_times[end] ? loss_idx : nothing
                integrand = update_integrand_and_dgrad(
                    res, sensealg, callback, integrand,
                    adj_prob, forward_sol, dgdu_discrete,
                    dgdp_discrete, dλ, dgrad, sensitivity_times[end],
                    loss_idx
                )
            end

            for i in (length(sensitivity_times) - 1):-1:1
                if ArrayInterface.ismutable(res)
                    res .+= quadgk(
                        integrand, sensitivity_times[i], sensitivity_times[i + 1],
                        atol = abstol, rtol = reltol
                    )[1]
                else
                    res += quadgk(
                        integrand, sensitivity_times[i], sensitivity_times[i + 1],
                        atol = abstol, rtol = reltol
                    )[1]
                end
                loss_idx = searchsortedlast(t, sensitivity_times[i])
                loss_idx = loss_idx >= firstindex(t) &&
                    t[loss_idx] == sensitivity_times[i] ? loss_idx : nothing
                if sensitivity_times[i] ∈ event_times
                    integrand = update_integrand_and_dgrad(
                        res, sensealg, callback,
                        integrand,
                        adj_prob, forward_sol, dgdu_discrete,
                        dgdp_discrete, dλ, dgrad, sensitivity_times[i],
                        loss_idx
                    )
                end
                if dgdp_discrete !== nothing && loss_idx !== nothing
                    (; y) = integrand
                    dgdp_discrete(dgdp_cache, y, p, t[loss_idx], loss_idx)
                    res .+= dgdp_cache
                end
            end
            # correction for start interval
            if sensitivity_times[1] != forward_sol.prob.tspan[1]
                res .+= quadgk(
                    integrand, forward_sol.prob.tspan[1], sensitivity_times[1],
                    atol = abstol, rtol = reltol
                )[1]
            end
        end
        if rcb !== nothing && !isempty(rcb.Δλas)
            iλ = zero(rcb.λ)
            out = zero(res')
            yy = similar(rcb.y)
            yy .= false
            for (Δλa, tt) in rcb.Δλas
                (; algevar_idxs) = rcb.diffcache
                iλ[algevar_idxs] .= Δλa
                forward_sol(yy, tt)
                vec_pjac!(out, iλ, yy, tt, integrand)
                res .+= out'
                iλ .= zero(eltype(iλ))
            end
        end
        return state_values(adj_sol)[end], res
    end
end

function update_p_integrand(integrand::AdjointSensitivityIntegrand, p)
    (;
        sol, adj_sol, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp,
    ) = integrand
    _use_full_p = hasproperty(sensealg, :diff_tunables) &&
        sensealg.diff_tunables isa Val{false} &&
        isscimlstructure(p) && !(p isa AbstractArray)
    if _use_full_p
        tunables, repack = p, identity
    elseif isscimlstructure(p) && !(p isa AbstractArray)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = p, identity
    end
    return AdjointSensitivityIntegrand(
        sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp, tunables, repack
    )
end

function update_integrand_and_dgrad(
        res, sensealg::QuadratureAdjoint, callbacks, integrand,
        adj_prob, sol, dgdu_discrete, dgdp_discrete, dλ, dgrad,
        ti, cur_time
    )
    for cb in callbacks.discrete_callbacks
        if ti ∈ cb.affect!.event_times
            integrand = _update_integrand_and_dgrad(
                res, sensealg, cb,
                integrand, adj_prob, sol,
                dgdu_discrete,
                dgdp_discrete, dλ, dgrad,
                ti, cur_time
            )
        end
    end
    for cb in callbacks.continuous_callbacks
        if ti ∈ cb.affect!.event_times ||
                ti ∈ cb.affect_neg!.event_times
            integrand = _update_integrand_and_dgrad(
                res, sensealg, cb,
                integrand, adj_prob, sol,
                dgdu_discrete,
                dgdp_discrete, dλ, dgrad,
                ti, cur_time
            )
        end
    end
    return integrand
end

function _update_integrand_and_dgrad(
        res, sensealg::QuadratureAdjoint, cb, integrand,
        adj_prob, sol, dgdu, dgdp, dλ, dgrad, t, cur_time
    )
    if ArrayInterface.ismutable(integrand.y)
        integrand.sol(integrand.y, t)
        integrand.adj_sol(integrand.λ, t)
    else
        integrand = AdjointSensitivityIntegrand(
            integrand.sol, integrand.adj_sol, integrand.p,
            integrand.sol(t), integrand.adj_sol(t), integrand.pf,
            integrand.f_cache, integrand.pJ, integrand.paramjac_config,
            integrand.sensealg, integrand.dgdp_cache, integrand.dgdp,
            integrand.tunables, integrand.repack
        )
    end

    indx, pos_neg = get_indx(cb, t)
    tprev = get_tprev(cb, indx, pos_neg)

    cb_autojacvec = supports_callback_vjp(sensealg.autojacvec) ?
                    sensealg.autojacvec : ReverseDiffVJP()
    cb_sensealg = setvjp(sensealg, cb_autojacvec)
    event_idxs = cb isa VectorContinuousCallback ?
        get_event_idx(cb, indx, pos_neg) : nothing
    cache_key = cb isa VectorContinuousCallback ? (pos_neg, indx) :
        (pos_neg, nothing)
    diffcaches = get_cb_diffcaches(cb, cb_autojacvec)

    wp = CallbackAffectPWrapper(cb, cb_autojacvec, pos_neg, event_idxs, tprev)

    # The quadrature pass runs after the adjoint solve, so it does not execute
    # reverse callbacks inside an integrator. Use the tracked forward left limit
    # explicitly here, matching the reverse-callback path in callback_tracking.jl.
    copy_to_integrator!(cb, integrand.y, integrand.p, indx, pos_neg)
    p_left = copy(integrand.p)

    _p = similar(integrand.p, size(integrand.p))
    _p .= false
    wp(_p, integrand.y, copy(p_left), t)

    w = CallbackAffectWrapper(cb, cb_autojacvec, pos_neg, event_idxs, tprev)
    _y = copy(integrand.y)
    w(_y, _y, integrand.p, t)
    _is_noop = _p == integrand.p && _y == integrand.y
    _is_noop && return integrand

    if _p != integrand.p
        diffcache_wp = diffcaches[cache_key][2]

        fakeSp = CallbackSensitivityFunctionPSwap(wp, cb_sensealg, diffcache_wp, sol.prob)
        #vjp with Jacobin given by dw/dp before event and vector given by grad
        res_vec = copy(vec(res))
        dgrad_p = similar(integrand.tunables)
        dgrad_p .*= false
        vecjacobian!(
            nothing, integrand.y, res_vec, copy(p_left), t, fakeSp;
            dgrad = dgrad_p, dy = nothing
        )
        res .= dgrad_p'
        integrand = update_p_integrand(integrand, p_left)
    end

    _y == integrand.y && return integrand

    # Create a fake sensitivity function to do the vjps needs to be done
    # to account for parameter dependence of affect function
    fakeS = CallbackSensitivityFunction(w, cb_sensealg, diffcaches[cache_key][1], sol.prob)
    if cur_time === nothing
        dλ .*= false
    elseif dgdu !== nothing # discrete cost
        dgdu(dλ, integrand.y, integrand.p, t, cur_time)
    else
        error("Please provide `dgdu` to use adjoint_sensitivities with `QuadratureAdjoint()` and callbacks.")
    end

    @assert dgdp === nothing

    # account for implicit events

    @. dλ = -dλ - integrand.λ
    dgrad_p = similar(integrand.tunables)
    dgrad_p .*= false
    vecjacobian!(dλ, integrand.y, dλ, integrand.p, t, fakeS; dgrad = dgrad_p)
    res .-= dgrad_p'
    return integrand
end
