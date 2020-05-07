using Interpolations

include("KinematicModel.jl")

function vars_state_conservative(m::KinematicModel, FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
        ρq_tot::FT
        ρq_liq::FT
        ρq_ice::FT
        ρq_rai::FT
        ρq_sno::FT
    end
end

function vars_state_auxiliary(m::KinematicModel, FT)
    @vars begin
        # defined in init_state_auxiliary
        p::FT
        z::FT
        x::FT
        # defined in update_aux
        u::FT
        w::FT
        q_tot::FT
        q_vap::FT
        q_liq::FT
        q_ice::FT
        q_rai::FT
        q_sno::FT
        e_tot::FT
        e_kin::FT
        e_pot::FT
        e_int::FT
        T::FT
        S::FT
        RH::FT
        rain_w::FT
        snow_w::FT
        # more diagnostics
        src_cloud_liq::FT
        src_cloud_ice::FT
        src_rain_acnv::FT
        src_snow_acnv::FT
        src_liq_rain_accr::FT
        src_liq_snow_accr::FT
        src_ice_snow_accr::FT
        src_ice_rain_accr::FT
        src_snow_rain_accr::FT
        src_rain_accr_sink::FT
        src_rain_evap::FT
        src_snow_subl::FT
        src_snow_melt::FT
        flag_cloud_liq::FT
        flag_cloud_ice::FT
        flag_rain::FT
        flag_snow::FT
        # helpers for bc
        ρe_init::FT
        ρq_tot_init::FT
    end
end

function init_kinematic_eddy!(eddy_model, state, aux, (x, y, z), t)
    FT = eltype(state)
    _grav::FT = grav(param_set)

    dc = eddy_model.data_config
    @inbounds begin

        # sounding data based on GATE III for moisture...
        z_in =
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                5.5,
                6.0,
                6.5,
                7.0,
                7.5,
                8.0,
                8.5,
                9.0,
                9.5,
                10.0,
                10.5,
                11.0,
                11.5,
                12.0,
                12.5,
                13.0,
                13.5,
                14.0,
                14.5,
                15.0,
                15.5,
                16.0,
                16.5,
                17.0,
                17.5,
                18.0,
                27.0,
            ] * 1000.0
        r_in =
            [
                16.5,
                16.5,
                13.5,
                12.0,
                10.0,
                8.7,
                7.1,
                6.1,
                5.2,
                4.5,
                3.6,
                3.0,
                2.3,
                1.75,
                1.3,
                0.9,
                0.5,
                0.25,
                0.125,
                0.065,
                0.003,
                0.0015,
                0.0007,
                0.0003,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
                0.0001,
            ] / 1000
        qt_in = r_in ./ (r_in .+ 1)
        # ... and temperature
        T_in = [
            299.184,
            294.836,
            294.261,
            288.773,
            276.698,
            265.004,
            253.930,
            243.662,
            227.674,
            214.266,
            207.757,
            201.973,
            198.278,
            197.414,
            198.110,
            198.110,
        ]
        z_T_in =
            [
                0.0,
                0.492,
                0.700,
                1.698,
                3.928,
                6.039,
                7.795,
                9.137,
                11.055,
                12.645,
                13.521,
                14.486,
                15.448,
                16.436,
                17.293,
                22.0,
            ] * 1000

        init_qt = LinearInterpolation(z_in, qt_in, extrapolation_bc = Flat())
        init_T = LinearInterpolation(z_T_in, T_in, extrapolation_bc = Flat())

        # density
        q_pt_0 = PhasePartition(init_qt(z))
        R_m, cp_m, cv_m, γ = gas_constants(param_set, q_pt_0)
        T::FT = init_T(z)
        ρ::FT = aux.p / R_m / T
        state.ρ = ρ

        # moisture
        state.ρq_tot = ρ * init_qt(z)
        state.ρq_liq = ρ * q_pt_0.liq
        state.ρq_ice = ρ * q_pt_0.ice
        state.ρq_rai = ρ * FT(0)
        state.ρq_sno = ρ * FT(0)

        #https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281998%29055%3C3283%3ATCRMOL%3E2.0.CO%3B2
        _Z::FT = FT(15000 + 1e-6)                  # TODO
        _X::FT = FT(10000)
        _xc::FT = FT(30000)
        _A::FT = FT(4.8 * 1e4) / FT(2.5)           # TODO
        _S::FT = FT(2.5 * 1e-2) * FT(1e-2) / FT(2) # TODO
        #_ρ_00::FT = FT(1)
        ρu::FT = FT(0)
        ρw::FT = FT(0)
        # velocity (derivative of streamfunction)
        # This is actually different than what comes out from taking a
        # derivative of Ψ from the paper. I have sin(π/2/X(x-xc)).
        # This setup makes more sense to me though.
        #if z < _Z
        if x >= (_xc + _X)
            ρu = -_A * FT(π) / _Z * cos(FT(π) / _Z * z) + _S * z
            ρw = FT(0)
        elseif x <= (_xc - _X)
            ρu = _A * FT(π) / _Z * cos(FT(π) / _Z * z) + _S * z
            ρw = FT(0)
        else
            ρu =
                -_A * FT(π) / _Z *
                cos(FT(π) / _Z * z) *
                sin(FT(π / 2) / _X * (x - _xc)) + _S * z
            ρw =
                _A * FT(π / 2) / _X *
                sin(FT(π) / _Z * z) *
                cos(FT(π / 2) / _X * (x - _xc))
        end
        #else
        #    ρu = _S * _Z
        #    ρw = FT(0)
        #end

        state.ρu = SVector(ρu, FT(0), ρw)
        u::FT = ρu / ρ
        w::FT = ρw / ρ

        # energy
        e_kin::FT = 1 // 2 * (u^2 + w^2)
        e_pot::FT = _grav * z
        e_int::FT = internal_energy(param_set, T, q_pt_0)
        e_tot::FT = e_kin + e_pot + e_int
        state.ρe = ρ * e_tot
    end
    return nothing
end

function kinematic_model_nodal_update_auxiliary_state!(
    m::KinematicModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    _grav::FT = grav(param_set)
    _T_freeze::FT = T_freeze(param_set)

    @inbounds begin

        if t == FT(0)
            aux.ρe_init = state.ρe
            aux.ρq_tot_init = state.ρq_tot
        end

        # velocity
        aux.u = state.ρu[1] / state.ρ
        aux.w = state.ρu[3] / state.ρ
        # water
        aux.q_tot = state.ρq_tot / state.ρ
        aux.q_liq = state.ρq_liq / state.ρ
        aux.q_ice = state.ρq_ice / state.ρ
        aux.q_rai = state.ρq_rai / state.ρ
        aux.q_sno = state.ρq_sno / state.ρ
        aux.q_vap = aux.q_tot - aux.q_liq - aux.q_ice
        # energy
        aux.e_tot = state.ρe / state.ρ
        aux.e_kin = 1 // 2 * (aux.u^2 + aux.w^2)
        aux.e_pot = _grav * aux.z
        aux.e_int = aux.e_tot - aux.e_kin - aux.e_pot
        # supersaturation
        q = PhasePartition(aux.q_tot, aux.q_liq, aux.q_ice)
        aux.T = air_temperature(param_set, aux.e_int, q)
        ts_neq = TemperatureSHumNonEquil(param_set, aux.T, state.ρ, q)
        # TODO: add super_saturation method in moist thermo
        aux.S = max(0, aux.q_vap / q_vap_saturation(ts_neq) - FT(1)) * FT(100)
        aux.RH = relative_humidity(ts_neq) * FT(100)

        aux.rain_w =
            terminal_velocity(param_set, rain_param_set, state.ρ, aux.q_rai)
        aux.snow_w =
            terminal_velocity(param_set, snow_param_set, state.ρ, aux.q_sno)

        # more diagnostics
        ts_eq = TemperatureSHumEquil(param_set, aux.T, state.ρ, aux.q_tot)
        q_eq = PhasePartition(ts_eq)

        aux.src_cloud_liq = conv_q_vap_to_q_liq_ice(liquid_param_set, q_eq, q)
        aux.src_cloud_ice = conv_q_vap_to_q_liq_ice(ice_param_set, q_eq, q)

        aux.src_rain_acnv = conv_q_liq_to_q_rai(rain_param_set, aux.q_liq)
        aux.src_snow_acnv =
            conv_q_ice_to_q_sno(param_set, ice_param_set, q, state.ρ, aux.T)

        aux.src_liq_rain_accr = accretion(
            param_set,
            liquid_param_set,
            rain_param_set,
            aux.q_liq,
            aux.q_rai,
            state.ρ,
        )
        aux.src_liq_snow_accr = accretion(
            param_set,
            liquid_param_set,
            snow_param_set,
            aux.q_liq,
            aux.q_sno,
            state.ρ,
        )
        aux.src_ice_snow_accr = accretion(
            param_set,
            ice_param_set,
            snow_param_set,
            aux.q_ice,
            aux.q_sno,
            state.ρ,
        )
        aux.src_ice_rain_accr = accretion(
            param_set,
            ice_param_set,
            rain_param_set,
            aux.q_ice,
            aux.q_rai,
            state.ρ,
        )

        aux.src_rain_accr_sink = accretion_rain_sink(
            param_set,
            ice_param_set,
            rain_param_set,
            aux.q_ice,
            aux.q_rai,
            state.ρ,
        )

        if aux.T < _T_freeze
            aux.src_snow_rain_accr = accretion_snow_rain(
                param_set,
                snow_param_set,
                rain_param_set,
                aux.q_sno,
                aux.q_rai,
                state.ρ,
            )
        else
            aux.src_snow_rain_accr = accretion_snow_rain(
                param_set,
                rain_param_set,
                snow_param_set,
                aux.q_rai,
                aux.q_sno,
                state.ρ,
            )
        end

        aux.src_rain_evap = evaporation_sublimation(
            param_set,
            rain_param_set,
            q,
            aux.q_rai,
            state.ρ,
            aux.T,
        )
        aux.src_snow_subl = evaporation_sublimation(
            param_set,
            snow_param_set,
            q,
            aux.q_sno,
            state.ρ,
            aux.T,
        )

        aux.src_snow_melt =
            snow_melt(param_set, snow_param_set, aux.q_sno, state.ρ, aux.T)

        aux.flag_cloud_liq = FT(0)
        aux.flag_cloud_ice = FT(0)
        aux.flag_rain = FT(0)
        aux.flag_snow = FT(0)
        if (aux.q_liq >= FT(0))
            aux.flag_cloud_liq = FT(1)
        end
        if (aux.q_ice >= FT(0))
            aux.flag_cloud_ice = FT(1)
        end
        if (aux.q_rai >= FT(0))
            aux.flag_rain = FT(1)
        end
        if (aux.q_sno >= FT(0))
            aux.flag_snow = FT(1)
        end
    end
end

function boundary_state!(
    ::RusanovNumericalFlux,
    m::KinematicModel,
    state⁺,
    aux⁺,
    n,
    state⁻,
    aux⁻,
    bctype,
    t,
    args...,
)
    FT = eltype(state⁻)
    @inbounds state⁺.ρq_rai = FT(0)
    @inbounds state⁺.ρq_sno = FT(0)

    # 1 - left     (x = 0,   z = ...)
    # 2 - right    (x = -1,  z = ...)
    # 3,4 - y boundary (periodic)
    # 5 - bottom   (x = ..., z = 0)
    # 6 - top      (x = ..., z = -1)

    state⁺.ρq_tot = state⁻.ρq_tot
    state⁺.ρq_liq = state⁻.ρq_liq
    state⁺.ρq_ice = state⁻.ρq_ice
    state⁺.ρ = state⁻.ρ

    if bctype == 1
        state⁺.ρu = SVector(state⁻.ρu[1], FT(0), FT(0))
        state⁺.ρe = aux⁻.ρe_init
        state⁺.ρq_tot = aux⁻.ρq_tot_init
    end
    if bctype == 2
        state⁺.ρu = SVector(state⁻.ρu[1], FT(0), FT(0))
        state⁺.ρe = aux⁻.ρe_init
        state⁺.ρq_tot = aux⁻.ρq_tot_init
    end
    if bctype == 5
        state⁺.ρu -= 2 * dot(state⁻.ρu, n) .* SVector(n)
        state⁺.ρe = state⁻.ρe
        state⁺.ρq_tot = state⁻.ρq_tot
    end
    if bctype == 6
        state⁺.ρe = aux⁻.ρe_init
        state⁺.ρq_tot = aux⁻.ρq_tot_init
        state⁺.ρu = SVector(state⁻.ρu[1], FT(0), state⁻.ρu[3])
    end
end

@inline function wavespeed(
    m::KinematicModel,
    nM,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    @inbounds begin
        u = state.ρu / state.ρ
        q_rai::FT = state.ρq_rai / state.ρ
        q_sno::FT = state.ρq_sno / state.ρ
        rain_w = terminal_velocity(param_set, rain_param_set, state.ρ, q_rai)
        snow_w = terminal_velocity(param_set, snow_param_set, state.ρ, q_sno)

        nu =
            nM[1] * u[1] +
            nM[3] * max(u[3], rain_w, snow_w, u[3] - rain_w, u[3] - snow_w)
    end
    return abs(nu)
end

@inline function flux_first_order!(
    m::KinematicModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    @inbounds begin
        q_rai::FT = state.ρq_rai / state.ρ
        q_sno::FT = state.ρq_sno / state.ρ
        rain_w = terminal_velocity(param_set, rain_param_set, state.ρ, q_rai)
        snow_w = terminal_velocity(param_set, snow_param_set, state.ρ, q_sno)

        # advect moisture ...
        flux.ρ = SVector(state.ρu[1], FT(0), state.ρu[3])
        flux.ρq_tot = SVector(
            state.ρu[1] * state.ρq_tot / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_tot / state.ρ,
        )
        flux.ρq_liq = SVector(
            state.ρu[1] * state.ρq_liq / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_liq / state.ρ,
        )
        flux.ρq_ice = SVector(
            state.ρu[1] * state.ρq_ice / state.ρ,
            FT(0),
            state.ρu[3] * state.ρq_ice / state.ρ,
        )
        flux.ρq_rai = SVector(
            state.ρu[1] * state.ρq_rai / state.ρ,
            FT(0),
            (state.ρu[3] / state.ρ - rain_w) * state.ρq_rai,
        )
        flux.ρq_sno = SVector(
            state.ρu[1] * state.ρq_sno / state.ρ,
            FT(0),
            (state.ρu[3] / state.ρ - snow_w) * state.ρq_sno,
        )
        # ... energy ...
        flux.ρe = SVector(
            state.ρu[1] / state.ρ * (state.ρe + aux.p),
            FT(0),
            state.ρu[3] / state.ρ * (state.ρe + aux.p),
        )
        # ... and don't advect momentum (kinematic setup)
    end
end

function source!(
    m::KinematicModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    FT = eltype(state)

    _grav::FT = grav(param_set)

    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)

    _cv_d::FT = cv_d(param_set)
    _cv_v::FT = cv_v(param_set)
    _cv_l::FT = cv_l(param_set)
    _cv_i::FT = cv_i(param_set)

    _T_0::FT = T_0(param_set)
    _T_freeze = T_freeze(param_set)

    @inbounds begin
        e_tot = state.ρe / state.ρ
        q_tot = state.ρq_tot / state.ρ
        q_liq = state.ρq_liq / state.ρ
        q_ice = state.ρq_ice / state.ρ
        q_rai = state.ρq_rai / state.ρ
        q_sno = state.ρq_sno / state.ρ
        u = state.ρu[1] / state.ρ
        w = state.ρu[3] / state.ρ
        ρ = state.ρ
        e_int = e_tot - 1 // 2 * (u^2 + w^2) - _grav * aux.z

        q = PhasePartition(q_tot, q_liq, q_ice)
        T = air_temperature(param_set, e_int, q)
        _Lf = latent_heat_fusion(param_set, T)
        # equilibrium state at current T
        ts_eq = TemperatureSHumEquil(param_set, T, state.ρ, q_tot)
        q_eq = PhasePartition(ts_eq)

        # zero out the source terms
        source.ρq_tot = FT(0)
        source.ρq_liq = FT(0)
        source.ρq_ice = FT(0)
        source.ρq_rai = FT(0)
        source.ρq_sno = FT(0)
        source.ρe = FT(0)

        # vapour -> cloud liquid water
        source.ρq_liq += ρ * conv_q_vap_to_q_liq_ice(liquid_param_set, q_eq, q)
        # vapour -> cloud ice
        source.ρq_ice += ρ * conv_q_vap_to_q_liq_ice(ice_param_set, q_eq, q)

        ## cloud liquid water -> rain
        acnv = ρ * conv_q_liq_to_q_rai(rain_param_set, q_liq)
        source.ρq_liq -= acnv
        source.ρq_tot -= acnv
        source.ρq_rai += acnv
        source.ρe -= acnv * _cv_l * (T - _T_0)

        ## cloud ice -> snow
        acnv = ρ * conv_q_ice_to_q_sno(param_set, ice_param_set, q, state.ρ, T)
        source.ρq_ice -= acnv
        source.ρq_tot -= acnv
        source.ρq_sno += acnv
        source.ρe -= acnv * (_cv_i * (T - _T_0) - _e_int_i0)

        # cloud liquid water + rain -> rain
        accr =
            ρ * accretion(
                param_set,
                liquid_param_set,
                rain_param_set,
                q_liq,
                q_rai,
                state.ρ,
            )
        source.ρq_liq -= accr
        source.ρq_tot -= accr
        source.ρq_rai += accr
        source.ρe -= accr * _cv_l * (T - _T_0)

        # cloud ice + snow -> snow
        accr =
            ρ * accretion(
                param_set,
                ice_param_set,
                snow_param_set,
                q_ice,
                q_sno,
                state.ρ,
            )
        source.ρq_ice -= accr
        source.ρq_tot -= accr
        source.ρq_sno += accr
        source.ρe -= accr * (_cv_i * (T - _T_0) - _e_int_i0)

        # cloud liquid water + snow -> snow or rain
        accr =
            ρ * accretion(
                param_set,
                liquid_param_set,
                snow_param_set,
                q_liq,
                q_sno,
                state.ρ,
            )
        if T < _T_freeze
            source.ρq_liq -= accr
            source.ρq_tot -= accr
            source.ρq_sno += accr
            source.ρe -= accr * (_cv_i * (T - _T_0) - _e_int_i0)
        else
            source.ρq_liq -= accr
            source.ρq_tot -= accr
            source.ρq_sno -= accr * (_cv_l / _Lf * (T - _T_freeze))
            source.ρq_rai += accr * (FT(1) + _cv_l / _Lf * (T - _T_freeze))
            source.ρe += -accr * _cv_l * (FT(2) * T - _T_freeze - _T_0)
        end

        # cloud ice + rain -> snow
        accr =
            ρ * accretion(
                param_set,
                ice_param_set,
                rain_param_set,
                q_ice,
                q_rai,
                state.ρ,
            )
        accr_rain_sink =
            ρ * accretion_rain_sink(
                param_set,
                ice_param_set,
                rain_param_set,
                q_ice,
                q_rai,
                state.ρ,
            )
        source.ρq_ice -= accr
        source.ρq_tot -= accr
        source.ρq_rai -= accr_rain_sink
        source.ρq_sno += accr + accr_rain_sink
        source.ρe -=
            accr_rain_sink * _Lf + accr * (_cv_i * (T - _T_0) - _e_int_i0)

        # rain + snow -> snow or rain
        if T < _T_freeze
            accr =
                ρ * accretion_snow_rain(
                    param_set,
                    snow_param_set,
                    rain_param_set,
                    q_sno,
                    q_rai,
                    state.ρ,
                )
            source.ρq_sno += accr
            source.ρq_rai -= accr
            source.ρe += accr * _Lf
        else
            accr =
                ρ * accretion_snow_rain(
                    param_set,
                    rain_param_set,
                    snow_param_set,
                    q_rai,
                    q_sno,
                    state.ρ,
                )
            source.ρq_rai += accr
            source.ρq_sno -= accr
            source.ρe -= accr * _Lf
        end

        # rain -> vapour
        evap =
            ρ * evaporation_sublimation(
                param_set,
                rain_param_set,
                q,
                q_rai,
                state.ρ,
                T,
            )
        source.ρq_rai += evap
        source.ρq_tot -= evap
        source.ρe -= evap * _cv_l * (T - _T_0)

        # snow -> vapour
        subl =
            ρ * evaporation_sublimation(
                param_set,
                snow_param_set,
                q,
                q_sno,
                state.ρ,
                T,
            )
        source.ρq_sno += subl
        source.ρq_tot -= subl
        source.ρe -= subl * (_cv_i * (T - _T_0) - _e_int_i0)

        # snow -> rain
        melt = ρ * snow_melt(param_set, snow_param_set, q_sno, state.ρ, T)

        source.ρq_sno -= melt
        source.ρq_rai += melt
        source.ρe -= melt * _Lf
    end
end

function main()
    # Working precision
    FT = Float64
    # DG polynomial order
    N = 4
    # Domain resolution and size
    Δx = FT(500)
    Δy = FT(1)
    Δz = FT(250)
    resolution = (Δx, Δy, Δz)
    # Domain extents
    xmax = 90000
    ymax = 10
    zmax = 16000
    # initial configuration
    wmax = FT(0.6)  # max velocity of the eddy  [m/s]
    θ_0 = FT(289) # init. theta value (const) [K]
    p_0 = FT(101500) # surface pressure [Pa]
    p_1000 = FT(100000) # reference pressure in theta definition [Pa]
    qt_0 = FT(7.5 * 1e-3) # init. total water specific humidity (const) [kg/kg]
    z_0 = FT(0) # surface height

    # time stepping
    t_ini = FT(0)
    t_end = FT(15 * 60) #FT(4 * 60 * 60)
    dt = FT(15)
    #CFL = FT(1.75)
    filter_freq = 1
    output_freq = 4 * 3

    # periodicity and boundary numbers
    periodicity_x = false
    periodicity_y = true
    periodicity_z = false
    idx_bc_left = 1
    idx_bc_right = 2
    idx_bc_front = 3
    idx_bc_back = 4
    idx_bc_bottom = 5
    idx_bc_top = 6

    driver_config = config_kinematic_eddy(
        FT,
        N,
        resolution,
        xmax,
        ymax,
        zmax,
        wmax,
        θ_0,
        p_0,
        p_1000,
        qt_0,
        z_0,
        periodicity_x,
        periodicity_y,
        periodicity_z,
        idx_bc_left,
        idx_bc_right,
        idx_bc_front,
        idx_bc_back,
        idx_bc_bottom,
        idx_bc_top,
    )
    solver_config = ClimateMachine.SolverConfiguration(
        t_ini,
        t_end,
        driver_config,
        ode_dt = dt,
        init_on_cpu = true,
        #Courant_number = CFL,
    )

    model = driver_config.bl

    mpicomm = MPI.COMM_WORLD

    # get state variables indices for filtering
    ρq_liq_ind = varsindex(vars_state_conservative(model, FT), :ρq_liq)
    ρq_ice_ind = varsindex(vars_state_conservative(model, FT), :ρq_ice)
    ρq_rai_ind = varsindex(vars_state_conservative(model, FT), :ρq_rai)
    ρq_sno_ind = varsindex(vars_state_conservative(model, FT), :ρq_sno)
    # get aux variables indices for testing
    q_tot_ind = varsindex(vars_state_auxiliary(model, FT), :q_tot)
    q_vap_ind = varsindex(vars_state_auxiliary(model, FT), :q_vap)
    q_liq_ind = varsindex(vars_state_auxiliary(model, FT), :q_liq)
    q_ice_ind = varsindex(vars_state_auxiliary(model, FT), :q_ice)
    q_rai_ind = varsindex(vars_state_auxiliary(model, FT), :q_rai)
    q_sno_ind = varsindex(vars_state_auxiliary(model, FT), :q_sno)
    S_ind = varsindex(vars_state_auxiliary(model, FT), :S)
    rain_w_ind = varsindex(vars_state_auxiliary(model, FT), :rain_w)
    snow_w_ind = varsindex(vars_state_auxiliary(model, FT), :snow_w)

    # filter out negative values
    cb_tmar_filter =
        GenericCallbacks.EveryXSimulationSteps(filter_freq) do (init = false)
            Filters.apply!(
                solver_config.Q,
                (:ρq_liq, :ρq_ice, :ρq_rai, :ρq_sno),
                solver_config.dg.grid,
                TMARFilter(),
            )
            nothing
        end

    # output for paraview

    # initialize base output prefix directory from rank 0
    vtkdir = abspath(joinpath(ClimateMachine.Settings.output_dir, "vtk"))
    if MPI.Comm_rank(mpicomm) == 0
        mkpath(vtkdir)
    end
    MPI.Barrier(mpicomm)

    step = [0]
    cb_vtk =
        GenericCallbacks.EveryXSimulationSteps(output_freq) do (init = false)
            out_dirname = @sprintf(
                "microphysics_test_4_mpirank%04d_step%04d",
                MPI.Comm_rank(mpicomm),
                step[1]
            )
            out_path_prefix = joinpath(vtkdir, out_dirname)
            @info "doing VTK output" out_path_prefix
            writevtk(
                out_path_prefix,
                solver_config.Q,
                solver_config.dg,
                flattenednames(vars_state_conservative(model, FT)),
                solver_config.dg.state_auxiliary,
                flattenednames(vars_state_auxiliary(model, FT)),
            )
            step[1] += 1
            nothing
        end

    # call solve! function for time-integrator
    result = ClimateMachine.invoke!(
        solver_config;
        user_callbacks = (cb_tmar_filter, cb_vtk),
        check_euclidean_distance = true,
    )

    ## supersaturation in the model
    #max_S = maximum(abs.(solver_config.dg.state_auxiliary[:, S_ind, :]))
    #@test max_S < FT(0.25)
    #@test max_S > FT(0)

    ## qt < reference number
    #max_q_tot = maximum(abs.(solver_config.dg.state_auxiliary[:, q_tot_ind, :]))
    #@test max_q_tot < FT(0.0077)

    ## no ice
    #max_q_ice = maximum(abs.(solver_config.dg.state_auxiliary[:, q_ice_ind, :]))
    #@test isequal(max_q_ice, FT(0))

    ## q_liq ∈ reference range
    #max_q_liq = max(solver_config.dg.state_auxiliary[:, q_liq_ind, :]...)
    #min_q_liq = min(solver_config.dg.state_auxiliary[:, q_liq_ind, :]...)
    #@test max_q_liq < FT(1e-3)
    #@test abs(min_q_liq) < FT(1e-5)

    ## q_rai ∈ reference range
    #max_q_rai = max(solver_config.dg.state_auxiliary[:, q_rai_ind, :]...)
    #min_q_rai = min(solver_config.dg.state_auxiliary[:, q_rai_ind, :]...)
    #@test max_q_rai < FT(3e-5)
    #@test abs(min_q_rai) < FT(3e-8)

    ## terminal velocity ∈ reference range
    #max_rain_w = max(solver_config.dg.state_auxiliary[:, rain_w_ind, :]...)
    #min_rain_w = min(solver_config.dg.state_auxiliary[:, rain_w_ind, :]...)
    #@test max_rain_w < FT(4)
    #@test isequal(min_rain_w, FT(0))
end

main()
