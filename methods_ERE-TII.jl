#=
Author: Emiliano Rojas Eng
These are the main mehtods for the paper 
"Tackling Informality and Inequality: A Directed Search Model" 2022
Last Update: July 24, 2023
=#

using JLD2, Interpolations, Plots, Optim, StatsBase, QuadGK, Distributions, 
StatsPlots, ProgressBars, Roots, DataFrames, GLM, CategoricalArrays, RegressionTables,
Images, ColorSchemes, Measures

"""
Main Code:
Updates Value Functions -> policy functions -> firm values -> market tightness -> VF
until convergence (of market tightness)
"""
function get_all(J_f, V_f, V_i, V_u, θ_f, params)

    error_θ = 1.0e3
    iters = 1
    
    while error_θ > tol && iters <= 25

        #gets new values of V_f, V_i, V_u
        iterate_workers!(V_f, V_i, V_u, θ_f, params); 

        #gets policy functions
        V_f, V_i, V_u, sector_f_pol, sector_i_pol, sector_u_pol, a_f_pol, a_i_pol, a_u_pol, w_ff_pol, w_if_pol, w_uf_pol,
        R_ff, R_fi, R_if, R_ii, R_uf, R_ui = 
        get_policy(V_f, V_i, V_u, θ_f, params);

        #get firm value
        update_J_f!(J_f, θ_f, w_ff_pol, a_f_pol, params);

        #update market tightness θ_f
        θ_f_old = copy(θ_f)
        update_θ!(θ_f, J_f, params)

        #calculates error
        error_θ = get_error_θ(θ_f, θ_f_old, sector_f_pol, sector_i_pol, sector_u_pol, w_ff_pol, w_if_pol, w_uf_pol, params)
        println("MAIN ITERATION ERROR IN ITERATION $iters IS $error_θ")
        iters += 1

    end

    return J_f, V_f, V_i, V_u, θ_f, error_θ
end

"""
This function updates the value function of all labor status {f, i, u} and types of workers (z, w, a)
A reward function is calculating for every and then value functions are updated until convergence
INPUT: Market Tightness θ_f and grid sizes N_z, N_w, N_a. and some guess for the initial Value functions
OUTPUT: updated values of V_f, V_i, V_u
"""
function iterate_workers!(V_f, V_i, V_u, θ_f, params)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    ## Initialize reward grids

    R_ff = zeros(N_z, N_w, N_a)
    R_fi = copy(R_ff)
    R_if = copy(R_ff)
    R_ii = copy(R_ff)
    R_uf = copy(R_ff)
    R_ui = copy(R_ff)

    V_bar_f = copy(R_ff)
    V_bar_i = copy(R_ff)
    V_bar_u = copy(R_ff)

    E_V_i = copy(R_ff)

    N_w_u_max = min(Integer(floor(rho * N_w) + 1), N_w)

    error = 1.0e2
    iters = 1

    while error > tol/2 && iters <= 200
        V_f_old = copy(V_f)
        V_i_old = copy(V_i)    
        V_u_old = copy(V_u)

        #first some auxiliary functions
        
        #calculates expected utility of labouring in the informal economy (given probability distribution of earnings)
        V_i_fun = linear_interpolation((z, w, a), V_i)
        for i_z = 1:N_z, i_a = 1:N_a
            #V_i_1d = interpolate(w, V_i[i_z, :, i_a], SteffenMonotonicInterpolation()) #Interpolation of Value function of informal
            E_V_i[i_z, :, i_a] .= quadgk(w -> informal_I_distribution(w)*V_i_fun(z[i_z], w, a[i_a]), w_min, w_max, rtol=1e-5)[1]
        end

        V_u_fun = linear_interpolation((z, w, a), V_i)
        for (i_z, i_w, i_a) in grid_index
            #calculates for every index i_w, which index corresponds to an unemployment insurance of rho*w[i_w]
            #V_u_1d = interpolate(w, V_u[i_z, :, i_a], SteffenMonotonicInterpolation()) #Interpolation of Value function of unemployment insurance
            V_bar_f[i_z, i_w, i_a] = δ_f * V_u_fun(z[i_z], rho * w[i_w], a[i_a]) + (1-δ_f) * V_f[i_z, i_w, i_a]  #unemployment with insurance or staying the same
        end

        V_bar_i = δ_i * V_u[:, fill(1, N_w), :] + (1-δ_i) * E_V_i #unemployment without insurance or staying the same
        V_bar_u = xi * V_u[:, fill(1, N_w), :] + (1-xi) * V_u # insurance expires or stays the same

        ## These are the rewards for every given flow

        #transition to informality
        R_fi = λ_f * prob_i * E_V_i + (1 - λ_f * prob_i) .* V_bar_f
        R_ii = λ_i * prob_i * E_V_i + (1 - λ_i * prob_i) .* V_bar_i
        R_ui = λ_u * prob_i * E_V_i + (1 - λ_u * prob_i) .* V_bar_u

        #transition to formality
    
        Threads.@threads for (i_z, i_w, i_a) in grid_index

            V_f_1d = linear_interpolation(w, V_f[i_z, :, i_a])#
            #V_f_1d = interpolate(w[2:end], V_f[i_z, 2:end, i_a], SteffenMonotonicInterpolation()) #Interpolation of Value function of formal
            p_θ_f_1d = linear_interpolation(w, p.(θ_f[i_z, :, i_a]))#
            #p_θ_f_1d = interpolate(w[2:end], p.(θ_f[i_z, 2:end, i_a]), SteffenMonotonicInterpolation()) #Interpolation of θ_f

            #calculates maximum wage feasible
            upper_bound = sum(θ_f[i_z, 2:end, i_a] .< 1.0e-5) > 0 ? w[2:end][θ_f[i_z, 2:end, i_a] .< 1.0e-5][1] : w[end]

            #formal to formal
            res_ff = Optim.optimize(
                x-> -(λ_f * p_θ_f_1d(x) .* V_f_1d(x) + (1 .- λ_f * p_θ_f_1d(x)) .* V_bar_f[i_z, i_w, i_a]), #value of searching wage x
                w[2],
                upper_bound
                );
            R_ff[i_z, i_w, i_a] = -Optim.minimum(res_ff)

            #informal to formal
            res_if = Optim.optimize(
                x-> -(λ_i * p_θ_f_1d(x) .* V_f_1d(x) + (1 .- λ_i * p_θ_f_1d(x)) .* V_bar_i[i_z, i_w, i_a]), #value of searching wage x
                w[2],
                upper_bound
                );
            R_if[i_z, i_w, i_a] = -Optim.minimum(res_if)

            #unemployment to formal
            if i_w <= N_w_u_max
            res_uf = Optim.optimize(
                x-> -(λ_u * p_θ_f_1d(x) .* V_f_1d(x) + (1 .- λ_u * p_θ_f_1d(x)) .* V_bar_u[i_z, i_w, i_a]), #value of searching wage x
                w[2],
                upper_bound
                );
            R_uf[i_z, i_w, i_a] = -Optim.minimum(res_uf)
            else
                R_uf[i_z, i_w, i_a] = 0
            end
        end

        ## gets the total reward given sector status. It is the maximum of transition to formality or informality. Formal sector = 1, Informal sector = 2
        ## gets the value function, optimal choice of wealth accumulation

        Threads.@threads for (i_z, i_w, i_a) in grid_index

            R_f_1d = linear_interpolation(a, max(R_ff[i_z, i_w, :], R_fi[i_z, i_w, :]))#
            #R_f_1d = interpolate(a, max(R_ff[i_z, i_w, :], R_fi[i_z, i_w, :]), SteffenMonotonicInterpolation())
            R_i_1d = linear_interpolation(a, max(R_if[i_z, i_w, :], R_ii[i_z, i_w, :]))#
            #R_i_1d = interpolate(a, max(R_if[i_z, i_w, :], R_ii[i_z, i_w, :]), SteffenMonotonicInterpolation())
            R_u_1d = linear_interpolation(a, max(R_uf[i_z, i_w, :], R_ui[i_z, i_w, :]))#
            #R_u_1d = interpolate(a, max(R_uf[i_z, i_w, :], R_ui[i_z, i_w, :]), SteffenMonotonicInterpolation())

            #upper bound of wealth is such that is inside [a_min, a_max] and less than the budget constraint
            upper_bound = min(max((I_u + w[i_w]+a[i_a])*(1+r), a[1]), a[N_a])

            #formal
            R_f_fun_1d(x) = R_f_fun(z[i_z], w[i_w], x)
            res_f = Optim.optimize(
                x-> -(u.(I_u + w[i_w] + a[i_a] .- x/(1+r)) + β*R_f_1d(x)), #value of holding x wealth
                a[1], #lower bound
                upper_bound #upper bound
                );
            V_f[i_z, i_w, i_a] = -Optim.minimum(res_f)

            #informal
            res_i = Optim.optimize(
                x-> -(u.(I_u + w[i_w] + a[i_a] .- x/(1+r)) + β*R_i_1d(x)), #value of holding x wealth
                a[1], #lower bound
                upper_bound #upper bound
                );
            V_i[i_z, i_w, i_a] = -Optim.minimum(res_i)

            #unemployed
            if i_w <= N_w_u_max
            res_u = Optim.optimize(
                x-> -(u.(I_u + w[i_w] + a[i_a] .- x/(1+r)) + β*R_u_1d(x)), #value of holding x wealth
                a[1], #lower bound
                upper_bound #upper bound
                );
            V_u[i_z, i_w, i_a] = -Optim.minimum(res_u)
            else
                V_u[i_z, i_w, i_a] = 0.0
            end
        end

        #Calculates the change in Value Functions. 
        e_f = quantile(vec(abs.(V_f - V_f_old)[2:end, 2:end, 2:end]), 0.99)
        e_i = quantile(vec(abs.(V_i - V_i_old)[:, :, 2:end]), 0.99)
        #The error of the unemployed value function only matters for the space that is visited. i.e where income < maximum insurance (rho * w[N_w])
        e_u = quantile(vec(abs.(V_u - V_u_old)[:, 1:N_w_u_max, 2:end]), 0.99)

        error = e_f + e_u + e_i

        if mod(iters, 30) == 0
            #=
            p1 = plot(z[1:end], mean(V_i, dims = [2, 3])[1:end])
            p2 = plot(w[2:end], mean(V_i, dims = [1, 3])[2:end])
            p3 = plot(a[2:end], mean(V_i, dims = [1, 2])[2:end])
            display(plot(p1, p2, p3, layout = (1, 3)))
            #display(plot(V_u[:, 1, 2:end]', legend = false))
            =#

            println("error in iteration $iters is e_f = $e_f, e_i = $e_i, e_u = $e_u")
            #println("error in iteration $iters is $error")
        else 
            nothing
        end

        iters += 1

    end

    return V_f, V_i, V_u
end

"""
This function obtains the policy functions of all labor status {f, i, u} and types of workers (z, w, a)
Its literally a copy paste from one interation of iterate_workers, but also saves the argmax of the decisions.
"""
function get_policy(V_f, V_i, V_u, θ_f, params)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    ## Initialize reward grids
    R_ff = zeros(N_z, N_w, N_a)
    R_fi = copy(R_ff)
    R_if = copy(R_ff)
    R_ii = copy(R_ff)
    R_uf = copy(R_ff)
    R_ui = copy(R_ff)

    V_bar_f = copy(R_ff)
    V_bar_i = copy(R_ff)
    V_bar_u = copy(R_ff)
    
    E_V_i = copy(R_ff)

    #initialize policy grids
    #sector choice is an integer (1 = formal, 2 = informal)
    sector_f_pol = zeros(Int32, N_z, N_w, N_a)
    sector_i_pol = copy(sector_f_pol)
    sector_u_pol = copy(sector_f_pol)

    #optimal wage to search for (if looking for a formal job)
    w_ff_pol = copy(R_ff)
    w_if_pol = copy(R_ff)
    w_uf_pol = copy(R_ff)

    #wealth storage policy
    a_f_pol = copy(R_ff)
    a_i_pol = copy(R_ff)
    a_u_pol = copy(R_ff)

    N_w_u_max = min(Integer(floor(rho * N_w) + 1), N_w)


    ## This is a copy paste from iterate_workers!, I will denote the additional code with ###!!!

    #first some auxiliary functions
    
    #calculates expected utility of labouring in the informal economy (given probability distribution of earnings)
    V_i_fun = linear_interpolation((z, w, a), V_i)
    for i_z = 1:N_z, i_a = 1:N_a
        #V_i_1d = interpolate(w, V_i[i_z, :, i_a], SteffenMonotonicInterpolation()) #Interpolation of Value function of informal
        E_V_i[i_z, :, i_a] .= quadgk(w -> informal_I_distribution(w)*V_i_fun(z[i_z], w, a[i_a]), w_min, w_max, rtol=1e-5)[1]
    end

    V_u_fun = linear_interpolation((z, w, a), V_i)
    for (i_z, i_w, i_a) in grid_index
        #calculates for every index i_w, which index corresponds to an unemployment insurance of rho*w[i_w]
        #V_u_1d = interpolate(w, V_u[i_z, :, i_a], SteffenMonotonicInterpolation()) #Interpolation of Value function of unemployment insurance
        V_bar_f[i_z, i_w, i_a] = δ_f * V_u_fun(z[i_z], rho * w[i_w], a[i_a]) + (1-δ_f) * V_f[i_z, i_w, i_a]  #unemployment with insurance or staying the same
    end   
    V_bar_i = δ_i * V_u[:, fill(1, N_w), :] + (1-δ_i) * E_V_i #unemployment without insurance or staying the same
    V_bar_u = xi * V_u[:, fill(1, N_w), :] + (1-xi) * V_u # insurance expires or stays the same

    ## These are the rewards for every given flow

    #transition to informality
    R_fi = λ_f * prob_i * E_V_i + (1 - λ_f * prob_i) .* V_bar_f
    R_ii = λ_i * prob_i * E_V_i + (1 - λ_i * prob_i) .* V_bar_i
    R_ui = λ_u * prob_i * E_V_i + (1 - λ_u * prob_i) .* V_bar_u

    #transition to formality

    Threads.@threads for (i_z, i_w, i_a) in grid_index

        V_f_1d = linear_interpolation(w, V_f[i_z, :, i_a])#
        #V_f_1d = interpolate(w[2:end], V_f[i_z, 2:end, i_a], SteffenMonotonicInterpolation()) #Interpolation of Value function of formal
        p_θ_f_1d = linear_interpolation(w, p.(θ_f[i_z, :, i_a]))#
        #p_θ_f_1d = interpolate(w[2:end], p.(θ_f[i_z, 2:end, i_a]), SteffenMonotonicInterpolation()) #Interpolation of θ_f

        #calculates maximum wage feasible
        upper_bound = sum(θ_f[i_z, 2:end, i_a] .< 1.0e-5) > 0 ? w[2:end][θ_f[i_z, 2:end, i_a] .< 1.0e-5][1] : w[end]

        #formal to formal
        res_ff = Optim.optimize(
            x-> -(λ_f * p_θ_f_1d(x) .* V_f_1d(x) + (1 .- λ_f * p_θ_f_1d(x)) .* V_bar_f[i_z, i_w, i_a]), #value of searching wage x
            w[2],
            upper_bound
            );
        R_ff[i_z, i_w, i_a] = -Optim.minimum(res_ff)
        w_ff_pol[i_z, i_w, i_a] = Optim.minimizer(res_ff) ###!!!

        #informal to formal
        #informal to formal
        res_if = Optim.optimize(
            x-> -(λ_i * p_θ_f_1d(x) .* V_f_1d(x) + (1 .- λ_i * p_θ_f_1d(x)) .* V_bar_i[i_z, i_w, i_a]), #value of searching wage x
            w[2],
            upper_bound
            );
        R_if[i_z, i_w, i_a] = -Optim.minimum(res_if)
        w_if_pol[i_z, i_w, i_a] = Optim.minimizer(res_if) ###!!!

        #unemployment to formal
        if i_w <= N_w_u_max
        res_uf = Optim.optimize(
            x-> -(λ_u * p_θ_f_1d(x) .* V_f_1d(x) + (1 .- λ_u * p_θ_f_1d(x)) .* V_bar_u[i_z, i_w, i_a]), #value of searching wage x
            w[2],
            upper_bound
            );
        R_uf[i_z, i_w, i_a] = -Optim.minimum(res_uf)
        w_uf_pol[i_z, i_w, i_a] = Optim.minimizer(res_uf) ###!!!
        else
            R_uf[i_z, i_w, i_a] = 0.0
            w_uf_pol[i_z, i_w, i_a] = 0.0
        end

        ## #gets the policy for sector selection. Formal sector = 1, Informal sector = 2
        sector_f_pol[i_z, i_w, i_a] = argmax([R_ff[i_z, i_w, i_a], R_fi[i_z, i_w, i_a]]) ###!!!
        sector_i_pol[i_z, i_w, i_a] = argmax([R_if[i_z, i_w, i_a], R_ii[i_z, i_w, i_a]]) ###!!!
        sector_u_pol[i_z, i_w, i_a] = argmax([R_uf[i_z, i_w, i_a], R_ui[i_z, i_w, i_a]]) ###!!!
    end

    ## gets the total reward given sector status. It is the maximum of transition to formality or informality. Formal sector = 1, Informal sector = 2

    Threads.@threads for (i_z, i_w, i_a) in grid_index

        R_f_fun = linear_interpolation(a, max(R_ff[i_z, i_w, :], R_fi[i_z, i_w, :]))#
        #R_f_fun = interpolate(a, max(R_ff[i_z, i_w, :], R_fi[i_z, i_w, :]), SteffenMonotonicInterpolation())
        R_i_fun = linear_interpolation(a, max(R_if[i_z, i_w, :], R_ii[i_z, i_w, :]))#
        #R_i_fun = interpolate(a, max(R_if[i_z, i_w, :], R_ii[i_z, i_w, :]), SteffenMonotonicInterpolation())
        R_u_fun = linear_interpolation(a, max(R_uf[i_z, i_w, :], R_ui[i_z, i_w, :]))#
        #R_u_fun = interpolate(a, max(R_uf[i_z, i_w, :], R_ui[i_z, i_w, :]), SteffenMonotonicInterpolation())

        #upper bound of wealth is such that is inside [a_min, a_max] and less than the budget constraint
        upper_bound = min(max((I_u + w[i_w]+a[i_a])*(1+r), a[1]), a[N_a])

        #formal
        res_f = Optim.optimize(
            x-> -(u.(I_u + w[i_w] + a[i_a] .- x/(1+r)) + β*R_f_fun(x)), #value of holding x wealth
            a[1], #lower bound
            upper_bound #upper bound
            );
        V_f[i_z, i_w, i_a] = -Optim.minimum(res_f)
        a_f_pol[i_z, i_w, i_a] = Optim.minimizer(res_f) ###!!!

        #informal
        res_i = Optim.optimize(
            x-> -(u.(I_u + w[i_w] + a[i_a] .- x/(1+r)) + β*R_i_fun(x)), #value of holding x wealth
            a[1], #lower bound
            upper_bound #upper bound
            );
        V_i[i_z, i_w, i_a] = -Optim.minimum(res_i)
        a_i_pol[i_z, i_w, i_a] = Optim.minimizer(res_i) ###!!!

        #unemployed
        if i_w <= N_w_u_max
        res_u = Optim.optimize(
            x-> -(u.(I_u + w[i_w] + a[i_a] .- x/(1+r)) + β*R_u_fun(x)), #value of holding x wealth
            a[1], #lower bound
            upper_bound #upper bound
            );
        V_u[i_z, i_w, i_a] = -Optim.minimum(res_u)
        a_u_pol[i_z, i_w, i_a] = Optim.minimizer(res_u) ###!!!
        else
            V_u[i_z, i_w, i_a] = 0.0
            a_u_pol[i_z, i_w, i_a] = 0.0
        end

    end

    return V_f, V_i, V_u, sector_f_pol, sector_i_pol, sector_u_pol, a_f_pol, a_i_pol, a_u_pol, w_ff_pol, w_if_pol, w_uf_pol,
    R_ff, R_fi, R_if, R_ii, R_uf, R_ui

end

"""
Updates the value of the formal firm (given value functions and policy functions)
"""
function update_J_f!(J_f, θ_f, w_ff_pol, a_f_pol, params)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    J_f_old = copy(J_f)
    profit = [(1-τ)*(y_f + z[i_z] - (1+τ_w)*w[i_w]) for i_z in 1:N_z, i_w in 1:N_w]

    Threads.@threads for (i_z, i_w, i_a) in grid_index
        if profit[i_z, i_w] < 0
            J_f[i_z, i_w, i_a] = 0
        else
            p_θ_f_1d = linear_interpolation(w, p.(θ_f[i_z, :, i_a]))#interpolate(w, p.(θ_f[i_z, :, i_a]), SteffenMonotonicInterpolation()) #Interpolation of θ_f
            J_f_1d = linear_interpolation(a, J_f_old[i_z, i_w, :])#interpolate(a, J_f_old[i_z, i_w, :], SteffenMonotonicInterpolation()) #Interpolation of Firm Value function
            J_f[i_z, i_w, i_a] = profit[i_z, i_w] + β*(1-δ_f)*(1-λ_f*p_θ_f_1d(w_ff_pol[i_z, i_w, i_a]))*J_f_1d(a_f_pol[i_z, i_w, i_a])
        end
    end

    return J_f
end


"""
This function updates the market tightness for submarket (z, w, a) given the firm value of 
operating in submarket (z, w, a) and the fixed cost of entering k. 
Free market entry is such that the expected value before entry is 0
"""
function update_θ!(θ_f, J_f, params)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    Threads.@threads for (i_z, i_w, i_a) in grid_index
        #condition for positive market tightness: profitability high enough
        if (β * J_f[i_z, i_w, i_a] >= (k_f)) & (i_w != 1)
            θ_f[i_z, i_w, i_a] = min(q_inv(k_f/(β * J_f[i_z, i_w, i_a])), 1.0e5)
        else
            θ_f[i_z, i_w, i_a] = 0.0
        end
    end

    return θ_f

end

"""
This function calculates the error of the main iteration. That is, the old θ_f vs the new θ_f
It is the square root of the sum of squared errors.
INPUT: θ_f and θ_f_old to compare
OUTPUT: The error of the iteration: error_θ
"""
function get_error_θ(θ_f, θ_f_old, sector_f_pol, sector_i_pol, sector_u_pol, w_ff_pol, w_if_pol, w_uf_pol, params)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    #matrix of all squared differences, norm given by p() is calculated
    loss_M = (p.(θ_f)-p.(θ_f_old)).^2
    error_θ = sqrt(sum(loss_M))

    p1 = plot(z[1:end], mean(p.(θ_f), dims = [2, 3])[1:end]*100)
    p2 = plot(w[1:end], mean(p.(θ_f), dims = [1, 3])[1:end]*100)
    p3 = plot(a[1:end], mean(p.(θ_f), dims = [1, 2])[1:end]*100)
    display(plot(p1, p2, p3, layout = (1, 3)))

    return error_θ
end

"""
Unconstrains the equilibrium
"""
function J_θ_uncons(V_f, V_i, V_u, J_f, θ_f, params)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    J_f_unc = copy(J_f)
    θ_f_unc = copy(θ_f)

    #gets policy functions
    V_f, V_i, V_u, sector_f_pol, sector_i_pol, sector_u_pol, a_f_pol, a_i_pol, a_u_pol, w_ff_pol, w_if_pol, w_uf_pol,
    R_ff, R_fi, R_if, R_ii, R_uf, R_ui = 
    get_policy(V_f, V_i, V_u, θ_f, params);

    J_f_old = copy(J_f)

    profit = [(1-τ)*(y_f + z[i_z] - (1+τ_w)*w[i_w]) for i_z in 1:N_z, i_w in 1:N_w]

    Threads.@threads for (i_z, i_w, i_a) in grid_index

        if profit[i_z, i_w] < 0
            J_f[i_z, i_w, i_a] = 0

        else
            p_θ_f_1d = linear_interpolation(w, p.(θ_f[i_z, :, i_a]))#interpolate(w, p.(θ_f[i_z, :, i_a]), SteffenMonotonicInterpolation()) #Interpolation of θ_f
            J_f_1d = linear_interpolation(a, J_f_old[i_z, i_w, :])#interpolate(a, J_f_old[i_z, i_w, :], SteffenMonotonicInterpolation()) #Interpolation of Firm Value function

            #if individual (z, w, a) finds it optimal to remain in the formal sector
            if (R_ff-R_fi)[i_z, i_w, i_a] >= 0
                J_f_unc[i_z, i_w, i_a] = profit[i_z, i_w] + β*(1-δ_f)*(1-λ_f*p_θ_f_1d(w_ff_pol[i_z, i_w, i_a]))*J_f_1d(a_f_pol[i_z, i_w, i_a])
            #if individual (z, w, a) tries to get out of the formal sector
            else
                J_f_unc[i_z, i_w, i_a] = profit[i_z, i_w] + β*(1-δ_f)*(1-λ_f * prob_i)*J_f_1d(a_f_pol[i_z, i_w, i_a])
            end
        end
    end

    θ_f_unc = update_θ!(θ_f_unc, J_f_unc, params)

    return J_f_unc, θ_f_unc
end

"""
unpacks parameters from the dictionary params
"""
function unpack_params(params)

    r = params["r"]
    β = params["β"]
    σ = params["σ"]
    𝛾 = params["𝛾"]
    δ_f = params["δ_f"]
    δ_i = params["δ_i"]
    xi = params["xi"]
    rho = params["rho"]
    λ_f = params["λ_f"]
    λ_i = params["λ_i"]
    λ_u = params["λ_u"]
    k_f = params["k_f"]
    y_f = params["y_f"]
    prob_i = params["prob_i"]
    
    #tax parameters
    τ = params["τ"]
    τ_w = params["τ_w"]
    I_u = params["I_u"]

    #informal earnings distribution parameters
    distribution_a = params["distribution_a"]
    distribution_b = params["distribution_b"]

    #Grid parameters
    z_min = params["z_min"]
    z_max = params["z_max"]
    w_min = params["w_min"]
    w_max = params["w_max"]
    a_min = params["a_min"]
    a_max = params["a_max"]
    N_z = params["N_z"]
    N_w = params["N_w"]
    N_a = params["N_a"]

    #other main functions
    u(c) = max(c, 1.0e-4)^(1-σ)/(1-σ) #consumption
    p(θ) = (1+θ^-𝛾)^-(1/𝛾) #probability of finding a job given market tightness
    q(θ) = (1+θ^𝛾)^-(1/𝛾) #probability of filling a job given market tightness
    q_inv(q) = (q^-𝛾-1)^(1/𝛾) #inverse of q
        
    #distribution of earnings of the informal sector
    informal_I_distribution(x) = pdf(Beta(distribution_a, distribution_b), x)

    z = range(z_min, stop=z_max, length=N_z)
    w = range(w_min, stop=w_max, length=N_w)
    #a = range(a_min, stop=a_max, length=N_a)
    #To get a finer grid on the part where most agents are
    a = vcat(range(a_min, stop=5, length=Integer(floor(N_a*0.50))), range(5 + (a_max-5)/ceil(N_a*0.50), stop=a_max, length=Integer(ceil(N_a*0.50))))

    grid_index = collect(Iterators.product(1:N_z, 1:N_w, 1:N_a))

    return r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a
end
     
"""
interpolates all functions from a previous grid to save iterations (a better guess)
Assumes a new N_z, N_w, N_a
"""
function resize_all(J_f, V_f, V_i, V_u, θ_f, params_new)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params_new)



    J_f_new = zeros(N_z, N_w, N_a)
    V_f_new = zeros(N_z, N_w, N_a)
    V_i_new = zeros(N_z, N_w, N_a)
    V_u_new = zeros(N_z, N_w, N_a)
    θ_f_new = zeros(N_z, N_w, N_a)

    z_old = range(z_min, stop=z_max, length=size(θ_f)[1])
    w_old = range(w_min, stop=w_max, length=size(θ_f)[2])
    a_old = range(a_min, stop=a_max, length=size(θ_f)[3])

    V_f_fun = linear_interpolation((z_old, w_old, a_old), V_f)
    V_i_fun = linear_interpolation((z_old, w_old, a_old), V_i)
    V_u_fun = linear_interpolation((z_old, w_old, a_old), V_u)
    J_f_fun = linear_interpolation((z_old, w_old, a_old), J_f)
    θ_f_fun = linear_interpolation((z_old, w_old, a_old), θ_f)

    #Pure interpolation
    Threads.@threads for (i_z, i_w, i_a) in grid_index
        J_f_new[i_z, i_w, i_a] = J_f_fun(z[i_z], w[i_w], a[i_a])
        V_f_new[i_z, i_w, i_a] = V_f_fun(z[i_z], w[i_w], a[i_a])
        V_i_new[i_z, i_w, i_a] = V_i_fun(z[i_z], w[i_w], a[i_a])
        V_u_new[i_z, i_w, i_a] = V_u_fun(z[i_z], w[i_w], a[i_a])
        θ_f_new[i_z, i_w, i_a] = θ_f_fun(z[i_z], w[i_w], a[i_a])
    end

    #copies the new interpolated values
    J_f = copy(J_f_new)
    V_f = copy(V_f_new)
    V_i = copy(V_i_new)
    V_u = copy(V_u_new)
    θ_f = copy(θ_f_new)

    return J_f, V_f, V_i, V_u, θ_f
end

"""
calculates gini coeficcient given a vector of incomes
"""
function gini_calculate(incomes)
    N = length(incomes)

    #gini formula
    gini = 1/N*(N + 1 - 2*(
        sum(sort(incomes) .* collect(N:-1:1)/sum(incomes))
    ))
    
    return gini
end

"""
calculates theil inequality index given a vector of incomes
"""
function theil(x)
    T = mean(x/mean(x) .* log.(x/mean(x)))
    return T
end

"""
calculates theil inequality index given a vector of formal and informal incomes
Returns a decomposition of the theil inequality index: 
within formal, within informal and between groups
"""
function theil_subgroups(formal, informal)
    N_f = length(formal)
    N_i = length(informal)
    N = N_f + N_i

    mu_f = mean(formal)
    mu_i = mean(informal)
    mu_all = mean([formal; informal])

    T_within_f = 
        N_f/N * mu_f/mu_all * theil(formal)
    T_within_i = 
        N_i/N * mu_i/mu_all * theil(informal)
    T_between =
        N_f/N * mu_f/mu_all * log(mu_f/mu_all) + 
        N_i/N * mu_i/mu_all * log(mu_i/mu_all)

    return T_within_f, T_within_i, T_between
end

"""
This function simulates an economy populated by N_ind agents for T periods. Then it returns the last (equilibrium)
values.
INPUT: N_ind (number of individuals), T (time periods to simulate), and equilibrium functions. V_f, V_i, V_u, θ_f and grid sizes N_z, N_w, N_a.
OUTPUT: vector of individual characteristics: z_ind, w_ind, a_ind, status_ind.
"""
function simulation_baseline(V_f, V_i, V_u, θ_f, N_ind, T, params)

    ## Starts the vectors of population

    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    z_ind = rand(Beta(distribution_a, distribution_b), N_ind)
    w_ind = zeros(N_ind)
    a_ind = zeros(N_ind)
    status_ind = rand([1, 2, 3], N_ind) #1 formal, 2 informal, 3 unemployed
    m_unemp = zeros(N_ind) #How many months unemployed. 0 if not unemployed

    ## Gets the policy functions

    all_funs = simulation_get_all_funs(V_f, V_i, V_u, θ_f, params)

    # Starts simulation
    println("Simulating...")
    for t = ProgressBar(1:T)
        simulation_transition_t!(z_ind, w_ind, a_ind, status_ind, m_unemp, params, all_funs, N_ind)    
    end

    #calculates next consumption for current population
    c_ind = simulation_consumption(z_ind, w_ind, a_ind, status_ind, params, all_funs, N_ind)

    return z_ind, w_ind, a_ind, c_ind, status_ind, m_unemp
end

"""
This function takes an economy in stationary state given old parameters (input: z_ind0, w_ind0, a_ind0, status_ind0)
Given a change in policy variables that changes the equilibrium value functions (previously calculated) (input: V_f_1, V_i_1, V_u_1, θ_f_1)
it then computes the new stationary state (output: z_ind, w_ind, a_ind, status_ind)
and records the transition (output: gini_w, gini_w_f, gini_w_i, gini_a, gini_a_f, gini_a_i)
"""
function simulation_policy(z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0, 
V_f_0, V_i_0, V_u_0, θ_f_0, V_f_1, V_i_1, V_u_1, θ_f_1, params_0, params_1, N_ind, T, padding)

    z_ind = copy(z_ind_0)
    w_ind = copy(w_ind_0)
    a_ind = copy(a_ind_0)
    c_ind = copy(c_ind_0)
    status_ind = copy(status_ind_0)
    m_unemp = copy(m_unemp_0)

    ## Begins defining the tracked series
    per_f = zeros(T+padding)
    per_i = zeros(T+padding)
    per_u = zeros(T+padding)
    tax = zeros(T+padding)
    Tw_within_f = zeros(T+padding)
    Tw_within_i = zeros(T+padding)
    Tw_between = zeros(T+padding)
    Ta_within_f = zeros(T+padding)
    Ta_within_i = zeros(T+padding)
    Ta_between = zeros(T+padding)
    Tc_within_f = zeros(T+padding)
    Tc_within_i = zeros(T+padding)
    Tc_between = zeros(T+padding)
    Tz_within_f = zeros(T+padding)
    Tz_within_i = zeros(T+padding)
    Tz_between = zeros(T+padding)
    mean_unemp = zeros(T+padding)
    mean_income_f = zeros(T+padding)
    mean_income_i = zeros(T+padding)
    mean_income_u = zeros(T+padding)
    mean_wealth_f = zeros(T+padding)
    mean_wealth_i = zeros(T+padding)
    mean_wealth_u = zeros(T+padding)
    mean_consumption_f = zeros(T+padding)
    mean_consumption_i = zeros(T+padding)
    mean_consumption_u = zeros(T+padding)
    mean_ability_f = zeros(T+padding)
    mean_ability_i = zeros(T+padding)
    mean_ability_u = zeros(T+padding)
    ## Ends defining the tracked series

    ## functions for the baseline

    all_funs_0 = simulation_get_all_funs(V_f_0, V_i_0, V_u_0, θ_f_0, params_0)

    # Simulation with baseline parameters 
    for t = 1:padding

        simulation_transition_t!(z_ind, w_ind, a_ind, status_ind, m_unemp, params_0, all_funs_0, N_ind)

        ## records tracked series
        c_ind = simulation_consumption(z_ind, w_ind, a_ind, status_ind, params_0, all_funs_0, N_ind)
        simulation_track_series!(z_ind, w_ind, a_ind, c_ind, status_ind, m_unemp, params_0,
        per_f, per_i, per_u, tax, Tw_within_f, Tw_within_i, Tw_between, Ta_within_f, Ta_within_i, Ta_between,
        Tc_within_f, Tc_within_i, Tc_between, Tz_within_f, Tz_within_i, Tz_between,
        mean_unemp, mean_income_f, mean_income_i, mean_income_u, mean_wealth_f, mean_wealth_i, mean_wealth_u, 
        mean_consumption_f, mean_consumption_i, mean_consumption_u, mean_ability_f, mean_ability_i, mean_ability_u,
        N_ind, t)

    end

    all_funs_1 = simulation_get_all_funs(V_f_1, V_i_1, V_u_1, θ_f_1, params_1)

    # Simulation with policy parameters 
    println("Simulating...")
    for t = ProgressBar(padding+1:T+padding)

        simulation_transition_t!(z_ind, w_ind, a_ind, status_ind, m_unemp, params_1, all_funs_1, N_ind)

        ## records tracked series
        c_ind = simulation_consumption(z_ind, w_ind, a_ind, status_ind, params_1, all_funs_1, N_ind)
        simulation_track_series!(z_ind, w_ind, a_ind, c_ind, status_ind, m_unemp, params_1,
        per_f, per_i, per_u, tax, Tw_within_f, Tw_within_i, Tw_between, Ta_within_f, Ta_within_i, Ta_between,
        Tc_within_f, Tc_within_i, Tc_between, Tz_within_f, Tz_within_i, Tz_between,
        mean_unemp, mean_income_f, mean_income_i, mean_income_u, mean_wealth_f, mean_wealth_i, mean_wealth_u, 
        mean_consumption_f, mean_consumption_i, mean_consumption_u, mean_ability_f, mean_ability_i, mean_ability_u,
        N_ind, t)

    end

    return z_ind, w_ind, a_ind, status_ind, m_unemp, per_f, per_i, per_u, tax, mean_unemp,
    Tw_within_f, Tw_within_i, Tw_between, Ta_within_f, Ta_within_i, Ta_between, 
    Tc_within_f, Tc_within_i, Tc_between, Tz_within_f, Tz_within_i, Tz_between,
    mean_income_f, mean_income_i, mean_income_u, mean_wealth_f, mean_wealth_i, mean_wealth_u, 
    mean_consumption_f, mean_consumption_i, mean_consumption_u, mean_ability_f, mean_ability_i, mean_ability_u
end


"""
Auxiliary function.
Gets all functions necesary for simulation
Input: V_f, V_i, V_u, θ_f, params
Output: all_funs, tuple containing
(params, a_f_pol_fun, a_i_pol_fun, a_u_pol_fun, R_ff_fun, R_fi_fun, R_if_fun, R_ii_fun, 
R_uf_fun, R_ui_fun, w_ff_pol_fun, w_if_pol_fun, w_uf_pol_fun, θ_f_fun, informal_I_distribution_rand)
"""
function simulation_get_all_funs(V_f, V_i, V_u, θ_f, params)

    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    V_f, V_i, V_u, sector_f_pol, sector_i_pol, sector_u_pol, a_f_pol, a_i_pol, a_u_pol, w_ff_pol, w_if_pol, w_uf_pol,
    R_ff, R_fi, R_if, R_ii, R_uf, R_ui = get_policy(V_f, V_i, V_u, θ_f, params);

    informal_I_distribution_rand = Beta(distribution_a, distribution_b)

    a_f_pol_fun = linear_interpolation((z, w, a), a_f_pol)
    a_i_pol_fun = linear_interpolation((z, w, a), a_i_pol)
    a_u_pol_fun = linear_interpolation((z, w, a), a_u_pol)

    R_ff_fun = linear_interpolation((z, w, a), R_ff)
    R_fi_fun = linear_interpolation((z, w, a), R_fi)
    R_if_fun = linear_interpolation((z, w, a), R_if)
    R_ii_fun = linear_interpolation((z, w, a), R_ii)
    R_uf_fun = linear_interpolation((z, w, a), R_uf)
    R_ui_fun = linear_interpolation((z, w, a), R_ui)

    w_ff_pol_fun = linear_interpolation((z, w, a), w_ff_pol)
    w_if_pol_fun = linear_interpolation((z, w, a), w_if_pol)
    w_uf_pol_fun = linear_interpolation((z, w, a), w_uf_pol)

    θ_f_fun = linear_interpolation((z, w, a), θ_f)

    all_funs = (a_f_pol_fun, a_i_pol_fun, a_u_pol_fun, R_ff_fun, R_fi_fun, R_if_fun, R_ii_fun, 
    R_uf_fun, R_ui_fun, w_ff_pol_fun, w_if_pol_fun, w_uf_pol_fun, θ_f_fun, informal_I_distribution_rand)

    return all_funs
end

"""
Auxiliary function.
This functions does all the simulation to get the population from t to t+1
INPUT: 
    vectors to be modified: z_ind, w_ind, a_ind, status_ind, m_unemp
    params
    all_funs, a tuple containing all interpolated policy functions
    N_ind, number of individuals
OUTPUT:
    modification of z_ind, w_ind, a_ind, status_ind, m_unemp
"""
function simulation_transition_t!(z_ind, w_ind, a_ind, status_ind, m_unemp, params, all_funs, N_ind)

    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    a_f_pol_fun, a_i_pol_fun, a_u_pol_fun, R_ff_fun, R_fi_fun, R_if_fun, R_ii_fun, 
    R_uf_fun, R_ui_fun, w_ff_pol_fun, w_if_pol_fun, w_uf_pol_fun, θ_f_fun, informal_I_distribution_rand = all_funs
    
    ##### This part is identical to the main simulate_model code
    Threads.@threads for j = 1:N_ind
        if status_ind[j] == 1

            #first does savings decision
            a_ind[j] = a_f_pol_fun(z_ind[j], w_ind[j], a_ind[j]) 

            if (R_ff_fun(z_ind[j], w_ind[j], a_ind[j]) - R_fi_fun(z_ind[j], w_ind[j], a_ind[j]) >= 0) #searches in formal
                w_ff_ind = w_ff_pol_fun(z_ind[j], w_ind[j], a_ind[j])
                p_finding = p(θ_f_fun(z_ind[j], w_ff_ind, a_ind[j]))
                if rand(Binomial(1, λ_f * p_finding)) == 1 #finds desired job
                    w_ind[j] = w_ff_ind
                elseif rand(Binomial(1, δ_f)) == 1 #goes to unemployment
                    status_ind[j] = 3
                    w_ind[j] = rho*w_ind[j]
                end #else wage and status stays the same
            else #tries to enter informality
                if rand(Binomial(1, λ_f * prob_i)) == 1
                    status_ind[j] = 2
                    w_ind[j] = rand(informal_I_distribution_rand)
                elseif rand(Binomial(1, δ_f)) == 1 #goes to unemployment
                    status_ind[j] = 3
                    w_ind[j] = rho*w_ind[j]
                end #else wage and status stays the same
            end

        elseif status_ind[j] == 2 #informal

            a_ind[j] = a_i_pol_fun(z_ind[j], w_ind[j], a_ind[j])

            if (R_if_fun(z_ind[j], w_ind[j], a_ind[j]) - R_ii_fun(z_ind[j], w_ind[j], a_ind[j]) >= 0) #tries to enter formality
                w_if_ind = w_if_pol_fun(z_ind[j], w_ind[j], a_ind[j])
                p_finding = p(θ_f_fun(z_ind[j], w_if_ind, a_ind[j]))
                if rand(Binomial(1, λ_i * p_finding)) == 1 #finds desired job
                    status_ind[j] = 1
                    w_ind[j] = w_if_ind
                elseif rand(Binomial(1, δ_i)) == 1 #goes to unemployment
                    status_ind[j] = 3
                    w_ind[j] = 0
                else
                    w_ind[j] = rand(informal_I_distribution_rand) #remains in informality
                end
            else
                if rand(Binomial(1, λ_i * prob_i)) == 1
                    w_ind[j] = rand(informal_I_distribution_rand) #remains in informality
                elseif rand(Binomial(1, δ_i)) == 1 #goes to unemployment
                    status_ind[j] = 3
                    w_ind[j] = 0
                else #stays the same
                    w_ind[j] = rand(informal_I_distribution_rand) #remains in informality
                end
                
            end

        elseif status_ind[j] == 3 #unemployed

            a_ind[j] = a_u_pol_fun(z_ind[j], w_ind[j], a_ind[j])

            if (R_uf_fun(z_ind[j], w_ind[j], a_ind[j]) - R_ui_fun(z_ind[j], w_ind[j], a_ind[j]) >= 0) #searches in formal
                w_uf_ind = w_uf_pol_fun(z_ind[j], w_ind[j], a_ind[j])
                p_finding = p(θ_f_fun(z_ind[j], w_uf_ind, a_ind[j]))
                if rand(Binomial(1, λ_u * p_finding)) == 1 #finds desired job
                    status_ind[j] = 1
                    w_ind[j] = w_uf_ind
                    m_unemp[j] = 0
                elseif rand(Binomial(1, xi)) == 1 #insurance expires
                    w_ind[j] = 0
                    m_unemp[j] += 1 #counts another month of unemployment
                else
                    #else insurance and status stays the same
                    m_unemp[j] += 1 #counts another month of unemployment
                end #else insurance and status stays the same

            else #tries to enter informality
                if rand(Binomial(1, λ_u * prob_i)) == 1
                    status_ind[j] = 2
                    w_ind[j] = rand(informal_I_distribution_rand)
                    m_unemp[j] = 0
                elseif rand(Binomial(1, xi)) == 1 #insurance expires
                    w_ind[j] = 0
                    m_unemp[j] += 1 #counts another month of unemployment
                else
                    #else insurance and status stays the same
                    m_unemp[j] += 1 #counts another month of unemployment
                end 
            end
        end
    end

    return nothing

end

"""
Auxiliary function.
This function calculates consumption in the next period given current parameters.
Remember that consumption happens first in the model, however, we would like to
know how consumption will behave given current labor market outcomes.
This function accomplishes this.
"""
function simulation_consumption(z_ind, w_ind, a_ind, status_ind, params, all_funs, N_ind)
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    a_f_pol_fun, a_i_pol_fun, a_u_pol_fun, R_ff_fun, R_fi_fun, R_if_fun, R_ii_fun, 
    R_uf_fun, R_ui_fun, w_ff_pol_fun, w_if_pol_fun, w_uf_pol_fun, θ_f_fun, informal_I_distribution_rand = all_funs

    c_ind = zeros(N_ind)

    Threads.@threads for j = 1:N_ind
        if status_ind[j] == 1
            c_ind[j] = I_u + w_ind[j] + a_ind[j] - a_f_pol_fun(z_ind[j], w_ind[j], a_ind[j])/(1+r)
        elseif  status_ind[j] == 2
            c_ind[j] = I_u + w_ind[j] + a_ind[j] - a_i_pol_fun(z_ind[j], w_ind[j], a_ind[j])/(1+r)
        else
            c_ind[j] = I_u + w_ind[j] + a_ind[j] - a_u_pol_fun(z_ind[j], w_ind[j], a_ind[j])/(1+r)
        end
    end

    return c_ind

end


"""
Auxiliary function.
tracks selected time series to plot
    INPUT: 
        all the vectors of population (z_ind, w_ind, a_ind, status_ind, m_unemp)
        params
        tracked series per_f, ..., mean_income_u
        t (what index to modify)
"""
function simulation_track_series!(z_ind, w_ind, a_ind, c_ind, status_ind, m_unemp, params,
per_f, per_i, per_u, tax, Tw_within_f, Tw_within_i, Tw_between, Ta_within_f, Ta_within_i, Ta_between,
Tc_within_f, Tc_within_i, Tc_between, Tz_within_f, Tz_within_i, Tz_between,
mean_unemp, mean_income_f, mean_income_i, mean_income_u, mean_wealth_f, mean_wealth_i, mean_wealth_u, 
mean_consumption_f, mean_consumption_i, mean_consumption_u, mean_ability_f, mean_ability_i, mean_ability_u,
N_ind, t)

    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    ## Begins recording the tracked series

    per_f[t] = sum(status_ind.== 1)/N_ind
    per_i[t] = sum(status_ind.== 2)/N_ind
    per_u[t] = sum(status_ind.== 3)/N_ind
    tax[t] = 
        sum(τ*(y_f .+ z_ind[status_ind .== 1] - w_ind[status_ind .== 1]*(1+τ_w))) + 
        sum(τ_w*(w_ind[status_ind .== 1])) -
        sum(w_ind[status_ind .== 3]) - #Net of payments of unemployment insurance
        I_u * N_ind #Net of payment of unconditional transfers

    Tw_within_f[t], Tw_within_i[t], Tw_between[t] = 
        theil_subgroups(w_ind[status_ind .== 1] .+ I_u, w_ind[status_ind .== 2] .+ I_u)

    Ta_within_f[t], Ta_within_i[t], Ta_between[t] = 
        theil_subgroups(a_ind[status_ind .== 1] .- (a_min - 1.0e-4), a_ind[status_ind .== 2] .- (a_min - 1.0e-4))

    Tc_within_f[t], Tc_within_i[t], Tc_between[t] = 
        theil_subgroups(c_ind[status_ind .== 1], c_ind[status_ind .== 2])

    Tz_within_f[t], Tz_within_i[t], Tz_between[t] = 
        theil_subgroups(z_ind[status_ind .== 1] .- (z_min - 1.0e-4), z_ind[status_ind .== 2] .- (z_min - 1.0e-4))

    mean_unemp[t] = mean(m_unemp[status_ind.== 3])

    mean_income_f[t] = mean(w_ind[status_ind .== 1]) .+ I_u
    mean_income_i[t] = mean(w_ind[status_ind .== 2]) .+ I_u
    mean_income_u[t] = mean(w_ind[status_ind .== 3]) .+ I_u

    mean_wealth_f[t] = mean(a_ind[status_ind .== 1])
    mean_wealth_i[t] = mean(a_ind[status_ind .== 2])
    mean_wealth_u[t] = mean(a_ind[status_ind .== 3])

    mean_consumption_f[t] = mean(c_ind[status_ind .== 1])
    mean_consumption_i[t] = mean(c_ind[status_ind .== 2])
    mean_consumption_u[t] = mean(c_ind[status_ind .== 3])

    mean_ability_f[t] = mean(z_ind[status_ind .== 1])
    mean_ability_i[t] = mean(z_ind[status_ind .== 2])
    mean_ability_u[t] = mean(z_ind[status_ind .== 3])


    ## Ends recording the tracked series

    return nothing

end

"""
This function calculates the welfare that each individual would get under the policy vs under baseline
"""
function simulation_policy_welfare(z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0, 
V_f_0, V_i_0, V_u_0, θ_f_0, V_f_1, V_i_1, V_u_1, θ_f_1, params_0, params_1, N_ind, T)

    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params_0)

    z_ind_baseline = copy(z_ind_0)
    w_ind_baseline = copy(w_ind_0)
    a_ind_baseline = copy(a_ind_0)
    c_ind_baseline = copy(c_ind_0)
    status_ind_baseline = copy(status_ind_0)
    m_unemp_baseline = copy(m_unemp_0)

    z_ind_policy = copy(z_ind_0)
    w_ind_policy = copy(w_ind_0)
    a_ind_policy = copy(a_ind_0)
    status_ind_policy = copy(status_ind_0)
    m_unemp_policy = copy(m_unemp_0)

    EC_ind = zeros(N_ind)

    consumption_path_baseline = zeros(N_ind, T)
    consumption_path_policy = zeros(N_ind, T)

    ## functions for the baseline
    all_funs_0 = simulation_get_all_funs(V_f_0, V_i_0, V_u_0, θ_f_0, params_0)
    ## functions for the policy
    all_funs_1 = simulation_get_all_funs(V_f_1, V_i_1, V_u_1, θ_f_1, params_1)

    # Starts simulation for baseline
    println("Simulating...")
    for t = ProgressBar(1:T)

        #Baseline
        simulation_transition_t!(z_ind_baseline, w_ind_baseline, a_ind_baseline, status_ind_baseline, 
        m_unemp_baseline, params_0, all_funs_0, N_ind)

        consumption_path_baseline[:, t] = simulation_consumption(z_ind_baseline, w_ind_baseline, a_ind_baseline, 
        status_ind_baseline, params_0, all_funs_0, N_ind)

        #Policy
        simulation_transition_t!(z_ind_policy, w_ind_policy, a_ind_policy, status_ind_policy, 
        m_unemp_policy, params_1, all_funs_1, N_ind)

        consumption_path_policy[:, t] = simulation_consumption(z_ind_policy, w_ind_policy, a_ind_policy, status_ind_policy, 
        params_1, all_funs_1, N_ind)

    end

    println("finding individual Equivalent Consumptions...")
    Threads.@threads for j in ProgressBar(1:N_ind)
        EC_ind[j] = find_zero(
            EC -> sum(β .^ collect(0:T-1) .* (u.(consumption_path_policy[j, :]) - u.(consumption_path_baseline[j, :] * (1 + EC)))), #The formula to get individual Equivalent Consumption
            (-1, 4), #Search space
            atol = 1/100
        )
    end

    println("finding aggregate Equivalent Consumptions...")
    EC_aggregate = find_zero(
        #The formula to get aggregate Equivalent Consumption matrix dimentions (MxT)*T
        EC -> sum((u.(consumption_path_policy) - u.(consumption_path_baseline * (1 + EC)))*(β .^ collect(0:T-1))),
        (-1, 4), #Search space
        atol = 1/1000
    )

    EC_aggregate_formal = find_zero(
        EC -> sum((u.(consumption_path_policy[status_ind_0 .== 1, :]) - u.(consumption_path_baseline[status_ind_0 .== 1, :] * (1 + EC)))*(β .^ collect(0:T-1))), #The formula to get aggregate Equivalent Consumption
        (-1, 4), #Search space
        atol = 1/1000
    )

    EC_aggregate_informal = find_zero(
        EC -> sum((u.(consumption_path_policy[status_ind_0 .== 2, :]) - u.(consumption_path_baseline[status_ind_0 .== 2, :] * (1 + EC)))*(β .^ collect(0:T-1))), #The formula to get aggregate Equivalent Consumption
        (-1, 4), #Search space
        atol = 1/1000
    )


    ## The EC without transitions
    #=
    EC_no_transition = find_zero(
        EC -> sum(u.(consumption_path_policy[:, T]) - u.(consumption_path_baseline[:, T] * (1 + EC))), #The formula to get aggregate Equivalent Consumption
        (-1, 4), #Search space
        atol = 1/10000
    )

    EC_no_transition_formal = find_zero(
        EC -> sum(u.(consumption_path_policy[status_ind_0 .== 1, T]) - u.(consumption_path_baseline[status_ind_0 .== 1, T] * (1 + EC))), #The formula to get aggregate Equivalent Consumption
        (-1, 4), #Search space
        atol = 1/10000
    )

    EC_no_transition_informal = find_zero(
        EC -> sum(u.(consumption_path_policy[status_ind_0 .== 2, T]) - u.(consumption_path_baseline[status_ind_0 .== 2, T] * (1 + EC))), #The formula to get aggregate Equivalent Consumption
        (-1, 4), #Search space
        atol = 1/10000
    )
    =#

    return EC_ind, EC_aggregate, EC_aggregate_formal, EC_aggregate_informal

end

function simulation_baseline_graba(V_f, V_i, V_u, θ_f, N_ind, T, params, ngraba)

    ## Starts the vectors of population

    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

    V_f, V_i, V_u, sector_f_pol, sector_i_pol, sector_u_pol, a_f_pol, a_i_pol, a_u_pol, w_ff_pol, w_if_pol, w_uf_pol,
    R_ff, R_fi, R_if, R_ii, R_uf, R_ui = 
    get_policy(V_f, V_i, V_u, θ_f, params);

    ######### Empieza Graficas ######################

    #This is the custum color gradient. Basically algae but the very first is yellow,
    #for informal sector.
    my_colors_1 = deepcopy(ColorSchemes.algae)
    my_colors_1.colors[1] = RGB{Float64}(1, 1, 0)

    ## Sector choice + wage choice, unemployed without insurance

    sector_choice_u = R_uf - R_ui #Which reward is greater determines where the agent goes
    sector_choice_u_fun = linear_interpolation((z, w, a), sector_choice_u)
    w_uf_pol_fun = linear_interpolation((z, w, a), w_uf_pol)
    p_θ_f_fun = linear_interpolation((z, w, a), p.(θ_f))
    a_heatmap = range(a_min, stop=a_max, length=1000)
    z_heatmap = range(z_min, stop=z_max, length=1000)
    sector_choice1_heatmap = zeros(1000, 1000)
    for i = 1:1000, j = 1:1000
        if sector_choice_u_fun(z_heatmap[i], 0.0, a_heatmap[j]) > 0.0
            sector_choice1_heatmap[i, j] = w_uf_pol_fun(z_heatmap[i], 0.0, a_heatmap[j])
        else
            sector_choice1_heatmap[i, j] = 0.0
        end
    end

    p1 = heatmap(z_heatmap, a_heatmap, sector_choice1_heatmap',
        c = cgrad(my_colors_1),
        #clim = (0, max(0.8, maximum(sector_choice1_heatmap))), #color limits
        clim = (0, 0.9),
        xlabel="ability (z)", ylabel="wealth (a)",
        title="a) Unemployed",
        titlefontsize = 13,
        guidefontsize = 10,
        colorbar_title = "wage searched",
        colorbar = false
    )
    annotate!([(0.01, 1, text("informal", 12, :left), :black), (0.30, 3, text("formal", 12, :left), :black)])

    ## informal sector + wage choice

    sector_choice_i = R_if - R_ii #Which reward is greater determines where the agent goes
    #sector_choice_i_smooth = smooths_wealth(sector_choice_i, params) #smooths in the wealth dimension for plot
    #sector_choice_i_fun = linear_interpolation((z, w, a), sector_choice_i_smooth)
    sector_choice_i_fun = linear_interpolation((z, w, a), sector_choice_i) #uncomment if you want to see it without smoothing
    w_if_pol_fun = linear_interpolation((z, w, a), w_if_pol)
    sector_choice_informal_heatmap = zeros(1000, 1000)
    for i = 1:1000, j = 1:1000
        if sector_choice_i_fun(z_heatmap[i], 0.0, a_heatmap[j]) > 0
            sector_choice_informal_heatmap[i, j] = 0.30#w_if_pol_fun(z_heatmap[i], 0.0, a_heatmap[j])
        else
            sector_choice_informal_heatmap[i, j] = 0.0
        end
    end
        
    p2 = heatmap(z_heatmap, a_heatmap, sector_choice_informal_heatmap',
        c=cgrad(my_colors_1),
        #clim = (0, min(0.8, maximum(sector_choice_informal_heatmap))), #color limits
        clim = (0, 0.9),
        xlabel="ability (z)", ylabel="wealth (a)",
        title="b) Informal worker",
        titlefontsize = 13,
        guidefontsize = 10,
        colorbar_title = "wage searched",
        colorbar = false
    )
    annotate!([(0.01, 1.0, text("informal", 12, :left), :black), (0.30, 3, text("formal", 12, :left), :black)])
    
    ## formal sector with low wage + wage choice
    sector_choice_f = R_ff - R_fi #Which reward is greater determines where the agent goes
    sector_choice_f_fun = linear_interpolation((z, w, a), sector_choice_f)
    w_ff_pol_fun = linear_interpolation((z, w, a), w_ff_pol)
    sector_choice_formal_heatmap = zeros(1000, 1000)
    for i = 1:1000, j = 1:1000
        if sector_choice_f_fun(z_heatmap[i], 0.15, a_heatmap[j]) > 0
            sector_choice_formal_heatmap[i, j] = w_ff_pol_fun(z_heatmap[i], 0.15, a_heatmap[j])
        else
            sector_choice_formal_heatmap[i, j] = 0.0
        end
    end

    p3 = heatmap(z_heatmap, a_heatmap, sector_choice_formal_heatmap',
        c=cgrad(my_colors_1),
        #clim = (0, max(0.8, maximum(sector_choice_formal_heatmap))), #color limits
        clim = (0, 0.9),
        xlabel="ability (z)", ylabel="wealth (a)",
        title="c) Formal worker, low wage",
        titlefontsize = 13,
        guidefontsize = 10,
        colorbar_title = "wage searched",
        colorbar = false
    )
    annotate!([(0.01, 1, text("informal", 12, :left), :black), (0.30, 3, text("formal", 12, :left), :black)])

    ## formal sector on the job search
    sector_choice_f = R_ff - R_fi #Which reward is greater determines where the agent goes
    sector_choice_f_fun = linear_interpolation((z, w, a), sector_choice_f)
    w_ff_pol_fun = linear_interpolation((z, w, a), w_ff_pol)
    sector_choice_formal2_heatmap = zeros(1000, 1000)
    reference_w = 0.505 #median wage, check in simulation
    for i = 1:1000, j = 1:1000
        if sector_choice_f_fun(z_heatmap[i], reference_w, a_heatmap[j]) > 0
            sector_choice_formal2_heatmap[i, j] = w_ff_pol_fun(z_heatmap[i], reference_w, a_heatmap[j])
        else
            sector_choice_formal2_heatmap[i, j] = 0.0
        end
    end

    p4 = heatmap(z_heatmap, a_heatmap, sector_choice_formal2_heatmap',
        #c = cgrad(my_colors_1),
        #clim = (0, 0.9),
        xlabel="ability (z)", ylabel="wealth (a)",
        title="d) Formal worker, median wage",
        titlefontsize = 13,
        guidefontsize = 10,
        colorbar_title = "wage searched",
        colorbar = false
    )

    p = plot(p1, p2, p3, p4, layout = (2, 2), link = :both,
    size = (750, 750))

    display(p)

    z_ind = rand(Beta(distribution_a, distribution_b), N_ind)
    #w_ind = rand(Uniform(w_min, w_max), N_ind)
    w_ind = zeros(N_ind)
    a_ind = zeros(N_ind)
    status_ind = rand([1, 2, 3], N_ind) #1 formal, 2 informal, 3 unemployed
    m_unemp = zeros(N_ind) #How many months unemployed. 0 if not unemployed

    ## Gets the policy functions

    all_funs = simulation_get_all_funs(V_f, V_i, V_u, θ_f, params)

    # Starts simulation
    println("Simulating...")
    for t = ProgressBar(1:T)
        simulation_transition_t!(z_ind, w_ind, a_ind, status_ind, m_unemp, params, all_funs, N_ind)    
    end


    per_form_graba = zeros(ngraba)
    per_u_graba = zeros(ngraba)
    for t = 1:ngraba
        simulation_transition_t!(z_ind, w_ind, a_ind, status_ind, m_unemp, params, all_funs, N_ind)

        per_f = sum(status_ind .== 1)/N_ind*100 #formal percentage
        per_i = sum(status_ind .== 2)/N_ind*100 #informal percentage

        per_informality = per_i ./(per_f + per_i)*100 #percentage of informality in the economy
        per_u = sum(status_ind .== 3)/N_ind*100 #unemployment percentage

        per_form_graba[t] = per_informality
        per_u_graba[t] = per_u
    end

    mean_per_formality = mean(per_form_graba)
    mean_per_u = mean(per_u_graba)

    return mean_per_formality, mean_per_u
end