#=
Author: Emiliano Rojas Eng
This is the code for the paper 
"Wage and Wealth Inequality in a High-Informality Economy: A Directed Search Model" 2022
Last Update: July 19, 2023
=#

using JLD2, Interpolations, Plots, Optim, StatsBase, QuadGK, Distributions, 
StatsPlots, ProgressBars, Roots, DataFrames, GLM, CategoricalArrays, RegressionTables,
Images, ColorSchemes, Measures, PrettyTables

include("methods_ERE-TII.jl")

"This is just a wrapper to print all plots"
function wrapper_main(policy_name)

####### Baseline Model parameters ########
@load "scenarios/baseline_ERE-TII.jld2"

J_f_0 = copy(J_f)
V_f_0 = copy(V_f)
V_i_0 = copy(V_i)
V_u_0 = copy(V_u)
θ_f_0 = copy(θ_f)
params_0 = copy(params)

if policy_name == "policy_tw_ERE-TII"
    @load "scenarios/policy_tw_ERE-TII.jld2"
    image_table = Images.load("EC_tables/$(policy_name)_table.png")
    image_table2 = Images.load("EC_tables/$(policy_name)_table2.png")
    policy_title = "Fiscal Incentive: Reduce payroll tax \$\\tau_w\$"

elseif policy_name == "policy_tau_ERE-TII"
    @load "scenarios/policy_tau_ERE-TII.jld2"
    image_table = Images.load("EC_tables/$(policy_name)_table.png")
    image_table2 = Images.load("EC_tables/$(policy_name)_table2.png")
    policy_title = "Fiscal Incentive: Reduce profit tax \$\\tau\$"

elseif policy_name == "policy_probi_ERE-TII"
    @load "scenarios/policy_probi_ERE-TII.jld2"
    image_table = Images.load("EC_tables/$(policy_name)_table.png")
    image_table2 = Images.load("EC_tables/$(policy_name)_table2.png")
    policy_title = "Punitive: Decrease probability of entering informality \$P_{inf}\$"

elseif policy_name == "policy_kf_ERE-TII"
    @load "scenarios/policy_kf_ERE-TII.jld2"
    image_table = Images.load("EC_tables/$(policy_name)_table.png")
    image_table2 = Images.load("EC_tables/$(policy_name)_table2.png")
    policy_title = "Barriers to Entry: Reduce fixed cost of \n entering formal market \$k_f\$"

elseif policy_name == "policy_insurance_ERE-TII"
    @load "scenarios/policy_insurance_ERE-TII.jld2"
    image_table = Images.load("EC_tables/$(policy_name)_table.png")
    image_table2 = Images.load("EC_tables/$(policy_name)_table2.png")
    policy_title = "Social Security: Unemployment Insurance"

elseif policy_name == "policy_Iu_ERE-TII"
    @load "scenarios/policy_Iu_ERE-TII.jld2"
    image_table = Images.load("EC_tables/$(policy_name)_table.png")
    image_table2 = Images.load("EC_tables/$(policy_name)_table2.png")
    policy_title = "Social Security: Universal Basic Income \$I_u\$"

else
    println("INVALID NAME")
    return nothing
end

J_f_1 = copy(J_f)
V_f_1 = copy(V_f)
V_i_1 = copy(V_i)
V_u_1 = copy(V_u)
θ_f_1 = copy(θ_f)
params_1 = copy(params)

N_ind = 100000
T = 1200
z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0 = simulation_baseline(V_f_0, V_i_0, V_u_0, θ_f_0, N_ind, T, params_0)

T_policy = 120
padding = 24

z_ind, w_ind, a_ind, status_ind, m_unemp, per_f, per_i, per_u, tax, mean_unemp,
Tw_within_f, Tw_within_i, Tw_between, Ta_within_f, Ta_within_i, Ta_between, 
Tc_within_f, Tc_within_i, Tc_between, Tz_within_f, Tz_within_i, Tz_between,
mean_income_f, mean_income_i, mean_income_u, mean_wealth_f, mean_wealth_i, mean_wealth_u, 
mean_consumption_f, mean_consumption_i, mean_consumption_u, mean_ability_f, mean_ability_i, mean_ability_u =
simulation_policy(z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0, 
V_f_0, V_i_0, V_u_0, θ_f_0, V_f_1, V_i_1, V_u_1, θ_f_1, params_0, params_1, N_ind, T_policy, padding)

############# Begin Plots for Scenario ##################
#########################################################

p1 = groupedbar([[per_u[1], per_u[end]] [per_i[1], per_i[end]] [per_f[1], per_f[end]]].*100, 
bar_position = :stack,
xticks = (1:2, ["before", "after"]),
color = [:purple :orange :green],
legend = :outerbottom,
label=["unemployed" "informal" "formal"],
title = "Formal, informal, and unemployed (%)",
subtitle = "as % of population"
)
hline!([0], color = :black, label = "")

#Uncomment to get the number of formal informal and unemployed inside barplot
annotate!([
    (1, (0+per_f[1]/2)*100, text(string(round(per_f[1]*100, digits = 1))*"%", 10)),
    (2, (0+per_f[end]/2)*100, text(string(round(per_f[end]*100, digits = 1))*"%", 10)),
    (1, (per_f[1]+per_i[1]/2)*100, text(string(round(per_i[1]*100, digits = 1))*"%", 10)),
    (2, (per_f[end]+per_i[end]/2)*100, text(string(round(per_i[end]*100, digits = 1))*"%", 10)),
    (1, (per_f[1]+per_i[1] + per_u[1]/2)*100, text(string(round(per_u[1]*100, digits = 1))*"%", 10, :white)),
    (2, (per_f[end]+per_i[end] + per_u[end]/2)*100, text(string(round(per_u[end]*100, digits = 1))*"%", 10, :white)),
])


#mean tightness plot
    r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
    prob_i, τ, τ_w, I_u, distribution_a, distribution_b,
    informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
    w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params_0)
tightness_0 = mean(p.(θ_f_0), dims = [1, 3])[2:end]*100
tightness_1 = mean(p.(θ_f_1), dims = [1, 3])[2:end]*100

p2 = plot(w[2:end], [tightness_0, tightness_1],
label = ["baseline" "policy"],
legend = :outerbottom,
color = [:black :red],
ylabel = "P(θ|w) in %",
ylim = (0.0, max(25, maximum(tightness_1))),
xlabel = "w\n",
guidefontsize = 9,
title = "Probability of getting a formal job\nby wage searched: P(θ|w)",
)

c_ind = simulation_consumption(z_ind, w_ind, a_ind, status_ind, params, simulation_get_all_funs(V_f, V_i, V_u, θ_f, params), N_ind)

#=
change_all= [(mean(w_ind)-mean(w_ind_0)),
(mean(a_ind)-mean(a_ind_0)),
(mean(c_ind)-mean(c_ind_0))]

change_formal = [(mean(w_ind[status_ind_0 .==1])-mean(w_ind_0[status_ind_0 .==1])),
(mean(a_ind[status_ind_0 .==1])-mean(a_ind_0[status_ind_0 .==1])),
(mean(c_ind[status_ind_0 .==1])-mean(c_ind_0[status_ind_0 .==1]))]

change_informal = [(mean(w_ind[status_ind_0 .== 2])-mean(w_ind_0[status_ind_0 .== 2])),
(mean(a_ind[status_ind_0 .== 2])-mean(a_ind_0[status_ind_0 .== 2])),
(mean(c_ind[status_ind_0 .== 2])-mean(c_ind_0[status_ind_0 .== 2]))]
=#

a_min = params_0["a_min"]

w_ind .+= params_1["I_u"]
w_ind_0 .+= params_0["I_u"]

agents_always_formal = (status_ind_0 .== 1) .& (status_ind .== 1)
agents_always_informal = (status_ind_0 .== 2) .& (status_ind .== 2)
agents_informal_to_formal = (status_ind_0 .== 2) .& (status_ind .== 1)

change_all= [(mean(w_ind)/mean(w_ind_0)),
(mean(a_ind .- a_min)/mean(a_ind_0 .- a_min)),
(mean(c_ind)/mean(c_ind_0))].*100 .- 100

change_always_formal = [(mean(w_ind[agents_always_formal])/mean(w_ind_0[agents_always_formal])),
(mean(a_ind[agents_always_formal] .- a_min)/mean(a_ind_0[agents_always_formal] .- a_min)),
(mean(c_ind[agents_always_formal])/mean(c_ind_0[agents_always_formal]))].*100 .- 100

change_always_informal = [(mean(w_ind[agents_always_informal])/mean(w_ind_0[agents_always_informal])),
(mean(a_ind[agents_always_informal] .- a_min)/mean(a_ind_0[agents_always_informal] .- a_min)),
(mean(c_ind[agents_always_informal])/mean(c_ind_0[agents_always_informal]))].*100 .- 100

change_informal_to_formal = [(mean(w_ind[agents_informal_to_formal])/mean(w_ind_0[agents_informal_to_formal])),
(mean(a_ind[agents_informal_to_formal] .- a_min)/mean(a_ind_0[agents_informal_to_formal] .- a_min)),
(mean(c_ind[agents_informal_to_formal])/mean(c_ind_0[agents_informal_to_formal]))].*100 .- 100

changes_matrix = [change_all change_always_formal change_always_informal change_informal_to_formal]

p3 = groupedbar(changes_matrix,
    ylim = (min(-5.0, minimum(changes_matrix)*1.05), max(40.0, maximum(changes_matrix)*1.05)),
    label=["all" "always formal" "always informal" "transitions from informal to formal"],
    legend = :outerbottom,
    xticks = (1:3, ["income", "wealth", "consumption"]),
    color = [:blue :green :orange :red],
    title = "Income, wealth and consumption,\n % change from baseline"
)
hline!([0], color = :black, label = "", linestyle = :dash)

#Changes in Theil inequality index
T_betweens = [Tw_between[end] - Tw_between[1], Ta_between[end] - Ta_between[1], 
Tc_between[end] - Tc_between[1]]
T_withins_f = [Tw_within_f[end] - Tw_within_f[1], Ta_within_f[end] - Ta_within_f[1], 
Tc_within_f[end] - Tc_within_f[1]]
T_withins_i = [Tw_within_i[end] - Tw_within_i[1], Ta_within_i[end] - Ta_within_i[1],
Tc_within_i[end] - Tc_within_i[1]]

Theil_matrix = [T_withins_i T_withins_f T_betweens]

#return Theil_matrix
plot_y_max = maximum(sum(Theil_matrix.*(Theil_matrix .> 0), dims = 2))*1.1
plot_y_min = minimum(sum(Theil_matrix.*(Theil_matrix .< 0), dims = 2))*1.1

p4 = groupedbar(Theil_matrix,
    ylim = (min(-0.05, plot_y_min), max(0.05, plot_y_max)),
    bar_position = :stack,
    label=["inequality within informal" "inequality within formal" "inequality between formal and informal"],
    legend = :outerbottom,
    xticks = (1:3, ["income", "wealth", "consumption"]),
    color = [:orange :green :red],
    title = "Change in inequality,\n theil index"
)
hline!([0], color = :black, label = "", linestyle = :dash)
scatter!(1:3, T_withins_i + T_withins_f + T_betweens,
markershape = :diamond,
color = :purple1,
markersize = 5,
label = "total")

#Plot 5 is a table with the results of EC (table is formatted manually in Excel)
p5 = plot(image_table,
border = :none,
title = "Welfare: Equivalent Consumptions")

#Plot 6 is a table with more results (From bootstrap_outer)
p6 = plot(image_table2,
border = :none,
title = "Additional statistics")

l = @layout [[a b]; [c d]; [e{0.16h} f{0.16h}]]
plot(p1, p2, p3, p4, p5, p6, layout = l,
titlefontsize = 11,
plot_title = policy_title,
size = (750, 1050),
left_margin = 5mm,
)
savefig("images/$(policy_name)_plot1.svg")

change_mean_unemp = mean_unemp[end] - mean(mean_unemp[1:padding])
change_tax = (tax[end]/mean(tax[1:padding]) - 1)*100
involuntary_inf = sum(z_ind[status_ind .== 2] .>= minimum(z_ind[status_ind .== 1]))/sum(status_ind .== 2)*100

df = DataFrame(
    A = ["Change in unemployment duration (# months)", "Change in tax revenue (net, %change)", "involuntary informality (as % of informal workers)"],
    B = round.([change_mean_unemp, change_tax, involuntary_inf], digits = 2)
)


println("Policy is:")
println(policy_name)
pretty_table(df, header = ["Concept", "Value"])

return nothing

end

function wrapper_EC(policy_name)
    ####### Baseline Model parameters ########
    @load "scenarios/baseline_ERE-TII.jld2"

    J_f_0 = copy(J_f)
    V_f_0 = copy(V_f)
    V_i_0 = copy(V_i)
    V_u_0 = copy(V_u)
    θ_f_0 = copy(θ_f)
    params_0 = copy(params)

    if policy_name == "policy_tw_ERE-TII"
        @load "scenarios/policy_tw_ERE-TII.jld2"

    elseif policy_name == "policy_tau_ERE-TII"
        @load "scenarios/policy_tau_ERE-TII.jld2"

    elseif policy_name == "policy_probi_ERE-TII"
        @load "scenarios/policy_probi_ERE-TII.jld2"

    elseif policy_name == "policy_kf_ERE-TII"
        @load "scenarios/policy_kf_ERE-TII.jld2"

    elseif policy_name == "policy_insurance_ERE-TII"
        @load "scenarios/policy_insurance_ERE-TII.jld2"

    elseif policy_name == "policy_Iu_ERE-TII"
        @load "scenarios/policy_Iu_ERE-TII.jld2"

    else
        println("INVALID NAME")
        return nothing
    end

    J_f_1 = copy(J_f)
    V_f_1 = copy(V_f)
    V_i_1 = copy(V_i)
    V_u_1 = copy(V_u)
    θ_f_1 = copy(θ_f)
    params_1 = copy(params)

    N_ind = 100000
    T = 1200
    z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0 = simulation_baseline(V_f_0, V_i_0, V_u_0, θ_f_0, N_ind, T, params_0)

    T_policy = 1200
    EC_ind, EC_aggregate, EC_aggregate_formal, EC_aggregate_informal = simulation_policy_welfare(z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0, 
    V_f_0, V_i_0, V_u_0, θ_f_0, V_f_1, V_i_1, V_u_1, θ_f_1, params_0, params_1, N_ind, T_policy)

    ##This is only to check that it calculates no EC when nothing changes
    #EC_ind, EC_aggregate, EC_no_transition = simulation_policy_welfare(z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0, 
    #V_f_0, V_i_0, V_u_0, θ_f_0, V_f_0, V_i_0, V_u_0, θ_f_0, params_0, params_0, N_ind, 1200)

    println("Policy is:")
    println(policy_name)
    println("EC_aggregate is:")
    println(EC_aggregate*100)
    println("EC_aggregate_formal is:")
    println(EC_aggregate_formal*100)
    println("EC_aggregate_informal is:")
    println(EC_aggregate_informal*100)

    status_ind_0_string = Vector{String}()
    for i =1:N_ind
        push!(status_ind_0_string, 
            status_ind_0[i] == 1 ? "formal" :
            status_ind_0[i] == 2 ? "informal" : 
                                "unemployed")
    end

    data = DataFrame((z_t0 = z_ind_0, w_t0 = w_ind_0, a_t0 =  a_ind_0, status_t0 = categorical(status_ind_0_string), EC_policy = EC_ind*100))
    ols_states = lm(@formula(EC_policy ~ z_t0 + w_t0 + a_t0), data)
    ols_status = lm(@formula(EC_policy ~ status_t0), data)
    ols_interactions = lm(@formula(EC_policy ~ status_t0*z_t0 + status_t0*w_t0 + status_t0*a_t0), data)

    #println("Table of policy $policy_name")
    #regtable(ols_status, ols_states, ols_interactions; renderSettings = latexOutput())
    #regtable(ols_interactions; renderSettings = latexOutput())

    return ols_states, ols_status
end

function bootstrap_outer(policy_name, num_bootstrap_samples)
    ####### Baseline Model parameters ########
    @load "scenarios/baseline_ERE-TII.jld2"

    J_f_0 = copy(J_f)
    V_f_0 = copy(V_f)
    V_i_0 = copy(V_i)
    V_u_0 = copy(V_u)
    θ_f_0 = copy(θ_f)
    params_0 = copy(params)

    if policy_name == "policy_tw_ERE-TII"
        @load "scenarios/policy_tw_ERE-TII.jld2"

    elseif policy_name == "policy_tau_ERE-TII"
        @load "scenarios/policy_tau_ERE-TII.jld2"

    elseif policy_name == "policy_probi_ERE-TII"
        @load "scenarios/policy_probi_ERE-TII.jld2"

    elseif policy_name == "policy_kf_ERE-TII"
        @load "scenarios/policy_kf_ERE-TII.jld2"

    elseif policy_name == "policy_insurance_ERE-TII"
        @load "scenarios/policy_insurance_ERE-TII.jld2"

    elseif policy_name == "policy_Iu_ERE-TII"
        @load "scenarios/policy_Iu_ERE-TII.jld2"

    else
        println("INVALID NAME")
        return nothing
    end

    J_f_1 = copy(J_f)
    V_f_1 = copy(V_f)
    V_i_1 = copy(V_i)
    V_u_1 = copy(V_u)
    θ_f_1 = copy(θ_f)
    params_1 = copy(params)

    N_ind = 100000
    T = 2400
    z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0 = simulation_baseline(V_f_0, V_i_0, V_u_0, θ_f_0, N_ind, T, params_0)

    function bootstrap_inner(policy_name)
        
        T_policy = 120
        padding = 24

        z_ind, w_ind, a_ind, status_ind, m_unemp, per_f, per_i, per_u, tax, mean_unemp,
        Tw_within_f, Tw_within_i, Tw_between, Ta_within_f, Ta_within_i, Ta_between, 
        Tc_within_f, Tc_within_i, Tc_between, Tz_within_f, Tz_within_i, Tz_between,
        mean_income_f, mean_income_i, mean_income_u, mean_wealth_f, mean_wealth_i, mean_wealth_u, 
        mean_consumption_f, mean_consumption_i, mean_consumption_u, mean_ability_f, mean_ability_i, mean_ability_u =
        simulation_policy(z_ind_0, w_ind_0, a_ind_0, c_ind_0, status_ind_0, m_unemp_0, 
        V_f_0, V_i_0, V_u_0, θ_f_0, V_f_1, V_i_1, V_u_1, θ_f_1, params_0, params_1, N_ind, T_policy, padding)

        change_mean_unemp = mean_unemp[end] - mean(mean_unemp[1:padding])
        change_tax = (tax[end]/mean(tax[1:padding]) - 1)*100
        involuntary_inf = sum(z_ind[status_ind .== 2] .>= minimum(z_ind[status_ind .== 1]))/sum(status_ind .== 2)*100

        return change_mean_unemp, change_tax, involuntary_inf
    end

    # Store the bootstrap sample means
    bootstrap_sample_means_1 = Float64[]
    bootstrap_sample_means_2 = Float64[]
    bootstrap_sample_means_3 = Float64[]

    # Generate bootstrap samples
    for _ in ProgressBar(1:num_bootstrap_samples)
        change_mean_unemp, change_tax, involuntary_inf = bootstrap_inner(policy_name) 
        push!(bootstrap_sample_means_1, change_mean_unemp)
        push!(bootstrap_sample_means_2, change_tax)
        push!(bootstrap_sample_means_3, involuntary_inf)
    end

    # Calculate mean and standard error of the bootstrap sample means
    bootstrap_mean_1 = mean(bootstrap_sample_means_1)
    bootstrap_mean_2 = mean(bootstrap_sample_means_2)
    bootstrap_mean_3 = mean(bootstrap_sample_means_3)

    bootstrap_standard_error_1 = std(bootstrap_sample_means_1)
    bootstrap_standard_error_2 = std(bootstrap_sample_means_2)
    bootstrap_standard_error_3 = std(bootstrap_sample_means_3)

    df = DataFrame(
        A = ["Change in unemployment duration (# months)", "Change in tax revenue (net, %change)", "involuntary informality (as % of informal workers)"],
        B = round.([bootstrap_mean_1, bootstrap_mean_2, bootstrap_mean_3], digits = 2),
        C = round.([bootstrap_standard_error_1, bootstrap_standard_error_2, bootstrap_standard_error_3], digits = 2)
    )

    println("Policy is:")
    println(policy_name)
    pretty_table(df, header = ["Concept", "Value", "Standard Deviation"])
    return df
end


wrapper_main("policy_tw_ERE-TII")
wrapper_main("policy_tau_ERE-TII")
wrapper_main("policy_probi_ERE-TII")
wrapper_main("policy_kf_ERE-TII")
wrapper_main("policy_insurance_ERE-TII")
wrapper_main("policy_Iu_ERE-TII")

#Uncomment to calculate all the EC and the regressions
#=
ols_states_1, ols_status_1 = wrapper_EC("policy_tw_ERE-TII")
ols_states_2, ols_status_2 = wrapper_EC("policy_tau_ERE-TII")
ols_states_3, ols_status_3 = wrapper_EC("policy_probi_ERE-TII")
ols_states_4, ols_status_4 = wrapper_EC("policy_kf_ERE-TII")
ols_states_5, ols_status_5 = wrapper_EC("policy_insurance_ERE-TII")
ols_states_6, ols_status_6 = wrapper_EC("policy_Iu_ERE-TII")

regtable(ols_states_1, ols_states_2, ols_states_3, ols_states_4,
ols_states_5, ols_states_6; 
renderSettings = latexOutput())
=#

#=
num_bootstrap_samples = 100
bootstrap_outer("policy_tw_ERE-TII", num_bootstrap_samples)
bootstrap_outer("policy_tau_ERE-TII", num_bootstrap_samples)
bootstrap_outer("policy_probi_ERE-TII", num_bootstrap_samples)
bootstrap_outer("policy_kf_ERE-TII", num_bootstrap_samples)
bootstrap_outer("policy_insurance_ERE-TII", num_bootstrap_samples)
bootstrap_outer("policy_Iu_ERE-TII", num_bootstrap_samples)
=#