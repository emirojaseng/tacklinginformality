#=
Author: Emiliano Rojas Eng
This is the code for the paper 
"Tackling Informality and Inequality: A Directed Search Model" 2022
Last Update: July 24, 2023
=#

using JLD2, Interpolations, Plots, Optim, StatsBase, QuadGK, Distributions, 
StatsPlots, ProgressBars, Roots, DataFrames, GLM, CategoricalArrays, RegressionTables,
Images, ColorSchemes, Measures
include("methods_ERE-TII.jl")

@load "scenarios/baseline_ERE-TII.jld2"

saveimages = false #true if you want to save images

#=
####### Baseline Model parameters ########
params2 = Dict([
        ("r", 1.04^(1/12)-1), 
        ("β", 0.99),     
        ("σ", 2.0),           
        ("𝛾", 0.65),          
        ("δ_f", 0.0112),        
        ("δ_i", 0.0112 * 2.73),  
        ("xi", 1.00),         
        ("rho", 0.0),          #E
        ("λ_f", 0.2824),        #E
        ("λ_i", 0.1942),        #E
        ("λ_u", 0.60),          #E
        ("k_f", 0.44),         #E
        ("y_f", 0.30),          #E
        ("prob_i", 0.85),
        ("τ", 0.30),          
        ("τ_z", 0.64),         #E
        ("τ_w", 0.196),       
        ("I_u", 0.01),      
        ("distribution_a", 0.7), #E
        ("distribution_b", 1.7), #E
        ("z_min", 0.0),     
        ("z_max", 1.0),     
        ("w_min", 0.0),      
        ("w_max", 1.0),         
        ("a_min", -1.756),       #E
        ("a_max", 16),        
        ("N_z", 18),             
        ("N_w", 35),            
        ("N_a", 35)             
])

#initial guess for grid values

params["N_z"] = 18
params["N_w"] = 35
params["N_a"] = 35
J_f, V_f, V_i, V_u, θ_f = resize_all(J_f, V_f, V_i, V_u, θ_f, params)

tol = 1.0e-1 # tolerance for VFI convergence
J_f, V_f, V_i, V_u, θ_f, error_θ = get_all(J_f, V_f, V_i, V_u, θ_f, params);
#jldsave("baseline_ERE-TII.jld2"; J_f, V_f, V_i, V_u, θ_f, params)
=#

######### Assumes model is already selected #############

r, β, σ, 𝛾, δ_f, δ_i, xi, rho, λ_f, λ_i, λ_u, k_f, y_f, 
prob_i, τ, τ_z, τ_w, I_u, distribution_a, distribution_b, 
informal_I_distribution, u, p, q, q_inv, z_min, z_max, w_min, 
w_max, a_min, a_max, z, w, a, grid_index, N_z, N_w, N_a = unpack_params(params)

V_f, V_i, V_u, sector_f_pol, sector_i_pol, sector_u_pol, a_f_pol, a_i_pol, a_u_pol, w_ff_pol, w_if_pol, w_uf_pol,
R_ff, R_fi, R_if, R_ii, R_uf, R_ui = 
get_policy(V_f, V_i, V_u, θ_f, params);

######### Starts plots ######################

#This is a custom color gradient. Basically algae but the very first is yellow,
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
sector_choice_i_fun = linear_interpolation((z, w, a), sector_choice_i) #uncomment if you want to see it without smoothing
w_if_pol_fun = linear_interpolation((z, w, a), w_if_pol)
sector_choice_informal_heatmap = zeros(1000, 1000)
for i = 1:1000, j = 1:1000
    if sector_choice_i_fun(z_heatmap[i], 0.0, a_heatmap[j]) > 0
        sector_choice_informal_heatmap[i, j] = w_if_pol_fun(z_heatmap[i], 0.0, a_heatmap[j])
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
reference_w = 0.52 #median wage, check in simulation
for i = 1:1000, j = 1:1000
    if sector_choice_f_fun(z_heatmap[i], reference_w, a_heatmap[j]) > 0
        sector_choice_formal2_heatmap[i, j] = w_ff_pol_fun(z_heatmap[i], reference_w, a_heatmap[j])
    else
        sector_choice_formal2_heatmap[i, j] = 0.0
    end
end

p4 = heatmap(z_heatmap, a_heatmap, sector_choice_formal2_heatmap',
    c = cgrad(my_colors_1),
    clim = (0, 0.9),
    xlabel="ability (z)", ylabel="wealth (a)",
    title="d) Formal worker, median wage",
    titlefontsize = 13,
    guidefontsize = 10,
    colorbar_title = "wage searched",
    colorbar = false
)

plot(p1, p2, p3, p4, layout = (2, 2), link = :both,
size = (750, 750))
saveimages ? savefig("images/labor_choices.svg") : 0

#################### J and θ ###########################

#algae but white if 0, to represent no formal market.
#my_colors_2 = deepcopy(ColorSchemes.algae)
#my_colors_2.colors[1] = RGB{Float64}(1, 1, 1)

my_colors_2 = deepcopy(ColorSchemes.inferno)
#my_colors_2.colors[1] = RGB{Float64}(1, 1, 1)

## J_f (z, w), highest a

J_f_fun = linear_interpolation((z, w, a), J_f)
z_heatmap = range(z_min, stop=z_max, length=1000)
w_heatmap = range(w_min, stop=w_max, length=1000)
J_zw1_heatmap = zeros(1000, 1000)
for i = 1:1000, j = 1:1000
    if sector_choice_f_fun(z_heatmap[i], w_heatmap[j], a_max) > 0
        J_zw1_heatmap[i, j] = J_f_fun(z_heatmap[i], w_heatmap[j], a_max)
    else
        #same as above, this condition is only here to help verify
        #Where the "exit to informal sector" happens
        J_zw1_heatmap[i, j] = J_f_fun(z_heatmap[i], w_heatmap[j], a_max)
    end
end
    
p5 = heatmap(z_heatmap, w_heatmap, J_zw1_heatmap',
    c=cgrad([:red, :white, :green, :darkgreen], [0.745, 0.90]),
    clim = (-30, 10),
    xlabel="ability (z)", ylabel="wage (w)",
    title="a) Firm value\nsubmarket (z, w), highest wealth",
    titlefontsize = 12,
    colorbar_title = "Firm value"
)
annotate!([
 (0.02, 0.95, text("Cost is too high to operate", 11, :left), :black),
 (0.02, 0.04, text("worker exits to the informal sector", 11, :left), :black),
 (0.30, 0.35, text("worker remains formal", 11, :left), :black)
])

## p(θ_f) (z, w), highest a

θ_f_fun = linear_interpolation((z, w, a), θ_f)
z_heatmap = range(z_min, stop=z_max, length=1000)
w_heatmap = range(w_min, stop=w_max, length=1000)
θ_zw1_heatmap = zeros(1000, 1000)
for i = 1:1000, j = 1:1000
    if sector_choice_f_fun(z_heatmap[i], w_heatmap[j], a_max) > 0
        θ_zw1_heatmap[i, j] = p.(θ_f_fun(z_heatmap[i], w_heatmap[j], a_max))
    else
        #same as above, this condition is only here to help verify
        #Where the "exit to informal sector" happens
        θ_zw1_heatmap[i, j] = p.(θ_f_fun(z_heatmap[i], w_heatmap[j], a_max))
    end
end
    
p6 = heatmap(z_heatmap, w_heatmap, θ_zw1_heatmap',
    c=cgrad(my_colors_2),
    clim = (0, 0.70),
    xlabel="ability (z)", ylabel="wage (w)",
    title="b) Labor market tightness P(θ)\nsubmarket (z, w), highest wealth",
    titlefontsize = 12,
    colorbar_title = "\nprobability of finding job",
    right_margin = 6Plots.mm
)

## J_f (z, w), lowest a

J_f_fun = linear_interpolation((z, w, a), J_f)
z_heatmap = range(z_min, stop=z_max, length=1000)
w_heatmap = range(w_min, stop=w_max, length=1000)
J_zw2_heatmap = zeros(1000, 1000)
for i = 1:1000, j = 1:1000
    if sector_choice_f_fun(z_heatmap[i], w_heatmap[j], a_min) > 0
        J_zw2_heatmap[i, j] = J_f_fun(z_heatmap[i], w_heatmap[j], a_min)
    else
        #same as above, this condition is only here to help verify
        #Where the "exit to informal sector" happens
        J_zw2_heatmap[i, j] = J_f_fun(z_heatmap[i], w_heatmap[j], a_min)
    end
end
    
p7= heatmap(z_heatmap, w_heatmap, J_zw2_heatmap',
    c=cgrad([:red, :white, :green, :darkgreen], [0.745, 0.90]),
    clim = (-30, 10),
    xlabel="ability (z)", ylabel="wage (w)",
    title="c) Firm value\nsubmarket (z, w), lowest wealth",
    titlefontsize = 12,
    colorbar_title = "Firm value"
)
annotate!([
 (0.03, 0.95, text("Cost is too high to operate", 11, :left), :black),
 (0.02, 0.04, text("worker exits to the informal sector", 11, :left), :black),
 (0.30, 0.25, text("worker remains formal", 11, :left), :black)
])

## p(θ_f) (z, w), lowest a

θ_f_fun = linear_interpolation((z, w, a), θ_f)
z_heatmap = range(z_min, stop=z_max, length=1000)
w_heatmap = range(w_min, stop=w_max, length=1000)
θ_zw_heatmap = zeros(1000, 1000)
for i = 1:1000, j = 1:1000
    if sector_choice_f_fun(z_heatmap[i], w_heatmap[j], a_min) > 0
        θ_zw_heatmap[i, j] = p.(θ_f_fun(z_heatmap[i], w_heatmap[j], a_min))
    else
        #same as above, this condition is only here to help verify
        #Where the "exit to informal sector" happens
        θ_zw_heatmap[i, j] = p.(θ_f_fun(z_heatmap[i], w_heatmap[j], a_min))
    end
end
    
p8 = heatmap(z_heatmap, w_heatmap, θ_zw_heatmap',
    c=cgrad(my_colors_2),
    clim = (0, 0.70),
    xlabel="ability (z)", ylabel="wage (w)",
    title="d) Labor market tightness P(θ)\nsubmarket (z, w), lowest wealth",
    titlefontsize = 12,
    colorbar_title = "\nprobability of finding job",
    right_margin = 6Plots.mm
)
#saveimages ? savefig("images/P_value_2.svg") : 0

plot(p5, p6, p7, p8, layout = (2, 2), link = :both,
size = (1000, 800))

saveimages ? savefig("images/firm_equilibrium.svg") : 0

######################## Begins Simulation #############################

## Simulates economy
N_ind = 100000 
T = 2400
z_ind, w_ind, a_ind, c_ind, status_ind, m_unemp = simulation_baseline(V_f, V_i, V_u, θ_f, N_ind, T, params)

## Distribution of w

gini_w = round(gini_calculate(w_ind)*100, digits = 1)
gini_w_f = round(gini_calculate(w_ind[status_ind .== 1])*100, digits = 1)
gini_w_i = round(gini_calculate(w_ind[status_ind .== 2])*100, digits = 1)

custom_range = range(minimum(w_ind), stop = maximum(w_ind), length = 200)
histogram(w_ind[status_ind .== 1], label = "formal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :green)
histogram!(w_ind[status_ind .== 2], label = "informal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :orange, linealpha = 0.7)
histogram!(w_ind[status_ind .== 3], label = "unemployed",
bins = custom_range, fillalpha = 0.0, linecolor = :purple)
title!("Distribution of labor income by employment status, histogram", titlefontsize = 11)
xlabel!("w (income)", xlabelfontsize = 10)
annotate!([
 (0.72, 3300, text("Gini w (all):          $(gini_w)", 8, :left), :black),
 (0.72, 3100, text("Gini w (formal):    $(gini_w_f)", 8, :left), :black),
 (0.72, 2900, text("Gini w (informal): $gini_w_i", 8, :left), :black),
])
saveimages ? savefig("images/distribution_w_1.svg") : 0

## Distribution of a

gini_a = round(gini_calculate(a_ind .- a_min)*100, digits = 1)
gini_a_f = round(gini_calculate(a_ind[status_ind .== 1] .- a_min)*100, digits = 1)
gini_a_i = round(gini_calculate(a_ind[status_ind .== 2] .- a_min)*100, digits = 1)

custom_range = range(minimum(a_ind), stop = maximum(a_ind), length = 200)
histogram(a_ind[status_ind .== 1], label = "formal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :green)
histogram!(a_ind[status_ind .== 2], label = "informal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :orange, linealpha = 0.7)
histogram!(a_ind[status_ind .== 3], label = "unemployed",
bins = custom_range, fillalpha = 0.0, linecolor = :purple)
title!("Distribution of wealth by employment status, histogram", titlefontsize = 11)
xlabel!("a (wealth)", xlabelfontsize = 10)
annotate!([
 (9.4, 3800, text("Gini a (all):          $gini_a", 8, :left), :black),
 (9.4, 3550, text("Gini a (formal):    $gini_a_f", 8, :left), :black),
 (9.4, 3300, text("Gini a (informal): $(gini_a_i)", 8, :left), :black),
])
saveimages ? savefig("images/distribution_a_1.svg") : 0

## Distribution of z

custom_range = range(minimum(z_ind), stop = maximum(z_ind), length = 200)

histogram(z_ind[status_ind .== 1], label = "formal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :green)
histogram!(z_ind[status_ind .== 2], label = "informal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :orange, linealpha = 0.7)
#histogram!(z_ind[status_ind .== 3], label = "unemployed",
#bins = custom_range, fillalpha = 0.0, linecolor = :purple)
title!("Distribution of ability by employment status, histogram", titlefontsize = 11)
xlabel!("z (ability)", xlabelfontsize = 10)
#vline!([minimum(z_ind[status_ind .== 1])])
saveimages ? savefig("images/distribution_z_1.svg") : 0

## Distribution of c

gini_c = round(gini_calculate(c_ind)*100, digits = 1)
gini_c_f = round(gini_calculate(c_ind[status_ind .== 1])*100, digits = 1)
gini_c_i = round(gini_calculate(c_ind[status_ind .== 2])*100, digits = 1)

custom_range = range(0.0, stop = maximum(c_ind), length = 200)
histogram(c_ind[status_ind .== 1], label = "formal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :green)
histogram!(c_ind[status_ind .== 2], label = "informal worker",
bins = custom_range, fillalpha = 0.0, linecolor = :orange, linealpha = 0.7)
histogram!(c_ind[status_ind .== 3], label = "unemployed",
bins = custom_range, fillalpha = 0.0, linecolor = :purple)
title!("Distribution of consumption by employment status, histogram", titlefontsize = 11)
xlabel!("c (consumption)", xlabelfontsize = 10)
annotate!([
 (0.75, 2000, text("Gini c (all):        $gini_c", 8, :left), :black),
 (0.75, 1850, text("Gini c (formal):  $gini_c_f", 8, :left), :black),
 (0.75, 1700, text("Gini c (informal): $gini_c_i", 8, :left), :black),
])
saveimages ? savefig("images/distribution_c_1.svg") : 0

Tw_within_f, Tw_within_i, Tw_between = 
    theil_subgroups(w_ind[status_ind .== 1], w_ind[status_ind .== 2])

Ta_within_f, Ta_within_i, Ta_between = 
    theil_subgroups(a_ind[status_ind .== 1] .- (a_min - 1.0e-4), a_ind[status_ind .== 2] .- (a_min - 1.0e-4))

Tc_within_f, Tc_within_i, Tc_between = 
    theil_subgroups(c_ind[status_ind .== 1], c_ind[status_ind .== 2])

Tz_within_f, Tz_within_i, Tz_between = 
    theil_subgroups(z_ind[status_ind .== 1] .- (z_min - 1.0e-4), z_ind[status_ind .== 2] .- (z_min - 1.0e-4))

T_betweens = [Tw_between, Ta_between, Tc_between, Tz_between,]
T_withins_f = [Tw_within_f, Ta_within_f, Tc_within_f, Tz_within_f,]
T_withins_i = [Tw_within_i, Ta_within_i, Tc_within_i, Tz_within_i,]

groupedbar([T_withins_i T_withins_f T_betweens],
    bar_position = :stack,
    label=["Inequality within informal" "Inequality within formal" "Inequality between formal and informal"],
    legend = :outerbottom,
    xticks = (1:4, ["Income", "Wealth", "Consumption", "Ability",]),
    color = [:orange :green :red],
    title = "Theil inequality index, disaggregation"
)
saveimages ? savefig("images/theil_baseline.svg") : 0

#To calculate voluntary informality
#sum(z_ind[status_ind .== 2] .< minimum(z_ind[status_ind .== 1])) / length(z_ind[status_ind .== 2])