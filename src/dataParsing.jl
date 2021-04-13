
using Statistics
using LaTeXStrings
using Measures
using Plots
using Plots.PlotMeasures
gr()

dropmean(A; dims=:) = dropdims(mean(A; dims=dims); dims=dims)
dropstd(A; dims=:) = dropdims(std(A; dims=dims); dims=dims)



# @userplot Ucurve
# @recipe function f(u::Ucurve)
#     x,y = u.args
#     # println(u.args)
#     legend := false

#     my = dropmean(y,dims=2)
#     sd = dropstd(y,dims=2)
#     ymax = max(y...)
#     multiplier = 10/(round(ymax,sigdigits=1))
#     print(multiplier)
#     my = my*multiplier
#     sd = sd*multiplier
#     ymax = ymax*multiplier
#     to_power_of = -Int(log10(multiplier))
#     power_label = raw"$ \times 10^"*"{$(to_power_of)}"*raw"$"
#     println(min(x...))
#     println(1.2*ymax)
#     @series begin
#         ribbon --> sd
#         # yformatter --> :scientific
#         ylims-->(0,ymax)
#         xticks --> range(min(x...),max(x...), length=3)
#         # yguide --> power_label
#         annotations := (1.1*min(x...),0.9*ymax,power_label, :black)
#         # xlims -->(0, max(x...))       
#         x,my 
#     end
# end 


# upd = [OptimalPlasticityRatios.fixedNormFluctuatingNoisyGD(qual,rat*dis,dis) for qual in cQuality, dis in disturbance_sizes, rat in sysfluct_ratios, rep in num_reps]


function extract_endmean(rec::OptimalPlasticityRatios.recorder; length_to_average=100)
    return mean(rec.value[end-100:end])
end



function set_ticksize(min_,max_, step; sigdigits=2)
    labs = [string(Int(round(i, sigdigits=sigdigits))) for i in min_:step:max_ ]
    return (collect(min_:step:max_), labs)
end


function make_single_plot(x_axis, ds::Array{Float64,2})
    my = dropmean(ds, dims=2)
    sd = dropstd(ds, dims=2)
    ymax = max(ds...)
    # println(ymax)
    to_power_of = ceil(-log10(ymax))
    multiplier = 10^(to_power_of)
    my = my*multiplier
    sd = sd*multiplier
    ymax = ymax*multiplier
    
    to_power_of = -Int(log10(multiplier))

    if ymax < 2
        my *= 10 ; sd *= 10; multiplier *= 10; ymax *=10; to_power_of -= 1 
    end

    st = 1
    nt = length(0:st:ymax)
    while (nt >6 || nt < 3)
        nt > 6 ? st*=2 : st /= 2
        nt = length(0:st:ymax)
    end

    power_label = raw"$ \times 10^"*"{$(to_power_of)}"*raw"$"
    p = plot(x_axis, my , ribbon=sd
            , ylims = (0,ymax)
            , xlims = (0, x_axis[end])
            , xticks = range(0, x_axis[end], length=5)
            , yticks = set_ticksize(0,ymax,st, sigdigits=1)
            , legend = false
            # , annotations = (2.5 *x_axis[1],ymax, power_label, :black)
            # ; margin = 5mm
            # , title = power_label
            # , titleloc = :left
            # , titlefont = font(8)
            , margin = 10pt
            )

    annotate!(p, 0.1, ymax, text(power_label, :black, :left, 10))

end


function make_all_plots(x_axis, recStore)
    
dd = extract_endmean.(recStore; length_to_average=100) # get final loss values from the recorders, averaged over some amount of time
dat(i,j) = dd[i,j,:,:]  #  cQuality, disturbance_sizes,  sys_fluct_ratios, num_reps
all_plots = [make_single_plot(x_axis, dat(i,j)) for i in 1:size(dd)[1], j in 1:size(dd)[2]]

# with_spaces = [all_plots[1:3], plot()




end
# points = Point2f0[[0.0, 0.0], [1., 0.0], [1., 1.], [0.0, 1.]]
# spoly!(axes[1,2],points, alpha=0.1, color = :skyblue2, :fxaa = true)