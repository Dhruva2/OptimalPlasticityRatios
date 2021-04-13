using Flux
using Tracker
using Statistics
using Plots
using Distributed
using OptimalPlasticityRatios



const cQuality = 1 ./[0.1,20, 50]
const sysfluct_ratios =  0.1:0.3:2.5  #0.1:0.3:2.5
const disturbance_sizes = [0.01, 0.02, 0.03]
const num_reps = 6

const numInputs = 12
const numOutputs = 10

const batch_size = 1000
const training_size = 1000
# const test_size = 500

const num_hidden_units = 20
@everywhere const training_cycles = 8000


#https://tutorials.juliadiffeq.org/html/introduction/03-optimizing_diffeq_code.html mutating broadcast look up and use

function make_student(num_hidden_units::Integer)
    return Chain(
  Dense(numInputs, num_hidden_units, σ),
  # Dense(num_hidden_units, num_hidden_units, σ),
  Dense(num_hidden_units, numOutputs))
end


students = [make_student(num_hidden_units) for qual in cQuality, dis in disturbance_sizes, rat in sysfluct_ratios, rep in 1:num_reps]
teachers = [make_student(num_hidden_units) for rep in 1:num_reps]

teacher_weights = [Tracker.data.(params(teach)) for teach in teachers]

for i in 1:num_reps
  [Flux.loadparams!(student, (1. + 0.001*randn(1)[1]).*teacher_weights[i]) for student in students[:,:,:,i]]
end


tr_data = randn(Float32,numInputs,training_size)

ranger = [1:m for m in size(students)]
li = [OptimalPlasticityRatios.studentTeacherInfo(students[i,j,k,m], teachers[m], Flux.mse)
for (i,j,k,m) in Base.product(ranger...)]


# li = OptimalPlasticityRatios.studentTeacherInfo.(students,(teacher,),(Flux.mse,))
tr = [OptimalPlasticityRatios.trainingDataOnly(tr_data, batch_size, training_cycles) for el in li]
### rat is ||Δc||: ||Δϵ|| ratio. dis is  ||Δϵ||

n_weights = OptimalPlasticityRatios.get_num_weights(li[1])
disturbance_sizes *= sqrt(n_weights)
println(disturbance_sizes)
upd = [OptimalPlasticityRatios.fixedNormFluctuatingNoisyGD(qual,rat*dis,dis) for qual in cQuality, dis in disturbance_sizes, rat in sysfluct_ratios, rep in 1:num_reps]



@everywhere to_record = NamedTuple{(:loss,)}((training_cycles-100:training_cycles,))

recs = OptimalPlasticityRatios.make_records_template.(li, (to_record,))
# @time out = OptimalPlasticityRatios.training_loop!.(upd, li, (tr,), recs )

@time  outs = pmap(OptimalPlasticityRatios.training_loop!, upd,li,tr,recs)
rmprocs(workers())

stores = first.(outs)
recs = last.(outs)
d = Dict{Symbol, Any}(:recs => recs)


using BSON

bson("nonlinear_call.bson", d)





include("dataParsing.jl")
recLossOnly = [el[:loss] for el in recs]
# # dataStore = extract_endmean.(recLossOnly)


ps = make_all_plots(sysfluct_ratios, recLossOnly)

# l = @layout [ a b c ; 
#               d e f ;
#               g h i ]


# qq = plot(ps..., layout=l) 


sp = deepcopy(ps)
for i = 1:3
  for j = 1:3
    sp[i,j] = ps[j,i]
  end
end

pplot(r::recorder) = plot(r.index, r.value)

