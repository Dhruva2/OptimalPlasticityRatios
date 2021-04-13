using Flux
using Tracker
using Statistics
using Distributed
using OptimalPlasticityRatios

const cQuality  = 1 ./[0.1,1, 5]
const sysfluct_ratios =  0.1:0.3:2.5  
const disturbance_sizes = [0.01, 0.02, 0.03]
const num_reps = 8

const numInputs = 12
const numOutputs = 10


const batch_size = 1000
const training_size = 1000
# const test_size = 500

const num_hidden_units = 20
const training_cycles = 8000

function make_student(num_hidden_units::Integer)
    return Chain(
  Dense(numInputs, numOutputs))
end


students = [make_student(num_hidden_units) for qual in cQuality, dis in disturbance_sizes, rat in sysfluct_ratios, rep in 1:num_reps]
teachers = [make_student(num_hidden_units) for rep in 1:num_reps]

teacher_weights = [Tracker.data.(params(teach)) for teach in teachers]

for i in 1:num_reps
  [Flux.loadparams!(student, (1. + 0.001*randn(1)[1]).*teacher_weights[i]) for student in students[:,:,:,i]]
end


# [Flux.loadparams!(student, (1. + 0.001*randn(1)[1]).*teacher_weights) for student in students]


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



to_record = NamedTuple{(:loss,)}((training_cycles-100:training_cycles,))

recs = OptimalPlasticityRatios.make_records_template.(li, (to_record,))
# @time out = OptimalPlasticityRatios.training_loop!.(upd, li, (tr,), recs )

@time  outs = map(OptimalPlasticityRatios.training_loop!, upd,li,tr,recs)

stores = first.(outs)
recs = last.(outs)
d = Dict{Symbol, Any}(:recs => recs)


using BSON

bson("linear_call.bson", d)





include("dataParsing.jl")
recLossOnly = [el[:loss] for el in recs]
dataStore = extract_endmean.(recLossOnly)


ps = make_all_plots(sysfluct_ratios, recLossOnly)

# l = @layout [ a b c ; 
#               d e f ;
#               g h i ]


# qq = plot(ps..., layout=l) 


# sp = deepcopy(ps)
# for i = 1:3
#   for j = 1:3
#     sp[i,j] = ps[j,i]
#   end
# end

pplot(r::recorder) = plot(r.index, r.value)