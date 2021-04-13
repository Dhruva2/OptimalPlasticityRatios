"""
    recorder(index<:AbstractRange, value::Array{T,1}, step_info::Mvector{2})

- Holds array of type {T} over the range indicated by index.
- Step holds two numbers: (i,j) where j = index[i]
- Arrays are initialised as undef{T}
"""

struct recorder{T, U<:Integer, R<:AbstractRange}
    # holds a set of timepoints, and associated data of type T. Also a name. everything is in arrays so quicker to assign and allows for immutability of this struct
    index::R
    value::Array{T,1}
    current_step::MArray{Tuple{2},U,1,2}   # statically sized array of size 1, with mutable entries
end 
recorder(index::T, value) where T <: AbstractRange = recorder(index,Array{typeof(value)}(undef, length(index)), MVector(1,index[1]))
# recorder(r::recorder, rg::AbstractRange) = recorder(r.index[findall(x -> x ∈ rg, r.index)], r.value[findall(x -> x ∈ rg, r.index)])
recorder(r::recorder, rg::AbstractRange) = recorder(rg, r.value[findall(x -> x ∈ rg, r.index)], MVector(1,rg[1]))

@recipe f(::Type{recorder}, r::recorder) = (r.index, r.value)

"""
    make_records_template(l::lossInfo, which::NamedTuple)
e.g. 

    make_records_template(l, NamedTuple{(:grad_norm, :params)}(1:4, 5:7)) 

records grad_norm at timesteps 1:4 and params at timesteps 5:7
#### Available things to record:
    :grad_norm, :loss, :training_loss, :test_loss, :local_task_difficulty
    :grad, :update, :sys_noise, :intr_noise, :params
"""
function make_records_template(l::lossInfo, which::NamedTuple)
    fields = Array{recorder,1}(undef,length(keys(which)))
    weights_template = Tracker.data.(params(get_model(l)))
    for (i,name) in enumerate(keys(which))
         if name ∈ (:grad, :test_grad, :update, :sys_noise, :intr_noise, :params, :full_training_grad)
            # make an array of weights templates for these 
            fields[i] = recorder(which[name], deepcopy(weights_template))
        elseif name ∈ (:grad_norm, :test_grad_norm, :loss, :training_loss, 
                       :test_loss, :local_task_difficulty, :update_norm,
                       :training_test_grad_correlation, :full_training_loss)
            # make an array of floats for these
            fields[i] = recorder(which[name], 0.)
        end
    end 
    return NamedTuple{keys(which)}(fields)
end



function make_some_model_templates(l::lossInfo, num_holders)
    # preallocate arrays for: update term, noise term, 
    m = get_model(l)
    template = Tracker.data.(params(m))
    r = recorder(1:num_holders, template)
    [r.value[i] = deepcopy(template) for i in r.index]
    for i in r.index #zero the nested array
        [fill!(el,0) for el in r.value[i]]
    end
    return r
end


function make_temp_storage(l::lossInfo ,u::updateInfo, tr::trainingInfo, rec )
    store = Dict{Symbol,Any}
    store = Dict((:lossInfo, :updateInfo, :trainingInfo) .=> (l,u,tr))
    store[:params] = params(get_model(l))
    return store
end


"""
    set_not_updated(store, recording_template)
makes store[:not_updated];  a set of names that are not updated by do_updates(), but need to be updated for add_records(). 
"""
function set_not_updated!(store, rec)
    # must come immediately after initialise_updates()
    # the names of the things we have to update:
    store[:not_updated] = setdiff(keys(rec), keys(store))
    # to check = reduce(∪, get_storage_dependencies.(store[:not_updated])...)
    to_check = get_storage_dependencies(store[:not_updated])
    store[:am_i_updated] = Dict(to_check .=> false)
    return store
end



"""
    add_records(rec,store, data, t)
within the training loop at time t. fill rec given values in store
data is only used for calculating local task difficulty
"""
function add_records!(rec, store, t::Integer)
    li = store[:lossInfo]
    tr = store[:trainingInfo]
    [store[:am_i_updated][name] = false for name in keys(store[:am_i_updated])]

    for name in keys(rec)
            # my_rec = rec[name]
            if t == rec[name].current_step[2]
                # println(name, t)
                store = fill_record!(store,rec[name],name)
                rec[name].current_step[1] += 1
                if length(rec[name].index) >= rec[name].current_step[1]
                    rec[name].current_step[2] = rec[name].index[rec[name].current_step[1]]
                end
            end 
    end

    return rec, store
end
"""
check whether to fill record at this timestep
"""
function fill_record!(store, my_rec,name)
    if name ∈ store[:not_updated] && store[:am_i_updated][name] == false
    # update store values if not already done     
        # store = @eval $(Symbol("get_",name,"!"))(store)
        # e.g. get_grad_norm!(store)
        # store = eval(:($(Symbol("get_",name,"!"))(store))) eval is global scope so this doesn't work
        store = getfield(WeightDrifter, Symbol("get_$(name)!"))(store[:updateInfo], store)
        # store = f(store)
        store[:am_i_updated][name] = true
    end
    if name ∈ (:grad, :test_grad, :full_training_grad)
        my_rec.value[my_rec.current_step[1]] = Tracker.data.(store[name][p] for p in store[:params])
    elseif name == :params
        my_rec.value[my_rec.current_step[1]] =  Tracker.data.(store[name])
    else
        my_rec.value[my_rec.current_step[1]] =  Tracker.data(store[name])
    end
    return store
end

