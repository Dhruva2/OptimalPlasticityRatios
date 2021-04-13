abstract type lossInfo end

struct studentTeacherInfo{S, T, F<:Function} <: lossInfo
    student::S
    teacher::T
    loss_fn::F
end

get_model(l::lossInfo) = @warn "specify the function get_model for your concrete lossInfo type"
get_model(s::studentTeacherInfo) = s.student 
get_teacher(s::studentTeacherInfo) = s.teacher
function get_num_weights(s::studentTeacherInfo)
    ## don't add activation layers to the number of weights
    return sum([prod(size(aa)) for aa in params(s.student) if length(size(aa)) > 1 && sum(aa) != 0])   
end
# get_num_weights(s::studentTeacherInfo) = sum(prod([size(a)...]) for a in params(s.student))

function get_loss(s::studentTeacherInfo, data)
    return s.loss_fn(s.student(data), s.teacher(data))
end

"""
    get_loss_and_grad(s::studentTeacherInfo, data, params)

params are the parameters with which to get gradient with respect to
"""
function get_loss_and_grad(s::studentTeacherInfo, data, pms)
    # specific for student-teacher type losses. pms is the tracked params with which to take the gradient with respect to
    # specific for training regimes that require gradients
    loss_holder = s.loss_fn(s.student(data), s.teacher(data)) #this assigns a 
    grad_holder = Tracker.gradient(() -> loss_holder, pms) # this should not use allocations
    return loss_holder, grad_holder
end



function get_loss_and_grad!(store, data; nest=false)
    # specific for student-teacher type losses. pms is the tracked params with which to take the gradient with respect to
    # specific for training regimes that require gradients
    store[:loss] = store[:lossInfo].loss_fn(store[:lossInfo].student(data), store[:lossInfo].teacher(data))
    store[:grad] = Tracker.gradient(() -> store[:loss], store[:params]; nest = nest ) 
    store[:loss] = Tracker.data(store[:loss])
    return store
end

