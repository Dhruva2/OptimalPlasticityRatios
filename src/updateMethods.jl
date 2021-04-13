"""
    initialise_updates!(store, ::updateInfo)
provides the time = 0 values of relevant quantities (e.g. loss, grad) in store. these are used for the time = 1 update. 
time = 1 quantities are calculated after the time=1 update. 

When writing a new initialise_updates! method for a different type of ::updateInfo, make sure to initialise (in store),
exactly those fields which are automatically updated by the do_updates! method
"""
function initialise_updates!(store::Dict{Symbol,Any}, upd::fixedNormFluctuatingNoisyGD)
    
    ## initialise storage holders
    data = get_training_batch(store[:trainingInfo])
    store = get_loss_and_grad!(store, data) #sets loss and grad
    store[:sys_noise], store[:intr_noise], store[:update] = make_some_model_templates(store[:lossInfo],3).value

    ## initialise time zero values for storage holders
    # set noise
    for s in (store[:sys_noise],store[:intr_noise])
        avoid_activations(s)
    end

    g_norm2, sys_noise_norm2, intr_noise_norm2 = Float32(0),Float32(0),Float32(0) #squared quantities
    for (k,p) in  enumerate(store[:params])
        if length(size(p)) > 1 #avoid activation layers
            g_norm2 += sum(store[:grad][p].^2)
            sys_noise_norm2 += sum(store[:sys_noise][k].^2)
            intr_noise_norm2 += sum(store[:intr_noise][k].^2)
        end
    end

    store[:grad_norm] = sqrt(Tracker.data(g_norm2))
    store[:sys_noise_norm] = sqrt(Tracker.data(sys_noise_norm2))
    store[:intr_noise_norm] = sqrt(Tracker.data(intr_noise_norm2))
    
"""
Want γ1, γ2: 
    γ1^2 + γ2^2 = upd.norm^2 :=m^2
    γ1/γ2 = upd.sn_ratio := r^2
Soln:
γ1 = sqrt(m^2(1 - 1/(1+r^2))) = m*sqrt(1 - (1/(1+r^2)))
γ2 = sqrt(m^2/1+r^2) 
"""
    store[:γ1] = upd.norm*sqrt(      1 - ( 1/(1+upd.sn_ratio^2) )      )
    store[:γ2] = upd.norm*sqrt(      1/(1 + upd.sn_ratio^2)    )

    return store
end

function initialise_updates!(store::Dict{Symbol,Any}, upd::vanillaGD)
    data = get_training_batch(store[:trainingInfo])
    store = get_loss_and_grad!(store, data)
    store[:update] = make_some_model_templates(store[:lossInfo],1).value[1]
    return store
end


function initialise_updates!(store::Dict{Symbol,Any}, upd::beelineLearning)
    data = get_training_batch(store[:trainingInfo])
    store[:loss] = get_loss(store[:lossInfo], data)
    store[:intr_noise], store[:update] = make_some_model_templates(store[:lossInfo],2).value
    avoid_activations(store[:intr_noise])
    store[:intr_noise_norm] = sqrt(sum(sum(el.^2) for el in store[:intr_noise]))
    store[:teacher_array] = [Tracker.data(el) for el in params(get_teacher(store[:lossInfo]))]
    store[:offset] = store[:teacher_array] - [Tracker.data(el) for el in store[:params]]
    store[:offset_norm] = sqrt(sum(sum(el.^2) for el in store[:offset]))
  

    return store
end


"""
    do_updates(store, ::updateInfo)

mutates values in the dictionary store, using loss function in ::lossInfo and update rule in ::updateInfo
"""
function do_updates!(store::Dict{Symbol,Any}, upd::fixedNormFluctuatingNoisyGD)
    # for fixed Norm noisy gradient descent with an intrinsic plasticity term 
    ## seed noise

# multiply things by approprite norms: find multipliers
    sys_noise_multiplier = store[:γ2]/store[:sys_noise_norm]
    grad_multiplier = -store[:γ1]/store[:grad_norm]
    # signal_proportion = upd.sn_ratio/(1 + upd.sn_ratio) # in terms of signal:noise ratio
    # sys_noise_multiplier = sqrt((upd.norm^2)*signal_proportion)/(store[:sys_noise_norm])
    # grad_multiplier = -sqrt((upd.norm^2)*(1-signal_proportion))/(store[:grad_norm])
    intr_noise_multiplier = upd.intrinsic_norm/store[:intr_noise_norm]

# update parameters
    for (k,p) in enumerate(store[:params])
        if length(size(p)) > 1 #avoid activation layers
            store[:update][k] = Tracker.data(grad_multiplier*store[:grad][p]) .+ 
                                    sys_noise_multiplier*store[:sys_noise][k] .+
                                    intr_noise_multiplier*store[:intr_noise][k]
            Tracker.update!(p,store[:update][k])
        end
    end
    ## get loss and gradient
    data = get_training_batch(store[:trainingInfo])
    store = get_loss_and_grad!(store, data)
    store[:loss] = Tracker.data(store[:loss]) 

    ## set noise for next iteration
    for s in (store[:sys_noise],store[:intr_noise])
        avoid_activations(s)
    end

    ### calculate norms of things for next iteration
    g_norm2, sys_noise_norm2, intr_noise_norm2 = Float32(0),Float32(0),Float32(0) #squared quantities
    for (k,p) in  enumerate(store[:params])
        if length(size(p)) > 1 #avoid activation layers
            g_norm2 += sum(store[:grad][p].^2)
            sys_noise_norm2 += sum(store[:sys_noise][k].^2)
            intr_noise_norm2 += sum(store[:intr_noise][k].^2)
        end
    end
    store[:grad_norm] = sqrt(Tracker.data(g_norm2))
    store[:sys_noise_norm] = sqrt(Tracker.data(sys_noise_norm2))
    store[:intr_noise_norm] = sqrt(Tracker.data(intr_noise_norm2))
    return store
end

function do_updates!(store::Dict{Symbol,Any}, upd::vanillaGD) 
    # update parameters
    for (k,p) in enumerate(store[:params])
        if length(size(p)) > 1 #avoid activation layer
            store[:update][k] = Tracker.data(-upd.learning_rate*store[:grad][p])
            Tracker.update!(p, store[:update][k])
            # Tracker.update!(p, -upd.learning_rate*store[:grad][p])
            # store[:update][k] = Tracker.data(-upd.learning_rate*store[:grad][p])
        end
    end

    # recalculate loss and grads at new parameter value
    data = get_training_batch(store[:trainingInfo])
    store = get_loss_and_grad!(store, data)
end

function do_updates!(store::Dict{Symbol,Any}, upd::beelineLearning)
    
    for (k,p) in enumerate(store[:params])
        if length(size(p)) > 1
            store[:update][k] = (upd.step_size/store[:offset_norm])*store[:offset][k] + 
            (upd.intrinsic_norm/store[:intr_noise_norm])*store[:intr_noise][k]
            Tracker.update!(p, store[:update][k])
            
            ## update offset for next iteration
            store[:offset][k] = store[:teacher_array][k] - Tracker.data(p)
        end
    end


    avoid_activations(store[:intr_noise]) # update noise for next iter
    store[:intr_noise_norm] = sqrt(sum(sum(el.^2) for el in store[:intr_noise]))
    store[:offset_norm] = sqrt(sum(sum(el.^2) for el in store[:offset]))
    data = get_training_batch(store[:trainingInfo])
    store[:loss] = get_loss(store[:lossInfo], data)
    return store
end


function get_test_loss!(::updateInfo, store)
    if store[:am_i_updated][:test_loss] == true
        return store
    end
    store[:test_loss] = get_loss(store[:lossInfo], get_test_batch(store[:trainingInfo]))
    store[:am_i_updated][:test_loss] = true
    return store
end

function get_grad_norm!(::updateInfo, store)
    if store[:am_i_updated][:grad_norm] == true
        return store
    end
    store[:grad_norm] = Tracker.data(sqrt(sum(sum(store[:grad][p].^2) for p in store[:params])))
    store[:am_i_updated][:grad_norm] = true
    return store
end


function get_update_norm!(::updateInfo, store)
    if store[:am_i_updated][:update_norm] == true
        return store
    end
    store[:update_norm] = sqrt(sum(sum(el.^2) for el in store[:update]))
    store[:am_i_updated][:update_norm] = true
    return store
end


function get_test_grad!(::updateInfo, store)
    if store[:am_i_updated][:test_grad] == true
        return store
    end
    store = get_test_loss!(store[:updateInfo], store)
    store[:test_grad] = Tracker.gradient( () -> store[:test_loss], store[:params]; nest=true)
    store[:am_i_updated][:test_grad] = true
    return store
end

function get_full_training_loss!(::updateInfo, store)
    if store[:am_i_updated][:full_training_loss] == true
        return store
    end
    store[:full_training_loss] = get_loss(store[:lossInfo], store[:trainingInfo].training_data)
    store[:am_i_updated][:full_training_loss] = true
    return store
end

function get_full_training_grad!(::updateInfo, store)
    if store[:am_i_updated][:full_training_grad] == true
        return store
    end
    store = get_full_training_loss!(store[:updateInfo], store)
    store[:full_training_grad] = Tracker.gradient( () -> store[:full_training_loss], store[:params] )
    store[:am_i_updated][:full_training_grad] = true
    return store
end

function get_full_training_grad_norm!(::updateInfo, store)
    if store[:am_i_updated][:full_training_grad_norm] == true
        return store
    end
    store = get_full_training_grad!(store[:updateInfo], store)
    store[:full_training_grad_norm] = sqrt(sum(sum(store[:full_training_grad][p].^2) for p in store[:params]))
    store[:am_i_updated][:full_training_grad_norm] = true
    return store
end

function get_test_grad_norm!(::updateInfo, store)
    if store[:am_i_updated][:test_grad_norm] == true
        return store
    end
        store = get_test_grad!(store[:updateInfo], store)
        store[:test_grad_norm] = sqrt(sum(sum(store[:test_grad][p].^2) for p in store[:params]))
        store[:am_i_updated][:test_grad_norm] = true
    return store
end

"""
    get_local_task_difficulty(store::Dict{Symbol,Any}, tr::trainingInfo, li::lossInfo)
note that this function requires that store[:update_norm] is updated in the do_updates!() function, not post-hoc
"""
function get_local_task_difficulty!(::updateInfo, store)
    tr = store[:trainingInfo]
    li = store[:lossInfo]
    store = get_test_grad!(store[:updateInfo], store) 
    store = get_test_grad_norm!(store[:updateInfo], store)
    store = get_update_norm!(store[:updateInfo], store)

    #  Hessian*update_direction
    store[:extra_tplt] = Tracker.gradient(() -> sum(sum(sum(store[:test_grad][p].*store[:update][k])) for (k,p) in enumerate(store[:params])), store[:params])

    store[:local_task_difficulty] = Tracker.data(  
                        (1/(2*store[:test_grad_norm]*store[:update_norm]^2))*
                        (sum(sum(sum(store[:extra_tplt][p].*store[:update][k])) 
                                        for (k,p) in enumerate(store[:params]))) )
    store[:am_i_updated][:local_task_difficulty] = true
    return store
end

function get_training_test_grad_correlation!(::updateInfo, store)
    # store = get_grad_norm!(store[:updateInfo], store)
    store = get_test_grad_norm!(store[:updateInfo], store) # also gets test grad
    store = get_full_training_grad_norm!(store[:updateInfo], store)
    temp = 0.
    for p in store[:params]
        temp += dot(store[:full_training_grad][p], store[:test_grad][p])
    end
    store[:training_test_grad_correlation] = Tracker.data(temp/(store[:full_training_grad_norm]*store[:test_grad_norm]) )
    store[:am_i_updated][:training_test_grad_correlation] = true
    return store
end

"""
solution to double updating: 
1. make updated = make_update_checker part of store
2. if local_task_difficulty comes first: when it evaluates store[:test_loss], change updated(:test_loss) to true
3. if it comes after test_loss, don't need to calculate test loss

so an if statement in local_task_difficulty
"""

