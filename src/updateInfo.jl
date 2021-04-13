

"""
    abstract type updateInfo
- abstract supertype for all types of weight update
- holds algorithm-dependent information about training procedure 
(e.g. am I using gradient descent or Reinforce etc, and what do I need for them)
"""
abstract type updateInfo  end

"""
    gradientDependentInfo <: updateInfo
abstract supertype for training procedures that require gradient calculation
"""
abstract type gradientDependentInfo <: updateInfo  end

"""
    gradientFreeInfo <: updateInfo
abstract supertype for training procedures that are gradient free
"""
abstract type gradientFreeInfo <: updateInfo end
"""
    fixedNormFluctuatingNoisyGD(sn_ratio::Float32, norm::Float32, intrinsic_norm::Float32) <: gradientDependentInfo
update = k1*normalised_gradient + k2*normalised_systematic_noise + k3*normalised_intrinsic_noise
- sn_ratio is k1/k2 
- norm is √(k1^2 + k2^2)
- intrinsic norm is norm(k3)
"""
struct fixedNormFluctuatingNoisyGD <: gradientDependentInfo
    sn_ratio::Float32 
    norm::Float32 
    intrinsic_norm::Float32 
end

"""
    fixedNormNoisyGD <: fixedNormFluctuatingNoisyGD
special case of fixedNormFluctuatingNoisyGD with intrinsic_norm = 0
"""
fixedNormNoisyGD(sn_ratio, norm) = fixedNormFluctuatingNoisyGD(sn_ratio,norm,0)
# fixedNormNoisyGD(;sn_ratio, norm) = fixedNormFluctuatingNoisyGD(sn_ratio,norm,0) #allow keyword arguments
# fixedNormNoisyGD(;sn_ratio, norm, intrinsic_norm) = fixedNormFluctuatingNoisyGD(sn_ratio,norm, intrinsic_norm)


non_updateable(upd::updateInfo) = (:lossInfo, :trainingInfo, :updateInfo, :updated)

"""
return things that are required in the store by :name, and have their own update method (i.e. things with potential conflicts wrt double updating)
"""
function get_storage_dependencies(names::Array{Symbol,1})
    to_return = []
    for name in names
        push!(to_return, name)
        if name == :local_task_difficulty
            push!(to_return,[:test_loss, :test_grad, :test_grad_norm, :update_norm]...)
        elseif name == :test_grad
            push!(to_return,[:test_loss]...)
        elseif name == :training_test_grad_correlation
            push!(to_return, [:full_training_grad_norm, :test_grad_norm, :test_grad, :full_training_grad, :full_training_loss]...)
        elseif name == :full_training_grad
            push!(to_return, [:full_training_loss]...)
        end 
    end  
    to_return = unique(to_return)
    return (to_return...,)
end


"""
    pnas_update(γ, T, N)
- Holds information about update from PNAS paper: Fundamental bounds on learning performance. 
- update = T(γ1*normalised_gradient + γ2*normalised_systematic_noise + γ3*√N/√T*normalised_intrinsic_noise)
- not used explicitly: converts to 'equivalent' fixedNormNoisyGD for training
"""
pnas_update(γ::Array{NN,1}, T::TT, N::UU) where NN<:Number where TT <: Number where UU <: Number = fixedNormFluctuatingNoisyGD(T*γ[1], T*γ[2], sqrt(N*T)*γ[3])
pnas_update(; γ::Array{NN,1}, T::TT, N::UU) where NN<:Number where TT <: Number where UU <: Number = fixedNormFluctuatingNoisyGD(T*γ[1], T*γ[2], sqrt(N*T)*γ[3])


# function pnas_update_iterable(;  γ1::Union{AbstractRange, Number}, γ2::Union{AbstractRange, Number}, γ3::Union{AbstractRange,Number},
#                         T::Union{AbstractRange, Number}, 
#                         N::Union{AbstractRange, Number,Array{TT,1}}, repeats::Integer=1
#                     ) where TT
#     return [pnas_update( [_γ1,_γ2,_γ3], _T, _N) for (_γ1,_γ2,_γ3,_T,_N,r) in Base.product(γ1, γ2, γ3, T, N,1:repeats)]
# end


cdw_update(c::Number, normalised_corr::Number) = fixedNormFluctuatingNoisyGD(
    (normalised_corr/(c*(1-normalised_corr^2))), c, 0.)

################################################################################################################################################

"""
Struct for vanilla gradient descent
"""

"""
    vanillaGD(learning_rate)
implements gradient descent on the training set
"""
struct vanillaGD <: gradientDependentInfo
    learning_rate::Float32
end


"""
    beelineLearning(step_size, intrinsic_noise_norm, beehive)
moves the weights directly in the direction of 'beehive' (a predefined set of weights)
meanwhile intrinsic gaussian noise with a specified total norm makes the weights fluctuates
"""
struct beelineLearning <: gradientFreeInfo
    step_size::Float32
    intrinsic_norm::Float32
end
# beelineLearning(s, i, teacher) = beelineLearning(s,i
