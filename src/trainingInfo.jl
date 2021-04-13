"""
 holds algorithm independent information about training procedure. e.g. training cycles, what to record, etc
"""
abstract type trainingInfo end


struct training_and_test{A<:AbstractArray, I<:Integer} <: trainingInfo
    training_data::A
    test_data::A
    batch_size::Union{I,Nothing}
    iters::I
    tr_data_length::I
    tst_data_length::I

    function training_and_test(training_data, test_data, batch_size, iters)
        # automatically compute data length
        new{typeof(training_data), typeof(iters)}(training_data, test_data, 
                                        batch_size, iters, size(training_data)[end], size(test_data)[end])
    end
end
"""
    trainingDataOnly(training_data, batch_size, iters)
    trainingDataOnly(training_data, iters)
without batch_size input it takes full batches
"""
trainingDataOnly(tr_data::T, batch_size, iters) where T<:AbstractArray = training_and_test(tr_data, tr_data, batch_size, iters)
trainingDataOnly(tr_data::T, iters) where T<:AbstractArray = trainingDataOnly(tr_data, batch_size, iters)

"""
    get_training_batch(tr::trainingInfo)
get a view of a tr.batch_size'd, randomly chosen subset of tr.training_data
"""
function get_training_batch(tr::trainingInfo)
    function get_the_batch(batch_size::Nothing)
        return @view tr.training_data[:,:]
    end

    function get_the_batch(batch_size::Integer)
        return @view tr.training_data[:,rand(1:tr.tr_data_length, tr.batch_size)]
    end

    get_the_batch(tr.batch_size)
end

"""
    get_training_batch!(existing_data, tr::trainingInfo)

mutating version of get_training_batch. mutates existing_data. recommended over get_training_batch for speed where possible
"""
function get_training_batch!(existing_data, tr::trainingInfo)
    function get_the_batch(batch_size::Nothing)
        # do nothing
    end

    function get_the_batch(batch_size::Integer)
        return @view tr.training_data[:,rand(1:tr.tr_data_length, tr.batch_size)]
    end

    get_the_batch(tr.batch_size)
end

"""
    get_test_data(tr::training_and_test)
get a view of the test data in tr
"""
function get_test_batch(tr::training_and_test)
    return @view tr.test_data[:,:]    
end


