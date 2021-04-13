"""
    training_loop!(::updateInfo, ::lossInfo, ::trainingInfo, recorder::NamedTuple)

Train model defined in lossInfo, using update rule defined in updateInfo, and with training parameters defined in trainingInfo
"""
function training_loop!(upd::updateInfo, li::lossInfo, tr::trainingInfo, rec::NamedTuple; cb = () -> ()) 
    store = make_temp_storage(li, upd, tr, rec)
    store = initialise_updates!(store, upd)
    store = set_not_updated!(store, rec)
    rec, store = add_records!(rec,store,0)
    
    for i in 1:tr.iters
        store = do_updates!(store, upd)
        rec, store = add_records!(rec,store,i)
    end
    println("$(tr.iters) iterations")
    return store, rec
end