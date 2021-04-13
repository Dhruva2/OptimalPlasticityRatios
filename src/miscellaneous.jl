    """"
    hold miscellaneous functions with no downstream dependencies. 
    """
    
    
    """
        avoid_activations(tensor_shaped_like_weights)
    randn! every weight as long as it's not an activation (defined as a 1-tensor, since normal weights are arranged as matrices between layers)
    """
    function avoid_activations(noiseTemp)
        [randn!(t) for t in noiseTemp if length(size(t)) > 1]
    end

