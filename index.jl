using PyCall
include("CKY.jl")
include("Eisner.jl")

pushfirst!(PyVector(pyimport("sys")."path"), "", "parsers")
@pyimport mst

function eisner(lambda)
    headsize = size(lambda,2)
    #println(headsize)
    heads = Array{Int}(undef, headsize)
    score, arcset = Eisner(size(lambda,2)-1, lambda)
    for arc in arcset
        heads[arc[2]]=arc[1]
    end
    heads
end

function cky(lambda)
    headsize = size(lambda,2)-1
    heads = Array{Int}(undef, headsize)
    score, arcset = CKY(size(lambda,2)-1, lambda)
    for arc in arcset
        heads[arc[2]]=arc[1]
    end
    heads
end

edmonds(lambda) = mst.mst(lambda')[2:end]
