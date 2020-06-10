module DUtils


using Pkg
using Revise
using Statistics: mean
using Base.Iterators: cycle
using Knet: Knet, AutoGrad, Data, param, param0, mat, RNN, value, nll, adam, minibatch, progress!, converge, Random, TimerOutputs, dir, zeroone, progress, sgd, load, save, gc, Param, KnetArray, gpu, Data, relu, training, dropout
using Base.Iterators: flatten
using IterTools: ncycle, takenth



struct CreateModel; end;


struct MyData1 end


struct DData2; d; DData2(d) = new(d); end;

function Base.iterate(f::DData2, s...)
    next = iterate(f.d, s...)
    next === nothing && return nothing
    ((x,y),state) = next
    return (x,y), state
end

Base.length(f::DData2) = length(f.d) # collect needs this



mutable struct GCGC; itr; curriter; gc_at; end

GCGC(itr, gc_at, ::CreateModel) = GCGC(itr, 0, gc_at)

function Base.iterate(f::GCGC, s...)
    next = iterate(f.itr, s...)
    next === nothing && return nothing
    (x,state) = next
    f.curriter += 1
    if(f.curriter % f.gc_at == 0)
        GC.gc()
    end
    return x , state
end

Base.length(f::GCGC) = length(f.itr)


println("DUtils loaded.")

end #module DUtils
