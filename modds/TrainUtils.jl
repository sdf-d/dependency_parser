module TrainUtils



using Pkg
using Revise
using Statistics: mean
using Base.Iterators: cycle
using Knet: Knet, AutoGrad, Data, param, param0, mat, RNN, value, nll, adam, minibatch, progress!, converge, Random, TimerOutputs, dir, zeroone, progress, sgd, load, save, gc, Param, KnetArray, gpu, Data, relu, training, dropout
using Base.Iterators: flatten
using IterTools: ncycle, takenth


using ..DUtils: MyData1, DData2, CreateModel




# For running experiments
function trainresults(file,model; o...)
    if (print("Train from scratch? "); readline()[1]=='y')
        takeevery(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 1)
        r = ((model(dtrn,MyData1()), model(dtst,MyData1()), zeroone(model,dtrn), zeroone(model,dtst))
             for x in takeevery(length(dtrn), progress(adam(model,repeat(dtrn,30)))))
        r = reshape(collect(Float32,flatten(r)),(4,:))
        Knet.save(file,"results",r)
        Knet.gc() # To save gpu memory
    else
        #isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
        r = Knet.load(file,"results")
    end
    println(minimum(r,dims=2))
    return r
end









end #module TrainUtils
