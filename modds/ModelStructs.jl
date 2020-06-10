module ModelStructs




using Pkg
using Revise
using Statistics: mean
using Base.Iterators: cycle
using Knet: Knet, AutoGrad, Data, param, param0, mat, RNN, value, nll, adam, minibatch, progress!, converge, Random, TimerOutputs, dir, zeroone, progress, sgd, load, save, gc, Param, KnetArray, gpu, Data, relu, training, dropout
using Base.Iterators: flatten
using IterTools: ncycle, takenth

using ..DUtils: MyData1, DData2, CreateModel



struct EmbMatT end
struct EmbMatPar end

struct Embed; w; end

Embed(vocab::Int,embed::Int)=Embed(param(embed,vocab))
Embed(wmat, ::EmbMatT) = Embed(wmat)
Embed(wmat, aType, ::EmbMatPar) = Embed(param(wmat, atype = aType))
Embed(wmat, ::EmbMatPar) = Embed(param(wmat))

(e::Embed)(x) = e.w[:,x]  # (B,T)->(X,B,T)->rnn->(H,B,T)

struct Dense; w; b; f; end
Dense(i::Int,o::Int,f=identity) = Dense(param(o,i), param0(o), f)
(d::Dense)(x) = d.f.(d.w * mat(x,dims=1) .+ d.b)

struct DenseDrop; w; b; f; dout; end
DenseDrop(i::Int,o::Int,f=identity; dropout=0.0) = DenseDrop(param(o,i), param0(o), f, dropout)
(d::DenseDrop)(x) = d.f.(Knet.dropout(d.w, d.dout) * mat(x,dims=1) .+ Knet.dropout(d.b, d.dout))

struct Linear; w; b; end

Linear(input::Int, output::Int)=Linear(param(output,input), param0(output))

(l::Linear)(x) = l.w * mat(x,dims=1) .+ l.b  # (H,B,T)->(H,B*T)->(V,B*T)
(l::Linear)(x,y)= quadl(l(x),y)[1]

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)
(c::Chain)(d,a,b) = mean(c(x,y) for (x,y) in d)
(c::Chain)(d, ::MyData1) = mean(c(x,y) for (x,y) in d)

MLP(i::Int, h::Int, o::Int) = Chain(Dense(i,h, Knet.relu), Dense(h,o,Knet.relu))
MLP2(i::Int, o::Int) = Dense(i,o, Knet.relu)

MLP2(i::Int, o::Int; dropout=0.0) = DenseDrop(i,o, Knet.relu, dropout=dropout)

struct TestRNN; rnn1; w; b; end;

TestRNN(aa::CreateModel)  = begin
    return TestRNN(RNN(50,200,bidirectional=true, rnnType = :lstm, numLayers=3), param(3), param(3))
end
(trnn::TestRNN)(x) = begin
    rnnout = trnn.rnn1(x)
    return rnnout
end
(trnn::TestRNN)(x,y) = mean((trnn(x)-y).^2)
(trnn::TestRNN)(d::DData2) = (mean(trnn(x,y) for (x,y) in d))

struct Biaff; mlphead; mlpdep; w; b; end;
#Biaff(mlph::MLP, mlpd::MLP, input::Int, output::Int)=Biaff(mlph, mlpd , param(output,input), param0(output))
Biaff(rhid::Int, mlph::Int, m::Int, input::Int, output::Int) =
    Biaff(MLP(rhid,mlph, m) , MLP(rhid,mlph, m) , param(m,m), param0(m))
Biaff(rhid::Int, mlph::Int, m::Int) =
    Biaff(MLP(rhid,mlph, m) , MLP(rhid,mlph, m) , param(m,m), param0(m))

(bi::Biaff)(x) = begin
    tts = size(x,3)
    bbs = size(x,2)
    archead = bi.mlphead(x)
    arcdep = bi.mlpdep(x[:,:,2:end])
    hms = size(archead,1)
    archeadp = permutedims(reshape(archead, hms, bbs, tts), [3, 1, 2])
    archeadpp = permutedims(reshape(archead, hms, bbs, tts), [3, 2, 1])
    archeadppr = reshape(archeadpp, bbs*tts, hms)
    WH = bi.w * arcdep
    WHp = permutedims(reshape(WH, size(WH,1), bbs, tts-1), [1,3,2])
    HWH = Knet.bmm(archeadp, WHp)
    Hb = archeadppr * bi.b
    Hbr = reshape(Hb, tts, 1 , bbs)
    S = HWH .+ Hbr
end

struct BiaffREL; mlpheadrel; mlpdeprel; U; W; b; end;
#Biaff(mlph::MLP, mlpd::MLP, input::Int, output::Int)=Biaff(mlph, mlpd , param(output,input), param0(output))

BiaffREL(rhid::Int, mlph::Int, numrel::Int) =
    BiaffREL(MLP(rhid,mlph, numrel) , MLP(rhid,mlph, numrel) , param(1,numrel), param(numrel,numrel*2), param0(numrel))

(bi::BiaffREL)(x, argmx, S) = begin
    tts = size(x,3)
    bbs = size(x,2)


    archead = bi.mlpheadrel(x)
    arcdep = bi.mlpdeprel(x[:,:,2:end])
    Sr1 = reshape(S, tts, (tts-1)*bbs)
    #hms = size(archead,1)
    archeadp = permutedims(reshape(archead, rms, bbs, tts), [3, 1, 2])
    archeadpp = permutedims(reshape(archead, rms, bbs, tts), [3, 2, 1])
    rms = size(archead,1)


    archeadppr = reshape(archeadpp, bbs*tts, hms)
    WH = bi.w * arcdep
    WHp = permutedims(reshape(WH, size(WH,1), bbs, tts-1), [1,3,2])
    HWH = Knet.bmm(archeadp, WHp)
    Hb = archeadppr * bi.b
    Hbr = reshape(Hb, tts, 1 , bbs)
    S = HWH .+ Hbr
    return (S,Srel)
end


fModel(tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) =
    Chain(RNN(tembed, rnnh; bidirectional=true, rnnType = :lstm,o...), Biaff(2*rnnh,mlp1h,mlp1o))

fModel2(wmat, tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) =
    Chain(Embed(wmat, EmbMatT()), RNN(tembed, rnnh; bidirectional=true, rnnType = :lstm,o...), Biaff(2*rnnh,mlp1h,mlp1o))

fModel3(wmat, tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) =
    Chain(Embed(wmat, KnetArray, EmbMatPar()), RNN(tembed, rnnh; bidirectional=true, rnnType = :lstm,dropout=0.3, o...), Biaff(2*rnnh,mlp1h,mlp1o))


fModel4(wmat, tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) =
    Chain(Embed(wmat, KnetArray, EmbMatPar()), RNN(tembed, rnnh; bidirectional=true, rnnType = :lstm,dropout=0.3,numLayers=3, o...), Biaff(2*rnnh,mlp1h,mlp1o))

fModel5(wmat, tembed::Int, rnnh::Int, mlp1h::Int, mlp1o::Int; o...) =
    Chain(Embed(wmat, KnetArray, EmbMatPar()), RNN(tembed, rnnh; bidirectional=false, rnnType = :lstm, numLayers=2, o...), Biaff(rnnh,mlp1h,mlp1o))




end #module ModelStructs
