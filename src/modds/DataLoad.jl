module DataLoad


using Pkg
using Revise
using Statistics: mean
using Base.Iterators: cycle
using Knet: Knet, AutoGrad, Data, param, param0, mat, RNN, value, nll, adam, minibatch, progress!, converge, Random, TimerOutputs, dir, zeroone, progress, sgd, load, save, gc, Param, KnetArray, gpu, Data, relu, training, dropout
using Base.Iterators: flatten
using IterTools: ncycle, takenth

using ..DUtils: MyData1, DData2, CreateModel



UDEPREL = Dict{String,Int8}(
    "root"       => 1,  # root
    "acl"        => 2,  # clausal modifier of noun (adjectival clause)
    "advcl"      => 3,  # adverbial clause modifier
    "advmod"     => 4,  # adverbial modifier
    "amod"       => 5,  # adjectival modifier
    "appos"      => 6,  # appositional modifier
    "aux"        => 7,  # auxiliary
    "case"       => 8,  # case marking
    "cc"         => 9,  # coordinating conjunction
    "ccomp"      => 10, # clausal complement
    "clf"        => 11, # classifier
    "compound"   => 12, # compound
    "conj"       => 13, # conjunct
    "cop"        => 14, # copula
    "csubj"      => 15, # clausal subject
    "dep"        => 16, # unspecified dependency
    "det"        => 17, # determiner
    "discourse"  => 18, # discourse element
    "dislocated" => 19, # dislocated elements
    "expl"       => 20,
     # expletive
    "fixed"      => 21, # fixed multiword expression
    "flat"       => 22, # flat multiword expression
    "goeswith"   => 23, # goes with
    "iobj"       => 24, # indirect object
    "list"       => 25, # list
    "mark"       => 26, # marker
    "nmod"       => 27, # nominal modifier
    "nsubj"      => 28, # nominal subject
    "nummod"     => 29, # numeric modifier
    "obj"        => 30, # object
    "obl"        => 31, # oblique nominal
    "orphan"     => 32, # orphan
    "parataxis"  => 33, # parataxis
    "punct"      => 34, # punctuation
    "reparandum" => 35, # overridden disfluency
    "vocative"   => 36, # vocative
    "xcomp"      => 37, # open clausal complement
    )

function get_UDEPREL_dict()
    UDEPREL = Dict{String,Int8}(
    "root"       => 1,  # root
    "acl"        => 2,  # clausal modifier of noun (adjectival clause)
    "advcl"      => 3,  # adverbial clause modifier
    "advmod"     => 4,  # adverbial modifier
    "amod"       => 5,  # adjectival modifier
    "appos"      => 6,  # appositional modifier
    "aux"        => 7,  # auxiliary
    "case"       => 8,  # case marking
    "cc"         => 9,  # coordinating conjunction
    "ccomp"      => 10, # clausal complement
    "clf"        => 11, # classifier
    "compound"   => 12, # compound
    "conj"       => 13, # conjunct
    "cop"        => 14, # copula
    "csubj"      => 15, # clausal subject
    "dep"        => 16, # unspecified dependency
    "det"        => 17, # determiner
    "discourse"  => 18, # discourse element
    "dislocated" => 19, # dislocated elements
    "expl"       => 20,
     # expletive
    "fixed"      => 21, # fixed multiword expression
    "flat"       => 22, # flat multiword expression
    "goeswith"   => 23, # goes with
    "iobj"       => 24, # indirect object
    "list"       => 25, # list
    "mark"       => 26, # marker
    "nmod"       => 27, # nominal modifier
    "nsubj"      => 28, # nominal subject
    "nummod"     => 29, # numeric modifier
    "obj"        => 30, # object
    "obl"        => 31, # oblique nominal
    "orphan"     => 32, # orphan
    "parataxis"  => 33, # parataxis
    "punct"      => 34, # punctuation
    "reparandum" => 35, # overridden disfluency
    "vocative"   => 36, # vocative
    "xcomp"      => 37, # open clausal complement
    )

    return UDEPREL
end


function load_data2(path)
    xtrain, ytrain = open(path) do f
    xtrain = []
    ytrain = []
    sentence = []
    arcs = []
    count = 1
    for i in enumerate(eachline(f))
      if i[2] == ""
        push!(xtrain, sentence)
        labels = zeros(count, count)
        push!(ytrain, arcs)
      elseif i[2][1] != '#'
        temp = split(i[2])
        if temp[1] == "1"
            sentence = []
            arcs = []
            push!(sentence, temp[2])
            push!(arcs, parse(Int64, temp[7]))
            count = 1
        else
            push!(sentence, temp[2])
            if isnumeric(temp[7][1])
                push!(arcs, parse(Int64, temp[7]))
            else
                push!(arcs, 0)
            end
            count += 1
        end
      end
    end
    xtrain, ytrain
    end
    xtrain, ytrain
end


function load_data3(path, UDEPREL)
    xtrain, ytrain, deprels = open(path) do f
    xtrain = []
    ytrain = []
    deprels = []
    sentence = []
    temprels = []
    arcs = []
    count = 1
    for i in enumerate(eachline(f))
      if i[2] == ""
        push!(xtrain, sentence)
        labels = zeros(count, count)
        push!(ytrain, arcs)
        push!(deprels, temprels)
      elseif i[2][1] != '#'
        temp = split(i[2])
        if temp[1] == "1"
            sentence = []
            arcs = []
            temprels = []
            push!(sentence, temp[2])
            push!(arcs, parse(Int64, temp[7]))
            tmp = split(temp[8], ":")
            push!(temprels, get(UDEPREL, lowercase(tmp[1]), 0))
            if get(UDEPREL, lowercase(tmp[1]), 0) == 0
                println(tmp[1])
            end
            count = 1
        else
            push!(sentence, temp[2])
            tmp = split(temp[8], ":")
            push!(temprels, get(UDEPREL, lowercase(tmp[1]), 0))

            if get(UDEPREL, lowercase(tmp[1]), 0) == 0
                println(tmp[1])
            end
            
            if isnumeric(temp[7][1])
                push!(arcs, parse(Int64, temp[7]))
            else
                push!(arcs, 0)
            end
            count += 1
        end
      end
    end
    xtrain, ytrain, deprels
    end
    xtrain, ytrain, deprels
end

function load_embed(path)
    wembed, wembedind = open(path) do f
        wembed = Dict()
        wembedind = []
        for i in enumerate(eachline(f))
            line = i[2]
            tokens = split(line)
            key = tokens[1]
            temp = Array{Float32, 1}()
            for token in tokens[2:end]
                tmp = tryparse(Float32, token)
                append!(temp, tmp)
            end
            wembed[key] = i[1]
            push!(wembedind,temp)
        end
        wembed, wembedind
    end
    wembed, wembedind
end

function take_input(sentence)
    x = []
    words = split(sentence)
    for word in words
        if occursin("'", word)
            if occursin("n't", word)
                push!(x, word[1:end-3])
                push!(x, word[end-2:end])
            else
                temp = split(word, "'")
                push!(x, temp[1])
                push!(x, "'" * temp[2])
            end
            elseif (word[end] >= 'z' || word[end] <= 'A') && !isnumeric(word[end])
            push!(x, word[1:end-1])
            push!(x, word[end:end])
        else
            push!(x, word)
        end
    end
    x
end


rootind = 399999

function getind(word; max=400000, root=false)
    abc = get(wembed,lowercase(word),-1)
    if root == true
        return max-1
    elseif (abc >= 0)
        return abc
    else
        return max
    end
end

wembed, wembedind = load_embed("glove.6B.100d.txt")

lents(datdat) = map(x->length(x[1]), datdat)

function minib(data4c,batchsize)
    datdat = sort(data4c, by=x->length(x[1]), rev=true)
    lentsdatdat = lents(datdat)
    uniqdat = unique(lentsdatdat)
    lennnum = [(i,count(x->x==i,lentsdatdat),j) for (j,i) in enumerate(uniqdat)]
    i=1
    j=0
    k=0
    batdat = []
    for (a,b,c) in lennnum
        bb = b
        j += b
        j = i+b
        k=j
        while bb >= batchsize
            j =i+batchsize-1

            push!(batdat, (cat([x[1] for x in datdat[i:j]]...,dims=1), cat([x[2] for x in datdat[i:j]]...,dims=2)))
            i=j+1
            bb -= batchsize
        end
        i += bb
    end
    batdat
end


function get_dtrn_dtst()
    data2 = load_data2("en-ud-train.conllu")
    data22 = zip((reshape(x,1,1,length(x)) for x in data2[1]),data2[2])
    #data22c = collect(data22)


    data4 = ((reshape(cat(rootind,map(getind,x), dims=3),1,length(x)+1),y) for (x,y) in data22)

    data4_1 = ((x,y .+ 1) for (x,y) in data4)

    data4c = collect(data4_1)

    datmb2 = minib(data4c,25)



    #wembedmat = zeros(Float32,length(wembedind[1]), length(wembedind))
    #for i=1:length(wembedind)
    #    wembedmat[:,i] = convert(Array{Float32,1},wembedind[i])
    #end

    datmb2sh = Knet.shuffle(datmb2)

    dtrn = datmb2sh[1:400]
    dtst = datmb2sh[401:470]

    return dtrn, dtst

end

function get_dtrn_dtst_300()
    data2 = load_data2("en-ud-train.conllu")
    data22 = zip((reshape(x,1,1,length(x)) for x in data2[1]),data2[2])
    #data22c = collect(data22)


    data4 = ((reshape(cat(rootind,map(getind,x), dims=3),1,length(x)+1),y) for (x,y) in data22)

    data4_1 = ((x,y .+ 1) for (x,y) in data4)

    data4c = collect(data4_1)

    datmb2 = minib(data4c,25)



    wembedmat = zeros(Float32,length(wembedind[1]), length(wembedind))
    for i=1:length(wembedind)
        wembedmat[:,i] = convert(Array{Float32,1},wembedind[i])
    end

    datmb2sh = Knet.shuffle(datmb2)

    dtrn = datmb2sh[1:400]
    dtst = datmb2sh[401:470]

    return dtrn, dtst

end

function get_wembedmat(filename)
    wembed, wembedind = load_embed(filename)
    wembedmat = zeros(Float32,length(wembedind[1]), length(wembedind))
    for i=1:length(wembedind)
        wembedmat[:,i] = convert(Array{Float32,1},wembedind[i])


    end
    return wembedmat

end





end #end module DataLoad
