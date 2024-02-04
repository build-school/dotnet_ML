namespace TorchSharp.FSharp

open TorchSharp
open TorchSharp.Tensor
open FsUtils

module FsModule =
    let Forward y (x : NN.ProvidedModule) = x.Forward(y)

    let ZeroGrad (x:NN.Sequential)  = x.ZeroGrad()

    let Linear (inputS : int64) (outputS : int64) (withBias : bool) =
        NN.Module.Linear (inputS, outputS, withBias)

    let Conv2D (inputChannel:int64)(outputChannel:int64)(kernelSize:int64)(stride:int64)(padding:int64)=
        NN.Module.Conv2D (inputChannel, outputChannel, kernelSize, stride, padding)

    let RelU(inplace : bool) = NN.Module.Relu inplace

    let MaxPool2D (kernelSize : list<int64>) (stride : list<int64>) =
        (kernelSize, stride)
        |> Tuple2.map List.toArray
        |> NN.Module.MaxPool2D


    let Sequential (modules:array<NN.Module>) = 
        NN.Module.Sequential(modules)

    
    let AdaptiveAvgPool2D(outputSize : list<int64>) =
        outputSize
        |> List.toArray
        |> NN.Module.AdaptiveAvgPool2D

    let LogSoftMax (dimension : int64) (tensor : TorchTensor) =
        NN.Module.LogSoftMax dimension

    let Dropout (isTraining:bool) (probability:double)=
        NN.Module.Dropout (isTraining, probability)
    
    let DropoutDefault (isTraining:bool) =
        Dropout isTraining 0.5
    
    let FeatureDropout = 
        NN.Module.FeatureDropout

    

    

    
