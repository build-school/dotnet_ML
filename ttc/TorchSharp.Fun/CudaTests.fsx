#r @"nuget: NUnit.Framework"
//#r @"nuget: TorchSharp, 0.99.3"
#r @"nuget: TorchSharp"
//#r @"nuget: TorchSharp.Fun"
#r @"E:\anibal\ttc\TorchSharp.Fun\TorchSharp.Fun\bin\Debug\net8.0\TorchSharp.Fun.dll"
//#r @"nuget: libtorch-cpu-win-x64"
//#r @"nuget: torchsharp-cuda-windows, 0.99.3"
#r @"nuget: torchsharp-cuda-windows"

open NUnit.Framework
open TorchSharp
open TorchSharp.Fun
open System

let ``model with buffer`` () =

    let device = if torch.cuda_is_available() then torch.CUDA else failwith "this test needs cuda"

    //component modules we need 

    let m1,(bufRef: Modules.Parameter ref)   =
       let l1 = torch.nn.Linear(10,10)
       let d = torch.nn.Dropout()
       let buf = new Modules.Parameter( torch.ones([|10L;10L|]),requires_grad=true)
       buf.name <- "buf"
       let bufRef: Modules.Parameter ref = ref buf
       F [] [box l1; box d; box bufRef] (fun (t: torch.Tensor) -> (t --> l1 --> d) + bufRef.Value),bufRef

    let m2 = 
        torch.nn.Linear(10,10)
        ->> torch.nn.Linear(10,10)
        ->> torch.nn.ReLU()

    let m = m1 ->> m2

    m.to'(device)

    let t1input = torch.rand([|10L|]).``to``(device)
    let t' = m.forward t1input
    ()