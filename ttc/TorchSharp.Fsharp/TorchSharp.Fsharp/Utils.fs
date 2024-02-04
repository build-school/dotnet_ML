module FsUtils
open TorchSharp
open TorchSharp.Tensor
open TorchSharp.NN

    module Tuple2 =
        let map f (a, b) = (f a, f b)
 
