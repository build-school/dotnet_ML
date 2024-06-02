#if INTERACTIVE
#r "nuget: torchsharp-cuda-windows" //, 0.98.1" //for gpu
#r "nuget: TorchSharp" //, 0.98.1"
#r "nuget: TfCheckpoint"   
#r "nuget: FsBERTTokenizer"
#r "nuget: FSharp.Data, 5.0.2"
#r "nuget: FsPickler.Json.nstd20, 7.0.1"
#r "nuget: FsPickler.nstd20, 7.0.1"
#endif

(*

v4: 增加層數與參數等等

*)
open System
open System.Runtime.InteropServices
open System.Collections.Generic
open System.IO
open TfCheckpoint
open TorchSharp
open System.Collections.Concurrent
open MBrace.FsPickler.nstd20
open System.IO
open FSharp.Data
open FSharp.Collections
open BERTTokenizer


//module UnmanagedMemoryHelper =
    
//    // 分配非托管内存
//    let allocateUnmanagedMemory (size: int) : IntPtr =
//        Marshal.AllocHGlobal(size)
    
//    // 释放非托管内存
//    let freeUnmanagedMemory (ptr: IntPtr) =
//        Marshal.FreeHGlobal(ptr)
    
//    // 将 byte 数组写入到非托管内存
//    let writeBytesToUnmanagedMemory (ptr: IntPtr) (bytes: byte[]) =
//        Marshal.Copy(bytes, 0, ptr, bytes.Length)
    
//    // 从非托管内存读取 byte 数组
//    let readBytesFromUnmanagedMemory (ptr: IntPtr) (length: int) : byte[] =
//        let bytes = Array.zeroCreate<byte> length
//        Marshal.Copy(ptr, bytes, 0, length)
//        bytes

//    // 存储 byte[][] 到非托管内存
//    let storeByteArrays (byteArrays: byte[][]) =
//        // 计算所需的总内存大小
//        let totalSize = byteArrays |> Array.sumBy (fun arr -> arr.Length)
//        let ptrs = Array.zeroCreate<IntPtr> byteArrays.Length
//        let basePtr = allocateUnmanagedMemory totalSize
//        let mutable offset = 0
//        byteArrays |> Array.iteri (fun i arr ->
//            let arrPtr = IntPtr(basePtr.ToInt64() + int64 offset)
//            writeBytesToUnmanagedMemory arrPtr arr
//            ptrs.[i] <- arrPtr
//            offset <- offset + arr.Length
//        )
//        (basePtr, ptrs, byteArrays |> Array.map (fun arr -> arr.Length))

//    // 读取存储在非托管内存中的 byte[][] 数据
//    let retrieveByteArrays (ptrs: IntPtr[]) (lengths: int[]) : byte[][] =
//        ptrs |> Array.mapi (fun i ptr ->
//            readBytesFromUnmanagedMemory ptr lengths.[i]
//        )


//open UnmanagedMemoryHelper

//let byteArrays = [| [| 0uy; 1uy; 2uy |]; [| 10uy; 20uy; 30uy; 40uy |] |]
//let (basePtr, ptrs, lengths) = storeByteArrays byteArrays
//let retrievedArrays = retrieveByteArrays ptrs lengths
//retrievedArrays |> Array.iter (fun arr ->
//    printfn "Retrieved array: %A" arr
//)
//freeUnmanagedMemory basePtr










let binarySerializer = FsPickler.CreateBinarySerializer()


let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
printfn $"torch devices is %A{device}"


let ver = 3
let weightDataDir = 
    match ver with
    | 4 ->
        @"C:\anibal\ttc\MyTrainedModel\yelp_review_all_csv.v4"
    | _ ->
        @"C:\anibal\ttc\MyTrainedModel\yelp_review_all_csv"


let ifLoadPretrainedModel =     
    let di = DirectoryInfo weightDataDir
    if not di.Exists then
        di.Create()
    (DirectoryInfo weightDataDir).GetFiles().Length <> 0

//tensor dims - these values should match the relevant dimensions of the corresponding tensors in the checkpoint
let HIDDEN      = 
    match ver with
    | 4 ->
        256L
    | _ ->
        128L
let VOCAB_SIZE  = 30522L    // see vocab.txt file included in the BERT download
let TYPE_SIZE   = 2L         // bert needs 'type' of token
let MAX_POS_EMB = 512L

//other parameters
let EPS_LAYER_NORM      = 1e-12
let HIDDEN_DROPOUT_PROB = 0.1
let N_HEADS             = 2L
let ATTN_DROPOUT_PROB   = 0.1
let ENCODER_LAYERS      = 
    match ver with
    | 4 ->
        4L
    | _ ->
        2L
let ENCODER_ACTIVATION  = torch.nn.Activations.GELU

let gradCap = 0.1f
let gradMin,gradMax = (-gradCap).ToScalar(),  gradCap.ToScalar()

// 定义权重初始化函数
let initialize_weights (module_: torch.nn.Module) =
    torch.set_grad_enabled(false) |> ignore// 禁用梯度跟踪
    module_.apply (fun m ->
        match m with
        | :? TorchSharp.Modules.Linear as linear ->
            linear.weight.normal_(0.0, 0.02) |> ignore
            if not (isNull linear.bias) then
                linear.bias.zero_() |> ignore
        | :? TorchSharp.Modules.Embedding as embedding ->
            embedding.weight.normal_(0.0, 0.02) |> ignore
        | :? TorchSharp.Modules.LayerNorm as layer_norm ->
            layer_norm.weight.fill_(1.0) |> ignore
            layer_norm.bias.zero_() |> ignore
        | :? TorchSharp.Modules.Dropout ->
            // Dropout 层没有权重，不需要初始化
            ()
        | _ -> ()
    )  |> ignore
    torch.set_grad_enabled(true)  |> ignore // 重新启用梯度跟踪


let load_model (model: torch.nn.Module) (fileDir: string) =
    let stateDict = Dictionary<string, torch.Tensor>()
    model.state_dict(stateDict) |> ignore

    stateDict
    |> Seq.iter (fun kvp ->
        let filePath = Path.Combine(fileDir, kvp.Key.Replace('/', '_') + ".pt")
        if File.Exists(filePath) then
            let tensor = torch.load(filePath)
            stateDict.[kvp.Key] <- tensor
        else
            printfn "Warning: %s not found" filePath
    )
    
    model.load_state_dict(stateDict) |> ignore

let save_model (model: torch.nn.Module) (fileDir: string) =
    //use stream = System.IO.File.Create(filePath)
    let di = DirectoryInfo fileDir
    if not di.Exists then
        di.Create()
    let stateDict = Dictionary<string, torch.Tensor> ()

    model.state_dict(stateDict) |> ignore
    stateDict
    |> Seq.iter (fun kvp ->
        torch.save(kvp.Value,Path.Join(fileDir, kvp.Key.Replace('/', '_') + ".pt"))        
    )


//Note: The module and variable names used here match the tensor name 'paths' as delimted by '/' for TF (see above), 
//for easier matching.
type BertEmbedding(ifInitWeight,dev:torch.Device) as this = 
    inherit torch.nn.Module("embeddings")
    
    let word_embeddings         = torch.nn.Embedding(VOCAB_SIZE,HIDDEN,padding_idx=0L,device=dev)
    let position_embeddings     = torch.nn.Embedding(MAX_POS_EMB,HIDDEN,device=dev)
    let token_type_embeddings   = torch.nn.Embedding(TYPE_SIZE,HIDDEN,device=dev)
    let LayerNorm               = torch.nn.LayerNorm([|HIDDEN|],EPS_LAYER_NORM,device=dev)
    let dropout                 = torch.nn.Dropout(HIDDEN_DROPOUT_PROB).``to``(dev)

    do 
        this.RegisterComponents()
        if ifInitWeight then
            this.initWeight ()

    member this.initWeight () =
        initialize_weights this
        

    member this.forward(input_ids:torch.Tensor, token_type_ids:torch.Tensor, position_ids:torch.Tensor) =   
    
        let embeddings =      
            (input_ids       --> word_embeddings)        +
            (token_type_ids  --> token_type_embeddings)  +
            (position_ids    --> position_embeddings)

        embeddings --> LayerNorm --> dropout             // the --> operator works for simple 'forward' invocations


type BertPooler(dev:torch.Device) as this = 
    inherit torch.nn.Module<torch.Tensor,torch.Tensor>("pooler")

    let dense = torch.nn.Linear(HIDDEN,HIDDEN,device=dev)
    let activation = torch.nn.Tanh().``to``(dev)

    let ``:`` = torch.TensorIndex.Colon
    let first = torch.TensorIndex.Single(0L)

    do
        this.RegisterComponents()

    override _.forward (hidden_states) =
        let first_token_tensor = hidden_states.index(``:``, first) //take first token of the sequence as the pooled value
        //task { printfn "first_token_tensor: %A" first_token_tensor.device} |> ignore
        first_token_tensor --> dense --> activation

//let dense = torch.nn.Linear(HIDDEN,HIDDEN)

//dense.GetType().FullName
//torch.nn.Embedding(VOCAB_SIZE,HIDDEN,padding_idx=0L).GetType().FullName


type BertModel(ifInitWeight,dev:torch.Device) as this =
    inherit torch.nn.Module("bert")

    let embeddings = new BertEmbedding(ifInitWeight, dev)
    let pooler = new BertPooler(dev)

    let encoderLayer = torch.nn.TransformerEncoderLayer(HIDDEN, N_HEADS, MAX_POS_EMB, ATTN_DROPOUT_PROB, activation=ENCODER_ACTIVATION).``to``(dev)    
    let encoder = torch.nn.TransformerEncoder(encoderLayer, ENCODER_LAYERS).``to``(dev)
    

    do
        this.RegisterComponents()
    
    member this.forward(input_ids:torch.Tensor, token_type_ids:torch.Tensor, position_ids:torch.Tensor,?mask:torch.Tensor) =
        let src = embeddings.forward(input_ids, token_type_ids, position_ids)
        //task { printfn "src: %A" src.device} |> ignore
        let srcBatchDim2nd = src.permute(1L,0L,2L) //PyTorch transformer requires input as such. See the Transformer docs
        //task { printfn "srcBatchDim2nd: %A" srcBatchDim2nd.device} |> ignore
        let encoded = match mask with None -> encoder.forward(srcBatchDim2nd, null, null) | Some mask -> encoder.forward(srcBatchDim2nd,mask, null)
        //task { printfn "encoded: %A" encoded.device} |> ignore
        let encodedBatchFst = encoded.permute(1L,0L,2L)
        //task { printfn "encodedBatchFst: %A" encodedBatchFst.device} |> ignore
        encodedBatchFst --> pooler


//pm0.GetType().Name //string
//pm1.GetType().Name

//pm1.name

open System
module Tensor = 
    //Note: ensure 't matches tensor datatype otherwise ToArray might crash the app (i.e. exception cannot be caught)
    let private _getData<'t when 't:>ValueType and 't: unmanaged and 't:struct and 't : (new:unit->'t) > (t:torch.Tensor) =
        let s = t.data<'t>()
        s.ToArray()

    let getData<'t when 't:>ValueType and 't: unmanaged and 't:struct and 't : (new:unit->'t)>  (t:torch.Tensor) =
        if t.device_type <> DeviceType.CPU then 
            //use t1 = t.clone()
            use t2 = t.cpu()
            _getData<'t> t2
        else 
            _getData<'t> t
  
    let setData<'t when 't:>ValueType and 't: unmanaged and 't:struct and 't : (new:unit->'t)> (t:torch.Tensor) (data:'t[]) =
        if t.device_type = DeviceType.CPU |> not then failwith "tensor has to be on cpu for setData"        
        let s = t.data<'t>()
        s.CopyFrom(data,0,0L)



let foldr = @"C:\anibal\ttc\yelp_review_all_csv"
let testCsv = Path.Combine(foldr,"test.csv")
let trainCsv = Path.Combine(foldr,"train.csv")
if File.Exists testCsv |> not then failwith $"File not found; path = {testCsv}"
printfn "%A" trainCsv


type YelpCsv = FSharp.Data.CsvProvider<Sample="a,b", HasHeaders=false, Schema="Label,Text">
type [<CLIMutable>] YelpReview = {Label:int; Text:string}
//need to make labels 0-based so subtract 1
let testSet = YelpCsv.Load(testCsv).Rows |> Seq.map (fun r-> {Label=int r.Label - 1; Text=r.Text}) |> Seq.toArray 
let trainSet = YelpCsv.Load(trainCsv).Rows |> Seq.map (fun r->{Label=int r.Label - 1; Text=r.Text}) |> Seq.toArray

let classes = trainSet |> Seq.map (fun x->x.Label) |> set
//classes.Display()
let TGT_LEN = classes.Count |> int64



let BATCH_SIZE = 
    match ver with
    | 4 ->
        256
    | _ ->
        128
let trainBatches = trainSet |> Array.chunkBySize BATCH_SIZE

let testBatches  = testSet  |> Array.chunkBySize BATCH_SIZE

let vocabFile = @"C:\anibal\ttc\vocab.txt"
let vocab = Vocabulary.loadFromFile vocabFile

let position_ids = torch.arange(MAX_POS_EMB,device=device).expand(int64 BATCH_SIZE,-1L)//.``to``(device)


//position_ids.data<int64>() //seq [0L; 1L; 2L; 3L; ...]

let inline b2f (b:YelpReview): Features =
    Featurizer.toFeatures vocab true (int MAX_POS_EMB) b.Text ""

type ContainerNoNeedGC () =
    static member val batchDict = new ConcurrentDictionary<string * int, YelpReview[]>() with get, set
    static member val featureDict = new ConcurrentDictionary<string * int, Features[]>() with get, set
    static member val labelDict = new ConcurrentDictionary<string * int, int[]>() with get, set


//Container.batchDict <- null
//Container.featureDict <- null
//Container.labelDict <- null

type Container () =
    static member batchDict = new ConcurrentDictionary<string * int, YelpReview[]>() 
    static member featureDict = new ConcurrentDictionary<string * int, Features[]>() 
    static member labelDict = new ConcurrentDictionary<string * int, int[]>() 

let featurePickle = Path.Join($"{weightDataDir}.pickle","feature")
let labelPickle = Path.Join($"{weightDataDir}.pickle","label")



//let (trainBasePtr, trainPtrs, trainLengths) = 
//    if (DirectoryInfo featurePickle).GetFiles().Length = 0 then //表示 feature pickle 是空的
//        storeByteArrays (
//            trainBatches 
//            |> Array.map (Array.map b2f)
//            |> Array.map binarySerializer.Pickle)
//    else
//        failwith "not yet support read"


let trainSetIDs =
    trainBatches
    |> Array.mapi (fun i c -> 
        //有了 unsafe 之後不用了 //
        let r = Container.batchDict.TryAdd (("train", i), c)
        //printfn $"{r}"
        "train", i
    )

//Container.batchDict

//Container.batchDict.TryAdd (("train", 123), [||])

let testSetIDs =
    testBatches
    |> Array.mapi (fun i c -> 
        //有了 unsafe 之後不用了 //
        let _ = Container.batchDict.TryAdd (("test", i), c)
        "test", i
    )

 //task {printfn "2 %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))} |> ignore
 //GC.Collect()
 //task {printfn "3 %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))} |> ignore


let inline toFeature (batchId:string * int) = 
    Container.featureDict.GetOrAdd(
        batchId
        , fun k -> 
            let fi = Path.Join(featurePickle, string (snd k) + ".pickle") |> FileInfo
            if fi.Exists then
                binarySerializer.UnPickle (File.ReadAllBytes fi.FullName)
            else
                let f = Container.batchDict[k] |> Array.map b2f
                File.WriteAllBytes(fi.FullName, binarySerializer.Pickle f)                
                f
    )

//let trainX = trainBatches |> Array.map (Array.map b2f)
//let testX = testBatches |> Array.map (Array.map b2f)
    

let inline toLabel (batchId:string * int) = 
    Container.labelDict.GetOrAdd(
        batchId
        , fun k -> 
            let fi = Path.Join(labelPickle, string (snd k) + ".pickle") |> FileInfo
            if fi.Exists then
                binarySerializer.UnPickle (File.ReadAllBytes fi.FullName)
            else
                let l = Container.batchDict[k] |> Array.map (fun x->x.Label)
                File.WriteAllBytes(fi.FullName, binarySerializer.Pickle l)                
                l
    )
    
//convert a batch to input and output (X, Y) tensors
let toXY (batch:YelpReview[]) = 
    let xs = batch |> Array.map b2f
    let d_tkns      = xs |> Seq.collect (fun f -> f.InputIds )  |> Seq.toArray
    let d_tkn_typs  = xs |> Seq.collect (fun f -> f.SegmentIds) |> Seq.toArray
    let tokenIds = torch.tensor(d_tkns,     dtype=torch.int).view(-1L,MAX_POS_EMB)        
    let sepIds   = torch.tensor(d_tkn_typs, dtype=torch.int).view(-1L,MAX_POS_EMB)
    let Y = torch.tensor(batch |> Array.map (fun x->x.Label), dtype=torch.int64).view(-1L)
    (tokenIds, sepIds), Y




let toXY2 dev (batchId:string * int)  =
    let xs = toFeature batchId
    let d_tkns      = xs |> Seq.collect (fun f -> f.InputIds )  |> Seq.toArray
    let d_tkn_typs  = xs |> Seq.collect (fun f -> f.SegmentIds) |> Seq.toArray
    let tokenIds = torch.tensor(d_tkns,     dtype=torch.int).view(-1L,MAX_POS_EMB)        
    let sepIds   = torch.tensor(d_tkn_typs, dtype=torch.int).view(-1L,MAX_POS_EMB)
    let Y = torch.tensor(toLabel batchId, dtype=torch.int64).view(-1L)
    (tokenIds,sepIds),Y


let toXY3 (dev:torch.Device) (xs:Features[]) (labels:int[]) =
    let d_tkns      = xs |> Seq.collect (fun f -> f.InputIds )  |> Seq.toArray
    let d_tkn_typs  = xs |> Seq.collect (fun f -> f.SegmentIds) |> Seq.toArray
    let tokenIds = torch.tensor(d_tkns,     dtype=torch.int, device=dev).view(-1L,MAX_POS_EMB).``to``(dev)       
    let sepIds   = torch.tensor(d_tkn_typs, dtype=torch.int, device=dev).view(-1L,MAX_POS_EMB).``to``(dev)     
    let Y = torch.tensor(labels, dtype=torch.int64, device=dev).view(-1L).``to``(dev)     
    (tokenIds,sepIds),Y


let inline toXY4 (dev:torch.Device, d_tkns:int[], d_tkn_typs:int[], labels:int[]) =    
    let tokenIds = torch.tensor(d_tkns,     dtype=torch.int, device=dev).view(-1L,MAX_POS_EMB)//.``to``(dev)       
    let sepIds   = torch.tensor(d_tkn_typs, dtype=torch.int, device=dev).view(-1L,MAX_POS_EMB)//.``to``(dev)     
    let Y = torch.tensor(labels, dtype=torch.int64, device=dev).view(-1L)//.``to``(dev)     
    (tokenIds,sepIds),Y




type BertClassification(ifInitWeight,dev:torch.Device) as this = 
    inherit torch.nn.Module("BertClassification")

    let bert = new BertModel(ifInitWeight, dev)


    let intermediate = 
        match ver with
        | 4 ->
            torch.nn.Linear(HIDDEN, HIDDEN*2L, device=dev) 
        | _ ->
            null
    let dropout2 = 
        match ver with
        | 4 ->
            torch.nn.Dropout(0.1).``to``(dev) 
        | _ ->
            null

    let proj = 
        match ver with
        | 4 ->
            torch.nn.Linear(HIDDEN*2L,TGT_LEN,device=dev)
        | _ ->
            torch.nn.Linear(HIDDEN,TGT_LEN,device=dev)

            

    do
        this.RegisterComponents()
        if ifInitWeight then
            initialize_weights bert |> ignore

    member _.LoadBertPretrained(preTrainedDataFilePath) =
        load_model this preTrainedDataFilePath 
    
    member _.forward(tknIds,sepIds,postionIds) =
        use encoded = bert.forward(tknIds,sepIds,postionIds)
        match ver with
        | 4 ->
            encoded --> intermediate --> dropout2 --> proj 
            //encoded --> intermediate --> proj 
        | _ ->
            encoded --> proj 


   

let class_accuracy (y:torch.Tensor) (y':torch.Tensor) =
    use i = y'.argmax(1L)
    let i_t = Tensor.getData<int64>(i)
    let m_t = Tensor.getData<int64>(y)
    Seq.zip i_t m_t 
    |> Seq.map (fun (a,b) -> if a = b then 1.0 else 0.0) 
    |> Seq.average

//adjustment for end of data when full batch may not be available
let adjPositions currBatchSize = if int currBatchSize = BATCH_SIZE then position_ids else torch.arange(MAX_POS_EMB,device=device).expand(currBatchSize,-1L)//.``to``(device)

let dispose ls = ls |> List.iter (fun (x:IDisposable) -> x.Dispose())

//run a batch through the model; return true output, predicted output and loss tensors
let processBatch (bertClassification:BertClassification) (ceLoss:Modules.CrossEntropyLoss) ((tkns:torch.Tensor,typs:torch.Tensor), y:torch.Tensor) =
    use tkns_d = tkns.``to``(device)
    use typs_d = typs.``to``(device)
    let y_d    = y.``to``(device)            
    let positions  = adjPositions tkns.shape.[0]
    if device <> torch.CPU then //these were copied so ok to dispose old tensors
        dispose [tkns; typs; y]
    let y' = bertClassification.forward(tkns_d, typs_d, positions)
    let loss = ceLoss.forward(y', y_d)   
    y_d, y', loss


let processBatch2 (bertClassification:BertClassification) (ceLoss:Modules.CrossEntropyLoss) (tkns:torch.Tensor) (typs:torch.Tensor) (y:torch.Tensor) =
    //use tkns_d = tkns.``to``(device)
    //use typs_d = typs.``to``(device)
    //let y_d    = y.``to``(device)            
    use positions  = adjPositions tkns.shape.[0]
    //if device <> torch.CPU then //these were copied so ok to dispose old tensors
    //    dispose [tkns; typs; y]
    let y' = bertClassification.forward(tkns, typs, positions)
    let loss = ceLoss.forward(y', y)   
    y', loss

//evaluate on test set; return cross-entropy loss and classification accuracy
let evaluate (bertClassification:BertClassification) (ceLoss:Modules.CrossEntropyLoss) =
    bertClassification.eval()
    let lss =
        testBatches 
        |> Seq.map toXY
        |> Seq.map (fun batch ->
            let y, y',loss = processBatch bertClassification ceLoss batch
            let ls = loss.ToDouble()
            let acc = class_accuracy y y'            
            dispose [y;y';loss]
            GC.Collect()
            ls,acc)
        |> Seq.toArray
    let ls  = lss |> Seq.averageBy fst
    let acc = lss |> Seq.averageBy snd
    ls,acc




module PROD =    

    let mutable EPOCHS = 10
    let mutable verbose = true
    let _model = new BertClassification(not ifLoadPretrainedModel, device)
    _model.``to``(device) |> ignore
    if ifLoadPretrainedModel then
        _model.LoadBertPretrained weightDataDir
    let _loss = torch.nn.CrossEntropyLoss()
    _loss.``to``(device) |> ignore
    let _opt = torch.optim.Adam(_model.parameters (), 0.001, amsgrad=true)  
    _opt.``to``(device) |> ignore

    let mutable e = 0
    

    let toxyed4 = 
        //let fi = FileInfo @$"{featurePickle}\all.pickler"
        //if fi.Exists then
        //    binarySerializer.UnPickle <| File.ReadAllBytes fi.FullName
        //else
            let arr = 
                trainSetIDs 
                |> Array.take 300
                |> Array.mapi (fun i b -> 
                    if i % 20 = 0 then
                        printfn $"batch {b}"
                    let fs = toFeature b
                    let ls = toLabel b
                    let d_tkns      = fs |> Seq.collect (fun f -> f.InputIds )  |> Seq.toArray
                    let d_tkn_typs  = fs |> Seq.collect (fun f -> f.SegmentIds) |> Seq.toArray
                    d_tkns, d_tkn_typs, ls
                ) 
            //Container.batchDict <- null
            //Container.featureDict <- null
            //Container.labelDict <- null
            GC.Collect()
            //File.WriteAllBytes (fi.FullName, binarySerializer.Pickle arr)
            arr
        //|> Array.map toXY4


    //task {printfn "2 %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))} |> ignore
    //GC.Collect()
    //task {printfn "3 %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))} |> ignore
    let train () =
    
        while e < EPOCHS do
            e <- e + 1
            _model.train()
            let losses4 = 
                toxyed4
                |> Seq.mapi (fun i (d_tkns, d_tkn_typs, labels) ->   
            
                    _opt.zero_grad ()   
                    use tokenIds = torch.tensor(d_tkns,     dtype=torch.int, device=device).view(-1L,MAX_POS_EMB)//.``to``(dev)       
                    use sepIds   = torch.tensor(d_tkn_typs, dtype=torch.int, device=device).view(-1L,MAX_POS_EMB)//.``to``(dev)     
                    use y = torch.tensor(labels, dtype=torch.int64, device=device).view(-1L)//.``to``(dev)     

                    let positions  = adjPositions tokenIds.shape.[0]

                    use y' = _model.forward(tokenIds, sepIds, positions)
                    use loss = _loss.forward(y', y)   
                    //let yl = processBatch2 _model _loss tokenIds sepIds y
                    //use y' = fst yl
                    //use loss = snd yl
                    let ls = loss.ToDouble()  
                    loss.backward()
                    _model.parameters() 
                    |> Seq.iter (fun t -> 
                        use _ = t.grad().clip(gradMin,gradMax) 
                        ()
                        )
                    use  t_opt = _opt.step ()
                    if verbose && i % 100 = 0 then
                        let acc = class_accuracy y y'
                        task {printfn "1 %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))} |> ignore
                        printfn $"Epoch: {e}, minibatch: {i}, ce: {ls}, accuracy: {acc}"  
                        //GC.Collect()
                    dispose [y; y';loss]
                    //task {printfn "2 %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))} |> ignore
                    GC.Collect()
                    //task {printfn "3 %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))} |> ignore
                    ls
                    )
                |> Seq.toArray
            let evalCE, evalAcc = evaluate _model _loss
            printfn $"Epoch {e} train: {Seq.average losses4}; eval acc: {evalAcc}"
            
            save_model _model weightDataDir


        printfn "Done train"
  
  
open PROD
train ()

(*
module TEST =
    
    

    let testBert = (new BertModel(not ifLoadPretrainedModel, torch.CPU))//.``to``(device)

    if ifLoadPretrainedModel then
        load_model testBert weightDataDir
    //bert.named_modules() 
    testBert.named_parameters() |> Seq.map (fun struct(n,x) -> n,x.shape) |> Seq.iter (printfn "%A")


    let (struct (pm0, pm1)) = testBert.named_parameters() |> Seq.item 0

    
    //loadWeights testBert tensors (int ENCODER_LAYERS) nameMap


    //let l0w = testBert.get_parameter("encoder.layers.0.self_attn.in_proj_weight") |> Tensor.getData<float32>

    testBert.eval()
    let (_tkns,_seps),_ = trainBatches |> Seq.head |> toXY
    //_tkns.shape
    //_tkns |> Tensor.getData<int64>
    let _testOut = testBert.forward(_tkns,_seps,position_ids.cpu()) //test is on cpu

//*)

