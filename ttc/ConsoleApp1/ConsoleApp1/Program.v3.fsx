#if INTERACTIVE
#r "nuget: torchsharp-cuda-windows" //, 0.98.1" //for gpu
#r "nuget: TorchSharp" //, 0.98.1"
#r "nuget: TfCheckpoint"   
#r "nuget: FsBERTTokenizer"
#r "nuget: FSharp.Data, 5.0.2"
#r "nuget: FsPickler.Json.nstd20, 7.0.1"
#r "nuget: FsPickler.nstd20, 7.0.1"
#endif

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


let binarySerializer = FsPickler.CreateBinarySerializer()


let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
printfn $"torch devices is %A{device}"


let weightDataDir = @"C:\anibal\ttc\MyTrainedModel\yelp_review_all_csv"
let ifLoadPretrainedModel = (DirectoryInfo weightDataDir).GetFiles().Length <> 0

//tensor dims - these values should match the relevant dimensions of the corresponding tensors in the checkpoint
let HIDDEN      = 128L
let VOCAB_SIZE  = 30522L    // see vocab.txt file included in the BERT download
let TYPE_SIZE   = 2L         // bert needs 'type' of token
let MAX_POS_EMB = 512L

//other parameters
let EPS_LAYER_NORM      = 1e-12
let HIDDEN_DROPOUT_PROB = 0.1
let N_HEADS             = 2L
let ATTN_DROPOUT_PROB   = 0.1
let ENCODER_LAYERS      = 2L
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



//Note: The module and variable names used here match the tensor name 'paths' as delimted by '/' for TF (see above), 
//for easier matching.
type BertEmbedding(ifInitWeight) as this = 
    inherit torch.nn.Module("embeddings")
    
    let word_embeddings         = torch.nn.Embedding(VOCAB_SIZE,HIDDEN,padding_idx=0L)
    let position_embeddings     = torch.nn.Embedding(MAX_POS_EMB,HIDDEN)
    let token_type_embeddings   = torch.nn.Embedding(TYPE_SIZE,HIDDEN)
    let LayerNorm               = torch.nn.LayerNorm([|HIDDEN|],EPS_LAYER_NORM)
    let dropout                 = torch.nn.Dropout(HIDDEN_DROPOUT_PROB)

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


type BertPooler() as this = 
    inherit torch.nn.Module<torch.Tensor,torch.Tensor>("pooler")

    let dense = torch.nn.Linear(HIDDEN,HIDDEN)
    let activation = torch.nn.Tanh()

    let ``:`` = torch.TensorIndex.Colon
    let first = torch.TensorIndex.Single(0L)

    do
        this.RegisterComponents()

    override _.forward (hidden_states) =
        let first_token_tensor = hidden_states.index(``:``, first) //take first token of the sequence as the pooled value
        first_token_tensor --> dense --> activation

let dense = torch.nn.Linear(HIDDEN,HIDDEN)

//dense.GetType().FullName
//torch.nn.Embedding(VOCAB_SIZE,HIDDEN,padding_idx=0L).GetType().FullName


type BertModel(ifInitWeight) as this =
    inherit torch.nn.Module("bert")

    let embeddings = new BertEmbedding(ifInitWeight)
    let pooler = new BertPooler()

    let encoderLayer = torch.nn.TransformerEncoderLayer(HIDDEN, N_HEADS, MAX_POS_EMB, ATTN_DROPOUT_PROB, activation=ENCODER_ACTIVATION)
    let encoder = torch.nn.TransformerEncoder(encoderLayer, ENCODER_LAYERS)

    do
        this.RegisterComponents()
    
    member this.forward(input_ids:torch.Tensor, token_type_ids:torch.Tensor, position_ids:torch.Tensor,?mask:torch.Tensor) =
        let src = embeddings.forward(input_ids, token_type_ids, position_ids)
        let srcBatchDim2nd = src.permute(1L,0L,2L) //PyTorch transformer requires input as such. See the Transformer docs
        let encoded = match mask with None -> encoder.forward(srcBatchDim2nd, null, null) | Some mask -> encoder.forward(srcBatchDim2nd,mask, null)
        let encodedBatchFst = encoded.permute(1L,0L,2L)
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



let BATCH_SIZE = 128
let trainBatches = trainSet |> Seq.chunkBySize BATCH_SIZE

let testBatches  = testSet  |> Seq.chunkBySize BATCH_SIZE

let vocabFile = @"C:\anibal\ttc\vocab.txt"
let vocab = Vocabulary.loadFromFile vocabFile

let position_ids = torch.arange(MAX_POS_EMB).expand(int64 BATCH_SIZE,-1L).``to``(device)


//position_ids.data<int64>() //seq [0L; 1L; 2L; 3L; ...]

let b2f (b:YelpReview): Features =
    Featurizer.toFeatures vocab true (int MAX_POS_EMB) b.Text ""

//convert a batch to input and output (X, Y) tensors
let toXY (batch:YelpReview[]) = 
    let xs = batch |> Array.map b2f
    let d_tkns      = xs |> Seq.collect (fun f -> f.InputIds )  |> Seq.toArray
    let d_tkn_typs  = xs |> Seq.collect (fun f -> f.SegmentIds) |> Seq.toArray
    let tokenIds = torch.tensor(d_tkns,     dtype=torch.int).view(-1L,MAX_POS_EMB)        
    let sepIds   = torch.tensor(d_tkn_typs, dtype=torch.int).view(-1L,MAX_POS_EMB)
    let Y = torch.tensor(batch |> Array.map (fun x->x.Label), dtype=torch.int64).view(-1L)
    (tokenIds,sepIds),Y





type BertClassification(ifInitWeight) as this = 
    inherit torch.nn.Module("BertClassification")

    let bert = new BertModel(ifInitWeight)
    let proj = torch.nn.Linear(HIDDEN,TGT_LEN)

    do
        this.RegisterComponents()
        //this.LoadBertPretrained()
        initialize_weights bert |> ignore

    member _.LoadBertPretrained(preTrainedDataFilePath) =
        load_model bert preTrainedDataFilePath 
    
    member _.forward(tknIds,sepIds,postionIds) =
        use encoded = bert.forward(tknIds,sepIds,postionIds)
        encoded --> proj 



   

let class_accuracy (y:torch.Tensor) (y':torch.Tensor) =
    use i = y'.argmax(1L)
    let i_t = Tensor.getData<int64>(i)
    let m_t = Tensor.getData<int64>(y)
    Seq.zip i_t m_t 
    |> Seq.map (fun (a,b) -> if a = b then 1.0 else 0.0) 
    |> Seq.average

//adjustment for end of data when full batch may not be available
let adjPositions currBatchSize = if int currBatchSize = BATCH_SIZE then position_ids else torch.arange(MAX_POS_EMB).expand(currBatchSize,-1L).``to``(device)

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

//evaluate on test set; return cross-entropy loss and classification accuracy
let evaluate (bertClassification:BertClassification) (ceLoss:Modules.CrossEntropyLoss) =
    bertClassification.eval()
    let lss =
        testBatches 
        |> Seq.map toXY
        |> Seq.map (fun batch ->
            let y,y',loss = processBatch bertClassification ceLoss batch
            let ls = loss.ToDouble()
            let acc = class_accuracy y y'            
            dispose [y;y';loss]
            GC.Collect()
            ls,acc)
        |> Seq.toArray
    let ls  = lss |> Seq.averageBy fst
    let acc = lss |> Seq.averageBy snd
    ls,acc




module TEST =
    
    

    let testBert = (new BertModel(not ifLoadPretrainedModel))//.``to``(device)

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




module PROD =    

    let mutable EPOCHS = 1
    let mutable verbose = true
    let _model = new BertClassification(not ifLoadPretrainedModel)


    _model.``to``(device) |> ignore
    let _loss = torch.nn.CrossEntropyLoss()

    let opt = torch.optim.Adam(_model.parameters (), 0.001, amsgrad=true)  



    let mutable e = 0

    let toxyed = 
        trainBatches 
        |> Seq.mapi (fun i b -> 
            printfn $"batch {i}"
            toXY b
        ) 
        |> Seq.take 300
        |> Seq.toArray




    let losses = 
        toxyed
        |> Seq.mapi (fun i batch ->                 
            opt.zero_grad ()   
            let y, y', loss = processBatch _model _loss batch
            let ls = loss.ToDouble()  
            loss.backward()
            _model.parameters() |> Seq.iter (fun t -> t.grad().clip(gradMin,gradMax) |> ignore)                            
            use  t_opt = opt.step ()
            if verbose && i % 100 = 0 then
                let acc = class_accuracy y y'
                printfn $"Epoch: {e}, minibatch: {i}, ce: {ls}, accuracy: {acc}"                            
            dispose [y;y';loss]
            GC.Collect()
            ls)
        |> Seq.toArray

//let evalCE,evalAcc = evaluate e
