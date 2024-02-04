using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;

namespace TorchClassifier.Rnn
{
    internal sealed class RnnModule : torch.nn.Module
    {
        // уровень сети 1.1
        private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _i2h;

        // уровень сети 1.2
        private readonly torch.nn.Module<torch.Tensor, torch.Tensor> _i2o;

        // уровень softmax
        private readonly LogSoftmax _softmax;

        // hidden parameters size (128 by default):
        private int _hiddenSize;

        public RnnModule(int inputSize, int hiddenSize, int outputSize) : base(nameof(RnnModule))
        {
            _hiddenSize = hiddenSize;
            _i2h = torch.nn.Linear(inputSize + hiddenSize, hiddenSize);
            _i2o = torch.nn.Linear(inputSize + hiddenSize, outputSize);
            _softmax = torch.nn.LogSoftmax(dim: 1);
            RegisterComponents();
        }


        public (torch.Tensor, torch.Tensor) forward(torch.Tensor input, torch.Tensor hidden)
        {
            var combined = torch.cat(new List<torch.Tensor> { input, hidden }, 1);
            var outputHidden = _i2h.forward(combined);
            var output = _i2o.forward(combined);
            output = _softmax.forward(output);
            return (output, outputHidden);
        }

        public torch.Tensor InitHidden()
        {
            return torch.zeros(1, _hiddenSize);
        }
    }
}
