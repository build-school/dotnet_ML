using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;

namespace TorchClassifier.Rnn
{
    internal class RnnTrainer
    {
        private double _learningRate = 0.009;

        private readonly RnnModule _rnn;

        private readonly torch.optim.Optimizer _optimizer;

        private readonly NLLLoss _criterion = torch.nn.NLLLoss();

        public RnnTrainer(RnnModule rnn)
        {
            _rnn = rnn;
            _criterion = torch.nn.NLLLoss();
            _optimizer = torch.optim.SGD(_rnn.parameters(), _learningRate);
        }

        public RnnTrainer(RnnModule rnn, double learningRate)
        {
            _rnn = rnn;
            _learningRate = learningRate;
            _criterion = torch.nn.NLLLoss();
            _optimizer = torch.optim.SGD(_rnn.parameters(), _learningRate);
        }


        public (torch.Tensor output, float loss) TrainNetwork(torch.Tensor categoryTensor, torch.Tensor lineTensor)
        {

            // hidden parameter:
            var hidden = _rnn.InitHidden();

            // result tensor:
            torch.Tensor finalOutput = null;

            // Clear the gradients before doing the back-propagation
            _optimizer.zero_grad();

            long rangeSize = lineTensor.size()[0];
            for (var i = 0; i < rangeSize; i++)
            {
                var (output, nextHidden) = _rnn.forward(lineTensor[i], hidden);
                finalOutput = output;
                hidden = nextHidden;
            }

            var loss = _criterion.forward(finalOutput, categoryTensor);
            loss.backward();

            // Add parameters gradients to their values, multiplied by learning rate
            //foreach (var parameter in _rnn.parameters())
            //{
            //    parameter.add_(parameter.grad(), -_learningRate);
            //}

            _optimizer.step();

            return (finalOutput, loss.item<float>());
        }
    }
}
