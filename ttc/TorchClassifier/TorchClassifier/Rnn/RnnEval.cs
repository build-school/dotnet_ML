using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;

namespace TorchClassifier.Rnn
{
    internal class RnnEval
    {
        private RnnModule _rnn;

        public RnnEval(RnnModule rnn)
        {
            _rnn = rnn;
        }

        // Just return an output given a line
        public torch.Tensor Evaluate(torch.Tensor lineTensor)
        {
            var hidden = _rnn.InitHidden();
            torch.Tensor finalOutput = null;

            var rangeSize = lineTensor.size()[0];
            for (var i = 0; i < rangeSize; i++)
            {
                var (output, nextHidden) = _rnn.forward(lineTensor[i], hidden);
                hidden = nextHidden;
                finalOutput = output;
            }

            return finalOutput;
        }

    }
}
