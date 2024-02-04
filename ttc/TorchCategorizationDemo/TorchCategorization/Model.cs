using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchCategorization;

public sealed class Model : Module<Tensor, Tensor>
{
	// This model is overkill for this problem, but it's a good example of how to build a model.
	// It's loosely based on the MNIST example from the TorchSharpExamples repository.

	private readonly Module<Tensor, Tensor> fc1 = Linear(1, 32);
	private readonly Module<Tensor, Tensor> fc2 = Linear(32, 2);

	private readonly Module<Tensor, Tensor> relu1 = ReLU();
	private readonly Module<Tensor, Tensor> relu2 = ReLU();
	private readonly Module<Tensor, Tensor> relu3 = ReLU();

	private readonly Module<Tensor, Tensor> dropout1 = Dropout(0.25);
	private readonly Module<Tensor, Tensor> dropout2 = Dropout(0.25);
	private readonly Module<Tensor, Tensor> dropout3 = Dropout(0.5);

	private readonly Module<Tensor, Tensor> logsm = LogSoftmax(1);

	public Model(string name, Device? device = null) : base(name)
	{
		RegisterComponents();

		if (device != null && device.type == DeviceType.CUDA)
		{
			this.to(device);
		}
	}

	public override Tensor forward(Tensor input)
	{
		var l12 = relu1.forward(input);
		var l14 = dropout1.forward(l12);

		var l22 = relu2.forward(l14);
		var l24 = dropout2.forward(l22);

		var l31 = fc1.forward(l24);
		var l32 = relu3.forward(l31);
		var l33 = dropout3.forward(l32);

		var l41 = fc2.forward(l33);

		return logsm.forward(l41);
	}
}
