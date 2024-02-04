using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchCategorization;

public static class Categorization
{
	private static int _epochs = 4;
	private static int _trainBatchSize = 64;
	private static int _testBatchSize = 128;

	private readonly static int _logInterval = 100;

	internal static void Run(int epochs, int timeout, string? logdir, string? dataset)
	{
		_epochs = epochs;

		if (string.IsNullOrEmpty(dataset))
		{
			dataset = "default";
		}

		var device = cuda.is_available() ? CUDA : CPU;

		Console.WriteLine();
		Console.WriteLine($"\tRunning categorization with {dataset} on {device.type} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
		Console.WriteLine();

		random.manual_seed(1);

		var writer = string.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

		if (device.type == DeviceType.CUDA)
		{
			_trainBatchSize *= 4;
			_testBatchSize *= 4;
		}

		Console.WriteLine($"\tCreating the model...");

		var model = new Model("model", device);

		Console.WriteLine($"\tPreparing training and test data...");
		Console.WriteLine();

		using (DataGenerator train = new DataGenerator(-1, 1, 1, 1, 1024, _trainBatchSize, device: device),
							test = new DataGenerator(-1, 1, 1, 1, 256, _testBatchSize, device: device))
		{

			TrainingLoop(dataset, timeout, writer, device, model, train, test);
		}
	}

	internal static void TrainingLoop(string dataset, int timeout, TorchSharp.Modules.SummaryWriter? writer, Device device, Module<Tensor, Tensor> model, DataGenerator train, DataGenerator test)
	{
		var optimizer = optim.Adam(model.parameters());

		var scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.7);

		Stopwatch totalTime = Stopwatch.StartNew();

		for (var epoch = 1; epoch <= _epochs; epoch++)
		{

			Train(model, optimizer, NLLLoss(reduction: Reduction.Mean), device, train, epoch, train.BatchSize, train.Size);
			Test(model, NLLLoss(reduction: nn.Reduction.Sum), writer, device, test, epoch, test.Size);

			Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");

			if (totalTime.Elapsed.TotalSeconds > timeout) break;
		}

		totalTime.Stop();
		Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");

		Console.WriteLine("Saving model to '{0}'", dataset + ".model.bin");
		model.save(dataset + ".model.bin");
	}

	private static void Train(
		Module<Tensor, Tensor> model,
		optim.Optimizer optimizer,
		Loss<Tensor, Tensor, Tensor> loss,
		Device device,
		IEnumerable<(Tensor, Tensor)> dataLoader,
		int epoch,
		long batchSize,
		int size)
	{
		model.train();

		int batchId = 1;

		Console.WriteLine($"Epoch: {epoch}...");

		foreach (var (data, target) in dataLoader)
		{
			using (var d = torch.NewDisposeScope())
			{
				optimizer.zero_grad();

				var prediction = model.forward(data);
				var output = loss.forward(prediction, target);

				output.backward();

				optimizer.step();

				if (batchId % _logInterval == 0)
				{
					Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle():F4}");
				}

				batchId++;
			}

		}
	}

	private static void Test(
		Module<Tensor, Tensor> model,
		Loss<Tensor, Tensor, Tensor> loss,
		TorchSharp.Modules.SummaryWriter? writer,
		Device device,
		IEnumerable<(Tensor, Tensor)> dataLoader,
		int epoch,
		int size)
	{
		model.eval();

		double testLoss = 0;
		int correct = 0;

		foreach (var (data, target) in dataLoader)
		{
			using (var d = torch.NewDisposeScope())
			{
				var prediction = model.forward(data);
				var output = loss.forward(prediction, target);
				testLoss += output.ToSingle();

				correct += prediction.argmax(1).eq(target).sum().ToInt32();
			}
		}

		Console.WriteLine($"Size: {size}, Total: {size}");

		Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");

		if (writer != null)
		{
			writer.add_scalar("MNIST/loss", (float)(testLoss / size), epoch);
			writer.add_scalar("MNIST/accuracy", (float)correct / size, epoch);
		}
	}
}