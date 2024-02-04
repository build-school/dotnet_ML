using System.Collections;
using TorchSharp;
using static TorchSharp.torch;

namespace TorchCategorization;

public sealed class DataGenerator : IEnumerable<(Tensor, Tensor)>, IDisposable
{
	/// <summary>
	/// Constructor
	/// </summary>
	/// <param name="batch_size">The batch size</param>
	/// <param name="device">The device, i.e. CPU or GPU to place the output tensors on.</param>
	public DataGenerator(double mean1, double std1, double mean2, double std2, int count, int batch_size = 32, torch.Device? device = null)
	{
		// Our data is small enough that we can generate it all in one go.

		// Go through the data and create tensors
		for (var i = 0; i < count;)
		{
			var take = Math.Min(batch_size, Math.Max(0, count - i));

			if (take < 1)
			{
				break;
			}

			using var distribution1Tensor = normal(mean1, std1, new long[] { take }, ScalarType.Float32, device: device);
			using var distribution2Tensor = normal(mean2, std2, new long[] { take }, ScalarType.Float32, device: device);
			var dataTensor = zeros(new long[] { take, 1 }, ScalarType.Float32, device: device);
			var lablTensor = zeros(new long[] { take }, ScalarType.Int64, device: device);
			for (int j = 0; j < take; j++)
			{
				if (j % 2 == 0)
				{
					dataTensor[j] = new float[] { (float)distribution1Tensor[j] };
					lablTensor[j] = 0L;
				}
				else
				{
					dataTensor[j] = new float[] { (float)distribution2Tensor[j] };
					lablTensor[j] = 1L;
				}
			}

			data.Add(dataTensor);
			labels.Add(lablTensor);

			i += take;
		}

		BatchSize = batch_size;
		Size = count;
	}

	public int Size { get; set; }

	public int BatchSize { get; private set; }

	private readonly List<Tensor> data = new();
	private readonly List<Tensor> labels = new();

	public IEnumerator<(Tensor, Tensor)> GetEnumerator()
	{
		return new DataEnumerator(data, labels);
	}

	IEnumerator IEnumerable.GetEnumerator()
	{
		return GetEnumerator();
	}

	public void Dispose()
	{
		data.ForEach(d => d.Dispose());
		labels.ForEach(d => d.Dispose());
	}

	private class DataEnumerator : IEnumerator<(Tensor, Tensor)>
	{
		public DataEnumerator(List<Tensor> data, List<Tensor> labels)
		{
			this.data = data;
			this.labels = labels;
		}

		public (Tensor, Tensor) Current
		{
			get
			{
				return curIdx == -1
					? throw new InvalidOperationException("Calling 'Current' before 'MoveNext()'")
					: (data[curIdx], labels[curIdx]);
			}
		}

		object IEnumerator.Current => Current;

		public void Dispose()
		{
		}

		public bool MoveNext()
		{
			curIdx += 1;
			return curIdx < data.Count;
		}

		public void Reset()
		{
			curIdx = -1;
		}

		private int curIdx = -1;
		private readonly List<Tensor> data;
		private readonly List<Tensor> labels;
	}
}
