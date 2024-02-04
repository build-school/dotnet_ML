namespace TorchCategorization;

internal static class Program
{
	static void Main(string[] args)
	{
		Categorization.Run(5, 60, null, null);
		Console.WriteLine("Done!");
	}
}
