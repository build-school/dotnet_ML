
using TorchClassifier;
using TorchClassifier.NamesDatabase;

System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);

// See https://aka.ms/new-console-template for more information
Console.WriteLine("Recurrent neural network (RNN) cyrillic first name classifier\n");

//var modelFileName = Path.Combine(Directory.GetCurrentDirectory(), "model.pt");
var modelFileName = @"D:\work\test\torch\TorchClassifier\TorchClassifier\data\NamesClassifier.dat";

var classifier = new NamesClassifier
{
    ModelFileName = modelFileName
};

classifier.Run();