using System.Reflection;
using TorchClassifier.NamesDatabase;
using TorchClassifier.Rnn;
using TorchSharp;
using TorchSharp.Modules;

namespace TorchClassifier
{
    public class NamesClassifierRecord
    {
        public string Name { get; set; } = string.Empty;

        public string Sex { get; set; } = string.Empty;
    }

    internal class NamesClassifier
    {
        // size of hidden parameters
        private readonly int _hiddenSize = 128;

        // If you set this too high, it might explode. If too low, it might not learn
        private readonly double _learningRate = 0.005;

        // maximum training count
        private readonly int _maxTrainCount = 100000;

        private List<string> _categories;

        private List<NamesClassifierRecord> _records;

        public const string DefaultModelFileName = "NamesClassifier.dat";

        public string ModelFileName { get; set; } = string.Empty;


        private torch.Tensor LineToTensor(string strLine)
        {
            var tensor = torch.zeros(strLine.Length, 1, NameLetters.All.Count());
            for (int li = 0; li < strLine.Length; li++)
            {
                tensor[li][0][NameLetters.IndexOf(strLine[li])] = 1;
            }
            return tensor;
        }

        private (string, int) GetCategoryFromOutput(torch.Tensor output)
        {
            // torch.Tensor out of Variable with.data
            var (top_n, top_i) = output.topk(1);
            var categoryIndex = top_i[0][0];

            var res_index = categoryIndex.item<long>();
            // var res_index = categoryIndex.ToScalar().ToInt32();
            //var res_index = categoryIndex;
            return (_categories[(int)res_index], (int)res_index);
        }

        private string RandomString(IList<string> strings)
        {
            var randomIndex = torch.randint(0, strings.Count - 1, new torch.Size((10, 1)));
            var someIndex = randomIndex[0].item<long>();
            return strings[(int)someIndex];
        }

        private (int, NamesClassifierRecord) RandomRecord()
        {
            // var randomIndex = torch.randint(0, _records.Count - 1, new torch.Size((10, 1)));
            var randomIndex = torch.randint(0, _records.Count - 1, new torch.Size((1, 1)));
            var someIndex = randomIndex[0].item<long>();
            return ((int)someIndex, _records[(int)someIndex]);
        }



        public (torch.Tensor output, float loss) Train(RnnTrainer rnnTrainer, string category, string line)
        {
            var categoryIndex = (long)_categories.IndexOf(category);
            var categoryTensor = torch.as_tensor(new List<long> { categoryIndex }, torch.ScalarType.Int64);
            return rnnTrainer.TrainNetwork(categoryTensor, LineToTensor(line));
        }

        public IEnumerable<(float, string)> Predict(RnnModule rnn, string line, int maxPredictions = 2)
        {
            var eval = new RnnEval(rnn);
            var output = eval.Evaluate(LineToTensor(line));

            //  Get top N categories
            var (topv, topi) = output.topk(maxPredictions, 1, true);
            var predictions = new List<(float, string)>();

            for (var i = 0; i < maxPredictions; i++)
            {
                var value = topv[0][i].item<float>();
                var categoryIndex = topi[0][i];
                var numCategoryIndex = categoryIndex.ToScalar().ToInt32();

                Console.WriteLine($"{_categories[numCategoryIndex]} => ({value})");
                predictions.Add((value, _categories[numCategoryIndex]));
            }

            return predictions;
        }


        public void LoadRecords()
        {
            var namesDatabase = new CommonNamesDatabase();
            namesDatabase.Load();

            _records = namesDatabase.Records
                .Where(x => !string.IsNullOrEmpty(x.Sex))
                .Select(x => new NamesClassifierRecord
                {
                    Sex = x.Sex ?? string.Empty,
                    Name = x.Name,
                })
                .ToList();
            _categories = _records.GroupBy(x => x.Sex).Select(x => x.Key).ToList();
        }



        public void Run()
        {
            Console.WriteLine("Initializing model...");
            //var rnn = new RnnModule(NameLetters.All.Length, _hiddenSize, _categories.Count);
            var rnn = new RnnModule(NameLetters.All.Length, _hiddenSize, 2);

            if (File.Exists(ModelFileName))
            {
                Console.WriteLine("Loading previously trained model...\n");
                _categories = new List<string> { "M", "F" };
                rnn.load(ModelFileName);
            }
            else
            {
                Console.WriteLine("Loading names database data...");
                LoadRecords();

                Console.WriteLine("Training neural network...\n");
                var trainer = new RnnTrainer(rnn, _learningRate);

                for (var i = 0; i < _maxTrainCount; i++)
                {
                    var (randomIndex, randomRecord) = RandomRecord();
                    var (output, loss) = Train(trainer, randomRecord.Sex, randomRecord.Name);
                    if (i % 1000 == 0)
                    {
                        var percent = (i * 100) / _maxTrainCount;
                        var (guess, guessIndex) = GetCategoryFromOutput(output);
                        var correct = guess == randomRecord.Sex ? "+" : $"- ({randomRecord.Sex})";
                        Console.WriteLine(
                            $"{percent:D}% ({i} of {_maxTrainCount}) loss={loss} line[{randomIndex}]={randomRecord.Name} / {guess} {correct}");
                    }
                }

                Console.WriteLine("Saving model...");
                if (string.IsNullOrEmpty(ModelFileName))
                {
                    ModelFileName = Path.Combine(Directory.GetCurrentDirectory(), DefaultModelFileName);
                }

                Console.WriteLine($"Saving model to \"{ModelFileName}\"");
                rnn.save(ModelFileName);
            }

            var testNames = new [] { "Марьяна", "Лена", "Ульяна", "Зубаригет", "Антон", "Сергей", "Света", "Мика", "Женя", "Руслан", "Святогор", "Святомир"};
            Console.WriteLine("\nTesting predict capabilities:");
            foreach (var testName in testNames)
            {
                Console.WriteLine();
                var defaultColor = Console.ForegroundColor;
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine($"Test word: \"{testName}\"");
                Console.ForegroundColor = defaultColor;

                var result = Predict(rnn, testName);
                var minValue = result.MinBy(x => Math.Abs(x.Item1));
                var predictedGender = minValue.Item2 == "M" ? "♂ male" : "♀ female";

                Console.ForegroundColor = ConsoleColor.DarkGreen;

                Console.Write($"\"{testName}\" is most probably ");
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write($"{predictedGender}\n");
                Console.ForegroundColor = defaultColor;

            }
        }

    }
}
