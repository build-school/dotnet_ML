using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using ExcelDataReader;

namespace TorchClassifier.NamesDatabase
{
    public class CommonNamesDatabase
    {
        private const string DatabaseFileName = "russian_names.xlsx";

        private readonly IList<FirstNameRecord> _firstNameRecords = new List<FirstNameRecord>();

        private IDictionary<string, FirstNameRecord>? _firstNamesDictionary = null;


        public CommonNamesDatabase()
        {
        }

        private void LoadFirstNames(Stream stream)
        {
            using var reader = ExcelReaderFactory.CreateReader(stream);
            {
                // clear if any:
                _firstNameRecords.Clear();
                ;
                //skip header row
                reader.Read();

                while (reader.Read())
                {
                    if (reader.IsDBNull(1))
                    {
                        continue;
                    }

                    var record = new FirstNameRecord
                    {
                        Name = reader.GetString(0),
                        Sex = reader.GetString(1),
                        Frequency = (int)reader.GetDouble(2)
                    };
                    _firstNameRecords.Add(record);
                }
            }
        }

        public void Load()
        {
            if (!_firstNameRecords.Any())
            {
                //From the assembly where this code lives!
                var names1 = GetType().Assembly.GetManifestResourceNames();

                //or from the entry point to the application - there is a difference!
                var names2 = Assembly.GetExecutingAssembly().GetManifestResourceNames();

                var resourceName = Assembly.GetExecutingAssembly().GetName().Name + ".data." + DatabaseFileName;
                using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName);
                {
                    LoadFirstNames(stream);
                }
            }
        }
        public void Load(string? filePath)
        {
            if (!_firstNameRecords.Any())
            {
                if (!string.IsNullOrEmpty(filePath) || !File.Exists(filePath))
                {
                    using (var stream = File.Open(filePath, FileMode.Open, FileAccess.Read))
                    {
                        LoadFirstNames(stream);
                    }
                }
            }
        }

        public FirstNameRecord? Find(string name)
        {
            if (_firstNamesDictionary == null)
            {
                _firstNamesDictionary = _firstNameRecords.ToDictionary(x => x.Name);
            }

            if (_firstNamesDictionary.TryGetValue(name, out FirstNameRecord? record))
            {
                return record;
            }
            return null;
        }

        public IEnumerable<FirstNameRecord> Records
        {
            get
            {
                return _firstNameRecords;
            }
        }
    }
}
