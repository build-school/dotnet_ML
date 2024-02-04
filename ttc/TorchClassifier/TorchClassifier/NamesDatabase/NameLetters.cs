using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchClassifier.NamesDatabase
{
    internal class NameLetters
    {
        public static char[] AsciiSmall = Enumerable.Range('a', 'z' - 'a' + 1).Select(i => (char)i).ToArray();
        public static char[] AsciiCaps = Enumerable.Range('A', 'Z' - 'A' + 1).Select(i => (char)i).ToArray();
        public static char[] RussianSmall = Enumerable.Range('а', 'я' - 'а' + 1).Select(i => (char)i).ToArray();
        public static char[] RussianCaps = Enumerable.Range('А', 'Я' - 'А' + 1).Select(i => (char)i).ToArray();
        public static char[] PunctuationLetters = " .,;'".ToCharArray();

        public static char[]? _allLetters;

        public static char[] All
        {
            get
            {
                if (null == _allLetters)
                {
                    _allLetters = Array.Empty<char>()
                        //.Concat(AsciiSmall)
                        //.Concat(AsciiCaps)
                        .Concat(RussianSmall)
                        .Concat(RussianCaps)
                        .ToArray();
                }
                return _allLetters;
            }
        }

        public static int IndexOf(char letter)
        {
            return Array.IndexOf(All, letter);
        }
    }
}
