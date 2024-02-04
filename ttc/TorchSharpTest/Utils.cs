using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TorchSharp;
using static TorchSharp.TensorExtensionMethods;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;

namespace TorchSharpTest;

internal static class Utils
{
    public static long Rnd_seed()
    {
        return new Random(Guid.NewGuid().GetHashCode()).NextInt64();
    }

    //public static torchvision.ITransform ToTensor()
    //{
    //    return new ToTensorClass();
    //}
    //class ToTensorClass : torchvision.ITransform
    //{
    //    public torch.Tensor forward(torch.Tensor input)
    //    {
    //        return input;
    //    }
    //}
    public class ImageFolder : Dataset
    {
        private readonly int _image_size;
        private readonly Dictionary<long, Dictionary<string, Tensor>> _imagesTensor;
        public override long Count => _imagesTensor.LongCount();

        public ImageFolder(int image_size)
        {
            _imagesTensor = new();
            _image_size = image_size;
        }

        public override Dictionary<string, Tensor> GetTensor(long index) => _imagesTensor[index];

        public void LoadImageFiles(string imageFilesPath, string patterns = "*.jpg")
        {
            //torchvision.io.DefaultImager = new io.SkiaImager(100);

            var dirInfo = new DirectoryInfo(imageFilesPath);
            if (dirInfo.Exists is false)
            {
                throw new Exception("image file not found");
            }
            var allFiles = dirInfo.EnumerateFiles(patterns, SearchOption.AllDirectories);

            var indexs = 0L;

            foreach (var file in allFiles)
            {
                //var img = torchvision.io.read_image(file.FullName, io.ImageReadMode.RGB);
                var img = ImageToTensor(file.FullName);

                var normals = transforms.Normalize(
                                        new double[] { 0.5, 0.5, 0.5 },
                                        new double[] { 0.5, 0.5, 0.5 }
                                        );
                var _transform = transforms.Compose(
                                    //# specify transforms, including resizing
                                    //# could be improved in future (resizing sus)
                                    transforms.Resize(_image_size),
                                    transforms.CenterCrop(_image_size),
                                    normals
                                    );
                var transformImageTensor = _transform.forward(img).squeeze();
                _imagesTensor.Add(indexs, new() { { "data", transformImageTensor } });
                indexs++;
            }
        }

        public static Tensor ImageToTensor(string filepath)
        {
            var image = Image.Load<Rgb24>(filepath);
            //image.Mutate(p => p.Crop(500, 500).Resize(500, 500));
            var width = image.Width;
            var height = image.Height;

            // 创建三个 float[] 数组分别存储每个通道的数据
            var dataR = new float[width * height];
            var dataG = new float[width * height];
            var dataB = new float[width * height];

            int index = 0;
            image.ProcessPixelRows(pixellinq =>
            {
                for (int y = 0; y < height; y++)
                {
                    var pixelRowSpan = pixellinq.GetRowSpan(y);
                    for (int x = 0; x < width; x++)
                    {
                        var pixel = pixelRowSpan[x];
                        dataR[index] = pixel.R / 255.0f;
                        dataG[index] = pixel.G / 255.0f;
                        dataB[index] = pixel.B / 255.0f;
                        index++;
                    }
                }
            });

            // 嵌套三个 float[] 数组创建 3 维 float[] 数组
            //var data = new float[3][];
            //data[0] = dataR.Select(x => x).ToArray();
            //data[1] = dataG.Select(x => x).ToArray();
            //data[2] = dataB.Select(x => x).ToArray();

            // 将 3 维 float[] 数组转换为 Tensor 对象
            return dataR.Concat(dataG).Concat(dataB).ToArray().ToTensor(new long[] { 1, 3, height, width });
        }

    }
}