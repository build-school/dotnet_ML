using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
namespace TorchSharpTest.Models;

internal static class Model_Util
{
    public const float UPPER_FAKE = 0.3F;
    public const float UPPER_REAL = 1.2F;
    public const float LOWER_REAL = 0.7F;

    public static void weights_init(Module model)
    {
        if (model is Conv2d conv)
        {
            init.normal_(conv.weight, 0.0, 0.02);
        }
        else if (model is BatchNorm2d norm)
        {
            init.normal_(norm.weight, 1.0, 0.02);
            init.constant_(norm.bias, 0);
        }
    }

    public static Tensor generate_fake_labels(long tensor_size)
    {
        return rand(tensor_size) * UPPER_FAKE;
    }

    public static Tensor generate_real_labels(long tensor_size)
    {
        return rand(tensor_size) * (UPPER_REAL - LOWER_REAL) + LOWER_REAL;
    }
}