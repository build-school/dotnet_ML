namespace TorchSharpTest;

internal static class Generator
{
    //public static void Generate()
    //{
    //    torch.random.manual_seed(Rnd_seed());
    //    // 8x8
    //    var image_size = 64;
    //    // 色彩通道 3 -> rgb
    //    var nc = 3;
    //    // 输入张量
    //    var nz = 128;
    //    // 特征大小
    //    var ngf = 64;

    // var _device = device(DeviceType.CPU); var fixed_noise = randn(64, nz, 1, 1, device: _device);
    // var gen = new Model.Generator("Generator", nz, ngf, nc).to(_device); gen.load("");
    // //torch.no_grad() var generated = gen.forward(fixed_noise).detach().cpu();

    // // 将模型输出转换为 numpy 数组 NDArray npOutput = NDArray.FromMultiDimArray<long>(generated.shape);

    // // 将 numpy 数组转换为图像的维度 NDArray npImage = npOutput.transpose(new int[] { 1, 2, 0});

    // // 将 numpy 数组加载为 Image<Rgb24> 类型 Image<Rgb24> image =
    // Image.LoadPixelData<Rgb24>(npImage.ToByteArray(), 32, 32);

    // //img_list.append(make_grid(generated, padding = 2, normalize = True))

    //}
}