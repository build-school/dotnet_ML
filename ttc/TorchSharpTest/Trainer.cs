using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using static TorchSharpTest.Models.Model;
using static TorchSharpTest.Models.Model_Util;
using static TorchSharpTest.Utils;

namespace TorchSharpTest;

internal static class Trainer
{
    public static void Train()
    {
        // torch.multiprocessing.freeze_support()
        random.manual_seed(Rnd_seed());
        //# -------- set parameters ----------#
        /*
         *parser.add_argument('--dataroot', type=str, default='./data/cropped_images/', help='path to data root directory')
         *parser.add_argument('--output_dir', type=str, default='./results/', help='results path')
         *parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
         *parser.add_argument('--batch_size', type=int, default=32, help='batch size')
         *parser.add_argument('--image_size', type=int, default=64, help='image size')
         *parser.add_argument('--no_soft_labels', action='store_true', help='do not use soft labels (use real labels)')
         *parser.add_argument('--epochs', type=int, default=100, help="number of epochs to run for")
         *parser.add_argument('--no_plots', action='store_true', help='do not plot anything')
         */
        var no_soft_labels = true;
        //# Root directory for dataset

        var dataroot = @"D:\迅雷下载\Mirror.2.Project.X.Steam\Mirror.2.Project.X.Steam\common\Mirror 2 Project X\X emoji DLC";

        //var results_path = "";

        //# Number of workers for the dataloader
        var workers = 16;
        //# Batch size during training
        //# DCGAN paper uses 128
        var batch_size = 32;
        //# Spatial size of training images (All images resized to this size)
        var image_size = 64;
        //# Epochs
        var num_epochs = 1000L;

        //# default model parameters
        //# Color images = 3 (for RGB)
        var nc = 3L;
        //# Size of z latent vector (generator input size)
        var nz = 128L;
        //# Size of feature maps in generator
        var ngf = 64L;
        //# Size of feature maps in discriminator
        var ndf = 64L;
        //# Learning rate
        var lrg = 0.0002;
        var lrd = 0.0002;
        //# Beta 1 hyperparam for adam
        var beta1 = 0.5;
        //# Number of GPUs
        var ngpu = 1;
        //# Specify device
        var _device = device(DeviceType.CUDA);
        
        var dataset = new ImageFolder(image_size);
        dataset.LoadImageFiles(dataroot);
        var dataloader = new DataLoader(dataset, batch_size, shuffle: true, num_worker: workers);

        //# Plot training images
        //var real_batch = dataloader.GetEnumerator();
        var (netG, netD) = init_models(ngpu, _device, nz, ngf, ndf, nc, verbose: true);

        //# Use BCE loss
        var loss = nn.BCELoss();
        var fixed_noise = randn(128, nz, 1, 1, device: _device);

        //# Real/fake labels - only used if not using soft labels
        var real_label = 1;
        var fake_label = 0;

        //# set up optimizers
        var optimizerD = optim.Adam(netD.parameters(), lr: lrd, beta1, beta2: 0.999);
        var optimizerG = optim.Adam(netG.parameters(), lr: lrg, beta1, beta2: 0.999);
 
        //# Lists to keep track of progress
        var img_list = new List<Image<Rgb24>>();
        var G_losses = new List<float>();
        var D_losses = new List<float>();
        var iters = 0;
        for (long epoch = 0; epoch <= num_epochs; epoch++)
        {
            var enumerator = dataloader.GetEnumerator();
            long i = 0;
            while (enumerator.MoveNext())
            {
                i++;
                netD.zero_grad();

                var real_cpu = enumerator.Current["data"].to(_device);  // only once call

                var m_batch_size = real_cpu.size(0);

                var d_output = netD.forward(real_cpu).view(-1);

                var label = full(m_batch_size, real_label, dtype: ScalarType.Float32, device: _device);

                var soft_real_labels = generate_real_labels(m_batch_size);
                Tensor lossD_real;
                if (no_soft_labels)
                {
                    lossD_real = loss.forward(d_output, label);
                    lossD_real.backward();
                }
                else
                {
                    lossD_real = loss.forward(d_output, soft_real_labels);
                    lossD_real.backward();
                }
                // D(X) 均值 , 非运行必须
                var D_x = d_output.to(DeviceType.CPU).mean().item<float>();

                //## Train with all-fake batch
                var noise = randn(batch_size, nz, 1, 1, device: _device);
                var fake = netG.forward(noise); //# plug in G
                label.fill_(fake_label);
                var soft_fake_labels = generate_fake_labels(batch_size);
                var output = netD.forward(fake.detach()).view(-1);

                Tensor lossD_fake;
                if (no_soft_labels)
                {
                    lossD_fake = loss.forward(output, label);
                    lossD_fake.backward();
                }
                else
                {
                    lossD_fake = loss.forward(output, soft_fake_labels);
                    lossD_fake.backward();
                }
                // D(G(z)) z1
                var D_G_z1 = output.to(DeviceType.CPU).mean().item<float>();

                var lossD = lossD_real + lossD_fake;
                optimizerD.step();

                netG.zero_grad();
                label.fill_(real_label);
                soft_real_labels = generate_real_labels(batch_size);

                //# Put fake batch into D
                output = netD.forward(fake).view(-1);

                //# Calculate G's loss based on this output

                Tensor lossG;
                if (no_soft_labels)
                {
                    lossG = loss.forward(output, label);
                    lossG.backward();
                }
                else
                {
                    lossG = loss.forward(output, soft_real_labels);
                    lossG.backward();
                }

                //# Calculate gradients for G
                var D_G_z2 = output.to(DeviceType.CPU).mean().item<float>();

                optimizerG.step();

                //# Output training stats, nice formatting by pytorch
                //if (i % 50 == 0)
                //{
                    Console.WriteLine(@$"进度: [ {epoch} / {num_epochs} ] [ {i} / {dataloader.Count}] Loss_D:{lossD.to(DeviceType.CPU).item<float>()} | Loss_G:{lossG.to(DeviceType.CPU).item<float>()} | D(x):{D_x} | D(G(z)):[ {D_G_z1} / {D_G_z2} ]");
                //}
                ////print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                // % (epoch, num_epochs, i, len(dataloader), lossD.item<long>(), lossG.item(),
                // D_x, D_G_z1, D_G_z2))

                // # Save loss function for plotting
                G_losses.Add(lossG.to(DeviceType.CPU).item<float>());
                D_losses.Add(lossD.to(DeviceType.CPU).item<float>());

                if(iters % 20 == 0)
                {
                    netG.to(DeviceType.CPU).save("./NetG.model");
                    netD.to(DeviceType.CPU).save("./netD.model");
                }
                //# Check how the generator is doing by saving G output
                /*
                
                //实时生成图片以监测训练状态

                if ((iters % 500 == 0) || ((epoch == num_epochs - 1) && (i == dataloader.Count - 1)))
                {
                    fake = netG.forward(fixed_noise).detach().cpu();
                    no_grad().Dispose();

                    // 将模型输出转换为 numpy 数组
                    NDArray npOutput = NDArray.FromMultiDimArray<long>(fake.shape);

                    // 将 numpy 数组转换为图像的维度
                    NDArray npImage = npOutput.transpose(new int[] { 1, 2, 0 });

                    // 将 numpy 数组加载为 Image<Rgb24> 类型
                    Image<Rgb24> image = Image.LoadPixelData<Rgb24>(npImage.ToByteArray(), image_size, image_size);

                    img_list.Add(image);
                }
                */
                iters++;
            }
        }
        netG.to(DeviceType.CPU).save("./NetG.model");
        netD.to(DeviceType.CPU).save("./netD.model");

    }
}