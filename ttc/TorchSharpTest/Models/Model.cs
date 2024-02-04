using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

//self namespace
using static TorchSharpTest.Models.Model_Util;

namespace TorchSharpTest.Models;

internal static class Model
{
    public class Generator : Module<Tensor, Tensor>
    {
        private readonly Sequential main;
        private readonly Device _device;
        public Generator(string name, long nz, long ngf, long nc, Device devices) : base(name)
        {
            RegisterComponents();
            _device = devices;
            if (_device.type == DeviceType.CUDA)
            {
                this.to(_device);
            }
            main = Sequential(
            ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias: false, device: _device),
            BatchNorm2d(ngf * 8, device: _device),
            ReLU(true),
            ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias: false, device: _device),
            BatchNorm2d(ngf * 4, device: _device),
            ReLU(true),
            ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias: false, device: _device),
            BatchNorm2d(ngf * 2, device: _device),
            ReLU(true),
            ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias: false, device: _device),
            BatchNorm2d(ngf, device: _device),
            ReLU(true),
            ConvTranspose2d(ngf, nc, 4, 2, 1, bias: false, device: _device),
            Tanh()
            );
        }

        public override Tensor forward(Tensor input) => main.forward(input.cuda());
    }

    public class Discriminator : Module<Tensor, Tensor>
    {
        private readonly Sequential main;
        private readonly Device _device;
        public Discriminator(string name, long ndf, long nc, Device devices) : base(name)
        {
            RegisterComponents();
            _device = devices;
            if (_device.type == DeviceType.CUDA)
            {
                this.to(_device);
            }
            main = Sequential(
            //# input is (nc) x 64 x 64
            Conv2d(nc, ndf, 4, 2, 1, bias: false, device: _device),
            LeakyReLU(0.2, inplace: false),
            //# state size. (ndf) x 32 x 32
            Conv2d(ndf, ndf * 2, 4, 2, 1, bias: false, device: _device),
            BatchNorm2d(ndf * 2, device: _device),
            LeakyReLU(0.2, inplace: false),
            Dropout2d(p: 0.2, inplace: false),
            //# state size. (ndf*2) x 16 x 16
            Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias: false, device: _device),
            BatchNorm2d(ndf * 4, device: _device),
            LeakyReLU(0.2, inplace: false),
            Dropout2d(p: 0.2, inplace: false),
            //# state size. (ndf*4) x 8 x 8
            Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias: false, device: _device),
            BatchNorm2d(ndf * 8, device: _device),
            LeakyReLU(0.2, inplace: false),
            Dropout2d(p: 0.2, inplace: false),
            //# state size. (ndf*8) x 4 x 4
            Conv2d(ndf * 8, 1, 4, 1, 0, bias: false, device: _device),
            Sigmoid()
            );
        }

        public override Tensor forward(Tensor input) => main.forward(input.cuda());
    }

    public static (Generator, Discriminator) init_models(int ngpu, Device device, long nz, long ngf, long ndf, long nc = 3, bool verbose = false)
    {
        //# init generator
        var netG = new Generator("Generator", nz, ngf, nc, device).to(device);
        //# init discriminator
        var netD = new Discriminator("Discriminator", ndf, nc, device).to(device);
        //# parallelize
        if ((device.type == DeviceType.CUDA) && (ngpu > 1))
        {
            //netG = nn.DataParallel(netG, list(range(ngpu)))
            //netD = (netD, list(range(ngpu)))
        }

        netG.apply(weights_init);
        netD.apply(weights_init);

        if (verbose)
        {
            Console.WriteLine(netG);
            Console.WriteLine(netD);
        }

        return (netG, netD);
    }
}