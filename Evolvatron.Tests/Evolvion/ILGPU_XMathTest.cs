using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class ILGPU_XMathTest
{
    private readonly ITestOutputHelper _output;

    public ILGPU_XMathTest(ITestOutputHelper output)
    {
        _output = output;
    }

    static void TanhKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        output[index] = XMath.Tanh(input[index]);
    }

    static void ExpKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        output[index] = XMath.Exp(input[index]);
    }

    [Fact]
    public void XMath_Tanh_Works_On_CUDA()
    {
        using var context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        Assert.NotNull(cudaDevice);

        using var accelerator = cudaDevice.CreateAccelerator(context);
        _output.WriteLine($"Using: {accelerator.Name}");

        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(TanhKernel);

        const int size = 10;
        using var inputBuffer = accelerator.Allocate1D<float>(size);
        using var outputBuffer = accelerator.Allocate1D<float>(size);

        var inputData = new float[] { -2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f, 3f, 4f, 5f };
        inputBuffer.CopyFromCPU(inputData);

        kernel(size, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();

        var outputData = outputBuffer.GetAsArray1D();
        _output.WriteLine($"XMath.Tanh executed successfully on {accelerator.Name}!");

        for (int i = 0; i < size; i++)
        {
            _output.WriteLine($"  tanh({inputData[i]:F2}) = {outputData[i]:F6}");
        }
    }

    [Fact]
    public void XMath_Exp_Works_On_CUDA()
    {
        using var context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        var cudaDevice = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        Assert.NotNull(cudaDevice);

        using var accelerator = cudaDevice.CreateAccelerator(context);
        _output.WriteLine($"Using: {accelerator.Name}");

        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ExpKernel);

        const int size = 5;
        using var inputBuffer = accelerator.Allocate1D<float>(size);
        using var outputBuffer = accelerator.Allocate1D<float>(size);

        var inputData = new float[] { -2f, -1f, 0f, 1f, 2f };
        inputBuffer.CopyFromCPU(inputData);

        kernel(size, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();

        var outputData = outputBuffer.GetAsArray1D();
        _output.WriteLine($"XMath.Exp executed successfully on {accelerator.Name}!");

        for (int i = 0; i < size; i++)
        {
            _output.WriteLine($"  exp({inputData[i]:F2}) = {outputData[i]:F6}");
        }
    }
}
