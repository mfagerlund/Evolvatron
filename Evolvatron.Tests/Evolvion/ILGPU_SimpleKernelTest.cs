using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class ILGPU_SimpleKernelTest
{
    private readonly ITestOutputHelper _output;

    public ILGPU_SimpleKernelTest(ITestOutputHelper output)
    {
        _output = output;
    }

    static void SimpleKernel(Index1D index, ArrayView<float> input, ArrayView<float> output)
    {
        output[index] = input[index] * 2.0f + 1.0f;
    }

    [Fact]
    public void Simple_Kernel_Works_On_CUDA()
    {
        using var context = Context.Create(builder => builder.Default().EnableAlgorithms().Math(MathMode.Fast32BitOnly));

        _output.WriteLine($"Available devices: {context.Devices.Length}");

        var cudaDevice = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);
        Assert.NotNull(cudaDevice);

        using var accelerator = cudaDevice.CreateAccelerator(context);
        _output.WriteLine($"Using: {accelerator.Name}");

        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SimpleKernel);

        const int size = 1024;
        using var inputBuffer = accelerator.Allocate1D<float>(size);
        using var outputBuffer = accelerator.Allocate1D<float>(size);

        var inputData = new float[size];
        for (int i = 0; i < size; i++)
            inputData[i] = i;

        inputBuffer.CopyFromCPU(inputData);

        kernel(size, inputBuffer.View, outputBuffer.View);
        accelerator.Synchronize();

        var outputData = outputBuffer.GetAsArray1D();

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(i * 2.0f + 1.0f, outputData[i]);
        }

        _output.WriteLine($"Simple kernel executed successfully on {accelerator.Name}!");
    }
}
