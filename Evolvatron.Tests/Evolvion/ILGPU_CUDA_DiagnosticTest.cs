using ILGPU;
using ILGPU.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace Evolvatron.Tests.Evolvion;

public class ILGPU_CUDA_DiagnosticTest
{
    private readonly ITestOutputHelper _output;

    public ILGPU_CUDA_DiagnosticTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void ILGPU_Can_Detect_CUDA_GPU()
    {
        using var context = Context.Create(builder => builder.Default());

        _output.WriteLine($"ILGPU Version: {typeof(Context).Assembly.GetName().Version}");
        _output.WriteLine($"Available devices: {context.Devices.Length}");
        _output.WriteLine("");

        foreach (var device in context.Devices)
        {
            _output.WriteLine($"Device: {device.Name}");
            _output.WriteLine($"  Type: {device.AcceleratorType}");
            _output.WriteLine($"  Memory: {device.MemorySize / (1024 * 1024)} MB");
            _output.WriteLine($"  Warp Size: {device.WarpSize}");
            _output.WriteLine($"  Max Threads/Group: {device.MaxNumThreadsPerGroup}");
            _output.WriteLine($"  Max Grid Size: {device.MaxGridSize}");
            _output.WriteLine("");
        }

        var cudaDevice = context.Devices
            .FirstOrDefault(d => d.AcceleratorType == AcceleratorType.Cuda);

        if (cudaDevice == null)
        {
            _output.WriteLine("FAILURE: No CUDA device found!");
            _output.WriteLine("Expected to find NVIDIA GeForce RTX 4090");
            Assert.Fail("No CUDA device detected by ILGPU");
        }

        _output.WriteLine($"CUDA device found: {cudaDevice.Name}");
        Assert.Contains("4090", cudaDevice.Name, StringComparison.OrdinalIgnoreCase);

        using var accelerator = cudaDevice.CreateAccelerator(context);
        _output.WriteLine($"\nCUDA accelerator created successfully!");
        _output.WriteLine($"  Name: {accelerator.Name}");
        _output.WriteLine($"  Type: {accelerator.AcceleratorType}");
        _output.WriteLine($"  Memory: {accelerator.MemorySize / (1024 * 1024)} MB");
        _output.WriteLine($"  Warp Size: {accelerator.WarpSize}");
    }

    [Fact]
    public void ILGPU_Preferred_Device_Selection()
    {
        using var context = Context.Create(builder => builder.Default());

        _output.WriteLine("Testing device selection strategies:");
        _output.WriteLine("");

        var preferredNotCPU = context.GetPreferredDevice(preferCPU: false);
        _output.WriteLine($"GetPreferredDevice(preferCPU: false):");
        _output.WriteLine($"  Name: {preferredNotCPU.Name}");
        _output.WriteLine($"  Type: {preferredNotCPU.AcceleratorType}");
        _output.WriteLine("");

        var preferredCPU = context.GetPreferredDevice(preferCPU: true);
        _output.WriteLine($"GetPreferredDevice(preferCPU: true):");
        _output.WriteLine($"  Name: {preferredCPU.Name}");
        _output.WriteLine($"  Type: {preferredCPU.AcceleratorType}");
        _output.WriteLine("");

        using var acc = preferredNotCPU.CreateAccelerator(context);
        _output.WriteLine($"Created accelerator from preferCPU=false:");
        _output.WriteLine($"  Name: {acc.Name}");
        _output.WriteLine($"  Type: {acc.AcceleratorType}");
    }
}
