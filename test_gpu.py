import torch
import time

print("Waking up GPU...")
_ = torch.tensor([1.0]).cuda()
print("GPU awake.")

# FIRST TEST (Slow, because it's building the math engine in VRAM)
x = torch.randn(500, 500, device='cuda')
start = time.time()
y = torch.matmul(x, x)
end = time.time()
print(f"First math test took: {(end - start)*1000:.2f} ms (Expect this to be slow)")

# SECOND TEST (Fast, because the engine is already built)
start = time.time()
y = torch.matmul(x, x)
end = time.time()
print(f"Second math test took: {(end - start)*1000:.2f} ms (Expect this to be <10 ms)")

used_vram = torch.cuda.memory_allocated(0) / (1024**3)
print(f"VRAM used: {used_vram:.4f} GB")

if (end - start) < 0.05: # If the second test took less than 50 milliseconds
    print("\nSUCCESS: The 3050 is running at full speed. Phase 1 begins.")
else:
    print("\nWARNING: GPU is still slow. We have a hardware bottleneck.")