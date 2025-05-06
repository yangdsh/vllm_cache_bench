import torch

for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.set_device(i)
        x = torch.randn(1024, 1024).cuda()
        y = torch.matmul(x, x)
        print(f"✅ GPU {i} OK: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"❌ GPU {i} FAILED: {e}")