import torch
print(torch.cuda.is_available())  # Should print True if GPU is available
print(torch.cuda.device_count())  # Should print a number > 0

def get_cuda_cores(device=0):
    props = torch.cuda.get_device_properties(device)
    sm_count = props.multi_processor_count
    cores_per_sm = get_cores_per_sm(props.major, props.minor)
    if cores_per_sm is None:
        print("Unknown GPU architecture: {}.{}".format(props.major, props.minor))
        return None
    return sm_count * cores_per_sm

def get_cores_per_sm(major, minor):
    # Based on NVIDIA documentation:
    if major == 2:  # Fermi
        return 48 if minor == 1 else 32
    elif major == 3:  # Kepler
        return 192
    elif major == 5:  # Maxwell
        return 128
    elif major == 6:  # Pascal
        return 128 if minor in [1, 2] else 64
    elif major == 7:  # Volta/Turing
        return 64
    elif major == 8:  # Ampere
        # This is a common configuration; adjust if needed
        return 64 if minor == 0 else 128
    # Add more architectures if needed
    return None

if torch.cuda.is_available():
    total_cores = get_cuda_cores()
    if total_cores:
        print("Total CUDA Cores:", total_cores)
else:
    print("CUDA is not available.")