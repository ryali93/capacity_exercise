import matplotlib.pyplot as plt
import numpy as np

def plot_batch(batch, figsize=(15, 12)):
    """
    Visualize a batch of RGBN images with their corresponding masks
    
    Args:
        batch: Dictionary from DataLoader containing 'rgbn', 'mask', etc.
        figsize: Figure size for matplotlib
    """
    rgbn = batch['rgbn']
    masks = batch['mask']
    alert_ids = batch['alert_id']
    product_types = batch['product_type']
    dates = batch['date']
    
    batch_size = rgbn.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=figsize)
    
    # Handle single sample case
    if batch_size == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(batch_size):
        # Extract RGB bands (0=Red, 1=Green, 2=Blue in our RGBN ordering)
        rgb = rgbn[i, :3].permute(1, 2, 0).numpy()
        
        # Enhance visualization (satellite imagery is often dark)
        rgb_enhanced = np.clip(rgb * 2.5, 0, 1)  # Simple brightness enhancement
        
        # Extract NIR band (band 3)
        nir = rgbn[i, 3].numpy()
        
        # Extract mask
        mask = masks[i].numpy()
        
        # Plot RGB composite
        axes[i, 0].imshow(rgb_enhanced)
        axes[i, 0].set_title(f"Alert {alert_ids[i]} - RGB | {product_types[i]} | {dates[i][:10]}")
        axes[i, 0].axis('off')
        
        # Plot NIR band (vegetation appears bright in NIR)
        im_nir = axes[i, 1].imshow(nir, cmap='RdYlGn', vmin=0, vmax=1)
        axes[i, 1].set_title("NIR Band")
        axes[i, 1].axis('off')
        
        # Plot ground truth mask
        # Mask values: 0=forest, 1=deforestation, 255=nodata
        mask_display = np.where(mask == 255, np.nan, mask)  # Hide nodata as transparent
        im_mask = axes[i, 2].imshow(mask_display, cmap='RdBu_r', vmin=0, vmax=1)
        axes[i, 2].set_title("Deforestation Mask")
        axes[i, 2].axis('off')
        
        # Add colorbar for mask
        plt.colorbar(im_mask, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

def analyze_sample_statistics(dataset, num_samples=50):
    """
    Analyze statistics of the loaded data for debugging and understanding
    
    Args:
        dataset: Sentinel2Dataset instance
        num_samples: Number of samples to analyze
    """
    print(f"Analyzing statistics from {num_samples} samples...")
    
    rgbn_values = []
    mask_values = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        rgbn_values.append(sample['rgbn'].numpy())
        mask_values.append(sample['mask'].numpy())
    
    rgbn_stack = np.stack(rgbn_values)
    mask_stack = np.stack(mask_values)
    
    print(f"\\nRGBN Statistics:")
    print(f"  Shape: {rgbn_stack.shape}")
    print(f"  Min/Max: {rgbn_stack.min():.4f} / {rgbn_stack.max():.4f}")
    print(f"  Mean: {rgbn_stack.mean():.4f}")
    print(f"  Std: {rgbn_stack.std():.4f}")
    
    print(f"\\nMask Statistics:")
    unique_vals, counts = np.unique(mask_stack, return_counts=True)
    total_pixels = mask_stack.size
    print(f"  Shape: {mask_stack.shape}")
    print(f"  Unique values and proportions:")
    for val, count in zip(unique_vals, counts):
        pct = 100 * count / total_pixels
        if val == 0:
            print(f"    {val} (Forest): {count:,} pixels ({pct:.1f}%)")
        elif val == 1:
            print(f"    {val} (Deforestation): {count:,} pixels ({pct:.1f}%)")
        elif val == 255:
            print(f"    {val} (NoData): {count:,} pixels ({pct:.1f}%)")
        else:
            print(f"    {val}: {count:,} pixels ({pct:.1f}%)")

print("✓ Visualization functions defined")