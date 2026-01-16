import os
import argparse
import numpy as np
from pathlib import Path
import json
import csv
from datetime import datetime
import subprocess
import sys

def find_images(directory):
    """Find all image files in directory."""
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    images = []
    for name in os.listdir(directory):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(os.path.join(directory, name))
    return sorted(images)

def read_ground_truth(gt_path):
    """
    Read ground truth from gt.txt file.
    Expected format: row,col,rotation,scale
    Returns dict with ground truth values.
    """
    with open(gt_path, 'r') as f:
        line = f.read().strip()
    
    values = [float(x.strip()) for x in line.split(',')]
    
    if len(values) == 2:
        return {'row': values[0], 'col': values[1], 'rotation': None, 'scale': None}
    elif len(values) == 3:
        return {'row': values[0], 'col': values[1], 'rotation': values[2], 'scale': None}
    elif len(values) >= 4:
        return {'row': values[0], 'col': values[1], 'rotation': values[2], 'scale': values[3]}
    else:
        raise ValueError(f"Invalid ground truth format in {gt_path}: {line}")

def compute_error(prediction, ground_truth):
    """
    Compute error metrics between prediction and ground truth.
    """
    errors = {}
    
    # Position error (Euclidean distance in pixels)
    pred_row = prediction['row']
    pred_col = prediction['col']
    gt_row = ground_truth['row']
    gt_col = ground_truth['col']
    
    position_error = np.sqrt((pred_row - gt_row)**2 + (pred_col - gt_col)**2)
    errors['position_error'] = position_error
    errors['row_error'] = abs(pred_row - gt_row)
    errors['col_error'] = abs(pred_col - gt_col)
    
    # Rotation error (if available)
    if ground_truth['rotation'] is not None:
        rotation_error = abs(prediction['rotation'] - ground_truth['rotation'])
        # Normalize to [-180, 180]
        if rotation_error > 180:
            rotation_error = 360 - rotation_error
        errors['rotation_error'] = rotation_error
    else:
        errors['rotation_error'] = None
    
    # Scale error (if available)
    if ground_truth['scale'] is not None:
        scale_error = abs(prediction['scale'] - ground_truth['scale'])
        errors['scale_error'] = scale_error
        errors['scale_error_percent'] = (scale_error / ground_truth['scale']) * 100
    else:
        errors['scale_error'] = None
        errors['scale_error_percent'] = None
    
    return errors

def run_test_on_folder(test_folder, dino_version, dino_script_path):
    """
    Run test on a single folder containing two images and gt.txt.
    Returns dict with test results.
    """
    # Find images
    images = find_images(test_folder)
    if len(images) < 2:
        return {'success': False, 'error': f'Found only {len(images)} image(s), need at least 2'}
    
    img1_path = images[0]
    img2_path = images[1]
    
    # Read ground truth
    gt_path = os.path.join(test_folder, 'gt.txt')
    if not os.path.exists(gt_path):
        return {'success': False, 'error': 'gt.txt not found'}
    
    try:
        ground_truth = read_ground_truth(gt_path)
    except Exception as e:
        return {'success': False, 'error': f'Error reading ground truth: {e}'}
    
    # Run dino.py as subprocess
    try:
        # Run dino.py on the test folder
        cmd = [sys.executable, dino_script_path, test_folder, '--dino-version', dino_version, '--quiet']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return {'success': False, 'error': f'dino.py failed: {result.stderr}'}
        
        # Read results from output file
        output_dir = os.path.join(test_folder, 'outputs')
        results_file = os.path.join(output_dir, 'results.txt')
        
        if not os.path.exists(results_file):
            return {'success': False, 'error': f'Results file not found: {results_file}'}
        
        with open(results_file, 'r') as f:
            line = f.read().strip()
        
        # Parse results: row,col,rotation,scale,score
        values = [float(x.strip()) for x in line.split(',')]
        if len(values) < 5:
            return {'success': False, 'error': f'Invalid results format: {line}'}
        
        prediction = {
            'row': values[0],
            'col': values[1],
            'rotation': values[2],
            'scale': values[3],
            'score': values[4]
        }
        
        # Compute errors
        errors = compute_error(prediction, ground_truth)
        
        return {
            'success': True,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'errors': errors,
            'img1': os.path.basename(img1_path),
            'img2': os.path.basename(img2_path)
        }
        
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Processing timeout (>300s)'}
    except Exception as e:
        import traceback
        return {'success': False, 'error': f'Error during processing: {e}\n{traceback.format_exc()}'}

def run_all_tests(root_folder, dino_version, dino_script_path, output_file=None):
    """
    Run tests on all subfolders in root_folder.
    """
    print(f"Running tests in: {root_folder}")
    print(f"DINO Version: {dino_version}")
    print(f"DINO Script: {dino_script_path}")
    print("-" * 80)
    
    # Find all subfolders
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir() and not f.name.startswith('.')]
    subfolders = sorted(subfolders)
    
    if len(subfolders) == 0:
        print("No subfolders found!")
        return
    
    print(f"Found {len(subfolders)} test folders\n")
    
    results = []
    
    for i, folder in enumerate(subfolders):
        folder_name = os.path.basename(folder)
        print(f"[{i+1}/{len(subfolders)}] Testing: {folder_name}...", end=" ")
        
        result = run_test_on_folder(folder, dino_version, dino_script_path)
        result['folder'] = folder_name
        results.append(result)
        
        if result['success']:
            print(f"✓ Position error: {result['errors']['position_error']:.2f} px")
        else:
            print(f"✗ {result['error']}")
    
    # Generate report
    print("\n" + "=" * 80)
    print("TEST REPORT")
    print("=" * 80)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        position_errors = [r['errors']['position_error'] for r in successful]
        print(f"\nPosition Error Statistics:")
        print(f"  Mean: {np.mean(position_errors):.2f} px")
        print(f"  Median: {np.median(position_errors):.2f} px")
        print(f"  Std Dev: {np.std(position_errors):.2f} px")
        print(f"  Min: {np.min(position_errors):.2f} px")
        print(f"  Max: {np.max(position_errors):.2f} px")
        
        # Rotation errors if available
        rotation_errors = [r['errors']['rotation_error'] for r in successful 
                          if r['errors']['rotation_error'] is not None]
        if rotation_errors:
            print(f"\nRotation Error Statistics:")
            print(f"  Mean: {np.mean(rotation_errors):.2f}°")
            print(f"  Median: {np.median(rotation_errors):.2f}°")
            print(f"  Std Dev: {np.std(rotation_errors):.2f}°")
        
        # Scale errors if available
        scale_errors_pct = [r['errors']['scale_error_percent'] for r in successful 
                           if r['errors']['scale_error_percent'] is not None]
        if scale_errors_pct:
            print(f"\nScale Error Statistics:")
            print(f"  Mean: {np.mean(scale_errors_pct):.2f}%")
            print(f"  Median: {np.median(scale_errors_pct):.2f}%")
    
    # Save detailed results to CSV
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_results_{timestamp}.csv"
    elif os.path.isdir(output_file):
        # If output is a directory, create a file inside it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_file, f"test_results_{timestamp}.csv")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Folder', 'Success', 'Image1', 'Image2',
            'Pred_Row', 'Pred_Col', 'Pred_Rotation', 'Pred_Scale', 'Score',
            'GT_Row', 'GT_Col', 'GT_Rotation', 'GT_Scale',
            'Position_Error', 'Row_Error', 'Col_Error', 'Rotation_Error', 'Scale_Error',
            'Error_Message'
        ])
        
        for r in results:
            if r['success']:
                writer.writerow([
                    r['folder'], 'Yes', r['img1'], r['img2'],
                    r['prediction']['row'], r['prediction']['col'],
                    r['prediction']['rotation'], r['prediction']['scale'],
                    r['prediction']['score'],
                    r['ground_truth']['row'], r['ground_truth']['col'],
                    r['ground_truth']['rotation'], r['ground_truth']['scale'],
                    r['errors']['position_error'], r['errors']['row_error'],
                    r['errors']['col_error'], r['errors']['rotation_error'],
                    r['errors']['scale_error'], ''
                ])
            else:
                writer.writerow([
                    r['folder'], 'No', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', r['error']
                ])
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimization tests on DINO cross-correlation")
    parser.add_argument("test_folder", help="Folder containing test subfolders")
    parser.add_argument("--dino-version", choices=["v2", "v3"], default="v2",
                       help="DINO model version to test")
    parser.add_argument("--dino-script", default="dino.py",
                       help="Path to dino.py script (default: dino.py)")
    parser.add_argument("--output", help="Output CSV file path (default: auto-generated)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.test_folder):
        print(f"Error: '{args.test_folder}' is not a directory")
        exit(1)
    
    if not os.path.isfile(args.dino_script):
        print(f"Error: DINO script not found: '{args.dino_script}'")
        exit(1)
    
    # Run tests
    run_all_tests(args.test_folder, args.dino_version, args.dino_script, args.output)
