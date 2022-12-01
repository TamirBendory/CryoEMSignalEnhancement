import sys, os, argparse, shutil
import pickle5 as pickle
#import numpy as np
#from scipy.io import savemat

# os.chdir('/scratch/guysharon/Work/CryoEMSignalEnhancement')
from src.cryo_signal_enhance import class_average

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # args = '--starfile /scratch/guysharon/Work/starfiles/dataset_10028 --output /scratch/guysharon/Work/Python/saved_test_data/tmp --basis_file_in /scratch/guysharon/Work/Python/saved_test_data/basis_10028'
    # sys.argv[1:] = args.split(' ')
    
    #%%#################### default values for arguments #%%####################
    opts = {
        #   GENERAL
            "N":                    2147483647,
            'verbose':              True,
        
        #   PREPROCESSING
            "downsample":           89,
            "batch_size":           2**15,
            "num_coeffs":           500,
            
        #   2D CLASS 
            "class_theta_step":     5,
            "num_classes":          3000,
            "class_size":           300,
            "class_gpu":            0,
    
        #   EM
            "ctf":                  True,
            "iter":                 7,
            "norm":                 True,
            "em_num_inputs":        150,
            "take_last_iter":       True,
            "em_par":               True,
            "em_gpu":               False,
            "num_class_avg":        1500,
            
        #   DEBUG
            'debug_verbose':        False,
            'random_seed':          -1,
            'sample':               False
        }
    opts_keys = list(opts.keys())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--starfile",type=str, required=True)
    parser.add_argument("--output",type=str, required=True)
    parser.add_argument("--basis_file_out",type=str)
    parser.add_argument("--basis_file_in",type=str)
    for i in range(len(opts_keys)):
        val = opts[opts_keys[i]]
        if type(val) == bool:
            parser.add_argument(f"--{opts_keys[i]}",type=str2bool,default=val)
        else:
            parser.add_argument(f"--{opts_keys[i]}",type=type(val),default=val)
    print(" ")
    
    #%%#################### help message #%%####################
    help_str = f"""
[--starfile] and [--output] are not optional, the rest are.
====== General options ====== 
                            --starfile           ("")    : Input images (.star)
                            --output             ("")    : Ouput images (.mrcs)
                            --basis_file_out     ("")    : Write sPCA basis to file, allowing to skip preprocessing next time (.pkl)
                            --basis_file_in      ("")    : Optional input file for sPCA basis. When given, skips preprocessing (.pkl)
                            --N                  (inf)   : Number of images to take from starfile
                            --verbose            ({opts['verbose']})  : Print progress to screen
====== Preprocessing ======
                            --downsample         ({opts['downsample']})    : Preprocessed image size
                            --batch_size         ({opts['batch_size']})  : Batch size for PSD noise estimation
                            --num_coeffs         ({opts['num_coeffs']})  : Number of sPCA coefficient
====== 2-D Classification ======
                            --class_theta_step   ({opts['class_theta_step']})     : Sampling rate for the in-plane angle
                            --num_classes        ({opts['num_classes']})  : Number of classes to find (<= N)
                            --class_size         ({opts['class_size']})   : Number of images in each class (<= N)
                            --class_gpu          ({opts['class_gpu']})     : Use available gpu for matrix calculations (recommended)
                                                             -1       = don't use gpu
                                                              0       = select gpu with most available memory
                                                              1,2,... = use specific gpu
====== Expectation Maximization ======
                            --ctf                ({opts['ctf']})  : Perform CTF correction
                            --iter               ({opts['iter']})     : Maximum number of iterations to perform
                            --norm               ({opts['norm']})  : Perform normalisation-error correction
                            --em_num_inputs      ({opts['em_num_inputs']})   : Number of images to use for EM (taken from best of each class) (<= class_size)
                            --em_par             ({opts['em_par']})  : Perform EM parallelly (recommended)
                            --num_class_avg      ({opts['num_class_avg']})  : Number of class averages to produce (<= num_classes)
====== Debug ======
                            --debug_verbose      ({opts['debug_verbose']})  : Prints debug messages

Usage:
    cryo_signal_enhance --starfile STARFILE --output OUTPUT [-h]
                       [--basis_file_out BASIS_FILE_OUT]
                       [--basis_file_in BASIS_FILE_IN] [--N N] [--verbose VERBOSE]
                       [--downsample DOWNSAMPLE] [--batch_size BATCH_SIZE]
                       [--num_coeffs NUM_COEFFS] [--class_theta_step CLASS_THETA_STEP]
                       [--num_classes NUM_CLASSES] [--class_size CLASS_SIZE]
                       [--class_gpu CLASS_GPU] [--ctf CTF] [--iter ITER]
                       [--norm NORM] [--em_num_inputs EM_NUM_INPUTS]
                       [--em_par EM_PAR] [--num_class_avg NUM_CLASS_AVG]
                       [--debug_verbose DEBUG_VERBOSE]
                       
Example usage:
    cryo_signal_enhance --starfile dir/some_star.star --output dir/class_averages.mrcs
    
    cryo_signal_enhance --starfile dir/some_star.star --output dir/class_averages.mrcs --num_classes 3000 --num_class_avg 2000
    
    cryo_signal_enhance ... --help
"""
    
    if '--help' in args or '-h' in args:
        print(help_str)
        return
    
    #%%#################### parse args #%%#################### 
    try:
        args = parser.parse_args()
    except:
        print(help_str)
        return
    
    # debug, check if small sample
    if args.sample == True:
        args.N = 1000
        args.num_classes = 10
        args.class_size = 20
        args.num_class_avg = 10
        args.em_num_inputs = 20
    
    for arg in vars(args):
        if arg in opts_keys:
            opts[arg] = getattr(args, arg)
    
    #%% input validation
    if (args.starfile == None):
        print("WARNING: Please provide a valid full-path for starfile")
        print(help_str)
        return 
    
    if (args.output == None):
        print("WARNING: Please provide a valid full-path for output")
        print(help_str)
        return 
    
    if '.star' not in args.starfile:
        args.starfile += '.star'
        
    if '.mrcs' not in args.output:
        args.output += '.mrcs'
    args.output = os.path.abspath(args.output)
        
    if not isvalid(opts):
        print(help_str)
        return
    
    if shutil.which('relion_refine') == None:
        print("ERROR: ")
        print("     'relion_refine' directory is not in path.")
        print("      You can add it to the path using:")
        print("      Python:   os.environ['PATH'] = 'relion_refine_dir_full_path:' + os.environ['PATH']")
        print("      Terminal: export PATH=relion_refine_dir_full_path:$PATH")
        print(" ")
        return
    
    #%% check for external basis file
    if (args.basis_file_in != None):
        if '.pkl' not in args.basis_file_in:
            args.basis_file_in += '.pkl'
            
        with open(args.basis_file_in, 'rb') as inp:
            basis_dict  = pickle.load(inp)
        class dummy:
            pass
        external_basis = dummy()
        external_basis.src = dummy()
        external_basis.freqs = basis_dict['freqs']
        external_basis.coeffs = basis_dict['coeffs']
        external_basis.src.n = basis_dict['n']
        external_basis.src.L = basis_dict['L']
        opts['debug'] = {'basis':external_basis}
        print("Using external sPCA basis from: '" + args.basis_file_in + "'")
    
    
    #%%#################### RUN #%%####################
    
    writeable_directory = os.path.dirname(os.path.abspath(args.output))
    ca_stack, metadata, basis = class_average( args.starfile , writeable_directory , opts )
    
    ####################### RUN #######################
    
    
    #%% wrap up
    
    # save stack to mrcs file
    ca_stack.save(args.output, overwrite=True)
    print("Wrote class averages to file: '" + args.output + "'")
    
    # save basis to pickle file
    if (args.basis_file_out != None):       
        if '.pkl' not in args.basis_file_out:
            args.basis_file_out += '.pkl'
        args.basis_file_out = os.path.abspath(args.basis_file_out)
        basis_dict = {"coeffs":basis.coeffs,"freqs":basis.freqs,"n":basis.src.n,"L":basis.src.L}
        with open(args.basis_file_out, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(basis_dict, outp)
        print("Wrote sPCA basis to file: '" + args.basis_file_out + "'")
        
    # save metadata to mat file
    #savemat(args.output+".mat", metadata)
    
    print(" ");
    
#%% check validity of arguments
def isvalid(opts):
    tf = True
    if opts['num_classes'] > opts['N']:
        print(f"ERROR: num_classes({opts['num_classes']}) > N({opts['N']})")
        tf = False
    if opts['class_size'] > opts['N']:
        print(f"ERROR: class_size({opts['class_size']}) > N({opts['N']})")
        tf = False
    if opts['num_class_avg'] > opts['num_classes']:
        print(f"ERROR: num_class_avg({opts['num_class_avg']}) > num_classes({opts['num_classes']})")
        tf = False
    if opts['em_num_inputs'] > opts['class_size']:
        print(f"ERROR: em_num_inputs({opts['em_num_inputs']}) > class_size({opts['class_size']})")
        tf = False
    return tf

#%%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#%% MAIN
if __name__ == "__main__":
    sys.exit(main())