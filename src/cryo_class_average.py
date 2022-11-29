### Written by Guy sharon, Tamir bendory, and Yoel shkolnisky
### Guysharon1995@gmail.com

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import eigsh
import math as math_utils

from src.aspire.source import RelionSource, ArrayImageSource
from src.aspire.source.mrcstack import MrcStack
from src.aspire.basis.fspca import FSPCABasis
from src.aspire.storage import StarFile
from src.aspire.image import Image
from src.aspire.noise import AnisotropicNoiseEstimator

import scipy, os, sys, copy, binascii, subprocess, mrcfile, stat
import concurrent.futures, time, logging, warnings, shutil, psutil
from scipy.spatial.transform import Rotation
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager
from datetime import datetime
from tqdm import tqdm
import progress.bar

#%%
        
def class_average(starfile_fullpath, writeable_dir, opts):
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    
    ############################### INIT ###############################
    global timings # this global variable will hold timings information for different stages
    if ('timings' not in globals()): timings = {}
    warnings.filterwarnings("ignore") # I get an annoying socket warning
    t_start = time.perf_counter()
    
    PrepareOpts(starfile_fullpath, writeable_dir, opts)
    
    #MonitorResources()
    ############################ PREPROCESS ############################
    
    my_log("Preprocessing {} images from {}...".format(opts['N'],starfile_fullpath), opts['verbose'])
    preprocessed_source = PreprocessData(starfile_fullpath,opts)
    
    ############################### sPCA ###############################
    
    my_log("Calculating sPCA basis...", opts['verbose'])
    basis = FSPCA(preprocessed_source,opts)
    
    ######################### 2D CLASSIFICATION ########################
    
    my_log("2D Classification...", opts['verbose'])
    classes, classes_ref, chosen_classes, original_chosen_classes = Classify_2D(basis, opts)
    
    ############################### EM ################################
    
    my_log("Expectation Maximization...", opts['verbose'])
    ca_stack = EM_class_average(classes, classes_ref, starfile_fullpath, opts)
    
    ############################# WRAP UP ##############################
    
    metadata = {}
    try:
        timings["total"] = time.perf_counter() - t_start
        my_log("Done.", True)
        print(" ")
        print(f"Runtime summary for {basis.src.n} projections of size {preprocessed_source.original_image_size}x{preprocessed_source.original_image_size} (downsampled to {basis.src.L}) ")
        
        print(f"Total Runtime                           {timings['total']:6.0f} seconds")
        print(f"    Preprocess                          {timings['preprocess']:6.0f} seconds")
        print(f"        Formatting starfile             {timings['get_starfile']:6.0f} seconds")
        print(f"        Phase flipping                  {timings['phase_flip']:6.0f} seconds")
        print(f"        Downsampling ({preprocessed_source.original_image_size:3}->{basis.src.L:3})         {timings['downsample']:6.0f} seconds")
        print(f"        Normalizing                     {timings['normalize']:6.0f} seconds")
        print(f"        Estimating noise                {timings['estimate_noise']:6.0f} seconds")
        print(f"        Whitening                       {timings['whiten']:6.0f} seconds")
        print(f"        Calculating pipeline            {timings['cache_preprocess']:6.0f} seconds")
        print(f"        Inverting contrast              {timings['invert_contrast']:6.0f} seconds")
        print(f"    Computing sPCA basis                {timings['spca']:6.0f} seconds")
        print(f"    2D Classification:                  {timings['class_2d']:6.0f} seconds")
        print(f"        Initial classification          {timings['initial_classification']:6.0f} seconds")
        print(f"        Sorting classes                 {timings['sort_classes']:6.0f} seconds")
        print(f"    Expectation Maximization            {timings['em']:6.0f} seconds")
        print(" ")
        
        metadata = {"starfile_fullpath": starfile_fullpath, "timings": timings, "opts": opts,
                   'raw_images_size': preprocessed_source.original_image_size,'num_images':basis.src.n,
                   'classes': classes,'classes_ref': classes_ref, 'chosen_classes': original_chosen_classes}
    except:
        timings = {}    
   
    return ca_stack, metadata, basis

#%%
def PrepareOpts(starfile_fullpath, writeable_dir, opts):
    if opts['norm']:
        opts['norm'] = 'norm'
    else:
        opts['norm'] = 'dont_check_norm'
    
    logger = logging.getLogger(__name__)
    logger.disabled = True
    logging.disable()
    starfile = StarFile(starfile_fullpath)
    os.chdir(os.path.dirname(os.path.abspath(starfile_fullpath)))
    N = len(starfile.get_block_by_index(-1))
    opts['N'] = min(N,opts['N'])
    opts["starfile"] = starfile
    opts['writeable_dir'] = writeable_dir
    
#%%
def PreprocessData(starfile_fullpath,opts):
    # source = RelionSource(starfile_fullpath, pixel_size=1,max_rows=opts["N"], data_folder = os.getcwd())
    # This function is in charge of taking the raw images, and transforming them to images that are easier to work with.
    # For example, raw images are usually too big, so we will downsample them.
    # Also, raw images contain CTF artifacts, which we also want to remove before processing the images.
    # Whereas we don't entirly remove the CTF from the raw images in this stage, we do corrct the sign of the phase (phase-flipping).   
    
    global timings
    if ('timings' not in globals()): timings = {}
    t_start = time.perf_counter()
    
    #
    timings['preprocess']           = 0
    timings['get_starfile']         = 0
    timings['phase_flip']           = 0
    timings['downsample']           = 0
    timings['normalize']            = 0
    timings['estimate_noise']       = 0
    timings['whiten']               = 0
    timings['cache_preprocess']     = 0
    timings['invert_contrast']      = 0 
    
    # if there exist debug basis, take it
    if ('debug' in opts.keys()) and ('basis' in opts['debug'].keys()):        
        source = custom_empty()
        source.starfile = opts["starfile"]
        source.original_image_size = ReadFromMRCS(source.starfile.get_block_by_index(-1)['_rlnImageName'][0]).shape[1]
        return source
    
    # get starfile
    t0 = time.perf_counter()
    my_log("    Preparing starfile in the correct format...", opts['verbose'])
    starfile           = GetStarfileInCorrectFormat(starfile_fullpath,opts) # because of an annoying bug of RelionSource
    tmp_starfile_fname = tempname(opts['writeable_dir']) + ".star"
    starfile.write(tmp_starfile_fname) # we need to write the starfile to a file so that RelionSource can read it.
    
    # if there exist debug preprocessed images, take them
    if ('debug' in opts.keys()) and ('preprocessed_source' in opts['debug'].keys()):
        my_warning(" USED EXTERNAL 'preprocessed_source'", opts['verbose'])
        source = opts['debug']['preprocessed_source']
        source.starfile = starfile
        os.remove(tmp_starfile_fname)
        return source
    
    # create RelionSource object
    source = RelionSource(tmp_starfile_fname, pixel_size=getPixA(starfile),max_rows=opts["N"], data_folder = os.getcwd())
    os.remove(tmp_starfile_fname)
    original_image_size = source.L
    timings['get_starfile'] = time.perf_counter() - t0
    
    # flags for operations (phase-flip, downsample, normalize, whiten, invert-contrast)
    if 'preprocess_flags' in opts.keys():
        flags = opts['preprocess_flags']
    else:
        flags = [1,1,0,1,0]
    
    # phase flip
    t0 = time.perf_counter()
    if flags[0]:
        my_log("    adding phase-flip to pipeline...", opts['verbose'])
        source.phase_flip()
    timings['phase_flip'] = time.perf_counter() - t0

    # downsample
    t0 = time.perf_counter()
    if flags[1]:
        my_log(f"    adding downsample to pipeline ({original_image_size}->{opts['downsample']})...", opts['verbose'])
        source.downsample(opts["downsample"])
    timings['downsample'] = time.perf_counter() - t0

    # normalize
    t0 = time.perf_counter()
    if flags[2]:
        my_log("    adding background normalization to pipeline...", opts['verbose'])
        source.normalize_background()
    timings['normalize'] = time.perf_counter() - t0

    # calculate pipeline
    t0 = time.perf_counter()
    source.cache(batch_size=opts['batch_size'],verbose=opts['verbose'])
    timings['cache_preprocess'] = time.perf_counter() - t0

    # estimate noise
    t0 = time.perf_counter()
    if flags[3]:
        my_log("    estimating white noise...", opts['verbose'])
        noise_estimator = AnisotropicNoiseEstimator(source,batchSize=opts["batch_size"])
    timings['estimate_noise'] = time.perf_counter() - t0

    # whiten
    t0 = time.perf_counter()
    if flags[3]:
        my_log("    adding whiten to pipeline...", opts['verbose'])
        source.whiten(noise_estimator.filter)
    timings['whiten'] = time.perf_counter() - t0
    
    # invert contrast
    t0 = time.perf_counter()
    if flags[4]:
        my_log("    inverting contrast (global phase flip)...", opts['verbose'])
        source.invert_contrast()
    timings['invert_contrast'] = time.perf_counter() - t0
    #print(source.images(0,1)[0])
    
    # calculate pipeline
    t0 = time.perf_counter()
    source.cache(batch_size=opts['batch_size'],verbose=opts['verbose'])
    timings['cache_preprocess'] = timings['cache_preprocess'] + time.perf_counter() - t0
    
    source.starfile = starfile # just so everything is in one place
    source.original_image_size = original_image_size

    timings['preprocess'] = time.perf_counter() - t_start
    return source

#%%
def FSPCA(source,opts):
    global timings
    if ('timings' not in globals()): timings = {}
    
    t_start = time.perf_counter()
    
    if ('debug' in opts.keys()) and ('basis' in opts['debug'].keys()):
        basis = opts['debug']['basis']
        my_log("    using external basis file", True)
    else:
        basis_src = ArrayImageSource(source.images(start=0,num=opts["N"]).data)
        basis = FSPCABasis(basis_src, components=opts['num_coeffs'], noise_var=0)
        basis.coeffs = basis.to_complex(basis.spca_coef)
        basis.freqs = basis.complex_angular_indices
    
    basis.starfile = source.starfile
    timings["spca"] = time.perf_counter() - t_start
    return basis
    
#%%
def GetStarfileInCorrectFormat(starfile_fullpath,opts):
    # What I do in this function is take the original starfile, which (in the new version format) contains 2 blocks
    # of information (take a look at it in a text editor), and remove the first block, which usually includes data
    # that is shared across the different projections.
    # I ASSUME THAT IF THERE ARE 2 BLOCKS, THE FIRST BLOCK IS METADETA AND THE SECOND ONE IS LOOP!
    starfile = StarFile(starfile_fullpath) # read the starfile
    
    # opts["N"] specifies the number of projections the user wants to work on (N=np.inf means all of them).
    # If the user has chosen less than the maximum number of projections, we should cut them from the starfile to prevent unncessery processing.
    num_rows = len(starfile.get_block_by_index(-1))
    loop_block_name = list(starfile.blocks.keys())[-1]
    starfile.blocks[loop_block_name] = starfile.blocks[loop_block_name].drop(range(int(np.min((opts["N"],num_rows))),num_rows))
    
    if len(starfile.blocks) == 2: # check if it contains 2 blocks of information. If it's not, it means its in the old format, and no processing is needed
        first_block_keys  = starfile.get_block_by_index(0).iloc[0,:].keys()
        second_block_keys = starfile.get_block_by_index(1).iloc[0,:].keys()
        for i in range(len(first_block_keys)):
            if first_block_keys[i] not in second_block_keys: # before disposing of this attribute, add in to the second block, to each particle
                starfile.get_block_by_index(1)[first_block_keys[i]] = starfile.get_block_by_index(0).iloc[0,:][first_block_keys[i]]
        
        starfile.blocks.popitem(False) # remove the first block, so that the starfile looks like the old format
        
    key_list = starfile.get_block_by_index(-1).iloc[0, :].keys() # get the keys of the (now only) last block. (all of the different keys are listed in the object 'relion_metadata_fields', in the file 'relion.py').
    int_key_list = []
    for i in range(len(key_list)):
        # becuase of another annoying bug, I have to make sure that every int is written without a decimal point
        # so first locate all of the keys which are of type 'int'
        if RelionSource.relion_metadata_fields.get(key_list[i]) == int:
            int_key_list.append(key_list[i])

    # run across the remaining projections in the starfile, and remove the decimal point from keys of type 'int'
    for i in range(len(starfile.get_block_by_index(-1))):
        for j in range(len(int_key_list)):
            val_str = starfile.get_block_by_index(-1).iloc[i, :][int_key_list[j]]
            starfile.get_block_by_index(-1)[int_key_list[j]][i] = str(int(float(val_str)))

    return starfile

#%%
def Classify_2D(basis,opts):
    warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
    
    global timings
    if ('timings' not in globals()): timings = {}
    t_start = time.perf_counter()
    calc_metric = opts['debug_verbose']
    
    math = np
    if opts['class_gpu'] >= 0:
        math = cp
        if (opts['class_gpu'] > 0):
            best_gpu_id = opts['class_gpu'] - 1
            gpus_memories = [cp.cuda.Device(best_gpu_id).mem_info[0]]*opts['class_gpu']
        else:
            gpus_memories = []
            for i in range(cp.cuda.runtime.getDeviceCount()):
                try:
                    gpus_memories.append(cp.cuda.Device(i).mem_info[0])
                except:
                    gpus_memories.append(-1)
            best_gpu_id = np.argmax(gpus_memories)
            
        os.environ['CUPY_GPU_MEMORY_LIMIT'] = "90%"
        cp.cuda.Device(best_gpu_id).use()
            
        if opts['debug_verbose']:
            print(f"DEBUG: Chosen GPU #{best_gpu_id+1} with mem = {gpus_memories[best_gpu_id]}")
    
    # init
    N = basis.src.n
    num_classes     = opts["num_classes"]
    class_size      = opts["class_size"]
    #np.random.seed(0)

    # choose images at random (unless debug indices exists)
    chosen_classes  = np.random.choice(N, num_classes,replace=False)
    # chosen_classes = range(0,num_classes) # DEBUG REMOVE
    if ('debug' in opts.keys()) and ('chosen_classes' in opts['debug'].keys()):
        chosen_classes = opts['debug']['chosen_classes']
        my_warning("    USED DEBUG 'chosen_classes'", True)

    # normalize coefficients
    coeffs = math.array(math.transpose(basis.coeffs))
    coeffs = coeffs - math.mean(coeffs,axis=0)
    coeffs = coeffs / math.std(coeffs,axis=0)
    coeffs = coeffs / math.sqrt(math.sum(math.abs(coeffs)**2,0))
    
    my_log(f"    received {coeffs.shape[1]} coefficient vectors of size {coeffs.shape[0]}...", opts['verbose'])

    # rotexp will be used in order to steer the coefficients (rotate the images)
    coeffs_c        = math.conj(coeffs)
    thetas          = math.linspace(0,2*math.pi,round(360/opts["class_theta_step"]))[None,:]
    rotexp          = math.exp(1j*math.outer(math.array(basis.freqs),thetas))
    
    # initial classification
    classes = math.zeros([opts["class_size"],num_classes], dtype=math.dtype("uint"))
    has_init_pb = 0
    for batch_size in range(50,0,-1):
        if not (num_classes % batch_size) == 0:
            continue
        try:
            heads_rot_coeffs = math.zeros((batch_size,thetas.shape[1],coeffs.shape[0]),dtype=math.complex64)
            t0 = time.perf_counter()
            for i in range(0,num_classes,batch_size):
                
                for j in range(batch_size):
                    c_id = chosen_classes[j+i]
                
                    # pick a projection and calculate each rotation for it
                    tmp                 = math.tile(coeffs[:,c_id],(rotexp.shape[1],1)).transpose()
                    heads_rot_coeffs[j] = math.multiply(tmp,rotexp).transpose().conj()

                # calculate correlations across the different rotations
                # and take maximum correlation between all rotations
                corrs_across_rots   = math.matmul(heads_rot_coeffs,coeffs)
                max_corrs           = math.max(math.real(corrs_across_rots),axis=1)
                corrs_across_rots   = None
                
                corrs_across_rots_c = math.matmul(heads_rot_coeffs,coeffs_c)
                max_corrs_c         = math.max(math.real(corrs_across_rots_c),axis=1)
                corrs_across_rots_c = None
                
                # calculate the reflections between the projections
                ref_vec            = 2*(max_corrs < max_corrs_c)-1

                # merge correlations between reflected and not reflected
                max_corrs[ref_vec==1] = max_corrs_c[ref_vec==1]

                for j in range(batch_size):
                    # find maximum between these correlations
                    classes[:,j+i] = argmaxk(max_corrs[j], class_size)

                progress_bar('                                 Initial Classification:',i+has_init_pb,num_classes, opts['verbose'])
                has_init_pb = 1
            break
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            if (opts['debug_verbose']):
                print(e)
            continue
        # for i in range(num_classes):
        #     c_id = chosen_classes[i]
            
        #     # pick a projection and calculate each rotation for it
        #     tmp                = math.tile(coeffs[:,c_id],(rotexp.shape[1],1)).transpose()
        #     head_rot_coeffs    = math.multiply(tmp,rotexp).transpose().conj()
            
        #     # calculate correlations across the different rotations
        #     corrs_across_rots   = math.matmul(head_rot_coeffs,coeffs)
        #     corrs_across_rots_c = math.matmul(head_rot_coeffs,coeffs_c)
            
        #     # take maximum correlation between all rotations
        #     max_corrs          = math.max(math.real(corrs_across_rots),axis=0)
        #     max_corrs_c        = math.max(math.real(corrs_across_rots_c),axis=0)
            
        #     # calculate the reflections between the projections
        #     ref_vec            = 2*(max_corrs < max_corrs_c)-1
            
        #     # merge correlations between reflected and not reflected
        #     max_corrs[ref_vec==1] = max_corrs_c[ref_vec==1]
            
        #     # find maximum between these correlations
        #     classes[:,i] = argmaxk(max_corrs, class_size)
            
        #     progress_bar('    Initial Classification:',i,num_classes, opts['verbose'])
    if (i + has_init_pb != num_classes - 1):
        progress_bar('                                 Initial Classification:',num_classes - 1,num_classes, opts['verbose'])
    timings["initial_classification"] = time.perf_counter() - t0
    
    if calc_metric:
        #grade classes before refinement
        rots, _ = calcRotsFromStarfile(basis.starfile)
        if ~np.any(rots == None):
            metrics = math.zeros((len(chosen_classes),1))
            for i in range(len(chosen_classes)):
                metrics[i] = math.mean(gradeClass(classes[0:100,i],math.array(rots),5,math))
            my_log("    DEBUG: chosen classes metrics (before refinement):  {0:.4}".format(math.mean(metrics)), opts['debug_verbose'])
      
    # sort classes
    t0 = time.perf_counter()
    classes_ref     = math.zeros([class_size,num_classes],dtype=bool)
    class_grade_est = math.zeros((num_classes,))
    corrs           = math.zeros([class_size,class_size,thetas.shape[1]])
    corrs_c         = math.zeros([class_size,class_size,thetas.shape[1]])
    rot_coeffs      = math.zeros((thetas.shape[1],coeffs.shape[0],class_size),dtype=math.complex64)
    for i in range(num_classes):
        coeffs_inds    = coeffs[:,classes[:,i]]
        coeffs_indsT   = coeffs_inds.transpose().conj()
        coeffs_indsT_c = math.conj(coeffs_indsT)
        
        for k in range(thetas.shape[1]):

            # rotate coefficients
            tmp            = math.tile(rotexp[:,k],(class_size,1)).transpose()
            rot_coeffs[k]  = math.multiply(coeffs_inds,tmp)
        
        # calc correlations
        corrs    = math.abs(math.matmul(coeffs_indsT,rot_coeffs))
        corrs_c  = math.abs(math.matmul(coeffs_indsT_c,rot_coeffs))
            
        class_grade_est[i], members_grade, classes_ref[:,i] = gradeClassByCorrs(corrs,corrs_c,opts,thetas,math)
        sorted_locs = math.argsort(members_grade)[::-1]
        classes[:,i] = classes[sorted_locs,i]
        
        progress_bar("                                 Sorting classes:       ",i,num_classes, opts['verbose'])

    classes_ref[:,classes_ref[0,:]==False] = ~classes_ref[:,classes_ref[0,:]==False]
    timings["sort_classes"] = time.perf_counter() - t0

    # keep the best opts["num_class_avg"] classes
    locs            = argmaxk(class_grade_est,opts["num_class_avg"],math)
    classes         = classes[:,locs]
    classes_ref     = classes_ref[:,locs]
    original_chosen_classes = copy.deepcopy(chosen_classes)
    
    if calc_metric:
        #grade classes after refinement
        if ~np.any(rots == None):
            metrics = math.zeros((classes.shape[1],1))
            for i in range(classes.shape[1]):
                metrics[i] = math.mean(gradeClass(classes[0:100,i],math.array(rots),5,math))
            my_log("    DEBUG: chosen classes metrics (after refinement):  {0:.4}".format(math.mean(metrics)), opts['debug_verbose'])
  
    if math == cp:
        classes = classes.get()
        classes_ref = classes_ref.get()
        #chosen_classes  = chosen_classes[locs.get()]
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    #else:
        #chosen_classes  = chosen_classes[locs]
    timings["class_2d"] = time.perf_counter() - t_start
    
    return classes, classes_ref, chosen_classes, original_chosen_classes
   
#%%
def argmaxk(v,k,math=np):
    ind = math.argpartition(v, -int(k))[-int(k):]
    ind = ind[math.argsort(v[ind])][::-1]
    return ind
    
#%%
def rotate(basis,coef_vec,angles):
    rotated_coef = np.zeros((len(coef_vec),len(angles)))
    for i in range(len(angles)):
        rotated_coef[:,i] = basis.rotate(coef_vec,angles[i])
    return rotated_coef
    
#%%
def gradeClassByCorrs(corrs,corrs_c,opts,thetas,math):
    if corrs.ndim == 4: # use the batched version
        batch_size = corrs.shape[0]
        max_corrs = math.max(corrs,axis=1)
        best_ths  = thetas[0,math.argmax(corrs,axis=1)]
        
        max_corrs_c = math.max(corrs_c,axis=1)
        best_ths_c  = thetas[0,math.argmax(corrs_c,axis=1)]
        
        is_reflected = max_corrs_c > max_corrs
        
        m = 2*is_reflected-1
        if math == cp:
            _ , V = math.linalg.eigh(2*is_reflected-1)
            V = V[:,:,0]
        else:
            _ , V = scipy.linalg.eigh(2*is_reflected-1,eigvals=(m.shape[0]-1,m.shape[0]-1))
        
        class_ref = (V > 0).transpose([1,0])
        
        #
        locs              = is_reflected
        
        ipr_mat           = best_ths
        ipr_mat[locs]     = best_ths_c[locs]
        
        c = math.cos(ipr_mat)
        s = math.sin(ipr_mat)
        l = 1-2*locs
        rot_mat = math.zeros((batch_size,2*ipr_mat.shape[1],2*ipr_mat.shape[1]))
        rot_mat[:,0::2,0::2] = c
        rot_mat[:,1::2,0::2] = math.multiply(s,l)
        rot_mat[:,0::2,1::2] = -s
        rot_mat[:,1::2,1::2] = math.multiply(c,l)
        rot_mat = (rot_mat+rot_mat.transpose([0,2,1]))/2
        
        #
        if math == cp:
            D,V = cp.linalg.eigh(rot_mat)
            D = math.array([math.diag(D[i,-2:]) for i in range(batch_size)])
            V = V[:,:,-2:]
        else:
            D,V = scipy.linalg.eigh(rot_mat,eigvals=(rot_mat.shape[0]-2,rot_mat.shape[0]-1))
        rot_mat_est = math.matmul(V,np.matmul(D,V.transpose([0,2,1])))
        diffs = rot_mat_est-rot_mat
        diffs = math.sqrt(diffs[:,0::2,0::2]**2+diffs[:,1::2,0::2]**2+diffs[:,0::2,1::2]**2+diffs[:,1::2,1::2]**2)
        members_grade = -math.mean(diffs,1)
        grade = math.sum(D,[1,2])
        
    else:
        max_corrs = math.max(corrs,axis=0)
        best_ths  = thetas[0,math.argmax(corrs,axis=0)]
        
        max_corrs_c = math.max(corrs_c,axis=0)
        best_ths_c  = thetas[0,math.argmax(corrs_c,axis=0)]
        
        is_reflected = max_corrs_c > max_corrs
        
        m = 2*is_reflected-1
        if math == cp:
            # _,V = math.linalg.eigh(2*is_reflected-1)
            # V = V[:,0]
            _,V = eigsh(m*1.0,  k=1, return_eigenvectors=True)
        else:
            _ , V = scipy.linalg.eigh(2*is_reflected-1,eigvals=(m.shape[0]-1,m.shape[0]-1))
        
        class_ref = V > 0
        class_ref = math.reshape(class_ref,(class_ref.shape[0],))
        
        #
        locs              = is_reflected
        
        ipr_mat           = best_ths
        ipr_mat[locs]     = best_ths_c[locs]
        
        c = math.cos(ipr_mat)
        s = math.sin(ipr_mat)
        l = 1-2*locs
        rot_mat = math.zeros((2*ipr_mat.shape[0],2*ipr_mat.shape[0]))
        rot_mat[0::2,0::2] = c
        rot_mat[1::2,0::2] = math.multiply(s,l)
        rot_mat[0::2,1::2] = -s
        rot_mat[1::2,1::2] = math.multiply(c,l)
        rot_mat = (rot_mat+rot_mat.transpose())/2
        
        #
        if math == cp:
            # D,V = cp.linalg.eigh(rot_mat)
            # D = D[-2:]
            # V = V[:,-2:]
            D,V = eigsh(rot_mat,  k=2, return_eigenvectors=True)
        else:
            D,V = scipy.linalg.eigh(rot_mat,eigvals=(rot_mat.shape[0]-2,rot_mat.shape[0]-1))
        rot_mat_est = math.matmul(V,np.matmul(math.diag(D),V.transpose()))
        diffs = rot_mat_est-rot_mat
        diffs = math.sqrt(diffs[0::2,0::2]**2+diffs[1::2,0::2]**2+diffs[0::2,1::2]**2+diffs[1::2,1::2]**2)
        members_grade = -math.mean(diffs,0)
        grade = math.sum(D)
    
    return grade, members_grade, class_ref

def firstEigV(m,math,n = 3):
    v = math.random.randn(m.shape[0],1)
    for i in range(n):
        v = math.matmul(m,v) 
        v = v / math.linalg.norm(v)
    
    #[v,~] = eigs(gather(m),1);
    #v = gpuArray(v);    

#%%
def calcRotsFromStarfile(starfile):
    data       = starfile.get_block_by_index(-1)
    rots       = np.zeros((3,3,len(data)))
    trots      = np.zeros((3,3,len(data)))
    
    rec = data.iloc[0,:]
    if (not '_rlnAngleRot' in rec.keys()) or (not '_rlnAngleTilt' in rec.keys()) or (not '_rlnAnglePsi' in rec.keys()):
        return None,None
    
    for i in range(len(data)):
        rec = data.iloc[i,:]
        rot = float(rec['_rlnAngleRot'])
        tilt = float(rec['_rlnAngleTilt'])
        psi = float(rec['_rlnAnglePsi'])
        m = Rotation.from_euler('zyz', [psi, tilt, rot], degrees=True).as_matrix()
        trots[:,:,i] = np.stack([m[:,1],m[:,0],-m[:,2]])
        rots[:,:,i] = trots[:,:,i].transpose()
       
    return rots,trots

#%%
def gradeClass(_class,rots,mask_deg = 5,math = np):
    member_grades = math.zeros((len(_class),1))
    mask = math.cos(mask_deg * math.pi / 180)
    for i in range(len(_class)):
        tmp = math.abs(cosBetweenViewDirs(rots[:,:,_class[i]],rots[:,:,_class]))
        member_grades[i] = math.mean(tmp > mask);
    return member_grades
    
def cosBetweenViewDirs(rot,rots):
    viewing_dir  = rot[:,2]
    viewing_dirs = rots[:,2,:]
    results = viewing_dir @ viewing_dirs;
    return results
    
#%%
def EM_class_average(classes, classes_ref, starfile_fullpath, opts):
    global timings
    if ('timings' not in globals()): timings = {}
    t_start = time.perf_counter()
    
    warnings.filterwarnings("ignore")
    ####################################################################
    if opts["ctf"]:
        opts["ctf"] = "--ctf"
    else:
        opts["ctf"] = ""
    
    if ("em_gpu" in opts.keys()) and (opts["em_gpu"]):
        opts["em_gpu"] = "--gpu 4" 
    else:
       opts["em_gpu"] = ""
    
    ####################################################################
    # init
    n_ca                    = opts["num_class_avg"]
    n_inputs                = opts["em_num_inputs"]
    datafiles_folder        = tempname(opts['writeable_dir'])
    #f[em_opts.n_ca,1]       = parallel.FevalFuture;
    tmpname                 = tempname()
    mrcsfname               = os.path.join(datafiles_folder,'em_inputs_images.mrcs')
    
    if np.mean(classes_ref[0:n_inputs,0:n_ca]) > 0.5:
        classes_ref = ~classes_ref
    
    
    ######################### find indicies of reflected images ######################### 
    n_reflected_projs = np.sum(classes_ref[0:n_inputs,0:n_ca] == 1)
    reflected_projs_inds_cat = np.zeros((n_reflected_projs,),dtype=int)
    idx = 0
    for i in range(n_ca):
        projs_inds = classes[find(classes_ref[0:n_inputs,i] == 1),i]
        reflected_projs_inds_cat[idx:idx+len(projs_inds)] = projs_inds
        idx = idx + len(projs_inds)
    reflected_projs_inds_cat,ic = np.unique(reflected_projs_inds_cat, return_inverse=True)
    
    ########################## read starfile ########################## 
    starfile    = StarFile(starfile_fullpath)
    sf_data     = starfile.get_block_by_index(-1)
    pixA        = getPixA(starfile)
    
    ########################## create mrcs for flipped images ##########################
    mrcs_paths = ["" for i in range(len(reflected_projs_inds_cat))]
    for i in range(len(reflected_projs_inds_cat)): 
        mrcs_paths[i] = sf_data.get("_rlnImageName")[reflected_projs_inds_cat[i]]
    if len(mrcs_paths) != 0:
        reflected_projs_cat = Image(np.flip(readFromMultipleMrcs(mrcs_paths, opts['verbose']).data,axis=1))
    
        WriteMRC(mrcsfname,reflected_projs_cat)
    
    ############################### loop ###############################
    tmp_im = ReadFromMRCS(sf_data.get("_rlnImageName")[0])
    image_size = tmp_im.data.shape[1]
    if not os.path.exists(datafiles_folder + "/em_input/"):
        os.mkdir(datafiles_folder + "/em_input/")
    if not os.path.exists(datafiles_folder + "/em_output/"):
        os.mkdir(datafiles_folder + "/em_output/")
    ca_stack = Image(np.zeros((n_ca,image_size,image_size)))
    dirIds = ["" for x in range(n_ca)]
    for i in range(n_ca):
        # create starfile
        dirIds[i]           = str(i)
        indir               = datafiles_folder + "/em_input/"  + dirIds[i]
        outdir              = datafiles_folder + "/em_output/" + dirIds[i]
        starfname           = indir + "/" + "in.star"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not os.path.exists(indir):
            os.mkdir(indir)
        
        ###########################################################
        l1 = np.sum(classes_ref[0:n_inputs,0:i])
        l2 = np.sum(classes_ref[0:n_inputs,0:i+1])
        inds = ic[l1:l2]
        curr_starfile = copy.deepcopy(starfile) # we don't want to affect the original starfile
        StarFileSetData(curr_starfile, sf_data.iloc[classes[0:n_inputs,i],:])
        jj = 0
        for j in range(n_inputs):
            if classes_ref[j,i] == True:
                curr_starfile.get_block_by_index(-1).iloc[j]['_rlnImageName'] = "{0:06}@{1}".format(inds[jj]+1,mrcsfname)
                jj += 1
        curr_starfile.write(starfname)
        
        # cmd = "source /opt/relion3.1/setup.sh; relion_refine --o {0} --i {1} {2} --pad 2 --iter {3} --tau2_fudge 2 --particle_diameter {4} --K 1 --zero_mask --oversampling 1 --psi_step 5 --offset_range 5 --sym c1 --offset_step 2 --{5} --scale --j 64 {6} --random_seed -1 --verb 0 & ".format(
        #         outdir + "/run" , starfname , opts["ctf"], opts["iter"] , int(np.ceil(0.5*pixA*image_size)) , opts["norm"], opts["em_gpu"])
        # p.append(subprocess.Popen(['bash', '-c', cmd],cwd="/",stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        #_ = subprocess.check_output(['bash', '-c', cmd], stderr=subprocess.DEVNULL)
        progress_bar("                                 Creating STAR files for EMs:          ", i, n_ca, opts['verbose'])
        ###########################################################

    # [p[i].communicate() for i in range(len(p))]
    # #_ = subprocess.check_output(['bash', '-c', cmds], stderr=subprocess.DEVNULL)
    # fetched_image = np.zeros(len(dirIds))
    # progress_bar("        fetching results...", 0, n_ca, opts['verbose'])
    # while ~np.all(fetched_image):
    #     for i in range(len(dirIds)):
    #         averagefname = "{0}/run_it{1:03d}_classes.mrcs".format(datafiles_folder + "/em_output/" + dirIds[i],opts['iter'])
    #         if path.exists(averagefname) and fetched_image[i] == 0:
    #             try:
    #                 ca_stack[i] = ReadMRC(averagefname).data
    #                 fetched_image[i] = 1
    #             except: 
    #                 pass
    #     progress_bar("        fetching results...", int(np.sum(fetched_image)-1), n_ca, opts['verbose'])

    warnings.filterwarnings("ignore")
    if (opts["em_par"]):
        #my_log('    Applying EM to each class...', opts['verbose'])
        particle_diameter_A = 0.5*pixA*image_size
        
        ## 1
        # t0 = time.perf_counter()
        
        # WriteEmBashFile(opts,datafiles_folder,particle_diameter_A)
        # st = os.stat('tmp_em_bash.sh')
        # os.chmod('tmp_em_bash.sh', st.st_mode | stat.S_IEXEC)
        # os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
        # _ = subprocess.check_output("./tmp_em_bash.sh", stderr=subprocess.DEVNULL)
        # os.remove("tmp_em_bash.sh")
        
        # fetched_image = np.zeros(n_ca)
        # while ~np.all(fetched_image):
        #     for i in range(n_ca):
        #         best_iter = opts["iter"]
        #         averagefname = "{0}/run_it{1:03d}_classes.mrcs".format(outdir,best_iter)
        #         if os.path.exists(averagefname) and fetched_image[i] == 0:
        #             try:
        #                 ca_stack[i] = ReadMRC(averagefname).data
        #                 fetched_image[i] = 1
        #             except: 
        #                 pass
        ##
        
        ## 2
        datas = []
        for i in range(n_ca):
            datas.append((datafiles_folder,dirIds[i],opts,particle_diameter_A))
        os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
        with concurrent.futures.ThreadPoolExecutor(max_workers=(os.cpu_count())) as executor:
            results = list( tqdm( executor.map(_rlnEM, datas), ascii=' #', bar_format ='{desc}|{bar}| {n_fmt}/{total_fmt}', total=n_ca,ncols=113, disable =(not opts['verbose']), desc = '                                 Applying EM to each class:             ') )
        for i in range(n_ca):
            ca_stack.data[i,:,:] = results[i].data
        ##
            
        ## 3
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     pool = [executor.submit(rlnEM,datafiles_folder,dirId,opts,0.5*pixA*image_size) for dirId in dirIds]
        
        # for i in range(n_ca):
        #     ca_stack.data[i,:,:] = pool[i].result().data
        ##
        
    else:
        for i in range(n_ca):
            ca_stack.data[i,:,:] = rlnEM(datafiles_folder,dirIds[i],opts,0.5*pixA*image_size).data
            progress_bar('                                 Applying EM to each class...          ',i,n_ca,opts['verbose'])
    
    timings["em"] = time.perf_counter() - t_start
    
    # clean up
    shutil.rmtree(datafiles_folder)
    
    return ca_stack
 
#%%
def WriteEmBashFile(opts,datafiles_folder,particle_diameter_A):    
    txt = f"""#!/bin/bash
source /opt/relion3.1/setup.sh
export NUMEXPR_MAX_THREADS={os.cpu_count()}
for i in {{0..{opts['num_class_avg']-1}}}
do
   eval "relion_refine --o {datafiles_folder}/em_output/$i/run --i {datafiles_folder}/em_input/$i/in.star {opts["ctf"]} --pad 2 --iter {opts["iter"]} --tau2_fudge 2 --particle_diameter {int(np.ceil(particle_diameter_A))} --K 1  --zero_mask  --oversampling 1 --psi_step 5 --offset_range 5 --sym c1 --offset_step 2 --{opts["norm"]} --scale --j 64 --random_seed -1 --verb 0 &"
done"""
    f = open("tmp_em_bash.sh", "w")
    f.write(txt)
    f.close()
    
def _rlnEM(data):
    return rlnEM(data[0],data[1],data[2],data[3])

def rlnEM(datafiles_folder,dirId,opts,particle_diameter_A):
    indir               = datafiles_folder + "/em_input/"  + dirId
    outdir              = datafiles_folder + "/em_output/" + dirId
    starfname           = indir + "/" + "in.star"
    cmd = "source /opt/relion3.1/setup.sh; relion_refine --o {0} --i {1} {2} --pad 2 --iter {3} --tau2_fudge 2 --particle_diameter {4} --K 1 --zero_mask --oversampling 1 --psi_step 5 --offset_range 5 --sym c1 --offset_step 2 --{5} --scale --j 64 --random_seed -1 --verb 0".format(
            outdir + "/run", starfname, opts["ctf"], opts["iter"], int(np.ceil(particle_diameter_A)) , opts["norm"])
    _ = subprocess.check_output(['bash', '-c', cmd], stderr=subprocess.DEVNULL)
    
    best_iter = opts["iter"]
    averagefname = "{0}/run_it{1:03d}_classes.mrcs".format(outdir,best_iter)
    ca = ReadMRC(averagefname)
    return ca

#%%
def SortByContrast(stack):
    n  = stack.shape[1]
    
    s, theta = my_RadiusNorm(n,my_fctr(n))
    idx = s <= 1/2
    
    c = np.zeros(stack.shape[0])
    for i in range(c.shape[0]):
        diffs = np.diff(stack[i],axis=0)**2
        c[i] = np.sum(diffs[idx[1:,:]])
        
    sorted_locs = np.argsort(-c)
    sorted_stack = Image(stack[sorted_locs])
    
    return sorted_stack, sorted_locs
    
#%%    
def tempname(writeable_dir = "", n = 14):
    # simply returns an n lengthed string of hexas
    hexas = str(binascii.b2a_hex(os.urandom(n)))[2:-1:]
    if not os.path.exists(f"{writeable_dir}/tmp"):
        os.mkdir(f"{writeable_dir}/tmp")
    os.mkdir(f"{writeable_dir}/tmp/{hexas}")
    return f"{writeable_dir}/tmp/{hexas}"

#%%
def find(boolarr):
    return [i for i, x in enumerate(boolarr) if x]

#%%
def readFromMultipleMrcs(mrcs_paths, verbose = True):
    q = ReadFromMRCS(mrcs_paths[0])
    projs = Image(np.zeros((len(mrcs_paths),q.shape[1],q.shape[2])))
    last_mrcs_path = ""
    mrcs_inds = []
    for i in range(len(mrcs_paths)):
        splitstr = mrcs_paths[i].split("@")
        mrcs_path = splitstr[1]
        if mrcs_path == last_mrcs_path:
            mrcs_inds = np.append(mrcs_inds,int(splitstr[0])-1)
            continue
        else:
            if len(mrcs_inds) != 0:
                inds = mrcs_inds - min(mrcs_inds)
                projs_from_mrcs = ReadMRC(last_mrcs_path,min(mrcs_inds),max(inds)+1)
                projs[i-len(inds):i,:,:] = projs_from_mrcs[inds,:,:]
            
            mrcs_inds = np.array([int(splitstr[0])-1])
            last_mrcs_path = mrcs_path
        progress_bar("                                 Creating MRCS file for flipped images:", i, len(mrcs_paths), verbose)

    progress_bar("                                 Creating MRCS file for flipped images:", len(mrcs_paths)-1, len(mrcs_paths), verbose)
    inds = mrcs_inds - min(mrcs_inds)
    projs_from_mrcs = ReadMRC(last_mrcs_path,min(mrcs_inds),max(inds)+1)
    projs[len(mrcs_paths)-len(inds):len(mrcs_paths),:,:] = projs_from_mrcs[inds,:,:]
    return projs

#%%
def ReadFromMRCS(atMRCSpath):
    tmp = atMRCSpath.split("@")
    mrc_stack = MrcStack(tmp[1])
    return mrc_stack.images(int(tmp[0])-1,1)

#%%
def ReadMRC(path,start = 0,num = np.inf):
    mrc_stack = MrcStack(path)
    return mrc_stack.images(start,num)

#%%
def WriteMRC(filename,images):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(images.data.astype(np.float32))

#%%
def StarFileSetData(starfile, data):
    data_block_name = list(starfile.blocks.keys())[-1]
    starfile.blocks[data_block_name] = data
    return starfile

#%%
def getPixA(starfile):
    # The pixel size (NOT DETECTOR SIZE) can be named differently in a starfile,
    # so we need to check every attribute name.
    # Also, the pixel size can be the detector size divided by the magnification * 1e4.
    # If the starfile contains more than one unique pixel size (tolerance = 1e-4),
    # raises an error.
    att_names = ["_pixA","_rlnPixelSize","_rlnImagePixelSize","_rlnMicrographOriginalPixelSize",
                  "pixA" ,"rlnPixelSize" ,"rlnImagePixelSize" ,"rlnMicrographOriginalPixelSize"]
    pixA_final = -1
    rlnDetectorPixelSize = -1
    rlnMagnification = -1
    for i in range(len(starfile)):
        
        rec = starfile.get_block_by_index(i)
        for j in range(len(att_names)):
            if att_names[j] in rec.keys():
                pixA = float(rec[att_names[j]][0])
                if (pixA_final == -1):
                    pixA_final = pixA;
                elif (abs(pixA_final-pixA) > 1e-4):
                    raise Exception('discrepancy in pixel size (two different results in the same starfile)')
        
        if '_rlnDetectorPixelSize' in rec.keys():
            rlnDetectorPixelSize = float(rec._rlnDetectorPixelSize[0])
        if 'rlnDetectorPixelSize' in rec.keys():
            rlnDetectorPixelSize = float(rec.rlnDetectorPixelSize[0])
        if '_rlnMagnification' in rec.keys():
            rlnMagnification = float(rec._rlnMagnification[0])
        if 'rlnMagnification' in rec.keys():
            rlnMagnification = float(rec.rlnMagnification[0])
            
        if rlnDetectorPixelSize != -1 and rlnMagnification != -1:
            pixA = rlnDetectorPixelSize / rlnMagnification * 1e4;
            if pixA_final == -1:
                pixA_final = pixA;
            elif np.abs(pixA_final-pixA) > 1e-4:
                raise Exception('discrepancy in pixel size (two different results in the same starfile)')
        
    if pixA_final == -1:
        raise Exception(f'''Pixel size not found in starfile.
Pixel size can be either attribute from the following list:
{att_names}
or can be calculated from "_rlnDetectorPixelSize" (or "rlnDetectorPixelSize") and "_rlnMagnification" (or rlnMagnification) using
pixel size = rlnDetectorPixelSize / rlnMagnification * 1e4''')
    
    return pixA_final

#%%
def my_phase_flip(source, starfile, opts):
    pixA = source.pixel_size
    n    = source.L
    
    BW = 1/(pixA/10)
    s, theta = my_RadiusNorm(n,my_fctr(n));
    s = s * BW
    s2 = s**2
    s4 = s**4
    
    ims    = source.images(0,source.n).data
    imshat = np.fft.fftshift(np.fft.fft2(ims),axes=(1,2))
    pb_title = datetime.now().strftime("%Y-%m-%d %H:%M:%S,000 INFO     phase-flipping...")
    for i in range(source.n):
        data = starfile.get_block_by_index(-1).iloc[i,:]
        A               = float(data['_rlnAmplitudeContrast'])
        Cs              = float(data['_rlnSphericalAberration'])
        DefocusAngle    = float(data['_rlnDefocusAngle'])*np.pi/180
        DefocusV        = float(data['_rlnDefocusV'])/10
        DefocusU        = float(data['_rlnDefocusU'])/10
        voltage         = float(data['_rlnVoltage'])
    
        _lambda = 1.22639/np.sqrt(voltage*1000+0.97845*voltage**2)
        h = my_cryo_CTF_Relion(_lambda,DefocusU,DefocusV,DefocusAngle,Cs,A,s2,s4,theta)
        
        imshat[i,:,:] = imshat[i,:,:]*np.sign(h)
        
        starfile.get_block_by_index(-1).iloc[i,:]['_rlnImageName'] = f"{i+1:06}@mrcs_pf.mrcs"
        
        progress_bar(pb_title,i,source.n, opts['verbose'])
        
    images_pf = Image(np.real(np.fft.ifft2(np.fft.ifftshift(imshat,axes=(1,2)))))
    
    images_pf.save("mrcs_pf.mrcs", overwrite=True)
    starfile.write("starfile_pf.star")
    return RelionSource("starfile_pf.star", pixel_size=pixA, max_rows=opts["N"])
        
def my_cryo_CTF_Relion(_lambda,DefocusU,DefocusV,DefocusAngle,Cs,A,s2,s4,theta):   
    DFavg = (DefocusU+DefocusV)/2
    DFdiff = (DefocusU-DefocusV)
    df = DFavg+DFdiff*np.cos(2*(theta-DefocusAngle))/2
    
    k2 = np.pi*_lambda*df
    k4 = 1570796.3267948965*Cs*_lambda**3
    chi = k4*s4-k2*s2
    h = np.sqrt(1-A**2)*np.sin(chi)-A*np.cos(chi)
    return h
    
def my_fctr(n):
    return np.ceil((np.array([n,n])+1)/2);

def my_RadiusNorm(n,org):
    x=int(org[0])
    arr = np.array(range(1-x,n-x+1))/n
    Y = np.tile(arr,(n,1))
    X = Y.transpose()
    r = np.sqrt(X**2+Y**2)
    theta = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            theta[i][j] = math_utils.atan2(Y[i][j],X[i][j])
    return r,theta

#%%
def progress_bar(title, current, total, is_show, bar_length=30):
    if (is_show == False): return
    global progress_bar_is_finished, progress_bar_start_datetime, progress_shadybar
    if (current == 0):
        progress_bar_is_finished = False
        progress_shadybar = progress.bar.Bar(title, max=total)
        dt                          = datetime.now()
        progress_bar_start_datetime = dt.strftime("%Y-%m-%d %H:%M:%S,")+f"{int(dt.microsecond/1000):03} INFO"
    if (progress_bar_is_finished):
        return
    progress_shadybar.next(1 + current - progress_shadybar.index)
    
    if (progress_bar_is_finished == False and progress_shadybar.remaining == 0):
        progress_bar_is_finished = True
        print(" ")
   
def progress_bar_OLD(title, current, total, is_show, bar_length=30):
    if (is_show == False): return
    global progress_bar_timer, progress_bar_start_datetime
    if (current == 0):
        progress_bar_timer          = time.perf_counter()
        dt                          = datetime.now()
        progress_bar_start_datetime = dt.strftime("%Y-%m-%d %H:%M:%S,")+f"{int(dt.microsecond/1000):03} INFO"

    current += 1
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    
    sys.stdout.flush()
    dur = int(time.perf_counter() - progress_bar_timer)
    sys.stdout.write('\r{0} {1} [{2}{3}] {4}/{5} [{6:3.0f}%] [{7}s]'.format(
        progress_bar_start_datetime,title,arrow,padding,current,total,fraction*100,dur))
    if current == total:
        print(" ")
    
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
        
def my_log(string, is_show):
    if (is_show == False): return
    logger = logging.getLogger(__name__)
    logging.disable(0)
    logger.disabled = False
    logger.info(string)
    logger.disabled = True
    logging.disable()
    
def my_warning(string, is_show):
    logger = logging.getLogger(__name__)
    logging.disable(0)
    logger.disabled = False
    logger.warning(string)
    logger.disabled = True
    logging.disable()
    
#%%
def SaveSPCABasisAsMat(fname,spca_basis):
    from scipy.io import savemat
    mat_basis = {'sPCA_data': {'Coeff': spca_basis.coeffs, 'Freqs': spca_basis.freqs}}
    savemat(fname, mat_basis)
    
def MonitorResources():
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1024/1024)

class custom_empty:
    pass

#%%
import glob

def find_file(fname):
    for f in glob.glob(fname, recursive=True):
        print(f)