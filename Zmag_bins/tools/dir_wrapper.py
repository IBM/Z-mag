import re, os, shutil

class directory_tree:
    """ Handle output results and checkpoints
    """

    def __init__(self, mpi_comm, compute_BEM: bool, first_run: bool, output_dir: str, energy_terms: list, for_vacuum = False):
        """ Initialize the class
            ====================
            Args
            ----
            mpi_comm:       MPI.Comm. MPI communicator
            compute_BEM:    bool. True to compute the BEM matrix
            first_run:      bool. True if this is a first run and not a restart from a checkpointed state
            output_dir:     str. Absolute path to the output directory
            energy_terms:   list. Terms contributing to the weak form
            for_vacuum:     bool. True to create directories for subsequent calculations in the vacuum
        """
        self.mpi_comm = mpi_comm
        self.compute_BEM = compute_BEM
        self.first_run = first_run
        self.output_dir = output_dir
        self.energy_terms = energy_terms
        self.for_vacuum = for_vacuum
    
    def _clear_non_mesh_files(self, dir_path):
        """Helper: Remove all files except .xdmf, .h5, .msh from directory"""
        if not os.path.isdir(dir_path):
            return
        
        protected_exts = {'.xdmf', '.h5', '.msh'}
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename.lower())
                if ext not in protected_exts:
                    os.remove(file_path)

    def get_directory_tree_structure(self):
        """ Define directory tree for outputing results and checkpoints
            ===========================================================
        """
        if self.first_run:
            # Set output directories
            checkpoint_dir = os.path.join(self.output_dir,'checkpoints_run_1')
            results_dir = os.path.join(self.output_dir,'results_run_1')
            path_dirs = {
                'checkpoint_dir': checkpoint_dir, 
                'results_dir': results_dir
                }
            if "demag" in self.energy_terms:
                BEM_dir_name = 'BEM_MPI_n{}'.format(self.mpi_comm.size)
                BEM_dir = os.path.join(self.output_dir,BEM_dir_name+'_run_1')
                path_dirs['BEM_dir'] = BEM_dir
            if self.for_vacuum:
                for_vacuum_dir = os.path.join(self.output_dir,'for_vacuum_run_1')
                path_dirs['for_vacuum_dir'] = for_vacuum_dir
        else:
            # Set loading and output directories
            dir_names = os.listdir(self.output_dir)
            last_checkpoint = 0
            for name in dir_names:
                res = re.search(r'checkpoints_run_(\d*)', name)
                if res:
                    num = int(res.group(1))
                    last_checkpoint = num if num > last_checkpoint else last_checkpoint
            self.mpi_comm.Barrier()
            loading_dir = os.path.join(self.output_dir,'checkpoints_run_{}'.format(last_checkpoint))
            checkpoint_dir = os.path.join(self.output_dir,'checkpoints_run_{}'.format(last_checkpoint+1))
            results_dir = os.path.join(self.output_dir,'results_run_{}'.format(last_checkpoint+1))
            path_dirs = {
                'checkpoint_dir': checkpoint_dir, 
                'results_dir': results_dir,
                'loading_dir': loading_dir
                }
            if "demag" in self.energy_terms:
                BEM_dir_name = 'BEM_MPI_n{}'.format(self.mpi_comm.size)
                BEM_dir = os.path.join(self.output_dir,BEM_dir_name+'_run_{}'.format(last_checkpoint+1))
                path_dirs['BEM_dir'] = BEM_dir
            if self.for_vacuum:
                for_vacuum_dir = os.path.join(self.output_dir,'for_vacuum_run_{}'.format(last_checkpoint+1))
                path_dirs['for_vacuum_dir'] = for_vacuum_dir
        # Return path to directories
        return path_dirs
    
    
    def create_directory_tree(self):
        """ Create directory tree for outputing results and checkpoints
            ===========================================================
        """
        # Get directories
        path_dirs = self.get_directory_tree_structure()
        checkpoint_dir = path_dirs['checkpoint_dir']
        results_dir = path_dirs['results_dir']
        
        # Create directories
        if self.mpi_comm.rank == 0:
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)
            if os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            else:
                # Clear everything in the results directory apart from mesh files
                self._clear_non_mesh_files(results_dir)
            if "demag" in self.energy_terms:
                BEM_dir = path_dirs['BEM_dir']
                if self.compute_BEM:
                    if os.path.isdir(BEM_dir):
                        shutil.rmtree(BEM_dir)
                    os.makedirs(BEM_dir)
            if self.for_vacuum:
                for_vacuum_dir = path_dirs['for_vacuum_dir']
                if os.path.isdir(for_vacuum_dir):
                    shutil.rmtree(for_vacuum_dir)
                os.makedirs(for_vacuum_dir)
            os.makedirs(checkpoint_dir)