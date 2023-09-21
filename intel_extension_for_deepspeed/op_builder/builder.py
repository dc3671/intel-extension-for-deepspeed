"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import time
import importlib
import shutil
import subprocess
from pathlib import Path
from deepspeed.ops.op_builder.builder import OpBuilder, TORCH_MAJOR, TORCH_MINOR

c2s_run = None

class SYCLOpBuilder(OpBuilder):
    def builder(self):
        try:
            from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension
        except ImportError:
            from intel_extension_for_pytorch.xpu.utils import DPCPPExtension

        print("dpcpp sources = {}".format(self.sources()))
        dpcpp_ext = DPCPPExtension(
            name=self.absolute_name(),
            sources=self.strip_empty_entries(self.sources()),
            include_dirs=self.strip_empty_entries(self.include_paths()),
            extra_compile_args={
                'cxx': self.strip_empty_entries(self.cxx_args()),
            },
            extra_link_args=self.strip_empty_entries(self.extra_ldflags()) + self.fixed_aotflags())
        return dpcpp_ext

    def is_sycl_src(self):
        if os.environ.get('USE_SYCL'):
            return True
        else:
            return False

    def sycl_extension(self, code_path, include_paths):
        c2s_cmd = 'c2s'

        cuda_inc_path = os.environ.get('CUDA_INC_PATH')
        cuda_inc_flag = " --cuda-include-path=" + f'{cuda_inc_path}'

        # get input and output folder
        from .fused_adam import FusedAdamBuilder
        ds_root_path =os.path.dirname(FusedAdamBuilder().deepspeed_src_path("../../"))
        ds_root_path = os.path.abspath(ds_root_path)
        sycl_ds_kernel_path = "third-party"
        sycl_link_path = os.path.join(ds_root_path, sycl_ds_kernel_path)

        extra_args = " --use-experimental-features=local-memory-kernel-scope-allocation "
        sycl_include_paths = []
        for include_path in include_paths:
            ds_inc_path = os.path.join(ds_root_path, include_path)
            sycl_inc_path = os.path.join(sycl_link_path, include_path)
            sycl_include_paths.append(sycl_inc_path)
            extra_args += " --extra-arg=" + "\"" +  "-I " + f'{ds_inc_path}' + "\""

        # code_path should be relative path
        cuda_kernel_path = os.path.join(ds_root_path, code_path)

        out_root = " --out-root=" + f'{sycl_link_path}'
        in_root = " --in-root=" + f'{ds_root_path}'

        # check if there is rule.YAML
        idex_path = Path(__file__).parent.absolute()
        rule_file = os.path.join(idex_path, 'rule.YAML')
        print("************************************* rule_file : ", f'{rule_file}')
        if os.path.exists(rule_file):
            extra_args += " --rule-file " + f'{rule_file}'

        sources = ""
        sycl_sources = []
        processes_running = []

        # add pre_process and post_process cmd scripts
        pre_process_script = os.path.join(idex_path, 'pre_process.sh')
        post_process_script = os.path.join(idex_path, 'post_process.sh')
        print('*'*30, 'pre_process_script: ', pre_process_script)
        print('*'*30, 'post_process_script: ', post_process_script)

        global c2s_run
        if c2s_run is None:
            c2s_run = True
        if os.path.exists(pre_process_script) and c2s_run:
            p = subprocess.Popen('source ' + f'{pre_process_script}', stdout=subprocess.PIPE, shell=True)
            p.wait()

        for source in os.scandir(cuda_kernel_path):
            if '.cu' in source.name or '.cpp' in source.name:
                # sources += f' {os.path.join(cuda_kernel_path, source.name)}'
                cuda_source = f' {os.path.join(cuda_kernel_path, source.name)}'
                sycl_kernel_name = source.name.replace('.cu', '.sycl.cpp')
                sycl_kernel_abs_path = os.path.join(sycl_link_path, code_path, sycl_kernel_name)
                sycl_sources.append(os.path.join(sycl_link_path, code_path, sycl_kernel_name))
                # import pdb
                # pdb.set_trace()
                if (os.path.exists(sycl_sources[-1])):
                    print(f'skip migrate {sycl_sources[-1]}, we already have one.')
                    continue
                trans_cmd = c2s_cmd + cuda_inc_flag + extra_args + in_root + out_root + cuda_source
                print("**** processing ", f'{trans_cmd}')
                p = subprocess.Popen(f'{trans_cmd}', stdout=subprocess.PIPE, shell=True)
                processes_running.append(p)

        # trans_cmd = c2s_cmd + cuda_inc_flag + extra_args + in_root + out_root + sources
        exit_codes = [p.wait() for p in processes_running]

        if os.path.exists(post_process_script) and c2s_run:
            p = subprocess.Popen('source ' + f'{post_process_script}', stdout=subprocess.PIPE, shell=True)
            p.wait()
            c2s_run = False

        print("----------------------------- c2s job done! -----------------------------")
        return sycl_sources, sycl_include_paths

    def version_dependent_macros(self):
        # Fix from apex that might be relevant for us as well, related to https://github.com/NVIDIA/apex/issues/456
        version_ge_1_1 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 0):
            version_ge_1_1 = ['-DVERSION_GE_1_1']
        version_ge_1_3 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 2):
            version_ge_1_3 = ['-DVERSION_GE_1_3']
        version_ge_1_5 = []
        if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR > 4):
            version_ge_1_5 = ['-DVERSION_GE_1_5']
        return version_ge_1_1 + version_ge_1_3 + version_ge_1_5

    def cxx_args(self):
        cxx_flags = ['-fsycl', '-fsycl-targets=spir64_gen', '-g', '-gdwarf-4', '-O3', '-std=c++17', '-fPIC', '-DMKL_ILP64', '-fno-strict-aliasing']
        if os.environ.get('USE_MKL_GEMM'):
            cxx_flags.append('-DUSE_MKL_GEMM')
        return cxx_flags

    def extra_ldflags(self):
        return ['-fPIC', '-Wl,-export-dynamic']

    def fixed_aotflags(self):
        return ['-fsycl', '-fsycl-targets=spir64_gen', '-fsycl-max-parallel-link-jobs=8', '-Xs', "-options -cl-poison-unsupported-fp64-kernels,cl-intel-enable-auto-large-GRF-mode", '-Xs', "-device pvc"]

    def load(self, verbose=True):
        from deepspeed.git_version_info import installed_ops, torch_info  # noqa: F401
        if installed_ops[self.name]:
            return importlib.import_module(self.absolute_name())
        else:
            return self.jit_load(verbose)

    def jit_load(self, verbose=True):
        if not self.is_compatible(verbose):
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to it not being compatible due to hardware/software issue. {self.error_log}"
            )
        try:
            import ninja  # noqa: F401
        except ImportError:
            raise RuntimeError(
                f"Unable to JIT load the {self.name} op due to ninja not being installed."
            )

        self.jit_mode = True
        from intel_extension_for_pytorch.xpu.cpp_extension import load

        start_build = time.time()
        # Recognize relative paths as absolute paths for jit load

        sources = [self.deepspeed_src_path(path) for path in self.sources()]
        extra_include_paths = [
            self.deepspeed_src_path(path) for path in self.include_paths()
        ]

        # Torch will try and apply whatever CCs are in the arch list at compile time,
        # we have already set the intended targets ourselves we know that will be
        # needed at runtime. This prevents CC collisions such as multiple __half
        # implementations. Stash arch list to reset after build.
        '''
        torch_arch_list = None
        if "TORCH_CUDA_ARCH_LIST" in os.environ:
            torch_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
            os.environ["TORCH_CUDA_ARCH_LIST"] = ""
        '''

        op_module = load(
            name=self.name,
            sources=self.strip_empty_entries(sources),
            extra_include_paths=self.strip_empty_entries(extra_include_paths),
            extra_cflags=self.strip_empty_entries(self.cxx_args()),
            # extra_cuda_cflags=self.strip_empty_entries(self.nvcc_args()),
            extra_ldflags=self.strip_empty_entries(self.extra_ldflags()),
            verbose=verbose)

        build_duration = time.time() - start_build
        if verbose:
            print(f"Time to load {self.name} op: {build_duration} seconds")
        '''
        # Reset arch list so we are not silently removing it for other possible use cases
        if torch_arch_list:
            os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch_list
        '''
        return op_module


def sycl_kernel_path(code_path):
    # Always return a path like "SYCL_KERNEL_PATH/..."
    SYCL_KERNEL_PATH = "third-party"
    abs_source_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    rel_target_path = os.path.join(SYCL_KERNEL_PATH, code_path)

    # Jit_load mode require absolute path. Use abs path for copy
    # To get the absolute path of deepspeed
    # We use a non-abstract builder class instance to call deepspeed_src_path()
    # FusedAdamBuilder is one of such class instance
    from .fused_adam import FusedAdamBuilder
    abs_target_path = FusedAdamBuilder().deepspeed_src_path(rel_target_path)

    sycl_link_path = os.path.join(
        os.path.dirname(FusedAdamBuilder().deepspeed_src_path("")),
        SYCL_KERNEL_PATH)
    if not os.path.exists(sycl_link_path):
        # Create directory and link for sycl kernel:
        #   deepspeed/ops/SYCL_KERNEL_PATH-->../../SYCL_KERNEL_PATH
        sycl_dir_path = os.path.join(os.path.dirname(sycl_link_path),
                                     "../../" + SYCL_KERNEL_PATH)

        os.mkdir(sycl_dir_path)
        os.symlink("../../" + SYCL_KERNEL_PATH, sycl_link_path, True)
        print("Create directory and link for sycl kernel:{}-->{}".format(
            sycl_link_path,
            sycl_dir_path))

    import filecmp
    if (os.path.exists(abs_target_path) and filecmp.cmp(abs_target_path,
                                                        abs_source_path)):
        print("skip copy, {} and {} have the same content".format(
            abs_source_path,
            abs_target_path))
        return rel_target_path

    print("Copying SYCL kernel file from {} to {}".format(abs_source_path,
                                                          abs_target_path))
    os.makedirs(os.path.dirname(abs_target_path), exist_ok=True)
    shutil.copyfile(abs_source_path, abs_target_path)

    # Prebuild install mode require paths relative to the setup.py directory. Use the relative path.
    return rel_target_path


def sycl_kernel_include(code_path):
    import intel_extension_for_pytorch  # noqa: F401
    abs_path = os.path.join(Path(__file__).parent.absolute(), code_path)
    return abs_path
