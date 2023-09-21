
# fix cg::thread_block_tile<threadsPerHead> to auto
find ./third-party/ -type f -exec sed -Ei "s/cg::\S*/auto/g" {} +

# migrate thread_rank() to get_local_linear_id()
find ./third-party/ -type f -exec sed -i "s/thread_rank()/get_local_linear_id()/g" {} +

# migrate shfl to shuffle
find ./third-party/ -type f -exec sed -Ei "s/\.shfl/\.shuffle/g" {} +

# fix __half to sycl::half
find ./third-party/ -type f -exec sed -Ei "s/__half/sycl::half/g" {} +

# fix half2_raw to half2
find ./third-party/ -type f -exec sed -Ei "s/half2_raw/half2/g" {} +

# migrate meta_group_size to get_group_range().size()
find ./third-party/ -type f -exec sed -Ei "s/meta_group_size[(][)]/get_group_range().size()/g" {} +

# fix max_warps and elems undeclared
# find ./third-party/ -name "*.cpp" -exec sed -i "s/max_warps \* elems/reduce::max_warps \* 4/g" {} +

# fix #include <c10/cuda/CUDAStream.h>
find ./third-party/ -type f -exec sed -Ei "s:#include <c10/cuda/CUDAStream.h>:#include <ipex.h>:g" {} +

# change group_local_memory to group_local_memory_for_overwrite
find ./third-party -type f -exec sed -i "s/group_local_memory</group_local_memory_for_overwrite</g" {} +

# fix narrow cast error in pt_binding.cpp
find ./third-party/ -type f -exec sed -i "s/inline size_t GetMaxTokenLength()/inline int GetMaxTokenLength()/g" {} +
find ./third-party/ -type f -exec sed -i "s/const size_t mlp_1_out_neurons/const int mlp_1_out_neurons/g" {} +

# fix attn_softmax_v2 lacking of iterations
find ./third-party/ -type f -exec sed -i "s/attn_softmax_v2<T>/attn_softmax_v2<T, iterations>/g" {} +
