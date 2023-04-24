#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <torch/extension.h>
#include <c10/core/ScalarType.h>

#include "inference_onednn_wrappers.hpp"
#include "inference_sycl_layers.h"
#include "lru_cache.hpp"
#include "dnnl_ext.hpp"

namespace std {

template <> struct hash<dnnl::memory::dims> {
  size_t operator()(dnnl::memory::dims const& vec) const {
    size_t seed = vec.size();
    for(auto& i : vec) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}

template <typename T>
T concat(const T& t1, at::ScalarType d) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back((int64_t)d);

  return t;
}

template <typename T>
T concat(const T& t1, bool b) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back(b);

  return t;
}

template <typename T>
T concat(const T& t1, int b) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.push_back(b);

  return t;
}

template <typename T>
T concat(const T& t1, const T& t2) {
  T t;
  t.insert(t.end(), t1.begin(), t1.end());
  t.insert(t.end(), t2.begin(), t2.end());

  return t;
}

template <typename T1, typename T2, typename ...Ts>
T1 concat(const T1& t1, const T2& t2, const Ts&... ts) {
  return concat(concat(t1, t2), ts...);
}

// onednn parameters
using primitive_cache = onednnContext::lru_cache<dnnl::memory::dims, dnnl::primitive>;
static int cache_capacity = 512;

enum class behavior {
  query, infer, plain, blocking
};

/* dnnl::memory::desc md_from(const dnnl::memory::dims m_sz, behavior b = behavior::plain) { */
/*   /1* auto m_sz = dims_from(tensor.sizes()); *1/ */
/*   /1* auto m_strides = dims_from(tensor.strides()); *1/ */
/*   auto m_strides = c10::TensorType::contiguousStridesOf(m_sz); */
/*   auto data_type = cast(tensor.scalar_type()); */

/*   if (b == behavior::query) { */
/*     return dnnl::memory::desc(m_sz, data_type, tag::any); */
/*   } else { */
/*     // Warning: Encoding conflict possible! */
/*     if (match_prepacked_weight_tag(m_sz) != tag::undef) { */
/*       size_t A = 0, B = 1, a = 3, b[2] = {2,4}; */
/*       auto dim0 = m_sz[A] * m_sz[a]; */
/*       auto dim1 = m_sz[B] * m_sz[b[0]] * m_sz[b[1]]; */
/*       return memory::desc({dim0, dim1}, data_type, tag::any/1* TODO:specify detail information *1/); */
/*     } else { */
/*       return memory::desc(m_sz, data_type, m_strides); */
/*     } */
/*   } */

/*   throw std::exception(); */
/* } */

template <typename T, bool bmm>
struct onednn_matmul_impl {
  static int _(sycl::queue handle,
        bool trans_src,
        bool trans_wgt,
        int m,
        int n,
        int k,
        const float alpha,
        const float beta,
        const T* src_ptr,
        const T* wgt_ptr,
        T* dst_ptr,
        int batch);
};

template <bool bmm>
struct onednn_matmul_impl<bf16, bmm> {
  static int _(sycl::queue handle,
        bool trans_src,
        bool trans_wgt,
        int m,
        int n,
        int k,
        const float alpha,
        const float beta,
        const bf16* src_ptr,
        const bf16* wgt_ptr,
        bf16* dst_ptr,
        int batch) 
  {
      /*
       * src, [m, k], m: batch, k: in_feature
       * wgt, [k, n], n: k: in_features, out_feature
       * dst, [m, n], m: batch, n: out_features
       */
      device dev = handle.get_device();
      context ctx = handle.get_context();
      dnnl::engine engine = dnnl::sycl_interop::make_engine(dev, ctx);
      dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, handle);

      static thread_local primitive_cache cached(cache_capacity);

      dnnl::memory::dims src_dims, wgt_dims, dst_dims;
  
      if constexpr (bmm) {
          src_dims = {batch, m, k};
          wgt_dims = {batch, k, n};
          dst_dims = {batch, m, n};
      } else {
          src_dims = {m, k};
          wgt_dims = {k, n};
          dst_dims = {m, n};
      }
      
      dnnl::memory::desc src_md, wgt_md, dst_md;

      // add lru_cache
      dnnl::primitive compute;

      auto key = concat(
          dst_dims, src_dims, wgt_dims,
          false, false, (int)dnnl::memory::data_type::bf16);

      auto i_compute = cached.find(key);

      if (i_compute == cached.end()) {
        if constexpr (bmm) {
          src_md = dnnl::memory::desc(
              src_dims,
              dnnl::memory::data_type::bf16,
              trans_src ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
          wgt_md = dnnl::memory::desc(
              wgt_dims,
              dnnl::memory::data_type::bf16,
              trans_wgt ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
          dst_md = dnnl::memory::desc(
              dst_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::abc);
        } else {
          src_md = dnnl::memory::desc(
              src_dims,
              dnnl::memory::data_type::bf16,
              trans_src ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
          wgt_md = dnnl::memory::desc(
              wgt_dims,
              dnnl::memory::data_type::bf16,
              trans_wgt ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
          dst_md = dnnl::memory::desc(
              dst_dims, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ab);
        } 

        dnnl::primitive_attr attr;
        if (alpha != 1.0f) {
          attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 0);
        }
        if (beta != 0.0f) {
          dnnl::post_ops po;
          po.append_sum(beta);
          attr.set_post_ops(po);
        }
        

        /* dnnl::matmul::desc matmul_d(src_md, wgt_md, dst_md); */
        /* matmul_d.data.prop_kind = dnnl_prop_kind_t::dnnl_forward_inference; */
        /* matmul::primitive_desc matmul_pd(matmul_d, attr, engine); */
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, src_md, wgt_md, dst_md, attr);
        compute = dnnl::matmul(matmul_pd);

        cached.insert(std::make_pair(key, compute));
      } else {
        compute = i_compute->second;
      }

      // add lru_cache end 
  
      dnnl::primitive_ext ext_compute(compute);
      auto src_mem = dnnl::memory(*ext_compute.src_desc(), engine, (void*)src_ptr);
      auto wgt_mem = dnnl::memory(*ext_compute.weights_desc(), engine, (void*)wgt_ptr);
      auto dst_mem = dnnl::memory(*ext_compute.dst_desc(), engine, (void*)dst_ptr);
  
      std::unordered_map<int, dnnl::memory> matmul_args;
      if (alpha != 1.0f) {
        float alpha_v(alpha);
        dnnl::memory alpha_mem({{1}, dnnl::memory::data_type::f32, {1}}, engine, &alpha_v);
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, alpha_mem});
      }
      matmul_args.insert({DNNL_ARG_SRC, src_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, wgt_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});
  
      compute.execute(stream, matmul_args);
      /* stream.wait(); */
  }
};


template <bool bmm>
struct onednn_matmul_impl<sycl::half, bmm> {
  static int _(sycl::queue handle,
        bool trans_src,
        bool trans_wgt,
        int m,
        int n,
        int k,
        const float alpha,
        const float beta,
        const sycl::half* src_ptr,
        const sycl::half* wgt_ptr,
        sycl::half* dst_ptr,
        int batch)
  {
      /*
       * src, [m, k], m: batch, k: in_feature
       * wgt, [k, n], n: k: in_features, out_feature
       * dst, [m, n], m: batch, n: out_features
       */
      device dev = handle.get_device();
      context ctx = handle.get_context();
      dnnl::engine engine = dnnl::sycl_interop::make_engine(dev, ctx);
      dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, handle);

      static thread_local primitive_cache cached(cache_capacity);

      dnnl::memory::dims src_dims, wgt_dims, dst_dims;
  
      if constexpr (bmm) {
          src_dims = {batch, m, k};
          wgt_dims = {batch, k, n};
          dst_dims = {batch, m, n};
      } else {
          src_dims = {m, k};
          wgt_dims = {k, n};
          dst_dims = {m, n};
      }
      
      dnnl::memory::desc src_md, wgt_md, dst_md;

      // add lru_cache
      dnnl::primitive compute;

      auto key = concat(
          dst_dims, src_dims, wgt_dims,
          false, false, (int)dnnl::memory::data_type::f16);

      auto i_compute = cached.find(key);

      if (i_compute == cached.end()) {
        if constexpr (bmm) {
          src_md = dnnl::memory::desc(
              src_dims,
              dnnl::memory::data_type::f16,
              trans_src ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
          wgt_md = dnnl::memory::desc(
              wgt_dims,
              dnnl::memory::data_type::f16,
              trans_wgt ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
          dst_md = dnnl::memory::desc(
              dst_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::abc);
        } else {
          src_md = dnnl::memory::desc(
              src_dims,
              dnnl::memory::data_type::f16,
              trans_src ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
          wgt_md = dnnl::memory::desc(
              wgt_dims,
              dnnl::memory::data_type::f16,
              trans_wgt ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
          dst_md = dnnl::memory::desc(
              dst_dims, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab);
        } 

        dnnl::primitive_attr attr;
        if (alpha != 1.0f) {
          attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 0);
        }
        if (beta != 0.0f) {
          dnnl::post_ops po;
          po.append_sum(beta);
          attr.set_post_ops(po);
        }
        

        /* dnnl::matmul::desc matmul_d(src_md, wgt_md, dst_md); */
        /* matmul_d.data.prop_kind = dnnl_prop_kind_t::dnnl_forward_inference; */
        /* matmul::primitive_desc matmul_pd(matmul_d, attr, engine); */
        auto matmul_pd = dnnl::matmul::primitive_desc(engine, src_md, wgt_md, dst_md, attr);
        compute = dnnl::matmul(matmul_pd);

        cached.insert(std::make_pair(key, compute));
      } else {
        compute = i_compute->second;
      }

      // add lru_cache end 
  
      dnnl::primitive_ext ext_compute(compute);
      auto src_mem = dnnl::memory(*ext_compute.src_desc(), engine, (void*)src_ptr);
      auto wgt_mem = dnnl::memory(*ext_compute.weights_desc(), engine, (void*)wgt_ptr);
      auto dst_mem = dnnl::memory(*ext_compute.dst_desc(), engine, (void*)dst_ptr);
  
      std::unordered_map<int, dnnl::memory> matmul_args;
      if (alpha != 1.0f) {
        float alpha_v(alpha);
        dnnl::memory alpha_mem({{1}, dnnl::memory::data_type::f32, {1}}, engine, &alpha_v);
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, alpha_mem});
      }
      matmul_args.insert({DNNL_ARG_SRC, src_mem});
      matmul_args.insert({DNNL_ARG_WEIGHTS, wgt_mem});
      matmul_args.insert({DNNL_ARG_DST, dst_mem});
  
      compute.execute(stream, matmul_args);
      /* stream.wait(); */
  }
};

template <typename T, bool bmm>
int onednn_matmul(sycl::queue handle,
                  bool trans_src,
                  bool trans_wgt,
                  int m,
                  int n,
                  int k,
                  const float alpha,
                  const float beta,
                  const T* src_ptr,
                  const T* wgt_ptr,
                  T* dst_ptr,
                  int batch) {
  return onednn_matmul_impl<T, bmm>::_(handle,
                                       trans_src,
                                       trans_wgt,
                                       m,
                                       n,
                                       k,
                                       alpha,
                                       beta,
                                       src_ptr,
                                       wgt_ptr,
                                       dst_ptr,
                                       batch); 
}


template <typename T>
int onednn_matmul_ex(sycl::queue handle,
                     bool trans_src,
                     bool trans_wgt,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr)
{
    onednn_matmul<T, false>(
        handle, trans_src, trans_wgt, m, n, k, alpha, beta, src_ptr, wgt_ptr, dst_ptr, 1);
}

template <typename T>
int onednn_batchgemm(sycl::queue handle,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr,
                     bool trans_src,
                     bool trans_wgt,
                     int batch)
{
    onednn_matmul<T, true>(
        handle, trans_src, trans_wgt, m, n, k, alpha, beta, src_ptr, wgt_ptr, dst_ptr, batch);
}


template int onednn_matmul_ex(sycl::queue handle,
                              bool trans_src,
                              bool trans_wgt,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const bf16* src_ptr,
                              const bf16* wgt_ptr,
                              bf16* dst_ptr);


template int onednn_matmul_ex(sycl::queue handle,
                              bool trans_src,
                              bool trans_wgt,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const sycl::half* src_ptr,
                              const sycl::half* wgt_ptr,
                              sycl::half* dst_ptr);


template int onednn_batchgemm(sycl::queue handle,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const bf16* src_ptr,
                              const bf16* wgt_ptr,
                              bf16* dst_ptr,
                              bool trans_src,
                              bool trans_wgt,
                              int batch);


template int onednn_batchgemm(sycl::queue handle,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const sycl::half* src_ptr,
                              const sycl::half* wgt_ptr,
                              sycl::half* dst_ptr,
                              bool trans_src,
                              bool trans_wgt,
                              int batch);


