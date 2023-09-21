
# disable PTX_AVAILABLE
find ./ -name "*.h" -exec sed -Ei "s:#define.*PTX_AVAILABLE:// \0:g" {} +

# fix inference_context.h to make it could be migrate
git apply << 'DIFF___'
diff --git a/csrc/transformer/inference/includes/inference_context.h b/csrc/transformer/inference/includes/inference_context.h
index aaf56855..001be555 100644
--- a/csrc/transformer/inference/includes/inference_context.h
+++ b/csrc/transformer/inference/includes/inference_context.h
@@ -12,7 +12,24 @@
 #include <vector>
 #include "cublas_v2.h"
 #include "cuda.h"
+#include <array>
+#include <unordered_map>
+namespace at {
+  namespace cuda {
+    dpct::queue_ptr getCurrentCUDAStream() {
+      auto device_type = c10::DeviceType::XPU;
+      c10::impl::VirtualGuardImpl impl(device_type);
+      c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
+      auto& queue = xpu::get_queue_from_stream(c10_stream);
+      return &queue;
+    }

+    dpct::queue_ptr getStreamFromPool(bool) {
+      // not implemented
+      return nullptr;
+    }
+  }
+}
 #define MEGABYTE (1024 * 1024)
 #define GIGABYTE (1024 * 1024 * 1024)

DIFF___
