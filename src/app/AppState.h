
// ============================================================================
// AppState.h — 宿主侧共享状态契约
//
// 原先全部堆在 main_progressive_octree.cpp（~1670 行）里的文件级全局状态、
// 共享类型与跨文件函数，按"目录即架构"的拆分收敛到这一处：
//   - src/app/AppState.cpp    全局定义
//   - src/app/CudaHost.cpp    initCuda/getUniforms/resetCUDA/updateOctree/
//                             renderCUDA/initCudaProgram/reload/reset
//   - src/loader/Loader.cpp   spawnLoader/spawnUploader（加载与上传线程）
//   - src/render/ImGuiPanels.cpp  Settings/Stats/Message 三个 ImGui 窗口
//   - src/app/main.cpp        main() + 拖放/命令行入口 + update/render lambda
//
// 本头文件只做声明与类型定义；定义在对应 .cpp 中，函数体与拆分前逐字一致。
// ============================================================================

#pragma once

#include <mutex>
#include <deque>
#include <vector>
#include <memory>
#include <atomic>
#include <string>

#include "cuda.h"
#include "cuda_runtime.h"

#include <glm/matrix.hpp>

#include "CudaPrint/CudaPrint.h"
#include "HostDeviceInterface.h"
#include "LasLoader.h"
#include "SimlodLoader.h"

struct GLRenderer;

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

constexpr uint64_t PINNED_MEM_POOL_SIZE = 200;         // pool of pointers to batches of pinned memory
constexpr uint64_t BATCH_STREAM_SIZE    = 50;          // ring buffer of batches that are async streamed to GPU
constexpr uint64_t MAX_BATCH_SIZE       = 1'000'000;   // each loaded batch comprises <size> points
constexpr int MAX_LOADQUEUE_SIZE        = 300;         // stop loading from disk if processing lags behind

// ---------------------------------------------------------------------------
// 共享类型
// ---------------------------------------------------------------------------

struct PinnedMemorySlot {
	void* memLocation;
	CUevent uploadEnd;

	// ReadFileEx with unbuffered flag has special alignment requirements.
	// memLocation + memOffset gives the correct beginning of the data
	uint64_t memOffset;
};

struct Point{
	float x;
	float y;
	float z;
	union{
		uint32_t color;
		uint8_t rgba[4];
	};
};

// 与 common/structures.cuh 中设备端 Point 是两份独立定义，仅靠 16 字节布局保持
// 兼容。此断言由 MSVC 检查宿主侧。
static_assert(sizeof(Point) == 16, "宿主 Point 必须与设备端 structures.cuh::Point 布局一致");

struct PointBatch{
	std::string file = "";
	int first = 0;
	int count = 0;
	std::shared_ptr<std::vector<Point>> points;
	PinnedMemorySlot pinnedMem;

	LasHeader lasHeader;
};

struct Settings{
	bool useHighQualityShading       = true;
	bool showBoundingBox             = false;
	bool doUpdateVisibility          = true;
	bool showPoints                  = true;
	bool colorByNode                 = false;
	bool colorByLOD                  = false;
	bool colorWhite                  = false;
	bool autoFocusOnLoad             = true;
	bool benchmarkRendering          = false;
	float LOD                        = 0.2f;
	float minNodeSize                = 64.0f;
	int pointSize                    = 1;
	float fovy                       = 60.0f;
	bool enableEDL                   = true;
	float edlStrength                = 0.8f;
};

struct PinnedMemPool{
	std::mutex mtx_pool;
	std::mutex mtx_register;

	std::deque<PinnedMemorySlot> pool;
	std::vector<PinnedMemorySlot> registered;

	PinnedMemPool();

	// put all registered batches back into the pool
	void refill();

	void reserveSlots(int numSlots);
	void reserveSlot();

	PinnedMemorySlot acquire();
	void release(PinnedMemorySlot slot);
	void release(std::vector<PinnedMemorySlot> slots);
};

// ---------------------------------------------------------------------------
// 全局状态（定义见 AppState.cpp）
// ---------------------------------------------------------------------------

extern CudaPrint cudaprint;

extern std::vector<std::string> paths;

extern bool lazInBatches;
extern std::deque<PointBatch> batchesToProcess;
extern std::deque<PointBatch> batchesInPinnedMemory;
extern std::deque<PointBatch> batchesInPageableMemory;
extern std::deque<PinnedMemorySlot> pinnedMemoryInUpload;
extern std::atomic_bool resetInProgress;
extern std::mutex mtx_uploader;
extern std::vector<std::unique_ptr<std::mutex>> mtx_loader;
extern std::mutex mtx_batchesToProcess;
extern std::mutex mtx_batchesInPinnedMemory;
extern std::mutex mtx_batchesInPageableMemory;
extern std::mutex mtx_pinnedMemoryInUpload;

extern int batchStreamUploadIndex;
extern CUdeviceptr cptr_points_ring[BATCH_STREAM_SIZE];

extern CUdevice device;
extern CUcontext context;
extern int numSMs;

extern CUdeviceptr cptr_buffer;
extern CUdeviceptr cptr_buffer_persistent;
extern CUdeviceptr cptr_nodes;
extern CUdeviceptr cptr_renderbuffer;
extern CUdeviceptr cptr_stats;
extern CUdeviceptr cptr_numBatchesUploaded;
extern CUdeviceptr cptr_batchSizes;
extern CUdeviceptr cptr_frameStart;
extern CUgraphicsResource cugl_colorbuffer;
extern CUevent ce_render_start, ce_render_end;
extern CUevent ce_update_start, ce_update_end;
extern cudaStream_t stream_upload, stream_download;

extern CudaModularProgram* cuda_program_update;
extern CudaModularProgram* cuda_program_render;
extern CudaModularProgram* cuda_program_reset;

// glm 1.0 的默认构造不再初始化为单位阵（0.9.9 会），显式给 identity
extern glm::mat4 transform;
extern glm::mat4 transform_updatebound;

extern Stats stats;
extern void* h_stats_pinned;

extern double t_drop_start;

extern Settings settings;

extern PinnedMemPool pinnedMemPool;

extern bool requestReset;
extern bool requestBenchmark;
extern std::atomic_bool requestStepthrough;
extern std::atomic_bool requestStep;
extern float renderingDuration;
extern uint32_t numPointsUploaded;
extern float loadStart;

extern float kernelUpdateDuration;
extern float totalUpdateDuration;
extern double minKernelUpdateDuration;
extern double maxKernelUpdateDuration;
extern double avgKernelUpdateDuration;
extern double cntKernelUpdateDuration;

extern float kernelRenderDuration;
extern float totalRenderDuration;
extern double minKernelRenderDuration;
extern double maxKernelRenderDuration;
extern double avgKernelRenderDuration;
extern double cntKernelRenderDuration;

extern std::atomic_uint64_t numPointsTotal;
extern std::atomic_uint64_t numPointsLoaded;
extern std::atomic_uint64_t numBytesTotal;
extern std::atomic_uint64_t numBytesLoaded;
extern std::atomic_uint64_t numThreadsLoading;
extern int numBatchesTotal;
extern int numBatchesProcessed;
extern bool lastBatchFinishedDevice;
extern uint64_t momentaryBufferCapacity;
extern uint64_t persistentBufferCapacity;
extern std::vector<double> processFrameTimes;

extern float lastFrameTime;
extern float timeSinceLastFrame;

extern float3 boxMin;
extern float3 boxMax;
extern float3 boxSize;

extern uint64_t frameCounter;

// ---------------------------------------------------------------------------
// 跨文件函数
// ---------------------------------------------------------------------------

void initCuda();
Uniforms getUniforms(std::shared_ptr<GLRenderer> renderer);
void resetCUDA(std::shared_ptr<GLRenderer> renderer);
void updateOctree(std::shared_ptr<GLRenderer> renderer);
void renderCUDA(std::shared_ptr<GLRenderer> renderer);
void initCudaProgram(std::shared_ptr<GLRenderer> renderer);
void reload();
void reset(std::shared_ptr<GLRenderer> renderer);

void spawnLoader(size_t i);
void spawnUploader(std::shared_ptr<GLRenderer> renderer);

void drawSettingsWindow(std::shared_ptr<GLRenderer> renderer);
void drawStatsWindow(std::shared_ptr<GLRenderer> renderer);
void drawMessageWindow();
