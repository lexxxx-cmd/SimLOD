
#include "AppState.h"

#include "unsuck.hpp"

using namespace std;

// ---------------------------------------------------------------------------
// 全局状态定义（声明见 AppState.h，初值与拆分前 main.cpp 顶部逐字一致）
// ---------------------------------------------------------------------------

CudaPrint cudaprint;

vector<string> paths = {
	"NONE",
	// "d:/dev/pointclouds/riegl/retz.las",
};

bool lazInBatches = false;
deque<PointBatch> batchesToProcess;
deque<PointBatch> batchesInPinnedMemory;
deque<PointBatch> batchesInPageableMemory;
deque<PinnedMemorySlot> pinnedMemoryInUpload;
atomic_bool resetInProgress;
mutex mtx_uploader;
vector<unique_ptr<mutex>> mtx_loader;
mutex mtx_batchesToProcess;
mutex mtx_batchesInPinnedMemory;
mutex mtx_batchesInPageableMemory;
mutex mtx_pinnedMemoryInUpload;

int batchStreamUploadIndex = 0;
CUdeviceptr cptr_points_ring[BATCH_STREAM_SIZE];

CUdevice device;
CUcontext context;
int numSMs;

CUdeviceptr cptr_buffer;
CUdeviceptr cptr_buffer_persistent;
CUdeviceptr cptr_nodes;
CUdeviceptr cptr_renderbuffer;
CUdeviceptr cptr_stats;
CUdeviceptr cptr_numBatchesUploaded;
CUdeviceptr cptr_batchSizes;
CUdeviceptr cptr_frameStart;
CUgraphicsResource cugl_colorbuffer;
CUevent ce_render_start, ce_render_end;
CUevent ce_update_start, ce_update_end;
cudaStream_t stream_upload, stream_download;

CudaModularProgram* cuda_program_update = nullptr;
CudaModularProgram* cuda_program_render = nullptr;
CudaModularProgram* cuda_program_reset  = nullptr;

// glm 1.0 的默认构造不再初始化为单位阵（0.9.9 会），显式给 identity
glm::mat4 transform = glm::mat4(1.0f);
glm::mat4 transform_updatebound = glm::mat4(1.0f);

Stats stats;
void* h_stats_pinned = nullptr;

double t_drop_start = 0.0;

Settings settings;

PinnedMemPool pinnedMemPool;

bool requestReset                  = false;
bool requestBenchmark              = false;
atomic_bool requestStepthrough     = false;
atomic_bool requestStep            = false;
float renderingDuration            = 0.0f;
uint32_t numPointsUploaded         = 0;
float loadStart                    = 0.0f;

float kernelUpdateDuration         = 0.0f;
float totalUpdateDuration          = 0.0f;
double minKernelUpdateDuration     = Infinity;
double maxKernelUpdateDuration     = 0.0;
double avgKernelUpdateDuration     = 0.0;
double cntKernelUpdateDuration     = 0.0;

float kernelRenderDuration         = 0.0f;
float totalRenderDuration          = 0.0f;
double minKernelRenderDuration     = Infinity;
double maxKernelRenderDuration     = 0.0;
double avgKernelRenderDuration     = 0.0;
double cntKernelRenderDuration     = 0.0;

atomic_uint64_t numPointsTotal     = 0;
atomic_uint64_t numPointsLoaded    = 0;
atomic_uint64_t numBytesTotal      = 0;
atomic_uint64_t numBytesLoaded     = 0;
atomic_uint64_t numThreadsLoading  = 0;
int numBatchesTotal                = 0;
int numBatchesProcessed            = 0;
bool lastBatchFinishedDevice       = false;
uint64_t momentaryBufferCapacity   = 0;
uint64_t persistentBufferCapacity  = 0;
vector<double> processFrameTimes;

float lastFrameTime = static_cast<float>(now());
float timeSinceLastFrame = 0.0;

float3 boxMin  = float3{InfinityF, InfinityF, InfinityF};
float3 boxMax  = float3{-InfinityF, -InfinityF, -InfinityF};
float3 boxSize = float3{0.0, 0.0, 0.0};

uint64_t frameCounter = 0;

// ---------------------------------------------------------------------------
// PinnedMemPool（声明见 AppState.h）
// ---------------------------------------------------------------------------

PinnedMemPool::PinnedMemPool(){

}

// put all registered batches back into the pool
void PinnedMemPool::refill(){
	pool.clear();
	pool.insert(
		pool.end(),
		make_move_iterator(registered.begin()),
		make_move_iterator(registered.end()));
}

void PinnedMemPool::reserveSlots(int numSlots){
	for(int i = 0; i < numSlots; i++){
		this->reserveSlot();
	}
}

void PinnedMemPool::reserveSlot(){
	uint64_t bytesPerSlot = MAX_BATCH_SIZE * sizeof(Point);
	// reserve some extra bytes in case of IO with special alignment
	uint64_t alignmentPadding = 1'048'576;

	void* pinnedMem;
	CUevent slotEvent;

	cuMemAllocHost((void**)&pinnedMem, bytesPerSlot + alignmentPadding);
	cuEventCreate(&slotEvent, 0);

	PinnedMemorySlot slot = {
		.memLocation = pinnedMem,
		.uploadEnd   = slotEvent,
	};
	pool.push_back(slot);

	lock_guard<mutex> lock(mtx_register);
	registered.push_back(slot);
}

PinnedMemorySlot PinnedMemPool::acquire(){
	lock_guard<mutex> lock(mtx_pool);

	if(pool.size() == 0){
		printfmt("pool is empty, allocating additional pinned memory slots \n");
		reserveSlot();
	}

	PinnedMemorySlot slot = pool.front();
	slot.memOffset = 0;

	pool.pop_front();

	return slot;
}

void PinnedMemPool::release(PinnedMemorySlot slot){
	lock_guard<mutex> lock(mtx_pool);

	pool.push_back(slot);
}

void PinnedMemPool::release(vector<PinnedMemorySlot> slots){

	if (slots.size() == 0) return;

	lock_guard<mutex> lock(mtx_pool);

	pool.insert(
		pool.end(),
		make_move_iterator(slots.begin()),
		make_move_iterator(slots.end()));
}
