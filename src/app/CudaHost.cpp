
#include "AppState.h"

#include "GLRenderer.h"
#include "CudaModularProgram.h"
#include "CudaPrint/CudaPrint.h"

#include "cudaGL.h"

#include "unsuck.hpp"

#include "LasLoader.h"
#include "SimlodLoader.h"

#include <format>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

void initCuda(){
	cuInit(0);
	cuDeviceGet(&device, 0);
	cuCtxCreate(&context, 0, device);
	cuStreamCreate(&stream_upload, CU_STREAM_NON_BLOCKING);
	cuStreamCreate(&stream_download, CU_STREAM_NON_BLOCKING);

	cuCtxGetDevice(&device);
	cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
}

Uniforms getUniforms(shared_ptr<GLRenderer> renderer){
	Uniforms uniforms;

	// glm 1.0 的默认构造不再初始化为单位阵（0.9.9 会），必须显式给出单位阵
	glm::mat4 world = glm::mat4(1.0f);
	glm::mat4 view = renderer->camera->view;
	glm::mat4 proj = renderer->camera->proj;
	glm::mat4 worldViewProj = proj * view * world;
	world = glm::transpose(world);
	view = glm::transpose(view);
	proj = glm::transpose(proj);
	worldViewProj = glm::transpose(worldViewProj);

	memcpy(&uniforms.world, &world, sizeof(world));
	memcpy(&uniforms.view, &view, sizeof(view));
	memcpy(&uniforms.proj, &proj, sizeof(proj));
	memcpy(&uniforms.transform, &worldViewProj, sizeof(worldViewProj));

	if(settings.doUpdateVisibility){
		transform_updatebound = worldViewProj;
	}

	glm::mat4 transform_inv_updatebound = glm::inverse(transform_updatebound);
	memcpy(&uniforms.transform_updateBound, &transform_updatebound, sizeof(transform_updatebound));
	memcpy(&uniforms.transformInv_updateBound, &transform_inv_updatebound, sizeof(transform_inv_updatebound));

	uniforms.width                    = static_cast<float>(renderer->width);
	uniforms.height                   = static_cast<float>(renderer->height);
	uniforms.fovy_rad                 = 3.1415f * renderer->camera->fovy / 180.0;
	uniforms.time                     = static_cast<float>(now());
	uniforms.boxMin                   = float3{0.0f, 0.0f, 0.0f};
	uniforms.boxMax                   = boxSize;
	uniforms.frameCounter             = frameCounter;
	uniforms.showBoundingBox          = settings.showBoundingBox;
	uniforms.doUpdateVisibility       = settings.doUpdateVisibility;
	uniforms.showPoints               = settings.showPoints;
	uniforms.colorByNode              = settings.colorByNode;
	uniforms.colorByLOD               = settings.colorByLOD;
	uniforms.colorWhite               = settings.colorWhite;
	uniforms.LOD                      = settings.LOD;
	uniforms.minNodeSize              = settings.minNodeSize;
	uniforms.pointSize                = settings.pointSize;
	uniforms.useHighQualityShading    = settings.useHighQualityShading;
	uniforms.persistentBufferCapacity = persistentBufferCapacity;
	uniforms.momentaryBufferCapacity  = momentaryBufferCapacity;
	uniforms.enableEDL                = settings.enableEDL;
	uniforms.edlStrength              = settings.edlStrength;

	return uniforms;
}

void resetCUDA(shared_ptr<GLRenderer> renderer){

	Uniforms uniforms = getUniforms(renderer);

	void* args[] = {
		&uniforms,
		&cptr_buffer_persistent,
		&cptr_nodes,
		&cptr_stats,
		&cudaprint.cptr,
		&cptr_numBatchesUploaded,
		&cptr_batchSizes,
	};

	// we only need one single thread to do the resetting on device
	uint32_t numGroups = 1;
	uint32_t workgroupSize = 1;

	auto res_launch = cuLaunchCooperativeKernel(cuda_program_reset->kernels["kernel"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		printfmt("CUDA kernel 'reset' failed.\n");
	}

	cuCtxSynchronize();
}

// incrementally updates the octree on the GPU
void updateOctree(shared_ptr<GLRenderer> renderer){

	// cuCtxSynchronize();

	Uniforms uniforms = getUniforms(renderer);

	int workgroupSize = 256;
	int numGroups     = 1 * numSMs;
	auto ptrPoints    = cptr_points_ring[0];

	void* args[] = {
		&uniforms, &ptrPoints,
		&cptr_buffer, &cptr_buffer_persistent,
		&cptr_nodes,
		&cptr_stats, &cptr_frameStart,
		&cudaprint.cptr,
		&cptr_numBatchesUploaded,
		&cptr_batchSizes,
	};

	cuEventRecord(ce_update_start, 0);
	auto res_launch = cuLaunchCooperativeKernel(cuda_program_update->kernels["kernel_construct"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str;
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(ce_update_end, 0);

	// benchmark kernel- slows down overall loading!
	if(requestBenchmark){
		cuCtxSynchronize();

		float duration;
		cuEventElapsedTime(&duration, ce_update_start, ce_update_end);

		kernelUpdateDuration    += duration;
		minKernelUpdateDuration = std::min(minKernelUpdateDuration, double(duration));
		maxKernelUpdateDuration = std::max(maxKernelUpdateDuration, double(duration));
		avgKernelUpdateDuration = (cntKernelUpdateDuration * avgKernelUpdateDuration + duration) / (cntKernelUpdateDuration + 1.0);
		cntKernelUpdateDuration += 1.0;
	}

	requestStep = false;
	numBatchesProcessed++;

	// cuCtxSynchronize();
}

// draw the octree with a CUDA kernel
void renderCUDA(shared_ptr<GLRenderer> renderer){

	Uniforms uniforms = getUniforms(renderer);

	static bool registered = false;
	static GLuint registeredHandle = -1;

	// interop 链路任一步静默失败都会导致画面全黑，逐级检查并打印
	auto checkCudaError = [](CUresult result, const char* what){
		if(result != CUDA_SUCCESS){
			const char* str;
			cuGetErrorString(result, &str);
			printfmt("CUDA interop error in {}: {} \n", what, str);
		}
	};

	CUresult r;

	r = cuGraphicsGLRegisterImage(
		&cugl_colorbuffer,
		renderer->view.framebuffer->colorAttachments[0]->handle,
		GL_TEXTURE_2D,
		CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
	checkCudaError(r, "cuGraphicsGLRegisterImage");

	// map OpenGL resources to CUDA
	vector<CUgraphicsResource> dynamic_resources = {cugl_colorbuffer};
	r = cuGraphicsMapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
	checkCudaError(r, "cuGraphicsMapResources");

	CUDA_RESOURCE_DESC res_desc = {};
	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	r = cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, cugl_colorbuffer, 0, 0);
	checkCudaError(r, "cuGraphicsSubResourceGetMappedArray");

	CUsurfObject output_surf;
	r = cuSurfObjectCreate(&output_surf, &res_desc);
	checkCudaError(r, "cuSurfObjectCreate");

	cuEventRecord(ce_render_start, 0);

	float time = static_cast<float>(now());
	int workgroupSize = 256;

	int maxActiveBlocksPerSM;
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
		cuda_program_render->kernels["kernel_render"], workgroupSize, 0);

	int numGroups = maxActiveBlocksPerSM * numSMs;

	void* args[] = {
		&cptr_renderbuffer,
		&uniforms,
		&cptr_nodes,
		&output_surf,
		&cptr_stats,
		&cptr_frameStart,
		& cudaprint.cptr
	};


	auto res_launch = cuLaunchCooperativeKernel(cuda_program_render->kernels["kernel_render"],
		numGroups, 1, 1,
		workgroupSize, 1, 1,
		0, 0, args);

	if(res_launch != CUDA_SUCCESS){
		const char* str;
		cuGetErrorString(res_launch, &str);
		printf("error: %s \n", str);
	}

	cuEventRecord(ce_render_end, 0);

	// benchmark kernel- slows down overall loading!
	if(requestBenchmark){
		cuCtxSynchronize();

		float duration;
		cuEventElapsedTime(&duration, ce_render_start, ce_render_end);

		kernelRenderDuration    += duration;
		minKernelRenderDuration = std::min(minKernelRenderDuration, double(duration));
		maxKernelRenderDuration = std::max(maxKernelRenderDuration, double(duration));
		avgKernelRenderDuration = (cntKernelRenderDuration * avgKernelRenderDuration + duration) / (cntKernelRenderDuration + 1.0);
		cntKernelRenderDuration += 1.0;
	}

	if(settings.benchmarkRendering){
		cuCtxSynchronize();
		cuEventElapsedTime(&renderingDuration, ce_render_start, ce_render_end);
	}

	cuSurfObjectDestroy(output_surf);
	cuGraphicsUnmapResources(static_cast<int>(dynamic_resources.size()), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

	cuGraphicsUnregisterResource(cugl_colorbuffer);
}

// compile kernels and allocate buffers
void initCudaProgram(shared_ptr<GLRenderer> renderer){

	// allocate most gpu buffers
	uint64_t nodesCapacity           = 200'000;
	uint64_t estimatedNodeSize       = 200;           // see struct Node in progressive_octree.cu, but some more just in case
	// momentary 分配器实测稳态需求 ~607MB（backlog 240MB + 其余），原值 300MB
	// 长期越界写入相邻显存。640MB = 实测需求 + ~5% 余量，使分配回到界内。
	// 注意：momentary Allocator 的 offset 为非原子、依赖全网格均匀调用的竞态
	// 收敛，无法做调用点级容量检查，安全性只能靠背板 ≥ 需求保证。
	// persistent 池为自适应分配，会自动吸收此变化。
	uint64_t cptr_buffer_bytes       = 640'000'000;
	uint64_t cptr_nodes_bytes        = nodesCapacity * estimatedNodeSize;
	uint64_t cptr_renderbuffer_bytes = 200'000'000;

	momentaryBufferCapacity = cptr_buffer_bytes;

	cuMemAlloc(&cptr_buffer                , cptr_buffer_bytes);
	cuMemAlloc(&cptr_nodes                 , cptr_nodes_bytes);
	cuMemAlloc(&cptr_renderbuffer          , cptr_renderbuffer_bytes);
	cuMemAlloc(&cptr_stats                 , sizeof(Stats));
	cuMemAlloc(&cptr_numBatchesUploaded    , 4);
	cuMemAlloc(&cptr_batchSizes            , 4 * BATCH_STREAM_SIZE);
	cuMemAlloc(&cptr_frameStart            , 8);
	cuMemAllocHost((void**)&h_stats_pinned , sizeof(Stats));

	// allocate ring buffer for point upload
	uint64_t cptr_points_bytes = MAX_BATCH_SIZE * sizeof(Point);
	CUdeviceptr devicemem = 0;

	cuMemAlloc(&devicemem, BATCH_STREAM_SIZE * cptr_points_bytes);

	for(uint64_t i = 0; i < BATCH_STREAM_SIZE; i++){
		cptr_points_ring[i] = devicemem + i * cptr_points_bytes;
	}

	// allocate persistent (over multiple frames) buffer with remaining GPU memory
	//
	// 自适应策略：吃掉当前可用显存，仅预留安全余量（旧行为是 available * 0.80，
	// 比例式余量不随机器规模伸缩——大显存机器浪费、小显存机器也未必合适）。
	// 余量用于覆盖：后续小分配(cudaprint/cubin 加载等)、桌面合成器与其他进程的
	// 动态需求、WDDM 逐页调度开销。余量过小会把整机推入共享内存页调度，反而劣化。
	size_t availableMem = 0;
	size_t totalMem = 0;
	cuMemGetInfo(&availableMem, &totalMem);

	constexpr double RELATIVE_MARGIN = 0.05;   // 总显存的 5%
	constexpr size_t ABSOLUTE_MARGIN = 256'000'000; // 且至少 256 MB
	size_t margin = std::max<size_t>(size_t(double(totalMem) * RELATIVE_MARGIN), ABSOLUTE_MARGIN);

	size_t cptr_buffer_persistent_bytes = (availableMem > margin)
		? availableMem - margin
		: availableMem / 2;

	persistentBufferCapacity = cptr_buffer_persistent_bytes;

	// 分配失败(如桌面临时占用大)则逐次砍半重试，不再静默吞掉返回值
	CUresult memResult = cuMemAlloc(&cptr_buffer_persistent, cptr_buffer_persistent_bytes);
	for(int retry = 0; memResult != CUDA_SUCCESS && retry < 3; retry++){
		const char* errStr = "";
		cuGetErrorString(memResult, &errStr);
		printfmt("cuMemAlloc(persistent, {} MB) failed: {} - retrying with half size \n",
			cptr_buffer_persistent_bytes / 1'000'000llu, errStr);

		cptr_buffer_persistent_bytes = cptr_buffer_persistent_bytes / 2;
		persistentBufferCapacity = cptr_buffer_persistent_bytes;
		memResult = cuMemAlloc(&cptr_buffer_persistent, cptr_buffer_persistent_bytes);
	}
	if(memResult != CUDA_SUCCESS){
		printfmt("FATAL: cuMemAlloc(persistent) failed after retries \n");
	}

	printfmt("persistent buffer: total {:8L} MB, available {:8L} MB, margin {:8L} MB, allocated {:8L} MB \n",
		totalMem / 1'000'000llu,
		availableMem / 1'000'000llu,
		margin / 1'000'000llu,
		cptr_buffer_persistent_bytes / 1'000'000llu);

	uint64_t total = cptr_buffer_bytes
		+ cptr_nodes_bytes
		+ cptr_renderbuffer_bytes
		+ sizeof(Stats)
		+ cptr_buffer_persistent_bytes;

	printfmt("cuMemAlloc(&cptr_buffer,            {:8L} MB);\n", cptr_buffer_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_nodes,             {:8L} MB);\n", cptr_nodes_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_renderbuffer,      {:8L} MB);\n", cptr_renderbuffer_bytes / 1'000'000llu);
	printfmt("cuMemAlloc(&cptr_stats,             {:8L}   );\n", sizeof(Stats));
	printfmt("cuMemAlloc(&cptr_buffer_persistent, {:8L} MB);\n", cptr_buffer_persistent_bytes / 1'000'000llu);
	printfmt("==============================================\n");
	printfmt("                                    {:8L} MB  \n", total / 1'000'000llu);
	printfmt("\n");

	cuda_program_update = new CudaModularProgram({
            .modules = {
                    "./kernels/progressive_octree_voxels.cu",
                    "./kernels/utils.cu",
		},
		.kernels = {"kernel_construct"}
	});

	cuda_program_reset = new CudaModularProgram({
            .modules = {
                    "./kernels/reset.cu",
                    "./kernels/utils.cu",
		},
		.kernels = {"kernel"}
	});

	cuda_program_render = new CudaModularProgram({
            .modules = {
                    "./kernels/render.cu",
                    "./kernels/utils.cu",
		},
		.kernels = {"kernel_render"}
	});

	cuEventCreate(&ce_render_start, 0);
	cuEventCreate(&ce_render_end, 0);
	cuEventCreate(&ce_update_start, 0);
	cuEventCreate(&ce_update_end, 0);

	cuGraphicsGLRegisterImage(&cugl_colorbuffer, renderer->view.framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
}

void reload(){

	printfmt("start loading \n");

	loadStart = static_cast<float>(now());
	totalUpdateDuration     = 0.0f;
	kernelUpdateDuration    = 0.0f;
	minKernelUpdateDuration = Infinity;
	maxKernelUpdateDuration = 0.0;
	avgKernelUpdateDuration = 0.0;
	cntKernelUpdateDuration = 0.0;

	totalRenderDuration     = 0.0f;
	kernelRenderDuration    = 0.0f;
	minKernelRenderDuration = Infinity;
	maxKernelRenderDuration = 0.0;
	avgKernelRenderDuration = 0.0;
	cntKernelRenderDuration = 0.0;

	lock_guard<mutex> lock_batchesToProcess(mtx_batchesToProcess);
	lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
	lock_guard<mutex> lock_batchesInPageableMemory(mtx_batchesInPageableMemory);
	lock_guard<mutex> lock_pinnedMemoryInUpload(mtx_pinnedMemoryInUpload);

	// reset workload and loaded data
	batchesToProcess.clear();
	batchesInPinnedMemory.clear();
	batchesInPageableMemory.clear();
	pinnedMemoryInUpload.clear();
	pinnedMemPool.refill();

	boxMin = float3{InfinityF, InfinityF, InfinityF};
	boxMax = float3{-InfinityF, -InfinityF, -InfinityF};
	boxSize = float3{0.0, 0.0, 0.0};

	numPointsTotal  = 0;
	numPointsLoaded = 0;
	numBatchesTotal = 0;
	numBytesTotal   = 0;
	numBytesLoaded  = 0;
	lazInBatches = false;

	// query bounding box and number of points in input files
	for(string path : paths){

		if(!fs::exists(path)) continue;

		numBytesTotal += fs::file_size(path);

		if(iEndsWith(path, "laz")){
			lazInBatches = true;
		}

		if(iEndsWith(path, "las") || iEndsWith(path, "laz")){

			LasHeader header = loadHeader(path);
			numPointsTotal += header.numPoints;

			boxMin.x = std::min(boxMin.x, float(header.min[0]));
			boxMin.y = std::min(boxMin.y, float(header.min[1]));
			boxMin.z = std::min(boxMin.z, float(header.min[2]));

			boxMax.x = std::max(boxMax.x, float(header.max[0]));
			boxMax.y = std::max(boxMax.y, float(header.max[1]));
			boxMax.z = std::max(boxMax.z, float(header.max[2]));

			for (uint64_t first = 0; first < header.numPoints; first += MAX_BATCH_SIZE) {
				PointBatch batch;
				batch.file = path;
				batch.first = static_cast<int>(first);
				batch.count = static_cast<int>(std::min(header.numPoints - first, MAX_BATCH_SIZE));
				batch.lasHeader = header;

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}

		}else if(iEndsWith(path, "simlod")){

			auto buffer = readBinaryFile(path, 0, 24);
			uint64_t numPoints = (fs::file_size(path) - 24) / 16;

			boxMin.x = std::min(boxMin.x, buffer->get<float>( 0));
			boxMin.y = std::min(boxMin.y, buffer->get<float>( 4));
			boxMin.z = std::min(boxMin.z, buffer->get<float>( 8));

			boxMax.x = std::max(boxMax.x, buffer->get<float>(12));
			boxMax.y = std::max(boxMax.y, buffer->get<float>(16));
			boxMax.z = std::max(boxMax.z, buffer->get<float>(20));

			numPointsTotal += numPoints;

			for(uint64_t first = 0; first < numPoints; first += MAX_BATCH_SIZE){
				PointBatch batch;
				batch.file = path;
				batch.first = static_cast<int>(first);
				batch.count = static_cast<int>(std::min(numPoints - first, MAX_BATCH_SIZE));

				batchesToProcess.push_back(batch);
				numBatchesTotal++;
			}
		}
	}

	boxSize.x = boxMax.x - boxMin.x;
	boxSize.y = boxMax.y - boxMin.y;
	boxSize.z = boxMax.z - boxMin.z;

	stats                   = Stats();
	numPointsUploaded       = 0;
	numBatchesProcessed     = 0;
	lastBatchFinishedDevice = false;
	batchStreamUploadIndex  = 0;
	processFrameTimes.clear();
}

void reset(shared_ptr<GLRenderer> renderer){
	// before locking, notify threads that they should not try to acquire the lock (otherwise we might wait for a long time to lock the others out)
	resetInProgress = true;

	// lock out upload threads and loader threads
	for (size_t i = 0; i < mtx_loader.size(); ++i) {
		mtx_loader[i]->lock();
	}
	lock_guard<mutex> lock_upload(mtx_uploader);

	// finish pending upload
	cuCtxSynchronize();

	// now reset octree-construction related state on device
	resetCUDA(renderer);
	cuCtxSynchronize();

	// read stats just to make sure device and host are on the same page
	cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
	memcpy(&stats, h_stats_pinned, sizeof(Stats));
	cuCtxSynchronize();

	// finally, reset host state to start reloading current data set from disk
	reload();

	requestReset = false;

	// notify other threads that they may now try to acquire the lock again
	resetInProgress = false;

	// unlock loader threads
	for (size_t i = 0; i < mtx_loader.size(); ++i) {
		mtx_loader[i]->unlock();
	}
}
