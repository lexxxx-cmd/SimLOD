
// 程序入口：main() + 拖放/命令行数据入口 + update/render lambda。
// 全局状态见 AppState.h；CUDA 宿主函数见 CudaHost.cpp；加载线程见 src/loader/Loader.cpp；
// ImGui 窗口绘制见 src/render/ImGuiPanels.cpp。

#include <iostream>
#include <locale.h>
#include <string>
#include <vector>
#include <thread>
#include <cmath>

#include "CudaModularProgram.h"
#include "GLRenderer.h"
#include "cudaGL.h"
#include "CudaPrint/CudaPrint.h"

#include "unsuck.hpp"

#include "AppState.h"

using namespace std;

// 双显卡(Optimus/可切换显卡)笔记本：强制 OpenGL 渲染走独立显卡。
// 否则 GL 上下文落在核显上，CUDA-GL interop 的 cuGraphicsGLRegisterImage 会静默失败，
// 表现为 kernel 正常运行、stats 正常，但画面全黑。
// 必须以 EXE 导出形式存在，驱动在创建 GL 上下文前读取。
extern "C" __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

int main(int argc, char** argv){

	auto renderer = make_shared<GLRenderer>();
	auto cpu = getCpuData();
	int numThreads = 2 * static_cast<int>(cpu.numProcessors);
	printfmt("cpu.numProcessors: {} \n", cpu.numProcessors);
	printfmt("launching {} loader threads \n", numThreads);


	renderer->controls->yaw    = -1.15;
	renderer->controls->pitch  = -0.57;
	renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
	renderer->controls->target = {
		boxSize.x * 0.5f,
		boxSize.y * 0.5f,
		boxSize.z * 0.1f
	};

	initCuda();
	initCudaProgram(renderer);

	pinnedMemPool.reserveSlots(PINNED_MEM_POOL_SIZE);

	mtx_loader.reserve(numThreads);
	for(int i = 0; i < numThreads; i++){
		mtx_loader.push_back(make_unique<mutex>());
		spawnLoader(i);
	}

	cudaprint.init();

	spawnUploader(renderer);

	reload();

	renderer->onFileDrop([&](vector<string> files){
		vector<string> pointCloudFiles;

		t_drop_start = now();
		printfmt("drop at {:.3f} \n", now());

		for(auto file : files){
			printfmt("dropped: {} \n", file);

			if(iEndsWith(file, "las") || iEndsWith(file, "laz")){
				pointCloudFiles.push_back(file);
			}else if(iEndsWith(file, "simlod")){
				pointCloudFiles.push_back(file);
			}
		}

		paths = pointCloudFiles;

		reset(renderer);

		if(settings.autoFocusOnLoad){
			renderer->controls->yaw = -1.15;
			renderer->controls->pitch = -0.57;
			renderer->controls->radius = sqrt(boxSize.x * boxSize.x + boxSize.y * boxSize.y + boxSize.z * boxSize.z);
			renderer->controls->target = {
				boxSize.x * 0.5f,
				boxSize.y * 0.5f,
				boxSize.z * 0.1f
			};
		}
	});

	// 命令行直接加载数据，便于自动化测试（与拖放等价；必须在 onFileDrop 注册之后调用）
	if(argc > 1){
		vector<string> files;
		for(int i = 1; i < argc; i++){
			files.push_back(argv[i]);
		}
		renderer->fileDropListeners.back()(files);
	}

	auto update = [&](){
		cudaprint.update();

		renderer->camera->fovy = settings.fovy;
		renderer->camera->update();
	};

	auto render = [&](){

		timeSinceLastFrame = static_cast<float>(now()) - lastFrameTime;
		lastFrameTime = static_cast<float>(now());

		renderer->view.framebuffer->setSize(renderer->width, renderer->height);

		glBindFramebuffer(GL_FRAMEBUFFER, renderer->view.framebuffer->handle);

		if(!lastBatchFinishedDevice){
			processFrameTimes.push_back(timeSinceLastFrame);
		}

		if(requestReset){
			reset(renderer);
		}

		renderCUDA(renderer);

		if(!lastBatchFinishedDevice){
			updateOctree(renderer);
		}

		if(!lastBatchFinishedDevice){
			totalUpdateDuration = 1000.0f * (static_cast<float>(now()) - loadStart);
		}else if(requestBenchmark && lastBatchFinishedDevice){
			requestBenchmark = false;
			printfmt("finished loading, disabling benchmarking. \n");
		}

		static int statsAge = 0;
		{
			// copy stats from gpu to cpu.
			// actually laggs behind because we do async copy.
			// lacks sync, but as long as bytes are updated atomically in multiples of 4 or 8 bytes,
			// results should be fine.

			// seems to be fine to add the async copy to the main stream?
			cuMemcpyDtoHAsync(h_stats_pinned, cptr_stats, sizeof(Stats), ((CUstream)CU_STREAM_DEFAULT));
			memcpy(&stats, h_stats_pinned, sizeof(Stats));

			statsAge = static_cast<int>(renderer->frameCount) - stats.frameID;

			bool newLastBatchFinishedDevice = stats.numPointsProcessed == uint64_t(numPointsTotal);
			if(stats.memCapacityReached){
				newLastBatchFinishedDevice = true;
			}
			if(newLastBatchFinishedDevice != lastBatchFinishedDevice){
				lastBatchFinishedDevice = newLastBatchFinishedDevice;
				printfmt("stats.numPointsProcessed = {} \n", stats.numPointsProcessed);
				printfmt("numPointsTotal = {} \n", uint64_t(numPointsTotal));
				printfmt("setting lastBatchFinishedDevice = {} \n", lastBatchFinishedDevice ? "true" : "false");
			}

			// 相机与可见性诊断（每 120 帧）
			static int dbgFrameCounter = 0;
			if((dbgFrameCounter++ % 120) == 0){
				auto c = renderer->controls;
				auto pos = c->getPosition();
				printfmt("DBG frame {}: visibleNodes={} visiblePoints={} radius={:.1f} target=({:.1f},{:.1f},{:.1f}) pos=({:.1f},{:.1f},{:.1f}) \n",
					dbgFrameCounter, stats.numVisibleNodes, stats.numVisiblePoints, c->radius,
					c->target.x, c->target.y, c->target.z, pos.x, pos.y, pos.z);

				Uniforms u = getUniforms(renderer);
				printfmt("DBG2 boxMax=({:.1f},{:.1f},{:.1f}) transform row2=({:.4f},{:.4f},{:.4f},{:.4f}) row3=({:.4f},{:.4f},{:.4f},{:.4f}) \n",
					u.boxMax.x, u.boxMax.y, u.boxMax.z,
					u.transform.rows[2].x, u.transform.rows[2].y, u.transform.rows[2].z, u.transform.rows[2].w,
					u.transform.rows[3].x, u.transform.rows[3].y, u.transform.rows[3].z, u.transform.rows[3].w);

				glm::dmat4 cw = renderer->camera->world;
				glm::dmat4 cv = renderer->camera->view;
				glm::dmat4 cp = renderer->camera->proj;
				printfmt("DBG3 camWorld col3=({:.3f},{:.3f},{:.3f},{:.3f}) view col3=({:.3f},{:.3f},{:.3f},{:.3f}) proj col0=({:.4f},{:.4f},{:.4f},{:.4f}) \n",
					cw[3].x, cw[3].y, cw[3].z, cw[3].w,
					cv[3].x, cv[3].y, cv[3].z, cv[3].w,
					cp[0].x, cp[0].y, cp[0].z, cp[0].w);
				cout.flush();
			}
		}

		if(Runtime::showGUI)
		{
			drawSettingsWindow(renderer);
		}

		if(Runtime::showGUI)
		{
			drawStatsWindow(renderer);
		}

		drawMessageWindow();

		frameCounter++;
	};

	renderer->loop(update, render);

	return 0;
}
