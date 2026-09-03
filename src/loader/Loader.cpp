
#include "AppState.h"

#include "GLRenderer.h"

#include "unsuck.hpp"
#include "LasLoader.h"
#include "SimlodLoader.h"
#include "laszip_api.h"

#include <mutex>
#include <deque>
#include <thread>

using namespace std;

void spawnLoader(size_t i) {

	auto cpu = getCpuData();

	thread t([&, i]() {
		cuCtxSetCurrent(context);

		while (true) {
			bool everythingIsDone = batchStreamUploadIndex == numBatchesTotal;
			bool processingLagsBehind = numPointsLoaded > stats.numPointsProcessed + BATCH_STREAM_SIZE * MAX_BATCH_SIZE;

			// if (processingLagsBehind) {
			// 	printfmt("processing lags behind\n");
			// }

			if (everythingIsDone || processingLagsBehind || resetInProgress.load()) {
				std::this_thread::sleep_for(1ms);
				continue;
			}

			// lock thread's own mutex
			lock_guard<mutex> lock_loader(*mtx_loader[i]);

			PointBatch batch;

			mtx_batchesToProcess.lock();
			// get batch to load, if load queue isn't full
			if (batchesToProcess.size() > 0) {
				batch = batchesToProcess.front();

				// if it's a laz file, limit loading to fewer thrads.
				int maxThreads = std::max(double(cpu.numProcessors) / 2.0, 1.0);
				if(batch.count > 0 && iEndsWith(batch.file, "laz"))
				if(numThreadsLoading >= maxThreads)
				{
					mtx_batchesToProcess.unlock();
					continue;
				}

				batchesToProcess.pop_front();
			}
			mtx_batchesToProcess.unlock();

			PinnedMemorySlot pinnedMemSlot = pinnedMemPool.acquire();
			Point* pinnedPoints = (Point*)pinnedMemSlot.memLocation;

			if (batch.count > 0) {
				// load points in batch

				int batchID = batch.first / MAX_BATCH_SIZE;
				double t_start = now();
				// printfmt("start loading batch {} at {:.3f} \n", batchID, t_start);

				numThreadsLoading++;
				if(iEndsWith(batch.file, "las")){
					void* target = (void*)pinnedPoints;

					double translation[3] = {-boxMin.x, -boxMin.y, -boxMin.z};
					loadLasNative(batch.file, batch.lasHeader, batch.first, batch.count, target, translation);
					numBytesLoaded += batch.count * batch.lasHeader.bytesPerPoint;

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);

				}else if(iEndsWith(batch.file, "laz")){
					laszip_POINTER laszip_reader = nullptr;
					laszip_header* header = nullptr;
					laszip_point* laz_point = nullptr;

					laszip_BOOL is_compressed;
					laszip_BOOL request_reader = true;

					laszip_create(&laszip_reader);
					laszip_request_compatibility_mode(laszip_reader, request_reader);
					laszip_open_reader(laszip_reader, batch.file.c_str(), &is_compressed);

					laszip_get_header_pointer(laszip_reader, &header);
					laszip_get_point_pointer(laszip_reader, &laz_point);
					laszip_seek_point(laszip_reader, batch.first);

					for (int i = 0; i < batch.count; i++) {
						double XYZ[3];
						laszip_read_point(laszip_reader);
						laszip_get_coordinates(laszip_reader, XYZ);

						Point point;
						point.x = static_cast<float>(XYZ[0] - boxMin.x);
						point.y = static_cast<float>(XYZ[1] - boxMin.y);
						point.z = static_cast<float>(XYZ[2] - boxMin.z);

						auto rgb = laz_point->rgb;
						point.rgba[0] = rgb[0] > 255 ? rgb[0] / 256 : rgb[0];
						point.rgba[1] = rgb[1] > 255 ? rgb[1] / 256 : rgb[1];
						point.rgba[2] = rgb[2] > 255 ? rgb[2] / 256 : rgb[2];

						//int intensity = laz_point->intensity;
						//point.rgba[0] = intensity / 200;
						//point.rgba[1] = intensity / 200;
						//point.rgba[2] = intensity / 200;


						pinnedPoints[i] = point;
					}

					laszip_close_reader(laszip_reader);

					batch.pinnedMem = pinnedMemSlot;

					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}else if(iEndsWith(batch.file, "simlod")){

					batch.pinnedMem = pinnedMemSlot;

					// At least on windows, this uses winapi to do unbuffered loading to maximize SSD perf
					uint64_t padding;
					loadFileNative(batch.file, 24llu + 16llu * uint64_t(batch.first), 16llu * batch.count, pinnedMemSlot.memLocation, &padding);
					batch.pinnedMem.memOffset = padding;

					numBytesLoaded += 16llu * batch.count;
					numPointsLoaded += batch.count;

					lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);
					batchesInPinnedMemory.push_back(batch);
				}

				// double t_end = now();
				// double millies = (t_end - t_start) * 1000.0;
				// printfmt("finished loading batch {} at {:.3f}. duration: {:.3f} ms \n", batchID, t_end, millies);

				numThreadsLoading--;
			}else {
				// give back pinned memory slot if we didn't use it
				pinnedMemPool.release(pinnedMemSlot);
			}

			std::this_thread::sleep_for(1ms);
		}

		});
	t.detach();
}

// Spawns a thread that uploads data to the GPU
// - Asynchronously schedules uploads in stream_upload
// - After every async copy, it also asynchronously updates the number of uploaded points
void spawnUploader(shared_ptr<GLRenderer> renderer) {
	double timestamp = now();

	thread t([&]() {

		vector<PinnedMemorySlot> availableSlots;

		while (true) {

			double t_start = now();
			double spintime = 0.0001;

			// actually seems to wait way, way longer than 100ns.
			// std::this_thread::sleep_for(100ns);

			// go to sleep if reset is in progress
			if (resetInProgress.load()) {
				std::this_thread::sleep_for(1ms);
				continue;
			}

			// this lock ensures that we don't reset and upload at the same time
			lock_guard<mutex> lock_uploader(mtx_uploader);

			// spin instead of sleep, because sleep takes too long
			while (now() < t_start + spintime) {
				// do some spinning to avoid attempting locks too often
			}

			if(requestStepthrough){
				if(requestStep){
					requestStep = false;
				}else{
					continue;
				}
			}

			// reclaim all pinned memory slots that are no longer needed and give them back to the pool
			mtx_pinnedMemoryInUpload.lock();
			while (!pinnedMemoryInUpload.empty() && cuEventQuery(pinnedMemoryInUpload.front().uploadEnd) == cudaSuccess) {
				availableSlots.push_back(pinnedMemoryInUpload.front());
				pinnedMemoryInUpload.pop_front();
			}
			mtx_pinnedMemoryInUpload.unlock();

			pinnedMemPool.release(availableSlots);
			availableSlots.clear();

			bool everythingIsDone = batchStreamUploadIndex == numBatchesTotal;
			bool processingLagsBehind = numPointsUploaded > stats.numPointsProcessed + BATCH_STREAM_SIZE * MAX_BATCH_SIZE;

			if (everythingIsDone) continue;
			if (processingLagsBehind) continue;

			auto t_00 = now();

			// acquire work, or keep spinning if there is none
			PointBatch batch;
			{
				lock_guard<mutex> lock_batchesInPinnedMemory(mtx_batchesInPinnedMemory);

				if (batchesInPinnedMemory.size() > 0) {
					batch = batchesInPinnedMemory.front();
					batchesInPinnedMemory.pop_front();
				} else {
					continue;
				}
			}

			// UPLOAD
			uint32_t targetSlot = batchStreamUploadIndex;
			int uploadRingIndex = targetSlot % BATCH_STREAM_SIZE;

			auto source     = ((uint8_t*)batch.pinnedMem.memLocation) + batch.pinnedMem.memOffset;
			auto target     = cptr_points_ring[uploadRingIndex];
			size_t byteSize = batch.count * sizeof(Point);

			cuMemcpyHtoDAsync(target, source, byteSize, stream_upload);

			// record upload event
			cuEventRecord(batch.pinnedMem.uploadEnd, stream_upload);

			// since we process N batches per frame and batches may have varying amounts of points,
			// we need to let the kernel know the size of each individual batch
			cuMemsetD32Async(cptr_batchSizes + 4 * uploadRingIndex, batch.count, 1, stream_upload);

			// also let the kernel know how many batches we uploaded
			cuMemsetD32Async(cptr_numBatchesUploaded, batchStreamUploadIndex + 1, 1, stream_upload);

			batchStreamUploadIndex++;
			numPointsUploaded += batch.count;

			lock_guard<mutex> lock_pinnedMemoryInUpload(mtx_pinnedMemoryInUpload);
			pinnedMemoryInUpload.push_back(batch.pinnedMem);
		}

		});
	t.detach();

	setThreadPriorityHigh(t);
}
