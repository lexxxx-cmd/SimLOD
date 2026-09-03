
// Settings / Stats / Message 三个 ImGui 窗口的绘制（从原 main.cpp 的 render lambda
// 中原样提取，函数体未改动；由 src/app/main.cpp 的 render lambda 调用）。

#include "AppState.h"

#include "GLRenderer.h"
#include "Runtime.h"

#include "unsuck.hpp"

#include <format>
#include <iomanip>
#include <sstream>

#ifdef __cpp_lib_format
#include <format>
#else
#include "fmt/core.h"
using namespace fmt;
#endif

using namespace std;

void drawSettingsWindow(shared_ptr<GLRenderer> renderer){

	auto windowSize = ImVec2(490, 280);
	ImGui::SetNextWindowPos(ImVec2(10, 300));
	ImGui::SetNextWindowSize(windowSize);

	ImGui::Begin("Settings");
	ImGui::Checkbox("Show Bounding Box",        &settings.showBoundingBox);
	ImGui::Checkbox("Update Visibility",        &settings.doUpdateVisibility);
	ImGui::Checkbox("Show Points",              &settings.showPoints);
	ImGui::Checkbox("Color by Node",            &settings.colorByNode);
	ImGui::Checkbox("Color by LOD",             &settings.colorByLOD);
	// ImGui::Checkbox("Color white",              &settings.colorWhite);
	ImGui::Checkbox("enable Eye Dome Lighting", &settings.enableEDL);
	ImGui::Checkbox("High-Quality-Shading",     &settings.useHighQualityShading);
	ImGui::Checkbox("Auto-focus on load",       &settings.autoFocusOnLoad);
	ImGui::Checkbox("Benchmark Rendering",      &settings.benchmarkRendering);

	if(ImGui::Button("Reset")){
		requestReset = true;
		requestStepthrough = false;
	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Reset + Benchmark")){
		requestReset = true;
		requestBenchmark = true;
		requestStepthrough = false;
	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Stepthrough")){
		if(lastBatchFinishedDevice){
			requestReset = true;
		}
		requestStepthrough = true;
		requestStep = true;
	}

	ImGui::Text("Test Data Views");

	// original size: 1920x1080
	// clip size:
	//     offset: 63, 38
	//     size: 1795, 1010
	// resized to 50%, saved as 80% jpeg quality

	if(ImGui::Button("Chiller - bird")){
		// position: 39.55564356573898, -4.472634983341328, 9.256686713258468
		renderer->controls->yaw    = -5.237;
		renderer->controls->pitch  = -0.542;
		renderer->controls->radius = 34.626;
		renderer->controls->target = { 9.595, 10.394, 0.295, };
	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Chiller - close")){
		// position: 19.21071216298619, -0.590067491220811, 1.5756389652824982
		renderer->controls->yaw    = -5.752;
		renderer->controls->pitch  = 0.090;
		renderer->controls->radius = 16.153;
		renderer->controls->target = { 11.035, 13.285, 2.828, };
	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Retz - bird")){
		// position: -442.751714425827, 1032.8670571391256, -310.45475033534075
		renderer->controls->yaw    = -1.808;
		renderer->controls->pitch  = -0.997;
		renderer->controls->radius = 1166.684;
		renderer->controls->target = { 691.401, 884.472, -80.610, };

	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Retz - close")){
		// position: 627.9994594851617, 802.2611126991757, 76.41850773669957
		renderer->controls->yaw    = 0.750;
		renderer->controls->pitch  = -0.418;
		renderer->controls->radius = 80.902;
		renderer->controls->target = { 572.854, 856.372, 52.416, };

	}

	if(ImGui::Button("Morro Bay - bird")){
		// position: 1602.1138827457712, -475.95272623084657, 2313.666990965035
		renderer->controls->yaw    = -0.207;
		renderer->controls->pitch  = -0.797;
		renderer->controls->radius = 3866.886;
		renderer->controls->target = { 2398.747, 2167.120, -394.165, };
	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Morro Bay - close")){
		// position: 2840.684032348224, 949.9487599422316, 81.9126308772043
		renderer->controls->yaw    = -11.270;
		renderer->controls->pitch  = -0.225;
		renderer->controls->radius = 93.982;
		renderer->controls->target = { 2750.218, 974.775, 76.230, };
	}


	if(ImGui::Button("Meroe - bird")){
		// position: -366.08263489517935, 261.79980089364733, 206.0866739972536
		renderer->controls->yaw    = -7.430;
		renderer->controls->pitch  = -0.617;
		renderer->controls->radius = 929.239;
		renderer->controls->target = { 480.880, 573.485, -15.254, };
	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Meroe - close")){
		// position: 386.9127932944783, 808.8478521019321, 16.78255342532026
		renderer->controls->yaw    = -4.527;
		renderer->controls->pitch  = -0.192;
		renderer->controls->radius = 44.011;
		renderer->controls->target = { 343.652, 800.906, 18.330, };
	}

	if(ImGui::Button("Endeavor - bird")){
		// position: 641.9867239682803, 464.3862478069415, 613.116113282369
		renderer->controls->yaw    = -6.045;
		renderer->controls->pitch  = -0.713;
		renderer->controls->radius = 187.827;
		renderer->controls->target = { 597.671, 602.508, 493.795, };
	}
	ImGui::SameLine(0.0f);

	if(ImGui::Button("Endeavor - close")){
		// position: 600.8022710580775, 597.6937750182759, 508.70460986245035
		renderer->controls->yaw    = -12.560;
		renderer->controls->pitch  = -0.018;
		renderer->controls->radius = 8.087;
		renderer->controls->target = { 600.751, 605.780, 508.563, };
	}

	ImGui::SliderFloat("minNodeSize", &settings.minNodeSize, 32.0f, 1024.0f);
	ImGui::SliderInt("Point Size", &settings.pointSize, 1, 10);
	ImGui::SliderFloat("FovY", &settings.fovy, 20.0f, 100.0f);
	ImGui::SliderFloat("EDL Strength", &settings.edlStrength, 0.0f, 3.0f);

	if(ImGui::Button("Copy Camera")){
		auto controls = renderer->controls;
		auto pos = controls->getPosition();
		auto target = controls->target;

		stringstream ss;
		ss<< std::setprecision(2) << std::fixed;
		ss << format("// position: {}, {}, {} \n", pos.x, pos.y, pos.z);
		ss << format("renderer->controls->yaw    = {:.3f};\n", controls->yaw);
		ss << format("renderer->controls->pitch  = {:.3f};\n", controls->pitch);
		ss << format("renderer->controls->radius = {:.3f};\n", controls->radius);
		ss << format("renderer->controls->target = {{ {:.3f}, {:.3f}, {:.3f}, }};\n", target.x, target.y, target.z);

		string str = ss.str();

#ifdef _WIN32
		toClipboard(str);
#endif
	}

	ImGui::End();
}

void drawStatsWindow(shared_ptr<GLRenderer> renderer){

	auto windowSize = ImVec2(490, 440);
	ImGui::SetNextWindowPos(ImVec2(10, 590));
	ImGui::SetNextWindowSize(windowSize);

	ImGui::Begin("Stats");

	{ // used/total mem progress
		size_t availableMem = 0;
		size_t totalMem = 0;
		cuMemGetInfo(&availableMem, &totalMem);
		size_t unavailableMem = totalMem - availableMem;

		string strProgress = format("{:3.1f} / {:3.1f}",
			double(unavailableMem) / 1'000'000'000.0,
			double(totalMem) / 1'000'000'000.0
		);
		float progress = static_cast<float>(static_cast<double>(unavailableMem) / static_cast<double>(totalMem));
		ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
		ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::Text("Used GPU Memory");
	}

	{ // loaded/total points

		string strProgress = format("{} M / {} M",
			numPointsLoaded / 1'000'000,
			numPointsTotal / 1'000'000
		);
		float progress = static_cast<float>(static_cast<double>(numPointsLoaded) / static_cast<double>(numPointsTotal));
		ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
		ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::Text("Loaded Points");
	}

	{ // processed/total points

		string strProgress = format("{} M / {} M",
			stats.numPointsProcessed / 1'000'000,
			numPointsTotal / 1'000'000
		);
		float progress = static_cast<float>(static_cast<double>(stats.numPointsProcessed) / static_cast<double>(numPointsTotal));
		ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), strProgress.c_str());
		ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::Text("Processed Points");
	}

	auto locale = getSaneLocale();
	uint32_t numEmptyLeaves = stats.numLeaves - stats.numNonemptyLeaves;

	auto toMS = [locale](double millies){
		string str = "-";

		if(millies > 0.0){
			str = format("{:.1Lf} ms", millies);
		}

		return leftPad(str, 15);
	};

	auto toM = [locale](double number){
		string str = format(locale, "{:.1Lf} M", number / 1'000'000.0);
		return leftPad(str, 14);
	};

	auto toB = [locale](double number) {
		string str = format(locale, "{:.1Lf} B", number / 1'000'000'000.0);
		return leftPad(str, 14);
	};

	auto toMB = [locale](double number){
		string str = format(locale, "{:.1Lf} MB", number / 1'000'000.0);
		return leftPad(str, 15);
	};
	auto toGB = [locale](double number){
		string str = format(locale, "{:.1Lf} GB", number / 1'000'000'000.0);
		return leftPad(str, 15);
	};

	auto toIntString = [locale](double number){
		string str = format(locale, "{:L}", number);
		return leftPad(str, 10);
	};

	double pointsSecUpdate = double(stats.numPointsProcessed) / (double(kernelUpdateDuration) / 1000.0);
	double pointsSecTotal = double(stats.numPointsProcessed) / (double(totalUpdateDuration) / 1000.0);
	double gbs_file = double(numBytesLoaded) / (totalUpdateDuration / 1000.0);
	double gbs_gpu = double(16llu * stats.numPointsProcessed) / (totalUpdateDuration / 1000.0);

	double millionPointsSecRendered = 1000.0 * float(stats.numVisiblePoints) / renderingDuration;
	double millionVoxelsSecRendered = 1000.0 * float(stats.numVisibleVoxels) / renderingDuration;
	double millionSamplesSecRendered = 1000.0 * (float(stats.numVisiblePoints + stats.numVisibleVoxels) / renderingDuration);

	if(!settings.benchmarkRendering){
		millionPointsSecRendered = 0.0;
		millionVoxelsSecRendered = 0.0;
		millionSamplesSecRendered = 0.0;
		renderingDuration = 0.0;
	}

	double M = 1'000'000.0;
	double B = 1'000'000'000.0;
	double MB = 1'000'000.0; // TIL: MB = 1'000'000 vs. MiB = 1024 * 1024
	double GB = 1'000'000'000.0;

	vector<vector<string>> table = {
		{"#update kernel duration  ", toMS(kernelUpdateDuration)                            , format("{:.1f}", kernelUpdateDuration)},
		{"    max                  ", toMS(maxKernelUpdateDuration)                         , format("{:.1f}", maxKernelUpdateDuration)},
		{"    avg                  ", toMS(avgKernelUpdateDuration)                         , format("{:.1f}", avgKernelUpdateDuration)},
		{"#update total duration   ", toMS(totalUpdateDuration)                             , format("{:.1f}", totalUpdateDuration)},
		{"points/sec update kernel ", toM(pointsSecUpdate)                                  , format("{:.1f}", totalUpdateDuration / M)},
		{"points/sec total         ", toM(pointsSecTotal)                                   , format("{:.1f}", pointsSecTotal / M)},
		{"GB/s (disk I/O)          ", toGB(gbs_file)                                        , format("{:.1f}", gbs_file / GB)},
		{"GB/s (gpu)               ", toGB(gbs_gpu)                                         , format("{:.1f}", gbs_gpu / GB)},
		{"=========================", " "                                                   , " "},
		{"#render kernel duration  ", toMS(kernelRenderDuration)                            , format("{:.1f}", kernelRenderDuration)},
		{"=========================", " "                                                   , " "},
		{"rendering duration       ", toMS(renderingDuration)                               , format("{:.1f}", renderingDuration)},
		{"    points / sec         ", toB(millionPointsSecRendered)                        , format("{:.1f}", millionPointsSecRendered / B)},
		{"    voxels / sec         ", toB(millionVoxelsSecRendered)                        , format("{:.1f}", millionVoxelsSecRendered / B)},
		{"    samples / sec        ", toB(millionSamplesSecRendered)                       , format("{:.1f}", millionSamplesSecRendered / B)},
		{"=========================", " "                                                   , " "},
		{"#points processed        ", toM(double(stats.numPointsProcessed))                 , format("{:.1f}", stats.numPointsProcessed / M)},
		{"#nodes                   ", toIntString(stats.numNodes)                           , format("{}", stats.numNodes)},
		{"    #inner               ", toIntString(stats.numInner)                           , format("{}", stats.numInner)},
		{"    #leaves (nonempty)   ", toIntString(stats.numNonemptyLeaves)                  , format("{}", stats.numNonemptyLeaves)},
		{"    #leaves (empty)      ", toIntString(numEmptyLeaves)                           , format("{}", numEmptyLeaves)},
		{"#chunks                  ", toIntString(stats.numNodes)                           , format("{}", stats.numNodes)},
		{"    #voxels              ", toIntString(stats.numChunksVoxels)                    , format("{}", stats.numChunksVoxels)},
		{"    #points              ", toIntString(stats.numChunksPoints)                    , format("{}", stats.numChunksPoints)},
		{"#samples                 ", toM(stats.numPoints + stats.numVoxels)                , format("{:.1f}", (stats.numPoints + stats.numVoxels) / M)},
		{"    #points              ", toM(stats.numPoints)                                  , format("{:.1f}", stats.numPoints / M)},
		{"    #voxels              ", toM(stats.numVoxels)                                  , format("{:.1f}", stats.numVoxels / M)},
		{"momentary buffer         ", toMB(stats.allocatedBytes_momentary)                  , format("{:.1f}", stats.allocatedBytes_momentary / MB)},
		{"persistent buffer        ", toMB(stats.allocatedBytes_persistent)                 , format("{:.1f}", stats.allocatedBytes_persistent / MB)},
		{"=========================", " "                                                   , " "},
		{"#visible nodes           ", toIntString(stats.numVisibleNodes)                    , format("{}", stats.numVisibleNodes)},
		{"    #inner               ", toIntString(stats.numVisibleInner)                    , format("{}", stats.numVisibleInner)},
		{"    #leaves              ", toIntString(stats.numVisibleLeaves)                   , format("{}", stats.numVisibleLeaves)},
		{"#visible samples         ", toM(stats.numVisiblePoints + stats.numVisibleVoxels)  , format("{:.1f}", (stats.numVisiblePoints + stats.numVisibleVoxels) / 1'000'000.0f)},
		{"    #points              ", toM(stats.numVisiblePoints)                           , format("{:.1f}", stats.numVisiblePoints / 1'000'000.0f)},
		{"    #voxels              ", toM(stats.numVisibleVoxels)                           , format("{:.1f}", stats.numVisibleVoxels / 1'000'000.0f)},
	};

	if(ImGui::Button("Copy Stats")){
		stringstream ss;
		for (int row = 0; row < table.size(); row++){
			for (int column = 0; column < 2; column++){
				ss << table[row][column];
			}
			ss << "\n";
		}


		string str = ss.str();
		toClipboard(str);
	}


	auto flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV;
	if (ImGui::BeginTable("table1", 3, flags)){
		ImGui::TableSetupColumn("AAA", ImGuiTableColumnFlags_WidthStretch);
		ImGui::TableSetupColumn("BBB", ImGuiTableColumnFlags_WidthStretch);
		ImGui::TableSetupColumn("CCC", ImGuiTableColumnFlags_WidthFixed);
		for (int row = 0; row < table.size(); row++){
			ImGui::TableNextRow();
			for (int column = 0; column < 2; column++){
				ImGui::TableSetColumnIndex(column);

				ImGui::Text(table[row][column].c_str());
			}

			ImGui::PushID(row);

			ImGui::TableSetColumnIndex(2);
			if (ImGui::SmallButton("c")) {
				string str = table[row][2];
				toClipboard(str);
			}

			ImGui::PopID();
		}
		ImGui::EndTable();
	}


	ImGui::End();
}

void drawMessageWindow(){

	if(stats.memCapacityReached || (lazInBatches && !lastBatchFinishedDevice))
	{
		ImGuiIO& io = ImGui::GetIO();
		ImGui::SetNextWindowSize(ImVec2(700, 100));
		// ImGui::SetNextWindowPos(ImVec2(300, 200));
		ImGui::SetNextWindowPos(
			ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y - 100),
			ImGuiCond_Always, ImVec2(0.5f,0.5f)
		);

		ImGui::Begin("Message");

		ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));

		if(stats.memCapacityReached){
#ifdef __cpp_lib_format
        string message = std::format("WARNING: Octree mem usage ({} MB) approaches total capacity ({} MB). Further points are ignored.",
					stats.allocatedBytes_persistent / 1'000'000llu,
					persistentBufferCapacity / 1'000'000llu
				);
#else
        string message = fmt::format("WARNING: Octree mem usage ({} MB) approaches total capacity ({} MB). Further points are ignored.",
                                     stats.allocatedBytes_persistent / 1'000'000llu,
                                     persistentBufferCapacity / 1'000'000llu
        );
#endif

			ImGui::Text(message.c_str());
		}

		if(lazInBatches){
			ImGui::Text("WARNING: Loading *.laz files. Load routines are optimized for *.las or *.simlod - laz may be slow.");
		}

		ImGui::Text(" ");

		ImGui::PopStyleColor();


		ImGui::End();
	}
}
