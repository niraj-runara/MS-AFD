// MS-AFD · M0 — Single slice.
// One MPS process: static arena -> FFN captured into a CUDA Graph -> persistent
// loop, timing each iteration. Boilerplate only; steps are stubbed with TODOs.
//
// Run under MPS (see scripts/start_mps.sh); CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
// caps this process to a fraction of the GPU's SMs (Step 1 — external to this file).

#include <cstdio>
#include "arena.h"
#include "ffn.h"

using namespace msafd;

int main(int argc, char** argv) {
    // --- Step 2: static 1 GB arena -------------------------------------------
    // Arena arena(kArenaBytes);
    // TODO: reserve fixed regions: weights, input_activation, output_activation, workspace.

    // --- Step 3: build one FFN, run once eagerly, verify vs reference ---------
    // FfnConfig cfg;
    // Ffn ffn(cfg /*, arena */);
    // ffn.forward(input, output, stream);
    // TODO: check output against a CPU/numpy reference.

    // --- Step 4: capture the FFN into a CUDA Graph ---------------------------
    // cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    //   ffn.forward(input, output, stream);
    // cudaStreamEndCapture(stream, &graph);
    // cudaGraphInstantiate(&graphExec, graph, 0);
    // TODO: confirm graph output == eager output.

    // --- Step 5 + 6: persistent loop, time every iteration -------------------
    // const int iters = 10000;
    // for (int i = 0; i < iters; ++i) {
    //     cudaEventRecord(start, stream);
    //     cudaGraphLaunch(graphExec, stream);
    //     cudaEventRecord(stop, stream);
    //     cudaEventSynchronize(stop);
    //     // record elapsed_ms[i]  -> write to CSV for bench/latency.py
    // }
    // TODO: dump per-iteration timings to latency.csv.

    printf("MS-AFD M0 slice: scaffold only — not implemented yet.\n");
    return 0;
}
