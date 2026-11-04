# Mini-Challenge 2 - High Performance Computing (hpc) HS26

## 1. Goals

In this mini-challenge you build a simple image classifier based on a vision transformer and investigate its performance on CPUs and GPUs.
You implement attention using basic PyTorch operations, visualize the computation, and identify bottlenecks in training.
You then compare compilation and optimized attention implementations, and investigate how data loading and measurement affect performance.

By completing the challenge, you should be able to:

* Implement and train a small vision transformer, explaining its operations and tensor shapes.
* Run reproducible CPU and GPU experiments, including HPC training jobs submitted through Slurm.
* Connect a computation graph to operator timings, GPU kernels, memory use, and GPU idle periods.
* Distinguish the performance of attention, a training step, and the complete input and training pipeline.
* Explain the effects of batch size, compilation, attention backends, data-loader processes, and synchronization.
* Measure profiling overhead and distinguish diagnostic measurements from representative training performance.
* Track experiments and compare performance alongside learning behaviour.

## 2. Setup

We will work locally on a CPU and on GPU nodes of the [Calculon Cluster](https://fhnw-hpc.pages.fhnw.ch/docs/software/).
Use a private fork of this repository and add @tomaszkacprzak as a Guest.

Set up the following components:

* **PyTorch and torchvision:** model implementation, automatic differentiation, training, and dataset loading. Use a CUDA-enabled installation on the GPU cluster.
* **Slurm:** resource allocation and batch jobs. Prepare a job script requesting GPUs, CPU cores, memory, and runtime, activating your environment, and saving its output.
* **Torchview:** forward computation graphs with tensor shapes. Install Graphviz where required for rendering.
* **TorchLens:** operation-level graph inspection and metadata. Record its version and check what the available capture and timing modes measure.
* **PyTorch Profiler:** CPU/GPU operator and kernel traces, input shapes, and memory information. Export traces for viewing, for example in Perfetto.
* **Weights & Biases (W&B):** configurations, training metrics, and system utilization. If compute nodes cannot reach the service, record offline and synchronize afterwards.
* **Places365-Standard:** the 256 x 256 image version, with 365 scene classes. Configure the shared dataset path through an environment variable or command-line argument. Use `torchvision.datasets.Places365` with `small=True` and `download=False` after the shared copy is prepared.

Use a small, reproducible subset for development and CPU experiments, and the full training split for the final GPU training experiment.
Keep the validation split separate from training.
Store checkpoints, measurements, graphs, and traces in your own output directory.

Record package versions, CPU/GPU models, GPU memory, CPU thread counts, and Slurm resource requests.
Run substantial cluster workloads inside allocations, not on the login node.
Request enough CPU resources for the training process and data-loader workers.

## 3. Background reading

Before starting, familiarize yourself with the following topics:

* PyTorch introduction: tensors, modules, automatic differentiation, optimizers, datasets, and training loops.
* Scaled dot-product attention: queries, keys, values, scaling, softmax, and multiple attention heads.
* A simple vision transformer: image patches, linear embeddings, positional information, attention, pooling, and classification.
* Slurm introduction and tutorial: allocations, job scripts, resource requests, submission, monitoring, and job output.
* Computation graphs and profiling: CPU time, GPU time, asynchronous execution, memory, and kernel launches.

## 4. Instructions for assignment

### General instructions

Implement training and measurement code as Python scripts that run locally and through Slurm.
Use a **report notebook** to load saved measurements, collate results, create plots and tables, and explain your findings.
Its figures should be reproducible without rerunning long GPU jobs.

Each step below is an experiment. For every experiment:

1. **Implement and run:** state the question and your prediction, implement the necessary functions, and record the commands or job configuration used.
2. **Present observations:** include a configuration table and the requested plots/tables, with units, labels, and variability. Associate graphs and traces with their runs.
3. **Write a reflection:** explain the observations using evidence. What limits performance? Did the result match your prediction? What can and cannot be concluded?

Typically compare **two to four configurations per experiment**, changing one factor at a time.
Choose values far enough apart to reveal different behaviour, such as batch sizes ranging from insufficient GPU work to a throughput plateau.
Use short exploratory runs to select values rather than reporting an exhaustive sweep.
If no transition or slowdown appears within your resource limits, explain that finding instead of assuming the expected behaviour occurred.

Keep image resolution, patch size, patch count, embedding dimension, head count, and model depth fixed after selecting the baseline.
**Do not perform a patch-count sweep.**
Use identical initialization/checkpoints and input batches for direct implementation comparisons, and record random seeds.
Disable stochastic transformations when checking numerical equivalence, so random changes do not obscure implementation errors.

#### Measurement protocol

* A **training step** includes forward propagation, loss, backward propagation, and the optimizer update. Label forward-only measurements separately.
* Distinguish an **attention benchmark**, a **training step with inputs already on the device**, and an **end-to-end step including input delivery**. Do not mix their speedup claims.
* Warm up before measuring steady-state performance. Report startup, compilation, and worker initialization separately when relevant.
* Repeat short benchmark measurements at least three times. Report a central value and variability, such as mean and standard deviation, and the number of measured steps and images. Full training jobs need not be repeated three times; report variability across their measurement windows separately.
* GPU execution is asynchronous. Use CUDA events for suitable device intervals or synchronize at the boundaries of a wall-clock measurement window. Avoid synchronization after every operation.
* Measure throughput over many steps without graph capture. Take separate short profiler traces to explain it; do not profile the entire training run.
* Distinguish CPU self time, CPU inclusive time, and GPU time. Nested timings and overlapping work cannot be summed as disjoint elapsed times.
* Report peak allocated GPU memory separately from reserved memory and sampled system statistics. Reset peak statistics at a documented boundary.
* Label data fetching, host-to-device transfer, forward, backward, and optimizer work with `torch.profiler.record_function` where useful. Main-process traces do not automatically capture the internals of loader subprocesses.

#### Required precision and implementations

All required experiments use **FP32 model parameters and floating-point inputs, no autocast/mixed precision, and highest float32 matrix-multiplication precision**:

```python
torch.set_float32_matmul_precision("highest")
```

Record these settings and keep them identical across required comparisons.
If you use convolutions, also disable their TF32 mode using the controls supported by your PyTorch version.
Reduced precision belongs to Advanced Goal 1.

| Implementation | Attention calculation | Compilation | Precision |
| --- | --- | --- | --- |
| Manual, eager | Basic PyTorch tensor operations | Disabled | FP32, highest |
| Manual, compiled | The same manual implementation | `torch.compile` | FP32, highest |
| SDPA | `F.scaled_dot_product_attention`, selected backends | Disabled for the core backend comparison | FP32, highest |

Keep surrounding projections and architecture identical.
Some fused SDPA backends require FP16/BF16 or particular hardware and shapes.
**An unsupported FP32 backend is a result to document, not a reason to change precision in the required experiments.**
The math backend and automatic selection are reference configurations; they may execute the same implementation.

### Experiment 1: Dataset, manual attention, and training loop

**Implement and run**

1. Load Places365 images and labels, apply a fixed-size transformation and normalization, and create training and validation loaders.
2. Build a small vision transformer: split images into patches, project patches with a linear layer, add positional information, apply one or a few attention blocks, pool tokens, and produce 365 class logits.
   Document your compact architecture. Residual connections and normalization are reasonable additions; keep them fixed afterwards.
3. Implement multi-head attention using basic PyTorch operations:

   ```text
   Q, K, V = linear projections of the tokens, arranged into heads
   scores = Q @ transpose(K) / sqrt(head_dimension)
   weights = softmax(scores, along the key-token dimension)
   output = weights @ V
   ```

   Combine heads and apply the output projection as needed.
   You may start with loops, but produce a vectorized version for the main experiments.
   Do not use `nn.MultiheadAttention`, SDPA, or compilation for this baseline.
   PyTorch supplies differentiation; writing CUDA kernels or backward functions is not required.
4. Implement loss, backward propagation, optimizer updates, validation, and checkpoint saving.
5. Run a small CPU development experiment. Check finite losses/gradients and demonstrate learning, for example by overfitting a tiny training subset before evaluating held-out images.

**Plots and tables**

* Example images, model configuration, parameter count, and tensor shapes through an attention block.
* Training loss and validation accuracy versus steps. Distinguish the overfitting check from validation results.

**Reflection**

How do images become tokens, how does attention mix information, and how are predictions produced?
What evidence shows that the implementation works?

### Experiment 2: CPU graph and profiling

**Implement and run**

1. Generate a Torchview graph using a representative input batch.
2. Run the manual, eager implementation on CPU with two sufficiently different batch sizes and a fixed CPU thread count.
3. Measure unprofiled training steps, then capture representative steps with PyTorch Profiler, recording shapes and memory.
4. Identify the three operations with the largest CPU self time and locate them on the graph. Include backward work and explain its relationship to the forward graph.

**Plots and tables**

* An annotated Torchview graph showing tensor shapes and expensive operations.
* Step time and throughput for both configurations.
* A collated operator table with call counts, shapes, self/inclusive times, and a representative trace.

**Reflection**

Which operations dominate and why? Does their relative importance change with batch size?
Why is one graph node not necessarily one kernel or one profiler event?

### Experiment 3: Run on a GPU through Slurm

**Implement and run**

1. Submit the manual, eager implementation as a one-GPU Slurm job, keeping model, precision, and comparison batches unchanged.
2. Move the model and inputs to the allocated GPU and verify the device actually used.
3. Repeat Experiment 2's measurements and capture CPU and CUDA activities. First compare inputs already on the device; measure transfers separately.
4. Record hardware and allocations. If the CPU baseline ran on a different machine, discuss that limitation or repeat it in a cluster CPU allocation.

**Plots and tables**

* CPU/GPU step time, throughput, and speedup at matching batch sizes.
* Operator/kernel tables and a timeline showing CPU launches, GPU execution, and idle gaps.

**Reflection**

Which bottlenecks changed? When does the GPU benefit the workload, and when does overhead limit it?
Distinguish CPU launch time from GPU execution time.

### Experiment 4: TorchLens and profiler overhead

**Implement and run**

1. Capture representative CPU/GPU forwards with TorchLens. Inspect shapes, module relationships, and timing metadata; compare with Torchview and the native profiler.
2. For one fixed GPU workload, compare four modes: no profiler, basic PyTorch Profiler, PyTorch Profiler with shapes/stacks, and TorchLens capture.
3. Time equivalent forward workloads in each mode. Exclude rendering/export, keep other options fixed, and record activation-retention settings.
4. Identify each timing source. Ordinary TorchLens operation timings are instrumented host times; state whether a supported native-profiler integration supplies GPU timings.

**Plots and tables**

* A TorchLens graph and selected operations' shapes and timing fields.
* Elapsed time, overhead relative to the uninstrumented run, and available memory measurements for all four modes.

**Reflection**

What does each tool reveal or hide? How much does observation change execution?
Which measurements support a GPU speedup claim?

### Experiment 5: Batch-size sweep

**Implement and run**

1. Keep the manual, eager model fixed and select up to four widely separated batch sizes.
2. Benchmark GPU training steps using batches already on the GPU, isolating the model from loading.
3. Seek a range from insufficient GPU work towards a throughput plateau or practical memory limit. Report measured steps and images.
4. Capture representative traces for a small and a large batch.

**Plots and tables**

* Step latency, throughput, peak allocated memory, and sampled GPU utilization versus batch size.
* A comparison of kernel durations and gaps in the selected traces.

**Reflection**

Why can a slower step produce higher throughput? What limits further improvement?
Does high utilization establish efficient use of the GPU's computational capacity?

### Experiment 6: Compile the manual implementation

**Implement and run**

1. Compare the unchanged manual model in eager mode with `torch.compile(model)`. Leave loading and logging outside compilation.
2. Check outputs and gradients against eager execution using identical weights/inputs and documented tolerances.
3. Use two batch sizes from Experiment 5, giving four configurations in total.
4. Measure first-call/compilation cost separately. Warm up each shape, including backward, before timing steady-state training.
5. Profile compiled steps and inspect kernel counts, fusion, graph breaks, and recompilation.

**Plots and tables**

* Eager/compiled throughput and speedup at each batch size, with compilation cost separately reported.
* Expensive-operation and kernel-launch comparisons from representative traces.

**Reflection**

What changed although the source model did not? How many steps recover compilation cost?
TorchLens capture may execute an eager path; establish compiled execution with the native profiler.
Explain any workload where compilation does not help.

### Experiment 7: SDPA and attention backends

**Implement and run**

1. Replace only the attention calculation with `torch.nn.functional.scaled_dot_product_attention` (SDPA). Preserve projections, shapes, scaling, and attention over all image patches.
2. Compare outputs and input/parameter gradients with the manual implementation before benchmarking.
3. Inspect math, FlashAttention, memory-efficient, and cuDNN backend availability. Request individual backends using `torch.nn.attention.sdpa_kernel`; capture diagnostics for unsupported combinations.
4. Without compilation or mixed precision, benchmark automatic selection, math, and up to two additional supported backends. Record requested and observed backends; do not count silent fallback as a different implementation.
5. Measure attention forward/backward and complete training steps. Compare with manual eager/compiled results at matching settings, and inspect a representative graph and trace.

**Plots and tables**

* Backend compatibility, including unsupported configurations and reasons.
* Attention time, training throughput, peak allocated memory, and speedup.
* Graph/trace differences in intermediates, operation boundaries, and kernels.

**Reflection**

How can the same equation produce different memory use and execution patterns?
Why might a large attention speedup give a small training-step speedup?
Explain how fusion affects profiling detail and why a backend name alone does not establish performance.

### Experiment 8: Deliberately starve the GPU with data loading

**Implement and run**

1. Fix one implementation and batch size. Read Places365 files from drive, decode/transform on CPU, and transfer batches to the GPU.
2. Compare up to four `DataLoader(num_workers=...)` values: zero, a small count, a moderate count, and a large count within your Slurm allocation. Zero loads in the main process; positive values create independent loader processes.
3. Hold storage location, transformations, shuffle order, pinning, and prefetch settings fixed. Record how worker count changes total prefetched data. Separate startup from steady state.
4. Measure waiting for `next(loader)`, transfers, end-to-end throughput, CPU activity, and GPU gaps. Include a matching resident-input reference to estimate the input pipeline's cost.
5. Explore drive access: compare shared storage with node-local scratch if available, or first-pass with repeated-pass reads at a selected worker count. Report cache conditions; repeated reads may come from RAM rather than the drive.
6. Capture TorchLens information under low and high worker counts. Compare per-operation shapes and timing metadata alongside the pipeline trace. The model graph does not automatically include file reads, worker queues, or loader waits; unchanged graph information can be a meaningful result.

**Plots and tables**

* Throughput, batch-time variability, loader-wait time, CPU utilization, and GPU utilization versus workers.
* A storage/cache comparison and traces showing input-related GPU gaps.
* Collated TorchLens operation measurements for the selected configurations, identifying host/device timing and work outside capture.

**Reflection**

Does a small worker count leave the GPU waiting? When do more workers stop helping or make it slower?
Distinguish file access, decoding, CPU contention, queueing, and transfers; do not assume all loading time is disk I/O.
How can the model graph remain unchanged while total throughput changes substantially?

### Experiment 9: Accidental synchronization

**Implement and run**

1. Compare two or three fixed-workload GPU runs: no per-step host extraction, `loss.item()` each step, and extraction at a larger interval.
2. Keep loss computation and updates identical. Accumulate detached metrics where necessary without retaining autograd graphs.
3. Keep console/network output outside the comparison or measure it separately, isolating scalar extraction from output overhead.
4. Measure throughput over windows and capture a short trace per variant.

**Plots and tables**

* Throughput and step-time variability versus extraction frequency.
* Trace excerpts locating synchronization and changes in CPU/GPU overlap.

**Reflection**

Why can reading one scalar affect execution? Does time attributed to `.item()` include waiting for earlier work?
Why might the impact depend on workload size?

### Experiment 10: Full training and W&B comparison

**Implement and run**

1. Submit separate full-dataset training jobs for manual eager attention, compiled manual attention, and your fastest supported FP32 SDPA backend.
2. Keep model, batch size, loader, optimizer, initial weights, and data-order seed identical. Choose a feasible common budget covering at least one complete training epoch per configuration; evaluate on the validation split.
3. Disable graph capture and continuous profiling. Log metrics at a reasonable interval using what you learned about synchronization.
4. Record loss, validation accuracy, throughput, elapsed time, peak allocated GPU memory, CPU/GPU utilization, and available system memory statistics in W&B. Include configurations, Slurm job IDs, and versions.

**Plots and tables**

* Learning curves against both examples processed and elapsed time.
* Collated performance results and utilization plots across training and validation.
* W&B run links and exported key figures in the notebook.

**Reflection**

Do short benchmarks predict complete training performance? Which bottleneck dominates now?
Does faster execution preserve comparable learning, allowing for floating-point differences?
Explain why sampled utilization or allocated memory alone does not establish efficiency.

### Optional Advanced Goal 1: Mixed precision and Tensor Cores

**Implement and run**

1. Add mixed precision with `torch.autocast`. Use gradient scaling for FP16 where appropriate and explain the choice for BF16.
2. Compare up to four supported configurations, for example FP32/math, BF16/math, BF16/a fused backend, and FP16/the same fused backend. Separate precision effects from backend effects where possible.
3. Fix compilation, model, batch size, and loading. Keep the FP32 baseline at highest precision; TF32 is a separate choice, not this baseline.
4. Check numerical differences, finite gradients, and learning behaviour. Inspect evidence for Tensor Core execution where available; lower precision or speedup alone is not proof.

**Plots and tables**

* Throughput, memory, and speedup relative to FP32, recording actual dtypes/backends.
* Output/gradient differences and learning curves over a common budget.

**Reflection**

How much improvement comes from precision and how much from a newly available backend?
What roles do memory traffic, hardware support, numerical range, and gradient scaling play?

### Optional Advanced Goal 2: DDP on two and four GPUs

**Implement and run**

1. Set up PyTorch DistributedDataParallel (DDP) under Slurm with one process per GPU, preferably on one node initially. Include a one-GPU baseline using the same DDP path.
2. Assign each rank its GPU and a data shard using a distributed sampler. Call `sampler.set_epoch(epoch)` when shuffling. Reduce metrics across ranks and avoid counting duplicated validation examples twice.
3. Fix model, attention, precision, and compilation. Record per-rank loader workers and total CPU/storage demand. Avoid accidentally multiplying Slurm tasks and launcher-created processes.
4. **Strong scaling:** run on 1, 2, and 4 GPUs with a fixed global batch and fixed total examples/steps. Divide the global batch across ranks.
5. **Weak scaling:** run on 1, 2, and 4 GPUs with fixed per-GPU batch and step count. Global batch and total work increase with GPU count.
6. Use synchronized wall-clock windows and the slowest rank's elapsed time. Count globally processed images once. Profile computation, gradient communication, and waiting.

**Plots and tables**

* GPU count, per-GPU/global batch, examples, wall time, and aggregate throughput.
* Strong-scaling speedup `S(p) = T(1) / T(p)`, with ideal speedup `p`, and efficiency `S(p) / p`.
* Weak-scaling step time and aggregate throughput, with ideal constant step time and linear throughput. Report efficiency `X(p) / (p * X(1))`, where `X` is global images per second.
* Representative traces and per-GPU utilization showing communication or imbalance.

**Reflection**

Why does scaling depart from the ideal? Consider per-GPU work, synchronization, overlap, input contention, and rank imbalance.
Why does weak scaling's changing global batch prevent a claim of identical optimization or convergence behaviour?

### Submission and final reflection

Submit implementation code, environment specification, Slurm scripts, the report notebook, and measurements needed to regenerate its figures.
Include representative graphs/traces or accessible artifact links and the W&B comparisons.
Do not commit the dataset or large generated artifacts to Git.

Finish with a reflection on the realization of the mini-challenge.
What went well? Where did problems occur? Where did you need more time than planned?
Which measurement changed your understanding of performance most?
What would you do differently, and what would you change in the assignment? Give concrete examples.
