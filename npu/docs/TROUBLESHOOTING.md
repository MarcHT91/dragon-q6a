# Troubleshooting

## The NPU stops working: `Failed to create device`

Symptoms:

- every new session fails with
  `QNN SetupBackend failed Failed to create device. Error: QNN_DEVICE_ERROR_INVALID_CONFIG`
  (followed by "nodes assigned to the default CPU EP, but fallback ... disabled");
- `dmesg` repeats `a300000.remoteproc:glink-edge: intent request timed out` every ~10 s.

The CDSP (the Hexagon DSP that hosts the HTP) has stopped responding.

**Cause seen on kernel 7.2.** Around the start or exit of a process that uses the NPU,
fastrpc often fails to unmap a DSP buffer:

```
qcom,fastrpc-cb a300000.remoteproc:glink-edge:fastrpc:compute-cb@3: unmmap pt 0x3fc800000 ERROR
```

Each such failure leaves memory mapped on the DSP side. After several of them (7-8 in our
case, counted since the last CDSP start) the CDSP hangs. Compile sessions leak too. Releasing
all sessions before exit (`npu_common` does this at `atexit`) reduces it but does not always
prevent it.

**Avoid it:** keep the model in one long-lived process; compile on a PC
([PC_TOOLS.md](PC_TOOLS.md)); do not run many short NPU processes back to back. Watch the
counter:

```bash
dmesg | grep -c unmmap          # all since boot
```

**Recover** (root). **First stop every process that uses the NPU** (your runtime, services,
scripts). Restarting the CDSP under a process that still holds sessions leaves those
sessions pointing at the old DSP instance; when that process later exits, fastrpc oopses in
`fastrpc_req_mem_unmap_impl` and the next NPU process oopses in `dma_alloc_attrs`. After
that only a reboot helps.

```bash
sudo sh -c 'echo stop > /sys/class/remoteproc/remoteproc1/state; sleep 2;
            echo start > /sys/class/remoteproc/remoteproc1/state; sleep 2;
            systemctl restart cdsprpcd'
```

`remoteproc1` is the CDSP (`cat /sys/class/remoteproc/remoteproc1/name`). If `cdsprpcd`
stays in `deactivating` or a process is stuck in state `D`, reboot.

`dmesg` may be restricted for normal users; `sudo sysctl kernel.dmesg_restrict=0` makes it
readable.

## Never kill a process while it is running on the NPU

Killing an NPU process in the middle of a graph execution (for example with `timeout`
around a slow test) can leave its DSP buffers pinned in kernel memory. `free` then shows
memory as used with no process owning it. Repeat it a few times and the OOM killer starts
killing everything else; a service that restarts on failure then loops and makes it worse.
Limit the number of calls instead of the wall time, let the process exit normally, and
reboot if `used` memory stays high after all NPU processes are gone. If you run the model
as a service, limit automatic restarts (systemd `StartLimitBurst`).

## One QNN runtime per process

Loading the ONNX Runtime QNN sessions and the native runner (`lfm3_native.py`) in the same
process maps every graph's weights twice. The DSP runs out of address space (`fastrpc memory
map ... failed`, `Failed to map weights buffer to device!`, `Failed to find available PD`),
and the process can end up as a zombie with threads stuck in QNN. The memory it had mapped
(here ~2 GB) is not freed until a reboot, and later loads can fail for the same reason even
in a clean process. Compare the two runtimes in separate processes (`test_native.py ort`,
then `test_native.py native`). If loads fail after such an incident, reboot.

## Never read `/sys/kernel/debug/dma_buf/bufinfo`

On kernel 7.2.9 reading it oopses in `dma_buf_debug_show` while holding the dma-buf list
lock. From then on every `dma_buf_export` (onnxruntime-qnn, cdsprpcd) blocks forever in
state `D`. Only a reboot fixes it.

## `QNN_CONTEXT_ERROR_CREATE_FROM_BINARY` when loading a context

- The context was compiled for a different target: offline compiles need `soc_model=35`,
  `vtcm_mb=2`.
- QNN version mismatch between the machine that compiled and the board: use the same
  `onnxruntime-qnn` version on both.
- Every failed load also counts as a leak (see above).

## `OrtEp has invalid ort_version_supported=0`

`onnxruntime-qnn` 2.3.0 with a newer onnxruntime. Install `onnxruntime==1.27.0`.

## Wrong outputs after rebuilding a graph

A stale `*_ctx.onnx` / `*_ctx_qnn.bin` was loaded instead of the new graph. The build
scripts delete them before compiling; if you build graphs by hand, delete them yourself.

## The model calls tools when it should not

Check that the first layers are int8 (`INT8_LAYERS=0,1,2,3,4,5`); with int4 everywhere the
model is strongly biased towards `<tool_call>` ([QUANTIZATION.md](QUANTIZATION.md)).
`validate_build.py` prints the first-token p(`<tool_call>`) for 10 prompts; the last three
should be near 0.

## Stray `</think>` with thinking disabled

With `enable_thinking=False` the template ends the prompt with `<think></think>`, but the
model sometimes still emits `</think>` (for example "Rome</think>Rome"). `chat.py` masks the
`<think>` and `</think>` tokens when thinking is off.

## Debugging habits that cost us time

- `pkill -f "python something.py"` also matches the shell that runs it if the pattern is in
  its command line. Anchor the pattern (`pgrep -f "^/path/to/python something.py"`).
- Long prompts: the prefill graphs only handle whole 16-token chunks; the remainder (up to
  15 tokens) goes through the decode graphs one by one, ~0.1 s each.
