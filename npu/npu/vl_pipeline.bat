@echo off
rem LFM3-VL language model -> NPU, PC part: calibration data, eval set, GPTQ, activation ranges, offline QNN compile.
rem Run from the workspace root (models\LFM3-VL-LM, data\coco_val0.parquet, vl\vision.safetensors + vl_enc.py, npu\, configs\).
rem Progress in build\status.txt, logs in build\*.log. Steps already done are skipped (delete their output to redo).
cd /d %~dp0..
set PYTHONUTF8=1
if "%PY%"=="" set PY=.venv\Scripts\python.exe
set M=models\LFM3-VL-LM
if "%WIKI%"=="" set WIKI=data\wikitext\wikitext-2-raw-v1\train-00000-of-00001.parquet
echo start %date% %time% > build\status.txt
if not exist build\vlcalib\calib_vl.jsonl.done (
  %PY% npu\gen_calib_vl.py %M% data\coco_val0.parquet build\vlcalib 128 160 > build\1_gen_vl.log 2>&1 || goto fail
  echo.> build\vlcalib\calib_vl.jsonl.done
)
echo 1 gen_vl ok %time% >> build\status.txt
if not exist build\calib_chat.jsonl (
  %PY% npu\gen_calib_lfm3.py %M% build\calib_chat.jsonl > build\2_gen_chat.log 2>&1 || goto fail
)
echo 2 gen_chat ok %time% >> build\status.txt
if not exist build\eval_vl\ref_logits.npz (
  %PY% npu\make_eval_vl.py %M% data\coco_val0.parquet build\eval_vl 8 64 128 > build\3_eval_vl.log 2>&1 || goto fail
)
echo 3 eval_vl ok %time% >> build\status.txt
if not exist build\gptq\layer23.npz (
  set CHAT=build\calib_chat.jsonl,build\vlcalib\calib_vl.jsonl
  %PY% npu\gptq_lfm3.py %M% %WIKI% build\gptq 128 512 > build\4_gptq.log 2>&1 || goto fail
)
set CHAT=
echo 4 gptq ok %time% >> build\status.txt
set REF_DEVICE=cuda
if not exist build\ranges.json (
  %PY% npu\calibrate_lfm3.py %M% build\ranges_a.json > build\5_ranges_a.log 2>&1 || goto fail
  %PY% npu\calibrate2_lfm3.py %M% build\calib_chat.jsonl,build\vlcalib\calib_vl.jsonl build\ranges_b.json build\gptq 24 640 > build\5_ranges_b.log 2>&1 || goto fail
  %PY% npu\merge_ranges.py build\ranges.json build\ranges_a.json build\ranges_b.json > build\5_merge.log 2>&1 || goto fail
)
echo 5 ranges ok %time% >> build\status.txt
set INT8_LAYERS=0,1,2,3,4,5
set INT8_MATS=configs\int8_mats_100M.json
%PY% npu\build_all_lfm3.py %M% build\ranges.json build\gptq build\npu 4096 6 > build\6_build.log 2>&1 || goto fail
%PY% npu\build_prefill_lfm3.py %M% build\ranges.json build\gptq build\npu 16 > build\6_prefill.log 2>&1 || goto fail
echo 6 build ok %time% >> build\status.txt
echo DONE >> build\status.txt
exit /b 0
:fail
echo FAILED %time% >> build\status.txt
exit /b 1
