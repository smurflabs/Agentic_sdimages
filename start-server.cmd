@echo off
rem Launched at boot by the "SD15-API" scheduled task (runs as SYSTEM).
cd /d "%~dp0"
rem Point at the Administrator's HF cache so SYSTEM doesn't re-download the model
set HF_HOME=C:\Users\Administrator\.cache\huggingface
set PYTHONUNBUFFERED=1
rem C:\Python313 has no stdlib; Administrator's HKCU registry points Python at this split
rem install's Lib/DLLs, but SYSTEM has no such entry, so supply the same paths here.
set PYTHONPATH=C:\Users\Administrator\AppData\Local\Programs\Python\Python313\Lib;C:\Users\Administrator\AppData\Local\Programs\Python\Python313\DLLs
".venv\Scripts\python.exe" -m uvicorn app:app --host 0.0.0.0 --port 8000 > server.log 2>&1
