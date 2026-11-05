# Start the SD 1.5 API on http://0.0.0.0:8000 (all interfaces; docs at /docs)
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
