# One-command local model server for nl2sh+.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
#         (uses models/gguf/nl2sh-1.5b.Q4_K_M.gguf on port 8080, 4 threads)
# Then:   python cli/nl2sh.py extract tar.gz to /tmp
param(
    [string]$Model = "D:\Model_finetuing\models\gguf\nl2sh-1.5b.Q4_K_M.gguf",
    [int]$Port = 8080,
    [int]$Threads = 4
)
$Server = "D:\Model_finetuing\tools\llama.cpp\llama-server.exe"
if (-not (Test-Path $Model)) { Write-Error "model not found: $Model"; exit 1 }
if (-not (Test-Path $Server)) { Write-Error "llama-server not found: $Server"; exit 1 }
Write-Output "Serving $Model on http://127.0.0.1:$Port (threads=$Threads). Close this window to stop."
& $Server --model $Model --port $Port --threads $Threads
