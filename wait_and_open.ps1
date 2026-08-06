# run.bat launches this hidden in the background instead of guessing a
# fixed delay before opening the browser - server startup time varies a lot
# (first run vs cached, machine speed, antivirus scanning), so a fixed sleep
# either opens too early (blank/connection-refused page) or wastes time.
# This polls the backend's health endpoint until it actually responds.

$maxWaitSeconds = 90
$url = "http://127.0.0.1:8000/"
$healthUrl = "http://127.0.0.1:8000/api/health"

for ($i = 0; $i -lt $maxWaitSeconds; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -eq 200) {
            Start-Process $url
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

Add-Type -AssemblyName PresentationFramework
[System.Windows.MessageBox]::Show(
    "서버가 $maxWaitSeconds 초 안에 응답하지 않았습니다.`n실행 중인 검은색 명령 프롬프트(run.bat) 창에 오류 메시지가 있는지 확인해 주세요.",
    "Magnetic - 시작 지연",
    "OK",
    "Warning"
) | Out-Null
