param(
  [Parameter(Mandatory=$true)][string]$FlightId,
  [string]$Base = "http://localhost:8000"
)

$k = 0
while ($true) {
  $body = @{
    flight_id = $FlightId
    k = $k
    lat = 40.7 + (Get-Random -Minimum -5 -Maximum 5)/100.0
    lon = -74.0 + (Get-Random -Minimum -5 -Maximum 5)/100.0
    alt = 50 + (Get-Random -Minimum 0 -Maximum 50)
    ts  = (Get-Date).ToUniversalTime().ToString("o")
  } | ConvertTo-Json
  try {
    Invoke-RestMethod -Method Post -Uri "$Base/telemetry" -ContentType "application/json" -Body $body | Out-Null
    Write-Host "k=$k posted"
  } catch {
    Write-Warning "POST /telemetry failed at k=$k : $($_.Exception.Message)"
  }
  $k += 1
  Start-Sleep -Milliseconds 300
}
