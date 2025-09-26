# ops/run_es.ps1
<# 
Utilities to manage the Elasticsearch dev container.
Usage (PowerShell):
  ./ops/run_es.ps1 start     # start (create if missing)
  ./ops/run_es.ps1 stop      # stop container
  ./ops/run_es.ps1 rm        # remove container
  ./ops/run_es.ps1 status    # show docker ps + ES health
#>

param([Parameter(Mandatory=$true)][ValidateSet("start","stop","rm","status")]$cmd)

$Image = "docker.elastic.co/elasticsearch/elasticsearch:8.14.0"
$Name  = "es-dev"

function Start-ES {
  if (-not (docker ps -a --format "{{.Names}}" | Select-String -SimpleMatch $Name)) {
    Write-Host "Creating container $Name..." -ForegroundColor Cyan
    docker run --name $Name -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -d $Image | Out-Null
  } else {
    Write-Host "Starting container $Name..." -ForegroundColor Cyan
    docker start $Name | Out-Null
  }
  Write-Host "Waiting 10s for ES to settle..." -ForegroundColor Yellow
  Start-Sleep -Seconds 10
  Invoke-RestMethod http://localhost:9200/ | Select-Object version, tagline | Format-List
}

function Stop-ES { docker stop $Name }
function Rm-ES   { docker rm -f $Name }
function Status-ES { docker ps; try { Invoke-RestMethod http://localhost:9200/ | Select-Object name,version,tagline } catch { Write-Host "ES not responding" -ForegroundColor Red } }

switch ($cmd) {
  "start"  { Start-ES }
  "stop"   { Stop-ES }
  "rm"     { Rm-ES }
  "status" { Status-ES }
}
