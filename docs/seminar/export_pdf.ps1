param([string]$Pptx, [string]$Pdf)
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($Pptx, $true, $false, $false)
$pres.SaveAs($Pdf, 32)   # ppSaveAsPDF
$pres.Close()
$app.Quit()
Write-Output "pdf written: $Pdf"
