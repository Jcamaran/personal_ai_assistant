# Test script for querying your personal AI assistant
# Usage: .\test_query.ps1

Write-Host "`nPersonal AI Assistant - Query Test`n" -ForegroundColor Cyan

# Function to query the API
function Test-Query {
    param([string]$question, [int]$topK = 5)
    
    Write-Host "Question: " -NoNewline -ForegroundColor Yellow
    Write-Host $question
    Write-Host "`nProcessing..." -ForegroundColor Gray
    
    $body = @{
        query = $question
        top_k = $topK
    } | ConvertTo-Json
    
    try {
        $startTime = Get-Date
        $response = Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -Body $body -ContentType 'application/json'
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        
        Write-Host "`nAnswer:" -ForegroundColor Green
        Write-Host $response.answer -ForegroundColor White
        
        Write-Host "`nSources:" -ForegroundColor Cyan
        foreach ($source in $response.sources) {
            Write-Host "  - $($source.metadata.file_name) (score: $([math]::Round($source.similarity_score, 3)))" -ForegroundColor DarkGray
            Write-Host "    Folder: $($source.metadata.folder)" -ForegroundColor DarkGray
        }
        
        Write-Host "`nProcessing time: $([math]::Round($response.processing_time, 2))s (API: $([math]::Round($elapsed, 2))s)" -ForegroundColor DarkGray
        Write-Host ("-" * 80) -ForegroundColor DarkGray
    }
    catch {
        Write-Host "`nError: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Example queries - customize these based on your Obsidian vault content
Test-Query "What is a RAG application?"
Test-Query "Tell me about Azure AI"
Test-Query "What are my capstone project ideas?"
Test-Query "Summarize my notes on LeetCode"

Write-Host "`nTest complete!`n" -ForegroundColor Green
