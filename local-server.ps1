$ErrorActionPreference = 'Stop'
$projectDir = [System.IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path))
$rootPrefix = $projectDir.TrimEnd('\') + '\'
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add('http://127.0.0.1:4173/')
$mimeTypes = @{
    '.css' = 'text/css; charset=utf-8'
    '.html' = 'text/html; charset=utf-8'
    '.js' = 'text/javascript; charset=utf-8'
    '.mjs' = 'text/javascript; charset=utf-8'
    '.png' = 'image/png'
    '.svg' = 'image/svg+xml'
    '.webp' = 'image/webp'
}

$listener.Start()
try {
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $response = $context.Response
        try {
            $requestPath = [System.Uri]::UnescapeDataString($context.Request.Url.AbsolutePath)
            $relativePath = if ($requestPath -eq '/') { 'index.html' } else { $requestPath.TrimStart('/') -replace '/', '\' }
            $filePath = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($projectDir, $relativePath))

            if (-not $filePath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or -not [System.IO.File]::Exists($filePath)) {
                $response.StatusCode = 404
                continue
            }

            $extension = [System.IO.Path]::GetExtension($filePath).ToLowerInvariant()
            $response.ContentType = if ($mimeTypes.ContainsKey($extension)) { $mimeTypes[$extension] } else { 'application/octet-stream' }
            $response.Headers.Add('Cache-Control', 'no-cache')
            $stream = [System.IO.File]::OpenRead($filePath)
            try {
                $response.ContentLength64 = $stream.Length
                $stream.CopyTo($response.OutputStream)
            } finally {
                $stream.Dispose()
            }
        } catch {
            try { $response.StatusCode = 500 } catch {}
        } finally {
            $response.Close()
        }
    }
} finally {
    $listener.Close()
}
