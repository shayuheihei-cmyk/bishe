Add-Type -AssemblyName System.Drawing

$originalPath = 'D:\Users\shayuheihei\Desktop\bishe\assets\五禽戏-握拳开始-最终首页.png'
$inpaintedPath = 'C:\Users\shayuheihei\.codex\generated_images\01a19cbb-b9b2-7081-92f2-1be092a84266\exec-87443f75-8881-4e65-afb8-4b8891e23635.png'
$outputPath = 'D:\Users\shayuheihei\Desktop\bishe\assets\五禽戏-首页-干净底图.png'

$original = [System.Drawing.Bitmap]::new($originalPath)
$inpainted = [System.Drawing.Bitmap]::new($inpaintedPath)
if ($original.Width -ne $inpainted.Width -or $original.Height -ne $inpainted.Height) {
  $original.Dispose()
  $inpainted.Dispose()
  throw 'The edited image size differs from the source image.'
}

$result = [System.Drawing.Bitmap]::new($original)
try {
  # Feather only the 260 × 105 px bottom-centre area that held the baked-in icon and label.
  for ($y = 832; $y -le 938; $y++) {
    for ($x = 688; $x -le 982; $x++) {
      $normalizedX = ($x - 835.0) / 147.0
      $normalizedY = ($y - 885.0) / 53.0
      $radius = [Math]::Sqrt($normalizedX * $normalizedX + $normalizedY * $normalizedY)
      $t = [Math]::Max(0.0, [Math]::Min(1.0, (1.0 - $radius) / 0.17))
      $blend = $t * $t * (3.0 - 2.0 * $t)
      if ($blend -le 0) { continue }

      $base = $original.GetPixel($x, $y)
      $clean = $inpainted.GetPixel($x, $y)
      $red = [int][Math]::Round($base.R * (1 - $blend) + $clean.R * $blend)
      $green = [int][Math]::Round($base.G * (1 - $blend) + $clean.G * $blend)
      $blue = [int][Math]::Round($base.B * (1 - $blend) + $clean.B * $blend)
      $result.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($red, $green, $blue))
    }
  }
  $result.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
  $result.Dispose()
  $original.Dispose()
  $inpainted.Dispose()
}
