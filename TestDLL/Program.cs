using System.Drawing;
using System.Runtime.InteropServices;
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Drawing.Imaging;
using System.Diagnostics;

[DllImport("VisionAlgoritiomMagnus.dll", CallingConvention = CallingConvention.Cdecl)]
 static extern void InitializeTemplate(IntPtr buffer, int width, int height, int channels, int minReduceArea);


[DllImport("VisionAlgoritiomMagnus.dll", CallingConvention = CallingConvention.Cdecl)]
 static extern void DoInspect(IntPtr buffer, int width, int height, int channels,
    int minReduceArea, bool bitwiseNot, bool toleranceRange,
    double tolerance1, double tolerance2, double tolerance3, double tolerance4, double toleranceAngle,
    double score, int maxPos, double maxOverlap,
    bool debugMode, bool subPixel, bool stopLayer1);

 void InspectImage(byte[] imageBuffer, int width, int height, int channels) {
    GCHandle handle = GCHandle.Alloc(imageBuffer, GCHandleType.Pinned);
    IntPtr bufferPtr = handle.AddrOfPinnedObject();

    try {
        DoInspect(bufferPtr, width, height, channels,
            256, false, false,
            40, 60, -110, -100, 30,
            0.5, 70, 0,
            false, false, false);
    } catch (Exception ex) {
        Console.WriteLine($"Error: {ex.Message}");
    } finally {
        handle.Free();
    }
 }

byte[] LoadImage(string path, ref int imageWidth, ref int imageHeight, ref int chanels)
{

    Bitmap bitmap = new Bitmap(path);
    // Chuyển ảnh thành grayscale nếu chưa phải là ảnh xám
    Bitmap grayscaleBitmap = new Bitmap(bitmap.Width, bitmap.Height);
    using (Graphics g = Graphics.FromImage(grayscaleBitmap))
    {
        var colorMatrix = new System.Drawing.Imaging.ColorMatrix(
            new float[][]
            {
                    new float[] { 0.299f, 0.299f, 0.299f, 0, 0 }, // R to Gray
                    new float[] { 0.587f, 0.587f, 0.587f, 0, 0 }, // G to Gray
                    new float[] { 0.114f, 0.114f, 0.114f, 0, 0 }, // B to Gray
                    new float[] { 0, 0, 0, 1, 0 }, // Alpha
                    new float[] { 0, 0, 0, 0, 1 }  // No change to alpha
            });
        var attributes = new ImageAttributes();
        attributes.SetColorMatrix(colorMatrix);
        g.DrawImage(bitmap, new Rectangle(0, 0, bitmap.Width, bitmap.Height), 0, 0, bitmap.Width, bitmap.Height, GraphicsUnit.Pixel, attributes);
    }

    // Chuyển bitmap xám thành mảng byte (grayscale)
    byte[] imageData = new byte[grayscaleBitmap.Width * grayscaleBitmap.Height];
    for (int y = 0; y < grayscaleBitmap.Height; y++)
    {
        for (int x = 0; x < grayscaleBitmap.Width; x++)
        {
            Color pixelColor = grayscaleBitmap.GetPixel(x, y);
            imageData[y * grayscaleBitmap.Width + x] = pixelColor.R; // Grayscale value
        }
    }

    // Pin mảng byte và truyền tới C++

    imageWidth = grayscaleBitmap.Width;
    imageHeight = grayscaleBitmap.Height;
    chanels = 1;
    return imageData;
}


static void SendTemplateImage(byte[] templateData, int width, int height, int channels, int minReduceArea = 256)
{
    GCHandle handle = GCHandle.Alloc(templateData, GCHandleType.Pinned);
    IntPtr bufferPtr = handle.AddrOfPinnedObject();

    try
    {
        InitializeTemplate(bufferPtr, width, height, channels, minReduceArea);
    }
    finally
    {
        handle.Free();
    }
}

// Sử dụng
int imgWidth = 0, imageHeight = 0, chanels = 1;
byte[] templateData =LoadImage("..\\..\\..\\Test Images\\Dst2.bmp",ref imgWidth,ref imageHeight,ref chanels);
SendTemplateImage(templateData, imgWidth, imageHeight, chanels); // Gửi template image dưới dạng grayscale

// Đọc ảnh vào một đối tượng Bitmap và chuyển đổi thành ảnh xám (grayscale)

byte[] inspectImage = LoadImage("..\\..\\..\\Test Images\\Src2.bmp", ref imgWidth, ref imageHeight, ref chanels);
Stopwatch sw = new Stopwatch();
sw.Start();
InspectImage(inspectImage, imgWidth, imageHeight, chanels);
Console.WriteLine($"Inspection time: {sw.ElapsedMilliseconds} ms");
sw.Restart();
InspectImage(inspectImage, imgWidth, imageHeight, chanels);
Console.WriteLine($"Inspection time: {sw.ElapsedMilliseconds} ms");
