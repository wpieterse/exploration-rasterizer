#include <stdint.h>
#include <math.h>

#include <string>
#include <fstream>

#include "exploration/rasterizer/math.hpp"
#include "exploration/rasterizer/library.hpp"

uint32_t PackColor(uint8_t Red, uint8_t Green, uint8_t Blue, uint8_t Alpha) {
  return (((uint32_t) Alpha) << 24) | (((uint32_t) Blue) << 16) |
         (((uint32_t) Green) << 8) | (((uint32_t) Red) << 0);
}

void UnpackColor(
  uint32_t PackedColor,
  uint8_t* Red,
  uint8_t* Green,
  uint8_t* Blue,
  uint8_t* Alpha
) {
  *Red   = (PackedColor >> 0) & 255;
  *Green = (PackedColor >> 8) & 255;
  *Blue  = (PackedColor >> 16) & 255;
  *Alpha = (PackedColor >> 24) & 255;
}

void ClearColorBuffer(FRenderTarget* RenderTarget) {
  for(int32_t CurrentY = 0; CurrentY < RenderTarget->Height; CurrentY++) {
    for(int32_t CurrentX = 0; CurrentX < RenderTarget->Width; CurrentX++) {
      RenderTarget->Color[CurrentX + CurrentY * RenderTarget->Width] =
        PackColor(0, 0, 0, 255);
    }
  }
}

void ClearDepthBuffer(FRenderTarget* RenderTarget) {
  for(int32_t CurrentY = 0; CurrentY < RenderTarget->Height; CurrentY++) {
    for(int32_t CurrentX = 0; CurrentX < RenderTarget->Width; CurrentX++) {
      RenderTarget->Depth[CurrentX + CurrentY * RenderTarget->Width] =
        RenderTarget->FarClipPlane;
    }
  }
}

void WriteTargaImage(
  std::string const& FileName,
  uint8_t*           Buffer,
  uint32_t           ImageWidth,
  uint32_t           ImageHeight
) {
  uint8_t TargaHeader[18] = { 0,
                              0,
                              2,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              (uint8_t) (ImageWidth & 255),
                              (uint8_t) (ImageWidth >> 8),
                              (uint8_t) (ImageHeight & 255),
                              (uint8_t) (ImageHeight >> 8),
                              24,
                              0x20 };

  FILE* OutputFile = fopen(FileName.c_str(), "wb");

  fwrite(TargaHeader, sizeof(uint8_t), sizeof(TargaHeader), OutputFile);
  fwrite(Buffer, sizeof(uint8_t), ImageWidth * ImageHeight * 3, OutputFile);

  fclose(OutputFile);
}

void WriteColorBufferToFile(
  FRenderTarget* RenderTarget, std::string const& FileName
) {
  uint32_t ImageWidth  = RenderTarget->Width;
  uint32_t ImageHeight = RenderTarget->Height;

  size_t   BufferSize = ImageWidth * ImageHeight * 3;
  uint8_t* Buffer     = new uint8_t[BufferSize];

  uint32_t OutputPixelIndex = 0;
  for(uint32_t CurrentY = 0; CurrentY < ImageHeight; CurrentY++) {
    for(uint32_t CurrentX = 0; CurrentX < ImageWidth; CurrentX++) {
      uint8_t Red   = 0;
      uint8_t Green = 0;
      uint8_t Blue  = 0;
      uint8_t Alpha = 0;

      UnpackColor(
        RenderTarget->Color[CurrentX + CurrentY * ImageWidth],
        &Red,
        &Green,
        &Blue,
        &Alpha
      );

      Buffer[OutputPixelIndex + 0] = Blue;
      Buffer[OutputPixelIndex + 1] = Green;
      Buffer[OutputPixelIndex + 2] = Red;

      OutputPixelIndex += 3;
    }
  }

  WriteTargaImage(FileName + "_color.tga", Buffer, ImageWidth, ImageHeight);

  delete[] Buffer;
}

void WriteDepthBufferToFile(
  FRenderTarget* RenderTarget, std::string const& FileName
) {
  uint32_t ImageWidth  = RenderTarget->Width;
  uint32_t ImageHeight = RenderTarget->Height;

  size_t   BufferSize = ImageWidth * ImageHeight * 3;
  uint8_t* Buffer     = new uint8_t[BufferSize];

  uint32_t OutputPixelIndex = 0;
  for(uint32_t CurrentY = 0; CurrentY < ImageHeight; CurrentY++) {
    for(uint32_t CurrentX = 0; CurrentX < ImageWidth; CurrentX++) {
      float NearClipPlane = RenderTarget->NearClipPlane;
      float FarClipPlane  = RenderTarget->FarClipPlane;

      float Depth =
        RenderTarget->Depth[CurrentX + CurrentY * ImageWidth] * 2.0f - 1.0f;

      uint8_t Color = (2.0f * NearClipPlane * FarClipPlane) /
                      (FarClipPlane + NearClipPlane -
                       Depth * (FarClipPlane - NearClipPlane)) *
                      255;

      Buffer[OutputPixelIndex + 0] = Color;
      Buffer[OutputPixelIndex + 1] = Color;
      Buffer[OutputPixelIndex + 2] = Color;

      OutputPixelIndex += 3;
    }
  }

  WriteTargaImage(FileName + "_depth.tga", Buffer, ImageWidth, ImageHeight);

  delete[] Buffer;
}

void CreateRenderTarget(
  FRenderTarget* RenderTarget,
  int32_t        Width,
  int32_t        Height,
  float          NearClipPlane,
  float          FarClipPlane
) {
  RenderTarget->Width  = Width;
  RenderTarget->Height = Height;

  RenderTarget->NearClipPlane = NearClipPlane;
  RenderTarget->FarClipPlane  = FarClipPlane;

  RenderTarget->Color = new uint32_t[Width * Height];
  ClearColorBuffer(RenderTarget);

  RenderTarget->Depth = new float[Width * Height];
  ClearDepthBuffer(RenderTarget);
}

void DestroyRenderTarget(FRenderTarget* RenderTarget) {
  delete[] RenderTarget->Depth;
  RenderTarget->Depth = nullptr;

  delete[] RenderTarget->Color;
  RenderTarget->Color = nullptr;

  RenderTarget->Height = 0;
  RenderTarget->Width  = 0;
}

static constexpr float INCH_TO_MM = 25.4f;

//
// Compute screen coordinates based on a physically-based camera model. See
// http://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera
//
void ComputeScreenCoordinate(
  float             FilmApertureWidth,
  float             FilmApertureHeight,
  uint32_t          ImageWidth,
  uint32_t          ImageHeight,
  FitResolutionGate FitFilm,
  float             NearClipPlane,
  float             FocalLength,
  float&            Top,
  float&            Bottom,
  float&            Left,
  float&            Right
) {
  float FilmAspectRation  = FilmApertureWidth / FilmApertureHeight;
  float DeviceAspectRatio = ImageWidth / (float) ImageHeight;

  Top   = ((FilmApertureHeight * INCH_TO_MM / 2) / FocalLength) * NearClipPlane;
  Right = ((FilmApertureWidth * INCH_TO_MM / 2) / FocalLength) * NearClipPlane;

  // Field of view (horizontal)
  float FieldOfViewHorizontal =
    2 * 180 / M_PI * atan((FilmApertureWidth * INCH_TO_MM / 2) / FocalLength);

  float ScaleX = 1.0f;
  float ScaleY = 1.0f;

  switch(FitFilm) {
    default:
    case FitResolutionGate::Fill:
      if(FilmAspectRation > DeviceAspectRatio) {
        ScaleX = DeviceAspectRatio / FilmAspectRation;
      } else {
        ScaleY = FilmAspectRation / DeviceAspectRatio;
      }

      break;
    case FitResolutionGate::Overscan:
      if(FilmAspectRation > DeviceAspectRatio) {
        ScaleY = FilmAspectRation / DeviceAspectRatio;
      } else {
        ScaleX = DeviceAspectRatio / FilmAspectRation;
      }

      break;
  }

  Right  *= ScaleX;
  Top    *= ScaleY;
  Bottom  = -Top;
  Left    = -Right;
}
