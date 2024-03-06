#include <stdint.h>
#include <math.h>

#include <chrono>
#include <iostream>
#include <fstream>

#include "exploration/rasterizer/math.hpp"
#include "exploration/rasterizer/library.hpp"

#include "exploration/rasterizer/models/cow/vertices.hpp"
#include "exploration/rasterizer/models/cow/vertex_indices.hpp"
#include "exploration/rasterizer/models/cow/texture_coordinates.hpp"
#include "exploration/rasterizer/models/cow/texture_coordinate_indices.hpp"

static constexpr uint32_t IMAGE_WIDTH  = 1024;
static constexpr uint32_t IMAGE_HEIGHT = 1024;

Matrix44f const WorldToCamera = { 0.707107,  -0.331295, 0.624695,   0,
                                  0,         0.883452,  0.468521,   0,
                                  -0.707107, -0.331295, 0.624695,   0,
                                  -1.63871,  -5.747777, -40.400412, 1 };

static constexpr uint32_t TriangleCount      = 3156;
static constexpr float    NearClippingPlane  = 1.0f;
static constexpr float    FarClippingPlane   = 1000.0f;
static constexpr float    FocalLength        = 20.0f;
static constexpr float    FilmApertureWidth  = 0.980f;
static constexpr float    FileApertureHeight = 0.735f;

struct Executor_Generic {
    static void CreateRenderTarget(
      FRenderTarget* RenderTarget,
      int32_t        Width,
      int32_t        Height,
      float          NearClipPlane,
      float          FarClipPlane
    ) {
      ::CreateRenderTarget(
        RenderTarget, Width, Height, NearClipPlane, FarClipPlane
      );
    }

    static void DestroyRenderTarget(FRenderTarget* RenderTarget) {
      ::DestroyRenderTarget(RenderTarget);
    }

    static void CreateSurface(
      FSurfaceTriangles* Surface,
      uint32_t           TriangleCount,
      Vec3f*             Vertices,
      uint32_t*          VertexIndices,
      Vec2f*             TextureCoordinates,
      uint32_t*          TextureCoordinateIndices
    ) {
      Surface->TriangleCount            = TriangleCount;
      Surface->Vertices                 = Vertices;
      Surface->VertexIndices            = VertexIndices;
      Surface->TextureCoordinates       = TextureCoordinates;
      Surface->TextureCoordinateIndices = TextureCoordinateIndices;
      Surface->PrivateData              = nullptr;
    }

    static void DestroySurface(FSurfaceTriangles* Surface) {
      Surface->PrivateData              = nullptr;
      Surface->TextureCoordinateIndices = nullptr;
      Surface->TextureCoordinates       = nullptr;
      Surface->VertexIndices            = nullptr;
      Surface->Vertices                 = nullptr;
      Surface->TriangleCount            = 0;
    }

    static void Process(
      Matrix44f const&   WorldToCamera,
      float              Left,
      float              Right,
      float              Top,
      float              Bottom,
      FRenderTarget*     RenderTarget,
      FSurfaceTriangles* Triangles
    ) {
      Rasterize_Generic(
        WorldToCamera, Left, Right, Top, Bottom, RenderTarget, Triangles
      );
    }
};

struct Executor_Optimized {
    static void CreateRenderTarget(
      FRenderTarget* RenderTarget,
      int32_t        Width,
      int32_t        Height,
      float          NearClipPlane,
      float          FarClipPlane
    ) {
      ::CreateRenderTarget(
        RenderTarget, Width, Height, NearClipPlane, FarClipPlane
      );
    }

    static void DestroyRenderTarget(FRenderTarget* RenderTarget) {
      ::DestroyRenderTarget(RenderTarget);
    }

    static void CreateSurface(
      FSurfaceTriangles* Surface,
      uint32_t           TriangleCount,
      Vec3f*             Vertices,
      uint32_t*          VertexIndices,
      Vec2f*             TextureCoordinates,
      uint32_t*          TextureCoordinateIndices
    ) {
      CreateSurface_Optimized(
        Surface,
        TriangleCount,
        Vertices,
        VertexIndices,
        TextureCoordinates,
        TextureCoordinateIndices
      );
    }

    static void DestroySurface(FSurfaceTriangles* Surface) {
      DestroySurface_Optimized(Surface);
    }

    static void Process(
      Matrix44f const&   WorldToCamera,
      float              Left,
      float              Right,
      float              Top,
      float              Bottom,
      FRenderTarget*     RenderTarget,
      FSurfaceTriangles* Triangles
    ) {
      Rasterize_Optimized(
        WorldToCamera, Left, Right, Top, Bottom, RenderTarget, Triangles
      );
    }
};

#if defined(__AVX2__)

struct Executor_AVX2 {
    static void CreateRenderTarget(
      FRenderTarget* RenderTarget,
      int32_t        Width,
      int32_t        Height,
      float          NearClipPlane,
      float          FarClipPlane
    ) {
      ::CreateRenderTarget(
        RenderTarget, Width, Height, NearClipPlane, FarClipPlane
      );
    }

    static void DestroyRenderTarget(FRenderTarget* RenderTarget) {
      ::DestroyRenderTarget(RenderTarget);
    }

    static void CreateSurface(
      FSurfaceTriangles* Surface,
      uint32_t           TriangleCount,
      Vec3f*             Vertices,
      uint32_t*          VertexIndices,
      Vec2f*             TextureCoordinates,
      uint32_t*          TextureCoordinateIndices
    ) {
      CreateSurface_AVX2(
        Surface,
        TriangleCount,
        Vertices,
        VertexIndices,
        TextureCoordinates,
        TextureCoordinateIndices
      );
    }

    static void DestroySurface(FSurfaceTriangles* Surface) {
      DestroySurface_AVX2(Surface);
    }

    static void Process(
      Matrix44f const&   WorldToCamera,
      float              Left,
      float              Right,
      float              Top,
      float              Bottom,
      FRenderTarget*     RenderTarget,
      FSurfaceTriangles* Triangles
    ) {
      Rasterize_AVX2(
        WorldToCamera, Left, Right, Top, Bottom, RenderTarget, Triangles
      );
    }
};

#endif

#if defined(__AVX512F__)

struct Executor_AVX512 {
    static void CreateRenderTarget(
      FRenderTarget* RenderTarget,
      int32_t        Width,
      int32_t        Height,
      float          NearClipPlane,
      float          FarClipPlane
    ) {
      ::CreateRenderTarget(
        RenderTarget, Width, Height, NearClipPlane, FarClipPlane
      );
    }

    static void DestroyRenderTarget(FRenderTarget* RenderTarget) {
      ::DestroyRenderTarget(RenderTarget);
    }

    static void CreateSurface(
      FSurfaceTriangles* Surface,
      uint32_t           TriangleCount,
      Vec3f*             Vertices,
      uint32_t*          VertexIndices,
      Vec2f*             TextureCoordinates,
      uint32_t*          TextureCoordinateIndices
    ) {
      Surface->TriangleCount            = TriangleCount;
      Surface->Vertices                 = Vertices;
      Surface->VertexIndices            = VertexIndices;
      Surface->TextureCoordinates       = TextureCoordinates;
      Surface->TextureCoordinateIndices = TextureCoordinateIndices;
      Surface->PrivateData              = nullptr;
    }

    static void DestroySurface(FSurfaceTriangles* Surface) {
      Surface->PrivateData              = nullptr;
      Surface->TextureCoordinateIndices = nullptr;
      Surface->TextureCoordinates       = nullptr;
      Surface->VertexIndices            = nullptr;
      Surface->Vertices                 = nullptr;
      Surface->TriangleCount            = 0;
    }

    static void Process(
      Matrix44f const&   WorldToCamera,
      float              Left,
      float              Right,
      float              Top,
      float              Bottom,
      FRenderTarget*     RenderTarget,
      FSurfaceTriangles* Triangles
    ) {
      Rasterize_AVX512(
        WorldToCamera, Left, Right, Top, Bottom, RenderTarget, Triangles
      );
    }
};

#endif

struct Executor_CUDA {
    static void CreateRenderTarget(
      FRenderTarget* RenderTarget,
      int32_t        Width,
      int32_t        Height,
      float          NearClipPlane,
      float          FarClipPlane
    ) {
      ::CreateRenderTarget(
        RenderTarget, Width, Height, NearClipPlane, FarClipPlane
      );
    }

    static void DestroyRenderTarget(FRenderTarget* RenderTarget) {
      ::DestroyRenderTarget(RenderTarget);
    }

    static void CreateSurface(
      FSurfaceTriangles* Surface,
      uint32_t           TriangleCount,
      Vec3f*             Vertices,
      uint32_t*          VertexIndices,
      Vec2f*             TextureCoordinates,
      uint32_t*          TextureCoordinateIndices
    ) {
      Surface->TriangleCount            = TriangleCount;
      Surface->Vertices                 = Vertices;
      Surface->VertexIndices            = VertexIndices;
      Surface->TextureCoordinates       = TextureCoordinates;
      Surface->TextureCoordinateIndices = TextureCoordinateIndices;
      Surface->PrivateData              = nullptr;
    }

    static void DestroySurface(FSurfaceTriangles* Surface) {
      Surface->PrivateData              = nullptr;
      Surface->TextureCoordinateIndices = nullptr;
      Surface->TextureCoordinates       = nullptr;
      Surface->VertexIndices            = nullptr;
      Surface->Vertices                 = nullptr;
      Surface->TriangleCount            = 0;
    }

    static void Process(
      Matrix44f const&   WorldToCamera,
      float              Left,
      float              Right,
      float              Top,
      float              Bottom,
      FRenderTarget*     RenderTarget,
      FSurfaceTriangles* Triangles
    ) {
      Rasterize_CUDA(
        WorldToCamera, Left, Right, Top, Bottom, RenderTarget, Triangles
      );
    }
};

template<typename Executor>
void Process(
  char const* Name,
  char const* OutputName,
  uint32_t    ImageWidth,
  uint32_t    ImageHeight,
  uint32_t    TriangleCount,
  Vec3f*      Vertices,
  uint32_t*   VertexIndices,
  Vec2f*      TextureCoordinates,
  uint32_t*   TextureCoordinateIndices
) {
  std::cerr << Name << " : ";

  Matrix44f CameraToWorld = WorldToCamera.inverse();

  // compute screen coordinates
  float Top    = 0.0f;
  float Bottom = 0.0f;
  float Left   = 0.0f;
  float Right  = 0.0f;

  ComputeScreenCoordinate(
    FilmApertureWidth,
    FileApertureHeight,
    ImageWidth,
    ImageHeight,
    FitResolutionGate::Overscan,
    NearClippingPlane,
    FocalLength,
    Top,
    Bottom,
    Left,
    Right
  );

  FRenderTarget RenderTarget;

  Executor::CreateRenderTarget(
    &RenderTarget, ImageWidth, ImageHeight, NearClippingPlane, FarClippingPlane
  );

  FSurfaceTriangles Triangles;

  Executor::CreateSurface(
    &Triangles,
    TriangleCount,
    Vertices,
    VertexIndices,
    TextureCoordinates,
    TextureCoordinateIndices
  );

  auto TimerStart = std::chrono::high_resolution_clock::now();

  Executor::Process(
    WorldToCamera, Left, Right, Top, Bottom, &RenderTarget, &Triangles
  );

  auto TimerEnd = std::chrono::high_resolution_clock::now();
  auto PassedTime =
    std::chrono::duration<double, std::milli>(TimerEnd - TimerStart).count();

  std::cerr << PassedTime << " ms" << std::endl;

  Executor::DestroySurface(&Triangles);

  WriteColorBufferToFile(&RenderTarget, OutputName);
  // WriteDepthBufferToFile(&RenderTarget, OutputName);

  Executor::DestroyRenderTarget(&RenderTarget);
}

int main(int argc, char** argv) {
  Process<Executor_Generic>(
    "CPU - Generic",
    "rasterizer_cpu_generic",
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    TriangleCount,
    CowVertices,
    CowVertexIndices,
    CowTextureCoordinates,
    CowTextureCoordinateIndices
  );

  Process<Executor_Optimized>(
    "CPU - Optimized",
    "rasterizer_cpu_optimized",
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    TriangleCount,
    CowVertices,
    CowVertexIndices,
    CowTextureCoordinates,
    CowTextureCoordinateIndices
  );

#if defined(__AVX2__)

  Process<Executor_AVX2>(
    "CPU - AVX2",
    "rasterizer_cpu_avx2",
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    TriangleCount,
    CowVertices,
    CowVertexIndices,
    CowTextureCoordinates,
    CowTextureCoordinateIndices
  );

#endif

#if defined(__AVX512F__)

  // Process<Executor_AVX512>(
  //   "CPU - AVX512",
  //   "rasterizer_cpu_avx512",
  //   IMAGE_WIDTH,
  //   IMAGE_HEIGHT,
  //   TriangleCount,
  //   CowVertices,
  //   CowVertexIndices,
  //   CowTextureCoordinates,
  //   CowTextureCoordinateIndices
  // );

#endif

  // Process<Executor_CUDA>(
  //   "GPU - CUDA",
  //   "rasterizer_gpu_cuda",
  //   IMAGE_WIDTH,
  //   IMAGE_HEIGHT,
  //   TriangleCount,
  //   CowVertices,
  //   CowVertexIndices,
  //   CowTextureCoordinates,
  //   CowTextureCoordinateIndices
  // );

  return 0;
}
