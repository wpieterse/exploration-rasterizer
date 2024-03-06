struct FRenderTarget {
    uint32_t Width;
    uint32_t Height;

    float NearClipPlane;
    float FarClipPlane;

    uint32_t* Color;
    float*    Depth;
};

struct FSurfaceTriangles {
    uint32_t TriangleCount;

    Vec3f*    Vertices;
    uint32_t* VertexIndices;

    Vec2f*    TextureCoordinates;
    uint32_t* TextureCoordinateIndices;

    void* PrivateData;
};

uint32_t PackColor(uint8_t Red, uint8_t Green, uint8_t Blue, uint8_t Alpha);

void UnpackColor(
  uint32_t PackedColor,
  uint8_t* Red,
  uint8_t* Green,
  uint8_t* Blue,
  uint8_t* Alpha
);

void CreateRenderTarget(
  FRenderTarget* RenderTarget,
  int32_t        Width,
  int32_t        Height,
  float          NearClipPlane,
  float          FarClipPlane
);

void DestroyRenderTarget(FRenderTarget* RenderTarget);

void WriteColorBufferToFile(
  FRenderTarget* RenderTarget, std::string const& FileName
);

void WriteDepthBufferToFile(
  FRenderTarget* RenderTarget, std::string const& FileName
);

enum class FitResolutionGate {
  Fill     = 0,
  Overscan = 1,
};

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
);

void Rasterize_Generic(
  Matrix44f const&   WorldToCamera,
  float              Left,
  float              Right,
  float              Top,
  float              Bottom,
  FRenderTarget*     RenderTarget,
  FSurfaceTriangles* Triangles
);

void CreateSurface_Optimized(
  FSurfaceTriangles* Surface,
  uint32_t           TriangleCount,
  Vec3f*             Vertices,
  uint32_t*          VertexIndices,
  Vec2f*             TextureCoordinates,
  uint32_t*          TextureCoordinateIndices
);

void DestroySurface_Optimized(FSurfaceTriangles* Surface);

void Rasterize_Optimized(
  Matrix44f const&   WorldToCamera,
  float              Left,
  float              Right,
  float              Top,
  float              Bottom,
  FRenderTarget*     RenderTarget,
  FSurfaceTriangles* Triangles
);

#if defined(__AVX2__)

void CreateSurface_AVX2(
  FSurfaceTriangles* Surface,
  uint32_t           TriangleCount,
  Vec3f*             Vertices,
  uint32_t*          VertexIndices,
  Vec2f*             TextureCoordinates,
  uint32_t*          TextureCoordinateIndices
);

void DestroySurface_AVX2(FSurfaceTriangles* Surface);

void Rasterize_AVX2(
  Matrix44f const&   WorldToCamera,
  float              Left,
  float              Right,
  float              Top,
  float              Bottom,
  FRenderTarget*     RenderTarget,
  FSurfaceTriangles* Triangles
);

#endif

#if defined(__AVX512F__)

void Rasterize_AVX512(
  Matrix44f const&   WorldToCamera,
  float              Left,
  float              Right,
  float              Top,
  float              Bottom,
  FRenderTarget*     RenderTarget,
  FSurfaceTriangles* Triangles
);

#endif

void Rasterize_CUDA(
  Matrix44f const&   WorldToCamera,
  float              Left,
  float              Right,
  float              Top,
  float              Bottom,
  FRenderTarget*     RenderTarget,
  FSurfaceTriangles* Triangles
);
