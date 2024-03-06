#if defined(__AVX512F__)

#  include <stdint.h>
#  include <math.h>

#  include <immintrin.h>

#  include <algorithm>
#  include <string>

#  include "exploration/rasterizer/math.hpp"
#  include "exploration/rasterizer/library.hpp"

static constexpr int32_t CHECKER_PATTERN_SIZE = 10;

struct FTransformVertexConstants {
    float RightLeftConstant;
    float TopBottomConstant;
};

static void TransformVertex(
  Vec3f const&                     VertexWorld,
  Matrix44f const&                 WorldToCamera,
  FTransformVertexConstants const& Constants,
  float const&                     NearClipPlane,
  uint32_t const&                  ImageWidth,
  uint32_t const&                  ImageHeight,
  Vec3f&                           VertexCamera,
  Vec3f&                           VertexRaster
) {
  WorldToCamera.multVecMatrix(VertexWorld, VertexCamera);

  // convert to screen space
  Vec2f VertexScreen;
  VertexScreen.x = NearClipPlane * VertexCamera.x / -VertexCamera.z;
  VertexScreen.y = NearClipPlane * VertexCamera.y / -VertexCamera.z;

  // now convert point from screen space to NDC space (in range [-1,1])
  Vec2f VertexNormalizedDeviceCoordinate;

  VertexNormalizedDeviceCoordinate.x =
    2 * VertexScreen.x / Constants.RightLeftConstant;

  VertexNormalizedDeviceCoordinate.y =
    2 * VertexScreen.y / Constants.TopBottomConstant;

  // convert to raster space
  VertexRaster.x = (VertexNormalizedDeviceCoordinate.x + 1) / 2 * ImageWidth;

  // in raster space y is down so invert direction
  VertexRaster.y = (1 - VertexNormalizedDeviceCoordinate.y) / 2 * ImageHeight;
  VertexRaster.z = -VertexCamera.z;
}

static float MinimumOf3(float Left, float Middle, float Right) {
  return std::min(Left, std::min(Middle, Right));
}

static float MaximumOf3(float Left, float Middle, float Right) {
  return std::max(Left, std::max(Middle, Right));
}

static float
EdgeFunction(Vec3f const& Vertex0, Vec3f const& Vertex1, Vec3f const& Vertex2) {
  return (Vertex1[0] - Vertex0[0]) * (Vertex2[1] - Vertex0[1]) -
         (Vertex1[1] - Vertex0[1]) * (Vertex2[0] - Vertex0[0]);
}

template<typename ElementType>
constexpr ElementType AlignUp(ElementType Value, uint64_t Alignment) {
  return (ElementType) (((uint64_t) Value + Alignment - 1) & ~(Alignment - 1));
}

template<typename ElementType>
constexpr ElementType AlignDown(ElementType Value, uint64_t Alignment) {
  return (ElementType) (((uint64_t) Value) & ~(Alignment - 1));
}

struct FEdge {
    static constexpr uint32_t StepXSize = 8;
    static constexpr uint32_t StepYSize = 1;

    __m256 OneStepX;
    __m256 OneStepY;

    __m256 Initialize(
      Vec3f const& Vertex0, Vec3f const& Vertex1, Vec3f const& Origin
    ) {
      // Edge setup
      float A = Vertex0.y - Vertex1.y;
      float B = Vertex1.x - Vertex0.x;
      float C = Vertex0.x * Vertex1.y - Vertex0.y * Vertex1.x;

      __m256 A_SIMD = _mm256_set1_ps(A);
      __m256 B_SIMD = _mm256_set1_ps(B);
      __m256 C_SIMD = _mm256_set1_ps(C);

      // Step deltas
      OneStepX = _mm256_set1_ps(A * StepXSize);
      OneStepY = _mm256_set1_ps(B * StepYSize);

      __m256 Increase_SIMD =
        _mm256_set_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
      __m256 Origin_X_Raw_SIMD = _mm256_set1_ps(Origin.x);
      __m256 Origin_X_SIMD = _mm256_add_ps(Origin_X_Raw_SIMD, Increase_SIMD);
      __m256 Origin_Y_SIMD = _mm256_set1_ps(Origin.y);

      __m256 ATimesX = _mm256_mul_ps(A_SIMD, Origin_X_SIMD);
      __m256 BTimesY = _mm256_mul_ps(B_SIMD, Origin_Y_SIMD);

      // Edge function values at origin
      return _mm256_add_ps(ATimesX, _mm256_add_ps(BTimesY, C_SIMD));
    }
};

void Rasterize_AVX512(
  Matrix44f const&   WorldToCamera,
  float              Left,
  float              Right,
  float              Top,
  float              Bottom,
  FRenderTarget*     RenderTarget,
  FSurfaceTriangles* Triangles
) {
  float const RightLeftMinus = Right - Left;
  float const RightLeftPlus  = Right + Left;
  float const TopBottomMinus = Top - Bottom;
  float const TopBottomPlus  = Top + Bottom;

  FTransformVertexConstants TransformVertexConstants {
    .RightLeftConstant = RightLeftMinus - RightLeftPlus / RightLeftMinus,
    .TopBottomConstant = TopBottomMinus - TopBottomPlus / TopBottomMinus,
  };

  __m256 Zero_SIMD = _mm256_set1_ps(0.0f);

  for(uint32_t TriangleIndex = 0; TriangleIndex < Triangles->TriangleCount;
      TriangleIndex++) {
    Vec3f const& Vertex0World =
      Triangles->Vertices[Triangles->VertexIndices[TriangleIndex * 3 + 0]];

    Vec3f const& Vertex1World =
      Triangles->Vertices[Triangles->VertexIndices[TriangleIndex * 3 + 1]];

    Vec3f const& Vertex2World =
      Triangles->Vertices[Triangles->VertexIndices[TriangleIndex * 3 + 2]];

    Vec3f Vertex0Camera;
    Vec3f Vertex1Camera;
    Vec3f Vertex2Camera;

    Vec3f Vertex0Raster;
    Vec3f Vertex1Raster;
    Vec3f Vertex2Raster;

    TransformVertex(
      Vertex0World,
      WorldToCamera,
      TransformVertexConstants,
      RenderTarget->NearClipPlane,
      RenderTarget->Width,
      RenderTarget->Height,
      Vertex1Camera,
      Vertex2Raster
    );

    TransformVertex(
      Vertex1World,
      WorldToCamera,
      TransformVertexConstants,
      RenderTarget->NearClipPlane,
      RenderTarget->Width,
      RenderTarget->Height,
      Vertex2Camera,
      Vertex1Raster
    );

    TransformVertex(
      Vertex2World,
      WorldToCamera,
      TransformVertexConstants,
      RenderTarget->NearClipPlane,
      RenderTarget->Width,
      RenderTarget->Height,
      Vertex0Camera,
      Vertex0Raster
    );

    float TriangleXMinimum =
      MinimumOf3(Vertex0Raster.x, Vertex1Raster.x, Vertex2Raster.x);

    float TriangleYMinimum =
      MinimumOf3(Vertex0Raster.y, Vertex1Raster.y, Vertex2Raster.y);

    float TriangleXMaximum =
      MaximumOf3(Vertex0Raster.x, Vertex1Raster.x, Vertex2Raster.x);

    float TriangleYMaximum =
      MaximumOf3(Vertex0Raster.y, Vertex1Raster.y, Vertex2Raster.y);

    if(TriangleXMinimum > RenderTarget->Width - 1 || TriangleXMaximum < 0 || TriangleYMinimum > RenderTarget->Height - 1 || TriangleYMaximum < 0) {
      continue;
    }

    Vec2f TextureCoordinate2 =
      Triangles->TextureCoordinates
        [Triangles->TextureCoordinateIndices[TriangleIndex * 3 + 0]];

    Vec2f TextureCoordinate1 =
      Triangles->TextureCoordinates
        [Triangles->TextureCoordinateIndices[TriangleIndex * 3 + 1]];

    Vec2f TextureCoordinate0 =
      Triangles->TextureCoordinates
        [Triangles->TextureCoordinateIndices[TriangleIndex * 3 + 2]];

    Vertex0Raster.z = 1 / Vertex0Raster.z;
    Vertex1Raster.z = 1 / Vertex1Raster.z;
    Vertex2Raster.z = 1 / Vertex2Raster.z;

    TextureCoordinate0 *= Vertex0Raster.z;
    TextureCoordinate1 *= Vertex1Raster.z;
    TextureCoordinate2 *= Vertex2Raster.z;

    float TriangleArea =
      EdgeFunction(Vertex0Raster, Vertex1Raster, Vertex2Raster);

    __m256 TriangleArea_SIMD = _mm256_set1_ps(TriangleArea);

    float ColumStep_01  = Vertex0Raster.y - Vertex1Raster.y;
    float ColumnStep_12 = Vertex1Raster.y - Vertex2Raster.y;
    float ColumnStep_20 = Vertex2Raster.y - Vertex0Raster.y;

    float RowStep_01 = Vertex1Raster.x - Vertex0Raster.x;
    float RowStep_12 = Vertex2Raster.x - Vertex1Raster.x;
    float RowStep_20 = Vertex0Raster.x - Vertex2Raster.x;

    uint32_t TriangleXStartInitial =
      std::max(int32_t(0), (int32_t) (std::floor(TriangleXMinimum)));

    uint32_t TriangleXEndInitial = std::min(
      int32_t(RenderTarget->Width) - 1, (int32_t) (std::floor(TriangleXMaximum))
    );

    uint32_t TriangleYStartInitial =
      std::max(int32_t(0), (int32_t) (std::floor(TriangleYMinimum)));

    uint32_t TriangleYEndInitial = std::min(
      int32_t(RenderTarget->Height) - 1,
      (int32_t) (std::floor(TriangleYMaximum))
    );

    uint32_t TriangleXStart =
      AlignDown(TriangleXStartInitial, FEdge::StepXSize);
    uint32_t TriangleXEnd = AlignUp(TriangleXEndInitial, FEdge::StepXSize);
    uint32_t TriangleYStart =
      AlignDown(TriangleYStartInitial, FEdge::StepYSize);
    uint32_t TriangleYEnd = AlignUp(TriangleYEndInitial, FEdge::StepYSize);

    Vec3f TriangleStartPoint(TriangleXStart + 0.5, TriangleYStart + 0.5, 0);

    float WeightedRowStep_0 =
      EdgeFunction(Vertex1Raster, Vertex2Raster, TriangleStartPoint);

    float WeightedRowStep_1 =
      EdgeFunction(Vertex2Raster, Vertex0Raster, TriangleStartPoint);

    float WeightedRowStep_2 =
      EdgeFunction(Vertex0Raster, Vertex1Raster, TriangleStartPoint);

    FEdge Edge_01;
    FEdge Edge_12;
    FEdge Edge_20;

    __m256 Weight_0_Row_SIMD =
      Edge_12.Initialize(Vertex1Raster, Vertex2Raster, TriangleStartPoint);

    __m256 Weight_1_Row_SIMD =
      Edge_20.Initialize(Vertex2Raster, Vertex0Raster, TriangleStartPoint);

    __m256 Weight_2_Row_SIMD =
      Edge_01.Initialize(Vertex0Raster, Vertex1Raster, TriangleStartPoint);

    for(uint32_t CurrentY  = TriangleYStart; CurrentY <= TriangleYEnd;
        CurrentY          += FEdge::StepYSize) {
      float WeightedColumStep_0 = WeightedRowStep_0;
      float WeightedColumStep_1 = WeightedRowStep_1;
      float WeightedColumStep_2 = WeightedRowStep_2;

      __m256 Weight_0_SIMD = Weight_0_Row_SIMD;
      __m256 Weight_1_SIMD = Weight_1_Row_SIMD;
      __m256 Weight_2_SIMD = Weight_2_Row_SIMD;

      for(uint32_t CurrentX  = TriangleXStart; CurrentX <= TriangleXEnd;
          CurrentX          += FEdge::StepXSize) {
        __m256 Weight_0_Mask_SIMD =
          _mm256_cmp_ps(Weight_0_SIMD, Zero_SIMD, _CMP_GE_OQ);

        __m256 Weight_1_Mask_SIMD =
          _mm256_cmp_ps(Weight_1_SIMD, Zero_SIMD, _CMP_GE_OQ);

        __m256 Weight_2_Mask_SIMD =
          _mm256_cmp_ps(Weight_2_SIMD, Zero_SIMD, _CMP_GE_OQ);

        __m256 Weight_Mask_SIMD = _mm256_or_si256(
          Weight_0_Mask_SIMD,
          _mm256_or_si256(Weight_1_Mask_SIMD, Weight_2_Mask_SIMD)
        );

        if(WeightedColumStep_0 >= 0 && WeightedColumStep_1 >= 0 && WeightedColumStep_2 >= 0) {
          float Weight0 = WeightedColumStep_0 / TriangleArea;
          float Weight1 = WeightedColumStep_1 / TriangleArea;
          float Weight2 = WeightedColumStep_2 / TriangleArea;

          float OneOverZ = Vertex0Raster.z * Weight0 +
                           Vertex1Raster.z * Weight1 +
                           Vertex2Raster.z * Weight2;

          float Z = 1 / OneOverZ;

          if(Z < RenderTarget->Depth[CurrentY * RenderTarget->Width + CurrentX]) {
            RenderTarget->Depth[CurrentY * RenderTarget->Width + CurrentX] = Z;

            Vec2f TextureCoordinate = TextureCoordinate0 * Weight0 +
                                      TextureCoordinate1 * Weight1 +
                                      TextureCoordinate2 * Weight2;

            TextureCoordinate *= Z;

            float PixelInterpolantX =
              (Vertex0Camera.x / -Vertex0Camera.z) * Weight0 +
              (Vertex1Camera.x / -Vertex1Camera.z) * Weight1 +
              (Vertex2Camera.x / -Vertex2Camera.z) * Weight2;

            float PixelInterpolantY =
              (Vertex0Camera.y / -Vertex0Camera.z) * Weight0 +
              (Vertex1Camera.y / -Vertex1Camera.z) * Weight1 +
              (Vertex2Camera.y / -Vertex2Camera.z) * Weight2;

            Vec3f PixelPoint(PixelInterpolantX * Z, PixelInterpolantY * Z, -Z);

            Vec3f PixelNormal = (Vertex1Camera - Vertex0Camera)
                                  .crossProduct(Vertex2Camera - Vertex0Camera);
            PixelNormal.normalize();

            Vec3f ViewDirection = -PixelPoint;
            ViewDirection.normalize();

            float NormalDotView =
              std::max(0.0f, PixelNormal.dotProduct(ViewDirection));

            float CheckerPattern =
              (fmod(TextureCoordinate.x * CHECKER_PATTERN_SIZE, 1.0) > 0.5) ^
              (fmod(TextureCoordinate.y * CHECKER_PATTERN_SIZE, 1.0) < 0.5);

            float TextureColor =
              0.3 * (1 - CheckerPattern) + 0.7 * CheckerPattern;

            NormalDotView *= TextureColor;

            RenderTarget->Color[CurrentY * RenderTarget->Width + CurrentX] =
              PackColor(
                NormalDotView * 255,
                NormalDotView * 255,
                NormalDotView * 255,
                255
              );
          }
        }

        WeightedColumStep_0 += ColumnStep_12;
        WeightedColumStep_1 += ColumnStep_20;
        WeightedColumStep_2 += ColumStep_01;

        Weight_0_SIMD = _mm256_add_ps(Weight_0_SIMD, Edge_12.OneStepX);
        Weight_1_SIMD = _mm256_add_ps(Weight_1_SIMD, Edge_20.OneStepX);
        Weight_2_SIMD = _mm256_add_ps(Weight_2_SIMD, Edge_01.OneStepX);
      }

      WeightedRowStep_0 += RowStep_12;
      WeightedRowStep_1 += RowStep_20;
      WeightedRowStep_2 += RowStep_01;

      Weight_0_Row_SIMD = _mm256_add_ps(Weight_0_Row_SIMD, Edge_12.OneStepY);
      Weight_1_Row_SIMD = _mm256_add_ps(Weight_1_Row_SIMD, Edge_20.OneStepY);
      Weight_2_Row_SIMD = _mm256_add_ps(Weight_2_Row_SIMD, Edge_01.OneStepY);
    }
  }
}

#endif
