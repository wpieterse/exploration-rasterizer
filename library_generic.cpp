#include <stdint.h>
#include <math.h>

#include <algorithm>
#include <string>

#include "exploration/rasterizer/math.hpp"
#include "exploration/rasterizer/library.hpp"

static constexpr int32_t CHECKER_PATTERN_SIZE = 10;

//
// Compute vertex raster screen coordinates. Vertices are defined in world
// space. They are then converted to camera space, then to NDC space (in the
// range [-1,1]) and then to raster space. The z-coordinates of the vertex in
// raster space is set with the z-coordinate of the vertex in camera space.
//
static void TransformVertex(
  Vec3f const&     VertexWorld,
  Matrix44f const& WorldToCamera,
  float            Left,
  float            Right,
  float            Top,
  float            Bottom,
  float            NearClipPlane,
  uint32_t         ImageWidth,
  uint32_t         ImageHeight,
  Vec3f&           VertexRaster
) {
  Vec3f VertexCamera;

  WorldToCamera.multVecMatrix(VertexWorld, VertexCamera);

  // convert to screen space
  Vec2f VertexScreen;
  VertexScreen.x = NearClipPlane * VertexCamera.x / -VertexCamera.z;
  VertexScreen.y = NearClipPlane * VertexCamera.y / -VertexCamera.z;

  // now convert point from screen space to NDC space (in range [-1,1])
  Vec2f VertexNormalizedDeviceCoordinate;

  VertexNormalizedDeviceCoordinate.x =
    2 * VertexScreen.x / (Right - Left) - (Right + Left) / (Right - Left);

  VertexNormalizedDeviceCoordinate.y =
    2 * VertexScreen.y / (Top - Bottom) - (Top + Bottom) / (Top - Bottom);

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
  return (Vertex2[0] - Vertex0[0]) * (Vertex1[1] - Vertex0[1]) -
         (Vertex2[1] - Vertex0[1]) * (Vertex1[0] - Vertex0[0]);
}

void Rasterize_Generic(
  Matrix44f const&   WorldToCamera,
  float              Left,
  float              Right,
  float              Top,
  float              Bottom,
  FRenderTarget*     RenderTarget,
  FSurfaceTriangles* Triangles
) {
  // Outer loop
  for(uint32_t TriangleIndex = 0; TriangleIndex < Triangles->TriangleCount;
      TriangleIndex++) {
    Vec3f const& Vertex0World =
      Triangles->Vertices[Triangles->VertexIndices[TriangleIndex * 3 + 0]];

    Vec3f const& Vertex1World =
      Triangles->Vertices[Triangles->VertexIndices[TriangleIndex * 3 + 1]];

    Vec3f const& Vertex2World =
      Triangles->Vertices[Triangles->VertexIndices[TriangleIndex * 3 + 2]];

    // Convert the vertices of the triangle to raster space
    Vec3f Vertex0Raster;
    Vec3f Vertex1Raster;
    Vec3f Vertex2Raster;

    TransformVertex(
      Vertex0World,
      WorldToCamera,
      Left,
      Right,
      Top,
      Bottom,
      RenderTarget->NearClipPlane,
      RenderTarget->Width,
      RenderTarget->Height,
      Vertex0Raster
    );

    TransformVertex(
      Vertex1World,
      WorldToCamera,
      Left,
      Right,
      Top,
      Bottom,
      RenderTarget->NearClipPlane,
      RenderTarget->Width,
      RenderTarget->Height,
      Vertex1Raster
    );

    TransformVertex(
      Vertex2World,
      WorldToCamera,
      Left,
      Right,
      Top,
      Bottom,
      RenderTarget->NearClipPlane,
      RenderTarget->Width,
      RenderTarget->Height,
      Vertex2Raster
    );

    // Precompute reciprocal of vertex z-coordinate
    Vertex0Raster.z = 1 / Vertex0Raster.z;
    Vertex1Raster.z = 1 / Vertex1Raster.z;
    Vertex2Raster.z = 1 / Vertex2Raster.z;

    // Prepare vertex attributes. Divde them by their vertex z-coordinate
    // (though we use a multiplication here because v.z = 1 / v.z)
    Vec2f TextureCoordinate0 =
      Triangles->TextureCoordinates
        [Triangles->TextureCoordinateIndices[TriangleIndex * 3 + 0]];

    Vec2f TextureCoordinate1 =
      Triangles->TextureCoordinates
        [Triangles->TextureCoordinateIndices[TriangleIndex * 3 + 1]];

    Vec2f TextureCoordinate2 =
      Triangles->TextureCoordinates
        [Triangles->TextureCoordinateIndices[TriangleIndex * 3 + 2]];

    TextureCoordinate0 *= Vertex0Raster.z;
    TextureCoordinate1 *= Vertex1Raster.z;
    TextureCoordinate2 *= Vertex2Raster.z;

    float TriangleXMinimum =
      MinimumOf3(Vertex0Raster.x, Vertex1Raster.x, Vertex2Raster.x);

    float TriangleYMinimum =
      MinimumOf3(Vertex0Raster.y, Vertex1Raster.y, Vertex2Raster.y);

    float TriangleXMaximum =
      MaximumOf3(Vertex0Raster.x, Vertex1Raster.x, Vertex2Raster.x);

    float TriangleYMaximum =
      MaximumOf3(Vertex0Raster.y, Vertex1Raster.y, Vertex2Raster.y);

    // The triangle is out of screen
    if(TriangleXMinimum > RenderTarget->Width - 1 || TriangleXMaximum < 0 || TriangleYMinimum > RenderTarget->Height - 1 || TriangleYMaximum < 0) {
      continue;
    }

    // Be careful
    // TriangleXMinimum/TriangleXMaximum/TriangleYMinimum/TriangleYMaximum can
    // be negative. Don't cast to uint32_t!
    uint32_t TriangleXStart =
      std::max(int32_t(0), (int32_t) (std::floor(TriangleXMinimum)));

    uint32_t TriangleXEnd = std::min(
      int32_t(RenderTarget->Width) - 1, (int32_t) (std::floor(TriangleXMaximum))
    );

    uint32_t TriangleYStart =
      std::max(int32_t(0), (int32_t) (std::floor(TriangleYMinimum)));

    uint32_t TriangleYEnd = std::min(
      int32_t(RenderTarget->Height) - 1,
      (int32_t) (std::floor(TriangleYMaximum))
    );

    float TriangleArea =
      EdgeFunction(Vertex0Raster, Vertex1Raster, Vertex2Raster);

    // Inner loop
    for(uint32_t CurrentY = TriangleYStart; CurrentY <= TriangleYEnd;
        CurrentY++) {
      for(uint32_t CurrentX = TriangleXStart; CurrentX <= TriangleXEnd;
          CurrentX++) {
        Vec3f PixelLocation(CurrentX + 0.5, CurrentY + 0.5, 0);

        float Weight_0 =
          EdgeFunction(Vertex1Raster, Vertex2Raster, PixelLocation);

        float Weight_1 =
          EdgeFunction(Vertex2Raster, Vertex0Raster, PixelLocation);
          
        float Weight_2 =
          EdgeFunction(Vertex0Raster, Vertex1Raster, PixelLocation);

        if(Weight_0 >= 0 && Weight_1 >= 0 && Weight_2 >= 0) {
          Weight_0 /= TriangleArea;
          Weight_1 /= TriangleArea;
          Weight_2 /= TriangleArea;

          float OneOverZ = Vertex0Raster.z * Weight_0 +
                           Vertex1Raster.z * Weight_1 +
                           Vertex2Raster.z * Weight_2;

          float Z = 1 / OneOverZ;

          // Depth-buffer test
          if(Z < RenderTarget->Depth[CurrentY * RenderTarget->Width + CurrentX]) {
            RenderTarget->Depth[CurrentY * RenderTarget->Width + CurrentX] = Z;

            Vec2f TextureCoordinate = TextureCoordinate0 * Weight_0 +
                                      TextureCoordinate1 * Weight_1 +
                                      TextureCoordinate2 * Weight_2;

            TextureCoordinate *= Z;

            Vec3f Vertex0Camera;
            Vec3f Vertex1Camera;
            Vec3f Vertex2Camera;

            WorldToCamera.multVecMatrix(Vertex0World, Vertex0Camera);
            WorldToCamera.multVecMatrix(Vertex1World, Vertex1Camera);
            WorldToCamera.multVecMatrix(Vertex2World, Vertex2Camera);

            float PixelInterpolantX =
              (Vertex0Camera.x / -Vertex0Camera.z) * Weight_0 +
              (Vertex1Camera.x / -Vertex1Camera.z) * Weight_1 +
              (Vertex2Camera.x / -Vertex2Camera.z) * Weight_2;

            float PixelInterpolantY =
              (Vertex0Camera.y / -Vertex0Camera.z) * Weight_0 +
              (Vertex1Camera.y / -Vertex1Camera.z) * Weight_1 +
              (Vertex2Camera.y / -Vertex2Camera.z) * Weight_2;

            // PixelPoint is in camera space
            Vec3f PixelPoint(PixelInterpolantX * Z, PixelInterpolantY * Z, -Z);

            // Compute the face normal which is used for a simple facing ratio.
            // Keep in mind that we are doing all calculation in camera space.
            // Thus the view direction can be computed as the point on the
            // object in camera space minus Vec3f(0), the position of the camera
            // in camera space.
            Vec3f PixelNormal = (Vertex1Camera - Vertex0Camera)
                                  .crossProduct(Vertex2Camera - Vertex0Camera);
            PixelNormal.normalize();

            Vec3f ViewDirection = -PixelPoint;
            ViewDirection.normalize();

            float NormalDotView =
              std::max(0.f, PixelNormal.dotProduct(ViewDirection));

            // The final color is the reuslt of the faction ration multiplied by
            // the checkerboard pattern.
            float CheckerPattern =
              (fmod(TextureCoordinate.x * CHECKER_PATTERN_SIZE, 1.0) > 0.5) ^
              (fmod(TextureCoordinate.y * CHECKER_PATTERN_SIZE, 1.0) < 0.5);

            float TextureColor =
              0.3 * (1 - CheckerPattern) + 0.7 * CheckerPattern;

            NormalDotView *= TextureColor;

            RenderTarget->Color[CurrentX + CurrentY * RenderTarget->Width] =
              PackColor(
                NormalDotView * 255,
                NormalDotView * 255,
                NormalDotView * 255,
                255
              );
          }
        }
      }
    }
  }
}
