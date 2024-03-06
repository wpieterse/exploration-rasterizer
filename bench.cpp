#include <stdint.h>

#include <celero/Celero.h>

#include "exploration/rasterizer/math.hpp"
#include "exploration/rasterizer/library.hpp"

CELERO_MAIN

static constexpr uint32_t IMAGE_WIDTH  = 320;
static constexpr uint32_t IMAGE_HEIGHT = 200;

class DemoSimpleFixture : public celero::TestFixture {
  public:
    DemoSimpleFixture() {
    }

    virtual void
    setUp(celero::TestFixture::ExperimentValue const* const ExperimentValue
    ) override {
    }

    virtual void tearDown() override {
    }

  protected:
};

static constexpr auto SampleCount    = 30;
static constexpr auto IterationCount = 30;

BASELINE_F(DemoSimple, Simple, DemoSimpleFixture, SampleCount, IterationCount) {
  celero::DoNotOptimizeAway(0 == 0);
}
