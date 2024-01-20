//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/surf/SurfaceClipper.test.cc
//---------------------------------------------------------------------------//
#include "orange/surf/SurfaceClipper.hh"

#include "orange/surf/CylAligned.hh"
#include "orange/surf/CylCentered.hh"
#include "orange/surf/PlaneAligned.hh"
#include "orange/surf/Sphere.hh"
#include "orange/surf/SphereCentered.hh"
#include "orange/surf/VariantSurface.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class SurfaceClipperTest : public ::celeritas::test::Test
{
  protected:
    using Real6 = Array<real_type, 6>;

    void SetUp() override {}

    template<class S>
    Real6 clip_inf(Sense sense, S const& surf)
    {
        this->bbox = BBox::from_infinite();
        this->clip(sense, surf);
        return this->flattened_bbox();
    }

    Real6 flattened_bbox() const
    {
        CELER_EXPECT(bbox);
        Real6 result;
        auto iter = result.begin();
        iter = std::copy(bbox.lower().begin(), bbox.lower().end(), iter);
        iter = std::copy(bbox.upper().begin(), bbox.upper().end(), iter);
        CELER_ASSERT(iter == result.end());
        return result;
    }

    BBox bbox;
    SurfaceClipper clip{&bbox};
};

TEST_F(SurfaceClipperTest, inside)
{
    EXPECT_VEC_SOFT_EQ((Real6{-inf, -inf, -inf, 4, inf, inf}),
                       this->clip_inf(Sense::inside, PlaneX{4}));
    EXPECT_VEC_SOFT_EQ((Real6{-3, -3, -inf, 3, 3, inf}),
                       this->clip_inf(Sense::inside, CCylZ{3}));
    EXPECT_VEC_SOFT_EQ((Real6{-3, -3, -3, 3, 3, 3}),
                       this->clip_inf(Sense::inside, SphereCentered{3}));
    EXPECT_VEC_SOFT_EQ((Real6{0.75, 1.75, -inf, 1.25, 2.25, inf}),
                       this->clip_inf(Sense::inside, CylZ{{1, 2, 3}, 0.25}));
    EXPECT_VEC_SOFT_EQ((Real6{0.75, 1.75, 2.75, 1.25, 2.25, 3.25}),
                       this->clip_inf(Sense::inside, Sphere{{1, 2, 3}, 0.25}));

    // Test with variant
    VariantSurface v{CCylY{4}};
    EXPECT_VEC_SOFT_EQ((Real6{-4, -inf, -4, 4, inf, 4}),
                       this->clip_inf(Sense::inside, v));
}

TEST_F(SurfaceClipperTest, outside)
{
    EXPECT_VEC_SOFT_EQ((Real6{4, -inf, -inf, inf, inf, inf}),
                       this->clip_inf(Sense::outside, PlaneX{4}));
    EXPECT_VEC_SOFT_EQ((Real6{-inf, -inf, -inf, inf, inf, inf}),
                       this->clip_inf(Sense::outside, CCylZ{3}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
