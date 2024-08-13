//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/detail/OrientedBoundingZone.test.cc
//---------------------------------------------------------------------------//
#include "orange/detail/OrientedBoundingZone.hh"

#include <limits>
#include <vector>

#include "corecel/data/CollectionBuilder.hh"
#include "corecel/data/CollectionMirror.hh"
#include "orange/detail/TransformRecordInserter.hh"
#include "celeritas/Types.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
class OrientedBoundingZoneTest : public ::celeritas::test::Test
{
  protected:
    OrientedBoundingZoneData<Ownership::value, MemSpace::host> storage_;
    OrientedBoundingZoneData<Ownership::const_reference, MemSpace::host>
        ref_storage_;
};

TEST_F(OrientedBoundingZoneTest, basic)
{
    CollectionBuilder<Real3> half_widths(&storage_.half_widths);
    auto inner_id = half_widths.push_back({1., 1., 1.});
    auto outer_id = half_widths.push_back({2., 2., 2.});

    TransformRecordInserter tri(&storage_.transforms, &storage_.reals);
    auto transform_id = tri(
        VariantTransform{std::in_place_type<Translation>, Real3{10, 20, 30}});

    ref_storage_ = storage_;

    OrientedBoundingZone obz(inner_id, outer_id, transform_id, &ref_storage_);

    // Test is_inside functions
    EXPECT_TRUE(obz.is_inside_inner({10.5, 20.5, 30.5}));
    EXPECT_TRUE(obz.is_inside_outer({10.5, 20.5, 30.5}));

    EXPECT_FALSE(obz.is_inside_inner({11.5, 21.5, 31.5}));
    EXPECT_TRUE(obz.is_inside_outer({11.5, 21.5, 31.5}));

    EXPECT_FALSE(obz.is_inside_inner({12.5, 22.5, 32.5}));
    EXPECT_FALSE(obz.is_inside_outer({12.5, 22.5, 32.5}));

    // Test safety distance functions
    EXPECT_SOFT_EQ(1.43, obz.safety_distance_inside({10.12, 20.09, 30.57}));

    EXPECT_SOFT_EQ(std::sqrt(1.1 * 1.1),
                   obz.safety_distance_outside({10, 20, 32.1}));
    EXPECT_SOFT_EQ(std::sqrt(0.2 * 0.2 + 1.1 * 1.1),
                   obz.safety_distance_outside({10, 18.8, 32.1}));
    EXPECT_SOFT_EQ(std::sqrt(0.3 * 0.3 + 0.2 * 0.2 + 1.1 * 1.1),
                   obz.safety_distance_outside({11.3, 18.8, 32.1}));

    // Check that we get zeros for points between the inner and outer boxes
    EXPECT_SOFT_EQ(0., obz.safety_distance_inside({11.5, 20, 30}));
    EXPECT_SOFT_EQ(0., obz.safety_distance_outside({11.5, 20, 30}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
