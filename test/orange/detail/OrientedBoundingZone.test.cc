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
#include "corecel/data/Ref.hh"
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
    using ObzReal3 = OrientedBoundingZone::ObzReal3;

    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    template<class T>
    using ItemsRef = Collection<T, Ownership::const_reference, MemSpace::host>;

    Items<ObzReal3> half_widths_;
    Items<TransformRecord> transforms_;
    Items<real_type> reals_;

    ItemsRef<ObzReal3> half_widths_ref_;
    ItemsRef<TransformRecord> transforms_ref_;
    ItemsRef<real_type> reals_ref_;
};

TEST_F(OrientedBoundingZoneTest, basic)
{
    CollectionBuilder<ObzReal3> all_half_widths(&half_widths_);
    auto inner_id = all_half_widths.push_back({1., 1., 1.});
    auto outer_id = all_half_widths.push_back({2., 2., 2.});

    TransformRecordInserter tri(&transforms_, &reals_);
    auto transform_id = tri(
        VariantTransform{std::in_place_type<Translation>, Real3{10, 20, 30}});

    half_widths_ref_ = half_widths_;
    transforms_ref_ = transforms_;
    reals_ref_ = reals_;

    OrientedBoundingZoneRecord obz_record{inner_id, outer_id, transform_id};

    OrientedBoundingZone::Storage storage{
        &half_widths_ref_, &transforms_ref_, &reals_ref_};

    OrientedBoundingZone obz(&obz_record, &storage);

    // Test senses
    EXPECT_EQ(SignedSense::inside, obz.calc_sense({10.5, 20.5, 30.5}));
    EXPECT_EQ(SignedSense::on, obz.calc_sense({11.5, 21.5, 31.5}));
    EXPECT_EQ(SignedSense::outside, obz.calc_sense({12.5, 22.5, 32.5}));

    // Test safety distance functions
    EXPECT_SOFT_EQ(1.43, obz.safety_distance_inside({10.12, 20.09, 30.57}));
    EXPECT_SOFT_EQ(1.1, obz.safety_distance_outside({10, 20, 32.1}));

    EXPECT_SOFT_NEAR(std::hypot(0.2, 1.1),
                     obz.safety_distance_outside({10, 18.8, 32.1}),
                     1.e-5);
    EXPECT_SOFT_NEAR(std::hypot(0.3, 0.2, 1.1),
                     obz.safety_distance_outside({11.3, 18.8, 32.1}),
                     1.e-5);

    // Check that we get zeros for points between the inner and outer boxes
    EXPECT_SOFT_EQ(0., obz.safety_distance_inside({11.5, 20, 30}));
    EXPECT_SOFT_EQ(0., obz.safety_distance_outside({11.5, 20, 30}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
