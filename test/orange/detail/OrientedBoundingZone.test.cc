//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
    using FastReal3 = OrientedBoundingZone::FastReal3;

    template<class T>
    using Items = Collection<T, Ownership::value, MemSpace::host>;
    template<class T>
    using ItemsRef = Collection<T, Ownership::const_reference, MemSpace::host>;

    Items<FastReal3> half_widths_;
    Items<TransformRecord> transforms_;
    Items<real_type> reals_;

    ItemsRef<FastReal3> half_widths_ref_;
    ItemsRef<TransformRecord> transforms_ref_;
    ItemsRef<real_type> reals_ref_;
};

TEST_F(OrientedBoundingZoneTest, basic)
{
    // Set up OBZ such that the bboxes are:
    // inner = BBox({9, 19, 29}, {11, 21, 31})
    // outer = BBox({8.1, 18.1, 30.1}, {12.1, 22.1, 32.1})
    FastReal3 inner_hw = {1., 1., 1.};
    FastReal3 outer_hw = {2., 2., 2.};

    TransformRecordInserter tri(&transforms_, &reals_);
    auto inner_offset_id = tri(VariantTransform{
        std::in_place_type<Translation>, Real3{1.0, 2.0, 3.0}});
    auto outer_offset_id = tri(VariantTransform{
        std::in_place_type<Translation>, Real3{1.1, 2.1, 3.1}});
    auto transform_id = tri(
        VariantTransform{std::in_place_type<Translation>, Real3{9, 18, 27}});

    transforms_ref_ = transforms_;
    reals_ref_ = reals_;

    OrientedBoundingZoneRecord obz_record{
        {inner_hw, outer_hw}, {inner_offset_id, outer_offset_id}, transform_id};

    OrientedBoundingZone::StoragePointers sp{&transforms_ref_, &reals_ref_};

    OrientedBoundingZone obz(obz_record, sp);

    // Test senses
    EXPECT_EQ(SignedSense::inside, obz.calc_sense({10.5, 20.5, 30.5}));
    EXPECT_EQ(SignedSense::on, obz.calc_sense({11.5, 21.5, 31.5}));
    EXPECT_EQ(SignedSense::outside, obz.calc_sense({12.5, 22.5, 32.5}));

    // Test safety distance functions
    EXPECT_SOFT_NEAR(
        0.43, obz.calc_safety_inside({10.12, 20.09, 30.57}), 1.e-5);

    EXPECT_SOFT_NEAR(0.1, obz.calc_safety_outside({10.1, 20.1, 32.2}), 1.e-5);

    EXPECT_SOFT_NEAR(
        std::hypot(1, 0.2), obz.calc_safety_outside({10.1, 17.1, 32.3}), 1.e-5);

    EXPECT_SOFT_NEAR(std::hypot(0.2, 1, 0.2),
                     obz.calc_safety_outside({12.3, 17.1, 32.3}),
                     1.e-5);

    // Check that we get zeros for points between the inner and outer boxes
    EXPECT_SOFT_EQ(0., obz.calc_safety_inside({11.5, 20, 30}));
    EXPECT_SOFT_EQ(0., obz.calc_safety_outside({11.5, 20, 30}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
