//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/univ/detail/SenseCalculator.test.cc
//---------------------------------------------------------------------------//
#include "orange/univ/detail/SenseCalculator.hh"

#include <type_traits>

#include "corecel/Assert.hh"
#include "corecel/cont/Span.hh"
#include "orange/OrangeGeoTestBase.hh"
#include "orange/OrangeTypes.hh"
#include "orange/SenseUtils.hh"
#include "orange/surf/LocalSurfaceVisitor.hh"
#include "orange/univ/VolumeView.hh"
#include "orange/univ/detail/CachedLazySenseCalculator.hh"
#include "orange/univ/detail/LazySenseCalculator.hh"
#include "orange/univ/detail/Types.hh"

#include "celeritas_test.hh"
#include "gtest/gtest.h"

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
// DETAIL TESTS
//---------------------------------------------------------------------------//

TEST(Types, OnFace)
{
    // Null face
    OnFace not_face;
    EXPECT_FALSE(not_face);
    EXPECT_FALSE(not_face.id());
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(not_face.sense(), DebugError);
    }
    EXPECT_NO_THROW(not_face.unchecked_sense());

    // On a face
    OnFace face{FaceId{3}, Sense::outside};
    EXPECT_TRUE(face);
    EXPECT_EQ(FaceId{3}, face.id());
    EXPECT_EQ(Sense::outside, face.sense());
    EXPECT_EQ(Sense::outside, face.unchecked_sense());
}

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

template<class MySenseCalculator>
class SenseCalculatorTest : public ::celeritas::test::OrangeGeoTestBase
{
  protected:
    VolumeView make_volume_view(LocalVolumeId v) const
    {
        CELER_EXPECT(v);
        auto const& host_ref = this->host_params();
        return VolumeView{host_ref, host_ref.simple_units[SimpleUnitId{0}], v};
    }

    LocalSurfaceVisitor make_surf_visitor() const
    {
        return LocalSurfaceVisitor(this->host_params(), SimpleUnitId{0});
    }

    //! Access the shared CPU storage space for senses
    Span<SenseValue> sense_storage()
    {
        return this->host_state().temp_sense[AllItems<SenseValue>{}];
    }

    template<class MSC = MySenseCalculator,
             std::enable_if_t<std::is_same_v<MSC, LazySenseCalculator>, bool> = true>
    MSC construct_sense_calculator(LocalSurfaceVisitor const& visit,
                                   VolumeView const& vol,
                                   Real3 const& pos,
                                   OnFace& face)
    {
        return MSC(visit, vol, pos, face);
    }

    template<class MSC = MySenseCalculator,
             std::enable_if_t<!std::is_same_v<MSC, LazySenseCalculator>, bool>
             = true>
    MSC construct_sense_calculator(LocalSurfaceVisitor const& visit,
                                   VolumeView const& vol,
                                   Real3 const& pos,
                                   OnFace& face)
    {
        return MSC(visit, vol, pos, this->sense_storage(), face);
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

using TestTypes = ::testing::
    Types<SenseCalculator, CachedLazySenseCalculator, LazySenseCalculator>;

TYPED_TEST_SUITE(SenseCalculatorTest, TestTypes, );

TYPED_TEST(SenseCalculatorTest, one_volume)
{
    using MySenseCalc = TypeParam;
    {
        typename TestFixture::OneVolInput geo_inp;
        this->build_geometry(geo_inp);
    }

    // Test this degenerate case (no surfaces)
    OnFace face;
    MySenseCalc calc_senses = this->construct_sense_calculator(
        this->make_surf_visitor(),
        this->make_volume_view(LocalVolumeId{0}),
        Real3{123, 345, 567},
        face);
    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(calc_senses(FaceId{0}), DebugError);
    }
}

TYPED_TEST(SenseCalculatorTest, two_volumes)
{
    using MySenseCalc = TypeParam;
    {
        typename TestFixture::TwoVolInput geo_inp;
        geo_inp.radius = 1.5;
        this->build_geometry(geo_inp);
    }

    // Note that since these have the same faces, the results should be the
    // same for both.
    VolumeView outer = this->make_volume_view(LocalVolumeId{0});
    VolumeView inner = this->make_volume_view(LocalVolumeId{1});

    {
        // Point is in the inner sphere
        Real3 pos{0, 0.5, 0};
        {
            OnFace face;
            MySenseCalc calc_senses = this->construct_sense_calculator(
                this->make_surf_visitor(), inner, pos, face);
            // Test inner sphere, not on a face
            auto result = calc_senses(FaceId{0});
            EXPECT_EQ(Sense::inside, result);
            EXPECT_FALSE(face);
        }
        {
            OnFace face;
            MySenseCalc calc_senses = this->construct_sense_calculator(
                this->make_surf_visitor(), outer, pos, face);
            // Test not-sphere, not on a face
            auto result = calc_senses(FaceId{0});
            EXPECT_EQ(Sense::inside, result);
            EXPECT_FALSE(face);
        }
    }
    {
        // Point is in on the boundary: should register as "on" the face
        Real3 pos{1.5, 0, 0};
        {
            OnFace face;
            MySenseCalc calc_senses = this->construct_sense_calculator(
                this->make_surf_visitor(), inner, pos, face);
            auto result = calc_senses(FaceId{0});
            EXPECT_EQ(Sense::outside, result);
            EXPECT_EQ(FaceId{0}, face.id());
            EXPECT_EQ(Sense::outside, face.sense());
            if constexpr (std::is_same_v<SenseCalculator, MySenseCalc>
                          || std::is_same_v<CachedLazySenseCalculator,
                                            MySenseCalc>)
            {
                calc_senses.flip_sense(FaceId{0});
                EXPECT_EQ(Sense::inside, calc_senses(FaceId{0}));
            }
        }
        {
            OnFace face{FaceId{0}, Sense::inside};
            MySenseCalc calc_senses = this->construct_sense_calculator(
                this->make_surf_visitor(), inner, pos, face);
            auto result = calc_senses(FaceId{0});
            EXPECT_EQ(Sense::inside, result);
            EXPECT_EQ(FaceId{0}, face.id());
            EXPECT_EQ(Sense::inside, face.sense());
            if constexpr (std::is_same_v<SenseCalculator, MySenseCalc>
                          || std::is_same_v<CachedLazySenseCalculator,
                                            MySenseCalc>)
            {
                calc_senses.flip_sense(FaceId{0});
                EXPECT_EQ(Sense::outside, calc_senses(FaceId{0}));
            }
        }
    }
    {
        OnFace face;
        // Point is in the outer sphere
        MySenseCalc calc_senses = this->construct_sense_calculator(
            this->make_surf_visitor(), inner, Real3{2, 0, 0}, face);
        {
            auto result = calc_senses(FaceId{0});
            EXPECT_EQ(Sense::outside, result);
            EXPECT_FALSE(face);
        }
    }
}

TYPED_TEST(SenseCalculatorTest, five_volumes)
{
    using MySenseCalc = TypeParam;
    this->build_geometry("five-volumes.org.json");
    // this->describe(std::cout);

    auto calc_senses = [&](VolumeView vol, Real3 pos, OnFace face = {}) {
        [[maybe_unused]] MySenseCalc calc_senses
            = this->construct_sense_calculator(
                this->make_surf_visitor(), vol, pos, face);
        for (FaceId cur_face : range(FaceId{vol.num_faces()}))
        {
            if constexpr (std::is_same_v<MySenseCalc, LazySenseCalculator>)
            {
                this->sense_storage()[cur_face.unchecked_get()]
                    = calc_senses(cur_face);
            }
            else if constexpr (std::is_same_v<MySenseCalc,
                                              CachedLazySenseCalculator>)
            {
                calc_senses(cur_face);
            }
        }
        return std::pair{this->sense_storage().first(vol.num_faces()), face};
    };

    // Volume definitions
    VolumeView vol_b = this->make_volume_view(LocalVolumeId{2});
    VolumeView vol_c = this->make_volume_view(LocalVolumeId{3});
    VolumeView vol_e = this->make_volume_view(LocalVolumeId{5});

    {
        // Point is in the inner sphere
        Real3 pos{-.25, -.25, 0};
        {
            // Test inner sphere
            auto&& [storage, face] = calc_senses(vol_e, pos);
            EXPECT_EQ("{-}", this->senses_to_string(storage));
            EXPECT_FALSE(face);
        }
        {
            // Test between spheres
            auto&& [storage, face] = calc_senses(vol_c, pos);
            EXPECT_EQ("{- -}", this->senses_to_string(storage));
        }
        {
            // Test square (faces: 3, 5, 6, 7, 8, 9, 10)
            auto&& [storage, face] = calc_senses(vol_b, pos);
            EXPECT_EQ("{- + - - - - +}", this->senses_to_string(storage));
        }
    }
    {
        // Point is between spheres, on square edge (surface 8)
        Real3 pos{0.5, -0.25, 0};
        {
            // Test inner sphere
            auto&& [storage, face] = calc_senses(vol_e, pos);
            EXPECT_EQ("{+}", this->senses_to_string(storage));
            EXPECT_FALSE(face);
        }
        {
            // Test between spheres
            auto&& [storage, face] = calc_senses(vol_c, pos);
            EXPECT_EQ("{- +}", this->senses_to_string(storage));
        }
        {
            // Test square (faces: 1 through 7)
            auto&& [storage, face] = calc_senses(vol_b, pos);
            EXPECT_EQ("{- + - - + - +}", this->senses_to_string(storage));
            EXPECT_EQ(FaceId{4}, face.id());
            EXPECT_EQ(Sense::outside, face.sense());
        }
        {
            // Test square with correct face (surface 8, face 4)
            auto&& [storage, face]
                = calc_senses(vol_b, pos, OnFace{FaceId{4}, Sense::outside});
            EXPECT_EQ("{- + - - + - +}", this->senses_to_string(storage));
            EXPECT_EQ(FaceId{4}, face.id());
            EXPECT_EQ(Sense::outside, face.sense());
        }
        {
            // Test square with flipped sense
            auto&& [storage, face]
                = calc_senses(vol_b, pos, OnFace{FaceId{4}, Sense::inside});
            EXPECT_EQ("{- + - - - - +}", this->senses_to_string(storage));
            EXPECT_EQ(FaceId{4}, face.id());
            EXPECT_EQ(Sense::inside, face.sense());
        }
        {
            // Test square with "incorrect" face that gets assigned anyway
            auto&& [storage, face]
                = calc_senses(vol_b, pos, OnFace{FaceId{1}, Sense::inside});

            EXPECT_EQ("{- - - - + - +}", this->senses_to_string(storage));
            EXPECT_EQ(FaceId{1}, face.id());
            EXPECT_EQ(Sense::inside, face.sense());
        }
        if (CELERITAS_DEBUG)
        {
            // Out-of-range face ID
            EXPECT_THROW(
                calc_senses(vol_b, pos, OnFace{FaceId{8}, Sense::inside}),
                DebugError);
        }
    }
    {
        // Point is exactly on the lower right corner of b. If a face isn't
        // given then the lower face ID will be the one considered "on".
        // +x = surface 9 = face 5
        // -y = surface 10 = face 6
        Real3 pos{1.5, -1.0, 0};
        {
            // Test natural sense
            auto&& [storage, face] = calc_senses(vol_b, pos);
            EXPECT_EQ("{- + - + + + +}", this->senses_to_string(storage));
            EXPECT_EQ(FaceId{5}, face.id());
            EXPECT_EQ(Sense::outside, face.sense());
        }
        {
            // Test with lower face, flipped sense
            auto&& [storage, face]
                = calc_senses(vol_b, pos, OnFace{FaceId{5}, Sense::inside});
            EXPECT_EQ("{- + - + + - +}", this->senses_to_string(storage));
            EXPECT_EQ(FaceId{5}, face.id());
            EXPECT_EQ(Sense::inside, face.sense());
        }
        {
            // Test with right face, flipped sense
            auto&& [storage, face]
                = calc_senses(vol_b, pos, OnFace{FaceId{6}, Sense::inside});
            EXPECT_EQ("{- + - + + + -}", this->senses_to_string(storage));
            EXPECT_EQ(FaceId{6}, face.id());
            EXPECT_EQ(Sense::inside, face.sense());
        }
    }
}
//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
