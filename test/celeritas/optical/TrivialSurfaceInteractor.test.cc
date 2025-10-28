//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/TrivialSurfaceInteractor.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/optical/surface/model/TrivialInteractor.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace ::celeritas::test;
//---------------------------------------------------------------------------//

class TrivialSurfaceInteractorTest : public ::celeritas::test::Test
{
};

TEST_F(TrivialSurfaceInteractorTest, interactor)
{
    using M = TrivialInteractionMode;
    using A = SurfaceInteraction::Action;

    // Check results for each mode
    auto check_results = [](Real3 const& dir, Real3 const& pol) {
        {
            auto result = TrivialInteractor{M::absorb, dir, pol}();
            EXPECT_EQ(A::absorbed, result.action);
        }
        {
            auto result = TrivialInteractor{M::transmit, dir, pol}();
            EXPECT_EQ(A::refracted, result.action);
            EXPECT_VEC_EQ(dir, result.direction);
            EXPECT_VEC_EQ(pol, result.polarization);
        }
        {
            auto result = TrivialInteractor{M::backscatter, dir, pol}();
            EXPECT_EQ(A::reflected, result.action);
            EXPECT_VEC_EQ(negate(dir), result.direction);
            EXPECT_VEC_EQ(negate(pol), result.polarization);
        }
    };

    check_results(make_unit_vector(Real3{0, 0, 1}),
                  make_unit_vector(Real3{1, -1, 0}));

    check_results(make_unit_vector(Real3{1, 3, 2}),
                  make_unit_vector(Real3{-3, 1, 0}));

    check_results(make_unit_vector(Real3{1, 1, 1}),
                  make_unit_vector(Real3{-2, 1, 1}));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
