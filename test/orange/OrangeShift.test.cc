//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/OrangeShift.test.cc
//---------------------------------------------------------------------------//
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/cont/Array.hh"
#include "corecel/cont/Range.hh"
#include "geocel/Types.hh"
#include "orange/OrangeTrackView.hh"

#include "OrangeGeoTestBase.hh"
#include "TestMacros.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//

class ShiftTrackerTest : public OrangeGeoTestBase
{
  protected:
    enum class BoundaryState
    {
        INSIDE = 0,
        OUTSIDE = 1
    };

    void SetUp() final { this->build_geometry("hex-array.org.json"); }

    CELER_FUNCTION static constexpr unsigned int invalid_id()
    {
        return static_cast<unsigned int>(-1);
    }

    void initialize(Real3 pos, Real3 dir)
    {
        auto track = this->make_geo_track_view();
        track = {pos, dir};
    }

    void distance_to_boundary(real_type& distance)
    {
        auto track = this->make_geo_track_view();
        distance = track.find_next_step().distance;
    }

    void move_to_point(real_type distance)
    {
        auto track = this->make_geo_track_view();
        track.move_internal(distance);
    }

    void move_across_surface(BoundaryState& boundary_state, unsigned int& cell)
    {
        auto track = this->make_geo_track_view();
        track.move_to_boundary();
        track.cross_boundary();

        if (!track.is_outside())
        {
            boundary_state = BoundaryState::INSIDE;
            cell = track.impl_volume_id().get();
        }
        else
        {
            boundary_state = BoundaryState::OUTSIDE;
            cell = invalid_id();
        }
    }
};

//---------------------------------------------------------------------------//
TEST_F(ShiftTrackerTest, host)
{
    std::vector<Real3> pos{
        {-0.5466, 1.1298, -1.8526},
        {1.5968, -4.3272, -3.0764},
        {-1.2053, -2.7582, -0.1715},
        {-2.3368, -1.7800, 1.2726},
        {4.0610, 1.5512, 2.8693},
        {-1.5469, 1.6592, -0.6909},
        {-3.6040, -0.7626, -1.7840},
        {4.3726, -2.5543, -0.0444},
        {1.7047, 1.6042, 4.4779},
        {-0.8630, -4.8264, 3.1796},
    };
    std::vector<Array<real_type, 2>> mu_phi{
        {0.215991, 1.114365},
        {-0.887921, 1.414178},
        {0.727041, 5.874378},
        {0.822052, 3.051407},
        {0.576156, 3.585084},
        {-0.243608, 0.901952},
        {0.486739, 2.328782},
        {0.966572, 4.876337},
        {-0.798997, 0.149136},
        {0.748980, 1.677583},
    };

    std::vector<unsigned int> steps(10, 0);

    for (auto n : range(pos.size()))
    {
        auto costheta = mu_phi[n][0];
        auto sintheta = std::sqrt(1 - costheta * costheta);
        auto phi = mu_phi[n][1];
        Real3 dir
            = {sintheta * std::cos(phi), sintheta * std::sin(phi), costheta};

        this->initialize(pos[n], dir);

        auto dbnd = std::numeric_limits<real_type>::max();
        auto cell = this->invalid_id();
        BoundaryState bnd_state = BoundaryState::INSIDE;

        while (bnd_state == BoundaryState::INSIDE)
        {
            this->distance_to_boundary(dbnd);
            this->move_across_surface(bnd_state, cell);

            ++steps[n];
        }
    }

    std::vector<unsigned int> ref_steps = {5, 3, 5, 5, 6, 5, 4, 4, 5, 3};
    EXPECT_VEC_EQ(ref_steps, steps);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
