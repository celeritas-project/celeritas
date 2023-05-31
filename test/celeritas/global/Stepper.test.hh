//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/Stepper.test.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"

#include "../TestEm3Base.hh"
#include "StepperTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class TestEm3StepperTestBase : public TestEm3Base, public StepperTestBase
{
  public:
    std::vector<Primary>
    make_primaries_with_energy(PDGNumber particle,
                               size_type count,
                               celeritas::units::MevEnergy energy) const
    {
        Primary p;
        p.particle_id = this->particle()->find(particle);
        CELER_ASSERT(p.particle_id);
        p.energy = energy;
        p.track_id = TrackId{0};
        p.position = {-22, 0, 0};
        p.direction = {1, 0, 0};
        p.time = 0;

        std::vector<Primary> result(count, p);
        for (auto i : range(count))
        {
            result[i].event_id = EventId{i};
        }
        return result;
    }

    // Return electron primaries as default
    std::vector<Primary>
    make_primaries_with_energy(size_type count,
                               celeritas::units::MevEnergy energy) const
    {
        return this->make_primaries_with_energy(pdg::electron(), count, energy);
    }
};

//---------------------------------------------------------------------------//
#define TestEm3NoMsc TEST_IF_CELERITAS_GEANT(TestEm3NoMsc)
class TestEm3NoMsc : public TestEm3StepperTestBase
{
  public:
    //! Make 10GeV electrons along +x
    std::vector<Primary> make_primaries(size_type count) const override
    {
        return this->make_primaries_with_energy(
            count, celeritas::units::MevEnergy{10000});
    }

    size_type max_average_steps() const override
    {
        return 100000;  // 8 primaries -> ~500k steps, be conservative
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = TestEm3Base::build_geant_options();
        opts.msc = MscModelSelection::none;
        return opts;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas