//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/track/SimData.hh"

#include "DiagnosticRngEngine.hh"
#include "Test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// FWD declarations
class ParticleParams;
class ParticleTrackView;
class SimParams;
class SimTrackView;
class PDGNumber;

//---------------------------------------------------------------------------//
/*!
 * Test harness base class for optical physics.
 *
 * Constructs particle params, particle track views, and add some
 * functionality for multiple tests.
 *
 * May be expanded to encompass material data if needed.
 */
namespace test
{
class OpticalTestBase : public Test
{
  public:
    //!@{
    //! Initialize and destroy
    OpticalTestBase();
    ~OpticalTestBase();
    //!@}

    //! Initialize particle state data with given energy
    ParticleTrackView
    make_particle_track_view(units::MevEnergy energy, PDGNumber pdg);

    //! Initialize sim track state
    SimTrackView make_sim_track_view(real_type step_len);

    //! Get particle params data
    std::shared_ptr<ParticleParams> const& particle_params() const
    {
        return particle_params_;
    }

    //! Get SimTrackView
    std::shared_ptr<SimParams> const& sim_params() const
    {
        return sim_params_;
    }

  private:
    std::shared_ptr<ParticleParams> particle_params_;
    std::shared_ptr<SimParams> sim_params_;
    HostVal<ParticleStateData> p_state_val_;
    HostRef<ParticleStateData> p_state_ref_;
    HostVal<SimStateData> sim_state_val_;
    HostRef<SimStateData> sim_state_ref_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
