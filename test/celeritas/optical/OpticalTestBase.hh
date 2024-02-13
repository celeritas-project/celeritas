//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>

#include "celeritas/Quantities.hh"
#include "celeritas/Types.hh"
#include "celeritas/phys/ParticleData.hh"

#include "DiagnosticRngEngine.hh"
#include "celeritas_test.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
// FWD declarations
class ParticleParams;
class ParticleTrackView;
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
    //! \name Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    //!@}

    //!@{
    //! Initialize and destroy
    OpticalTestBase();
    ~OpticalTestBase();
    //!@}

    //! Initialize particle state data with given energy
    ParticleTrackView
    make_particle_track_view(units::MevEnergy energy, PDGNumber pdg);

    //! Get particle params data
    std::shared_ptr<ParticleParams> particle_params()
    {
        return particle_params_;
    }

    //! Get random number generator with clean counter
    RandomEngine& reset_rng()
    {
        rng_.reset_count();
        return rng_;
    }

  private:
    std::shared_ptr<ParticleParams> particle_params_;
    HostVal<ParticleStateData> state_val_;
    HostRef<ParticleStateData> state_ref_;
    RandomEngine rng_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
