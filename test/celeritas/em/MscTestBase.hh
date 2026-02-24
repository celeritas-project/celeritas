//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/MscTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <random>

#include "corecel/data/StateDataStore.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "celeritas/RootTestBase.hh"
#include "celeritas/geo/CoreGeoTrackView.hh"
#include "celeritas/geo/GeoData.hh"
#include "celeritas/grid/RangeCalculator.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleTrackView.hh"
#include "celeritas/phys/PhysicsTrackView.hh"
#include "celeritas/track/SimData.hh"
#include "celeritas/track/SimTrackView.hh"

#include "Test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness base class for multiple scattering models.
 */
class MscTestBase : public RootTestBase
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using MevEnergy = units::MevEnergy;
    //!@}

  public:
    //!@{
    //! Initialize and destroy
    MscTestBase();
    virtual ~MscTestBase();
    //!@}

    std::string_view gdml_basename() const final { return "four-steel-slabs"; }

    SPConstTrackInit build_init() override { CELER_ASSERT_UNREACHABLE(); }
    SPConstAction build_along_step() override { CELER_ASSERT_UNREACHABLE(); }

    //! Get random number generator with clean counter
    RandomEngine& rng()
    {
        rng_.reset_count();
        return rng_;
    }

    // Access particle track data
    ParticleTrackView make_par_view(PDGNumber pdg, MevEnergy energy) const;

    // Access physics track data
    PhysicsTrackView
    make_phys_view(ParticleTrackView const& par,
                   std::string const& matname,
                   HostCRef<PhysicsParamsData> const& host_ref) const;

    // Access geometry track data
    GeoTrackView make_geo_view(real_type r) const;

  private:
    template<template<Ownership, MemSpace> class S>
    using StateStore = StateDataStore<S, MemSpace::host>;

    StateStore<PhysicsStateData> physics_state_;
    StateStore<ParticleStateData> particle_state_;
    StateStore<GeoStateData> geo_state_;
    StateStore<SimStateData> sim_state_;
    RandomEngine rng_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
