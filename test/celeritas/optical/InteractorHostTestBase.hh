//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/InteractorHostTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>

#include "corecel/cont/Span.hh"
#include "corecel/data/CollectionStateStore.hh"
#include "corecel/data/StackAllocator.hh"
#include "corecel/random/DiagnosticRngEngine.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/ParticleData.hh"
#include "celeritas/optical/ParticleTrackView.hh"
#include "celeritas/optical/SimTrackView.hh"

#include "Test.hh"

namespace celeritas
{
namespace optical
{
namespace test
{
using namespace celeritas::test;

//---------------------------------------------------------------------------//
/*!
 * A test base for optical interactions.
 *
 * Manages the direction and track view of an incident photon, and provides
 * access to a diagnostic RNG engine.
 */
class InteractorHostBase
{
  public:
    //!@{
    //! \name Type aliases
    using RandomEngine = DiagnosticRngEngine<std::mt19937>;
    using Energy = units::MevEnergy;
    using Action = Interaction::Action;
    //!@}

  public:
    //! Initialize test base
    InteractorHostBase();

    //! Clean up test base
    virtual ~InteractorHostBase() = default;

    //! Get random number generator with clean counter
    RandomEngine& rng();

    //! Set incident photon direction
    void set_inc_direction(Real3 const& dir);

    //! Set incident photon energy
    void set_inc_energy(Energy energy);

    //! Set incident photon polarization
    void set_inc_polarization(Real3 const& pol);

    //! Get incident photon direction
    Real3 const& direction() const;

    //! Get incident photon track view
    ParticleTrackView const& particle_track() const;

    //! Get simulation track view
    SimTrackView const& sim_track() const;

    //!@{
    //! Check direction and polarizations are physical
    void check_direction_polarization(Real3 const& dir, Real3 const& pol) const;
    void check_direction_polarization(Interaction const& interaction) const;
    //!@}

  private:
    template<template<Ownership, MemSpace> class S>
    using StateStore = CollectionStateStore<S, MemSpace::host>;

    StateStore<ParticleStateData> ps_;
    StateStore<SimStateData> ss_;

    RandomEngine rng_;
    Real3 inc_direction_;
    std::shared_ptr<ParticleTrackView> pt_view_;
    std::shared_ptr<SimTrackView> st_view_;
};

class InteractorHostTestBase : public InteractorHostBase, public Test
{
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
