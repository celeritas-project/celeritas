//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/InteractorHostTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <random>

#include "celeritas/Quantities.hh"
#include "celeritas/optical/Interaction.hh"
#include "celeritas/optical/TrackView.hh"

#include "DiagnosticRngEngine.hh"
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
class InteractorHostTestBase : public Test
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
    InteractorHostTestBase();

    //! Clean up test base
    virtual ~InteractorHostTestBase() = default;

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
    TrackView const& photon_track() const;

    //!@{
    //! Check direction and polarizations are physical
    void check_direction_polarization(Real3 const& dir, Real3 const& pol) const;
    void check_direction_polarization(Interaction const& interaction) const;
    //!@}

  private:
    RandomEngine rng_;
    Real3 inc_direction_;
    std::shared_ptr<TrackView> pt_view_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace optical
}  // namespace celeritas
