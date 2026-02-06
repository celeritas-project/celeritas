//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/FieldTestBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include <cmath>

#include "corecel/data/StateDataStore.hh"
#include "corecel/math/ArrayUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleData.hh"
#include "celeritas/phys/ParticleTrackView.hh"

#include "Test.hh"

namespace celeritas
{
class ParticleParams;
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Base class for testing fields.
 */
class FieldTestBase : virtual public Test
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using MevEnergy = units::MevEnergy;
    //!@}

  public:
    // Access particle data
    SPConstParticle const& particle();
    inline SPConstParticle const& particle() const;

    // Create a particle track
    ParticleTrackView make_particle_view(PDGNumber pdg, MevEnergy energy);

    // Calculate field radius
    template<class GTV, class Field>
    inline real_type calc_field_curvature(ParticleTrackView const& particle,
                                          GTV const& geo,
                                          Field const& calc_field) const;

  protected:
    // Build particles
    [[nodiscard]] virtual SPConstParticle build_particle() const = 0;

  private:
    //// TYPE ALIASES ////
    using ParStateStore = StateDataStore<ParticleStateData, MemSpace::host>;

    //// DATA ////

    SPConstParticle particle_;
    ParStateStore par_state_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
//! Access particle params (if already created)
auto FieldTestBase::particle() const -> SPConstParticle const&
{
    CELER_EXPECT(particle_);
    return particle_;
}

//! Calculate the radius of the field curvature for the particle/geometry
template<class GTV, class Field>
real_type FieldTestBase::calc_field_curvature(ParticleTrackView const& particle,
                                              GTV const& geo,
                                              Field const& calc_field) const
{
    auto field_strength = norm(calc_field(geo.pos()));
    return native_value_from(particle.momentum())
           / (std::fabs(native_value_from(particle.charge())) * field_strength);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
