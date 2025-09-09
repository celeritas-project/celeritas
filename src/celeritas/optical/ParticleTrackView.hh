//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ParticleTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/math/ArraySoftUnit.hh"
#include "geocel/Types.hh"
#include "celeritas/Quantities.hh"

#include "ParticleData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Properties of a single optical photon.
 */
class ParticleTrackView
{
  public:
    //!@{
    //! \name Type aliases
    using Energy = units::MevEnergy;
    //!@}

    //! Data for initializing a particle track
    struct Initializer
    {
        Energy energy;
        Real3 polarization{0, 0, 0};
    };

  public:
    inline CELER_FUNCTION
    ParticleTrackView(NativeRef<ParticleStateData> const&, TrackSlotId);

    // Initialize the particle
    inline CELER_FUNCTION ParticleTrackView& operator=(Initializer const&);

    // Access the kinetic energy [MeV]
    CELER_FORCEINLINE_FUNCTION Energy energy() const;

    // Access the polarization
    CELER_FORCEINLINE_FUNCTION Real3 const& polarization() const;

    // Change the photon's energy [MeV]
    inline CELER_FUNCTION void energy(Energy);

    // Change the photon's polarization
    inline CELER_FUNCTION void polarization(Real3 const&);

  private:
    NativeRef<ParticleStateData> const& states_;
    TrackSlotId track_slot_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct from dynamic particle properties.
 */
CELER_FUNCTION
ParticleTrackView::ParticleTrackView(NativeRef<ParticleStateData> const& states,
                                     TrackSlotId tid)
    : states_(states), track_slot_(tid)
{
    CELER_EXPECT(track_slot_ < states_.size());
}

//---------------------------------------------------------------------------//
/*!
 * Initialize the particle.
 */
CELER_FUNCTION ParticleTrackView&
ParticleTrackView::operator=(Initializer const& init)
{
    CELER_EXPECT(init.energy >= zero_quantity());
    CELER_EXPECT(is_soft_unit_vector(init.polarization));

    states_.energy[track_slot_] = value_as<Energy>(init.energy);
    states_.polarization[track_slot_] = init.polarization;
    return *this;
}

//---------------------------------------------------------------------------//
/*!
 * Access the kinetic energy [MeV].
 */
CELER_FUNCTION auto ParticleTrackView::energy() const -> Energy
{
    return Energy{states_.energy[track_slot_]};
}

//---------------------------------------------------------------------------//
/*!
 * Access the polarization.
 */
CELER_FUNCTION Real3 const& ParticleTrackView::polarization() const
{
    return states_.polarization[track_slot_];
}

//---------------------------------------------------------------------------//
/*!
 * Change the particle's kinetic energy.
 */
CELER_FUNCTION
void ParticleTrackView::energy(Energy energy)
{
    CELER_EXPECT(energy >= zero_quantity());
    states_.energy[track_slot_] = value_as<Energy>(energy);
}

//---------------------------------------------------------------------------//
/*!
 * Change the particle's polarization.
 */
CELER_FUNCTION
void ParticleTrackView::polarization(Real3 const& polarization)
{
    CELER_EXPECT(is_soft_unit_vector(polarization));
    states_.polarization[track_slot_] = polarization;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
