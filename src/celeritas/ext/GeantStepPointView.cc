//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepPointView.cc
//---------------------------------------------------------------------------//
#include "GeantStepPointView.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Update attributes from logical volume.
 */
void GeantStepPointView::update_from_volume(G4LogicalVolume const& lv)
{
    sp_.SetMaterial(lv.GetMaterial());
    sp_.SetMaterialCutsCouple(lv.GetMaterialCutsCouple());
    sp_.SetSensitiveDetector(lv.GetSensitiveDetector());
}

//---------------------------------------------------------------------------//
/*!
 * Update attributes from the touchable's logical volume if possible.
 *
 * If the step point has an associated touchable, and that touchable is inside
 * the geometry, it updates. Otherwise, it clears the corresponding attributes.
 */
void GeantStepPointView::update_from_volume()
{
    G4LogicalVolume const* lv = nullptr;
    if (auto* touch = sp_.GetTouchable())
    {
        // The physical volume could be null if post-step is outside
        if (auto* pv = touch->GetVolume())
        {
            lv = pv->GetLogicalVolume();
        }
    }
    if (lv)
    {
        this->update_from_volume(*lv);
    }
    else
    {
        sp_.SetMaterial(nullptr);
        sp_.SetMaterialCutsCouple(nullptr);
        sp_.SetSensitiveDetector(nullptr);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Update mass and charge from particle definition.
 */
void GeantStepPointView::update_from_particle(GeantParticleView const& particle)
{
    sp_.SetMass(particle.mass().value() * CLHEP::MeV);
    sp_.SetCharge(particle.charge().value() * CLHEP::eplus);
}

//---------------------------------------------------------------------------//
/*!
 * Clear unsupported attributes to invalid sentinel values.
 *
 * This sets attributes that Celeritas does not currently track to sentinel
 * values to indicate they are unavailable to sensitive detectors.
 */
void GeantStepPointView::clear_unsupported()
{
    // Time since track was created
    sp_.SetLocalTime(std::numeric_limits<double>::infinity());
    // Time in rest frame since track was created
    sp_.SetProperTime(std::numeric_limits<double>::infinity());
    // Speed (TODO: use ParticleView)
    sp_.SetVelocity(std::numeric_limits<double>::infinity());
    // Safety distance
    sp_.SetSafety(std::numeric_limits<double>::infinity());
    // Polarization (default to zero)
    sp_.SetPolarization(G4ThreeVector());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
