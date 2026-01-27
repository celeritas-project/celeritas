//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantStepPointView.cc
//---------------------------------------------------------------------------//
#include "GeantStepPointView.hh"

#include <G4LogicalVolume.hh>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Update attributes from logical volume.
 */
void GeantStepPointView::update_from_volume(G4LogicalVolume const& lv)
{
    step_point_.SetMaterial(lv.GetMaterial());
    step_point_.SetMaterialCutsCouple(lv.GetMaterialCutsCouple());
    step_point_.SetSensitiveDetector(lv.GetSensitiveDetector());
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
    if (auto* touch = step_point_.GetTouchable())
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
        step_point_.SetMaterial(nullptr);
        step_point_.SetMaterialCutsCouple(nullptr);
        step_point_.SetSensitiveDetector(nullptr);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Update mass and charge from particle definition.
 */
void GeantStepPointView::update_from_particle(GeantParticleView const& particle)
{
    step_point_.SetMass(particle.mass().value() * CLHEP::MeV);
    step_point_.SetCharge(particle.charge().value() * CLHEP::eplus);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
