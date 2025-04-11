//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/AlongStepFactory.cc
//---------------------------------------------------------------------------//
#include "AlongStepFactory.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>

#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/QuantityIO.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/alongstep/AlongStepCylMapFieldMscAction.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/alongstep/AlongStepRZMapFieldMscAction.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/ext/GeantUnits.hh"
#include "celeritas/ext/GeantVolumeMapper.hh"
#include "celeritas/field/CylMapFieldInput.hh"
#include "celeritas/field/RZMapFieldInput.hh"
#include "celeritas/field/UniformFieldData.hh"
#include "celeritas/io/ImportData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a function to return the field strength.
 *
 * The function is evaluated whenever Celeritas is set up (which is after
 * Geant4 physics is initialized).
 */
UniformAlongStepFactory::UniformAlongStepFactory(FieldFunction f)
    : get_field_(std::move(f))
{
    CELER_EXPECT(get_field_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with field strength and the volumes where field is present.
 */
UniformAlongStepFactory::UniformAlongStepFactory(FieldFunction f,
                                                 VecVolumeFunction volumes)
    : get_field_(std::move(f)), get_volumes_(std::move(volumes))
{
    CELER_EXPECT(get_field_);
    CELER_EXPECT(get_volumes_);
}

//---------------------------------------------------------------------------//
/*!
 * Emit an along-step action.
 *
 * The action will embed the linear propagator if the magnetic field strength
 * is zero (or the accessor is unset).
 */
auto UniformAlongStepFactory::operator()(
    AlongStepFactoryInput const& input) const -> result_type
{
    // Get the field strength in tesla (or zero if accessor is undefined)
    auto field = this->get_field();
    auto magnitude = norm(field.strength);

    // Get the volumes where field is present
    auto volumes = this->get_volumes();

    if (magnitude > 0)
    {
        // Get the IDs of the volumes with field
        if (!volumes.empty())
        {
            field.volumes.reserve(volumes.size());
            GeantVolumeMapper find_volume(*input.geometry);
            for (auto const* lv : volumes)
            {
                CELER_ASSERT(lv);
                auto vol = find_volume(*lv);
                CELER_VALIDATE(
                    vol,
                    << "failed to find volume corresponding to Geant4 volume "
                    << lv->GetName() << " while setting up uniform field");
                CELER_LOG(debug) << "Found volume " << lv->GetName()
                                 << " that will be assigned a uniform field";
                field.volumes.push_back(vol);
            }
        }

        // Create a uniform field
        CELER_LOG(info)
            << "Creating along-step action with field strength " << magnitude
            << " T in "
            << (field.volumes.empty() ? "all" : std::to_string(volumes.size()))
            << " volumes";

        return celeritas::AlongStepUniformMscAction::from_params(
            input.action_id,
            *input.geometry,
            *input.material,
            *input.particle,
            field,
            celeritas::UrbanMscParams::from_import(
                *input.particle, *input.material, *input.imported),
            input.imported->em_params.energy_loss_fluct);
    }
    else
    {
        CELER_LOG(info) << "Creating along-step action with no field";
        return celeritas::AlongStepGeneralLinearAction::from_params(
            input.action_id,
            *input.material,
            *input.particle,
            celeritas::UrbanMscParams::from_import(
                *input.particle, *input.material, *input.imported),
            input.imported->em_params.energy_loss_fluct);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get the field params (used for converting to celeritas::inp).
 */
auto UniformAlongStepFactory::get_field() const -> FieldInput
{
    return get_field_ ? get_field_() : FieldInput{};
}

//---------------------------------------------------------------------------//
/*!
 * Get the volumes where field is present
 */
auto UniformAlongStepFactory::get_volumes() const -> VecVolume
{
    return get_volumes_ ? get_volumes_() : VecVolume{};
}

//---------------------------------------------------------------------------//
/*!
 * Emit an along-step action with a non-uniform magnetic field.
 *
 * The action will embed the field propagator with a RZMapField.
 */
RZMapFieldAlongStepFactory::RZMapFieldAlongStepFactory(RZMapFieldFunction f)
    : get_fieldmap_(std::move(f))
{
    CELER_EXPECT(get_fieldmap_);
}

//---------------------------------------------------------------------------//
/*!
 * Emit an along-step action.
 */
auto RZMapFieldAlongStepFactory::operator()(
    AlongStepFactoryInput const& input) const -> result_type
{
    CELER_LOG(info) << "Creating along-step action with a RZMapField";

    return celeritas::AlongStepRZMapFieldMscAction::from_params(
        input.action_id,
        *input.material,
        *input.particle,
        get_fieldmap_(),
        celeritas::UrbanMscParams::from_import(
            *input.particle, *input.material, *input.imported),
        input.imported->em_params.energy_loss_fluct);
}

//---------------------------------------------------------------------------//
/*!
 * Get the field params (used for converting to celeritas::inp).
 */
RZMapFieldInput RZMapFieldAlongStepFactory::get_field() const
{
    return this->get_fieldmap_();
}

//---------------------------------------------------------------------------//
/*!
 * Emit an along-step action with a non-uniform magnetic field.
 *
 * The action will embed the field propagator with a CylMapField.
 */
CylMapFieldAlongStepFactory::CylMapFieldAlongStepFactory(CylMapFieldFunction f)
    : get_fieldmap_(std::move(f))
{
    CELER_EXPECT(get_fieldmap_);
}

//---------------------------------------------------------------------------//
/*!
 * Emit an along-step action.
 */
auto CylMapFieldAlongStepFactory::operator()(
    AlongStepFactoryInput const& input) const -> result_type
{
    CELER_LOG(info) << "Creating along-step action with a CylMapField";

    return celeritas::AlongStepCylMapFieldMscAction::from_params(
        input.action_id,
        *input.material,
        *input.particle,
        get_fieldmap_(),
        celeritas::UrbanMscParams::from_import(
            *input.particle, *input.material, *input.imported),
        input.imported->em_params.energy_loss_fluct);
}

//---------------------------------------------------------------------------//
/*!
 * Get the field params (used for converting to celeritas::inp).
 */
CylMapFieldInput CylMapFieldAlongStepFactory::get_field() const
{
    return this->get_fieldmap_();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
