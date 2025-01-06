//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/AlongStepFactory.cc
//---------------------------------------------------------------------------//
#include "AlongStepFactory.hh"

#include <CLHEP/Units/SystemOfUnits.h>

#include "corecel/io/Logger.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/math/QuantityIO.hh"
#include "geocel/g4/Convert.hh"
#include "celeritas/alongstep/AlongStepGeneralLinearAction.hh"
#include "celeritas/alongstep/AlongStepRZMapFieldMscAction.hh"
#include "celeritas/alongstep/AlongStepUniformMscAction.hh"
#include "celeritas/em/params/UrbanMscParams.hh"
#include "celeritas/ext/GeantUnits.hh"
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
 * Emit an along-step action.
 *
 * The action will embed the linear propagator if the magnetic field strength
 * is zero (or the accessor is unset).
 */
auto UniformAlongStepFactory::operator()(
    AlongStepFactoryInput const& input) const -> result_type
{
    // Get the field strength in tesla (or zero if accessor is undefined)
    auto field_params = this->get_field();
    auto magnitude
        = native_value_to<units::FieldTesla>(norm(field_params.field));

    if (magnitude > zero_quantity())
    {
        // Create a uniform field
        CELER_LOG(info) << "Creating along-step action with field strength "
                        << magnitude;
        return celeritas::AlongStepUniformMscAction::from_params(
            input.action_id,
            *input.material,
            *input.particle,
            field_params,
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
UniformFieldParams UniformAlongStepFactory::get_field() const
{
    return get_field_ ? get_field_() : UniformFieldParams{};
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
}  // namespace celeritas
