//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/io/ImportOpticalProcess.hh"

#include "OpticalModel.hh"

namespace celeritas
{
class MaterialParams;
//---------------------------------------------------------------------------//
/*!
 * Optical absorption model.
 *
 * Models absorption of optical photons in a material based on the materials
 * absorption (attenuation) length. The absorption length determines the
 * step length, and calling the interaction simply kills the track and
 * deposits the energy.
 */
class AbsorptionModel : public OpticalModel
{
  public:
    //! Construct an absorption model with the action id.
    AbsorptionModel(ActionId id, MaterialParams const&);

    //! Execute the absorption interaction on the host
    void execute(OpticalParams const&, OpticalStateHost&) const override final;

    //! Execute the absorption interaction on the device
    void
    execute(OpticalParams const&, OpticalStateDevice&) const override final;

    //! Label of the absorption model.
    std::string_view label() const override final
    {
        return "optical-absorption";
    }

    //! Description of the absorption model.
    std::string_view description() const override final
    {
        return "optical photon absorbed by material";
    }

    //! Label of the absorption process.
    static std::string_view process_label() { return "Absorption"; }

    //! IPC of the absorption process.
    constexpr static ImportOpticalProcessClass process_class()
    {
        return ImportOpticalProcessClass::absorption;
    }

    // No data used by absorption
};
//---------------------------------------------------------------------------//
}  // namespace celeritas
