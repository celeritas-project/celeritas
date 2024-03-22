//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OpticalModel.hh"
#include "OpticalProcess.hh"

namespace celeritas
{
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
    void execute(CoreParams const&, CoreStateHost&) const override final;

    //! Execute the absorption interaction on the device
    void execute(CoreParams const&, CoreStateDevice&) const override final;

    //! Label of the absorption model.
    std::string label() const override final
    {
        return "optical-absorption";
    }

    //! Description of the absorption model.
    std::string description() const override final
    {
        return "optical photon absorbed by material";
    }

    //! Label of the absorption process.
    static std::string process_label()
    {
        return "Absorption";
    }

    //! IPC of the absorption process.
    constexpr static ImportProcessClass process_class()
    {
        return ImportProcessClass::absorption;
    }

    // No data used by absorption
};

// Define the corresponding absorption process.
using AbsorptionProcess = OpticalProcessInstance<AbsorptionModel>;

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
AbsorptionModel::AbsorptionModel(ActionId id, MaterialParams const&)
    : OpticalModel(id)
{}
//---------------------------------------------------------------------------//
}  // namespace celeritas
