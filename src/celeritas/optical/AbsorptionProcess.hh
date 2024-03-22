//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionProcess.hh
//---------------------------------------------------------------------------//
#pragma once

#include "OpticalProcess.hh"

namespace celeritas
{

class AbsorptionModel : public OpticalModel
{
  public:
    AbsorptionModel(ActionId id, MaterialParams const& materials)
        : OpticalModel(id)
    {
    }

    void execute(CoreParams const&, CoreStateHost&) const override final;

    void execute(CoreParams const&, CoreStateDevice&) const override final;

    std::string label() const override final
    {
        return "optical-absorption";
    }

    std::string description() const override final
    {
        return "optical photon absorbed by material";
    }

    static std::string process_label()
    {
        return "Absorption";
    }

    constexpr static ImportProcessClass process_class()
    {
        return ImportProcessClass::absorption;
    }

    // No data used by absorption
};

using AbsorptionProcess = OpticalProcessInstance<AbsorptionModel>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
