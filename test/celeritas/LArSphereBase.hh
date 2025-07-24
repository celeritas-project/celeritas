//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/LArSphereBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/optical/CoreParams.hh"
#include "celeritas/optical/ModelImporter.hh"
#include "celeritas/optical/PhysicsParams.hh"
#include "celeritas/phys/ProcessBuilder.hh"

#include "GeantTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Test harness for liquid argon sphere with optical properties.
 *
 * This class requires Geant4 to import the data. MSC is on by default.
 */
class LArSphereBase : public GeantTestBase
{
  protected:
    std::string_view geometry_basename() const override
    {
        return "lar-sphere";
    }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto result = GeantTestBase::build_geant_options();
        result.optical.absorption = true;
        result.optical.rayleigh_scattering = true;
        result.optical.wavelength_shifting.enable = true;
        result.optical.wavelength_shifting2.enable = true;
        return result;
    }

    GeantImportDataSelection build_import_data_selection() const override
    {
        auto result = GeantTestBase::build_import_data_selection();
        result.processes |= GeantImportDataSelection::optical;
        return result;
    }

    SPConstOpticalPhysics build_optical_physics() override
    {
        using IMC = celeritas::optical::ImportModelClass;

        optical::PhysicsParams::Input input;
        input.materials = this->optical_material();
        input.action_registry = this->optical_action_reg().get();

        optical::ModelImporter importer(
            this->imported_data(), this->optical_material(), this->material());

        for (IMC imc : {IMC::absorption, IMC::rayleigh})
        {
            if (auto builder = importer(imc))
            {
                input.model_builders.push_back(*builder);
            }
        }

        return std::make_shared<optical::PhysicsParams>(std::move(input));
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
