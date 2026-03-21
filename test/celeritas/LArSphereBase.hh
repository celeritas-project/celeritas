//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/LArSphereBase.hh
//---------------------------------------------------------------------------//
#pragma once

#include "celeritas/ext/GeantImporter.hh"
#include "celeritas/optical/CoreParams.hh"
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
    std::string_view gdml_basename() const override { return "lar-sphere"; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto result = GeantTestBase::build_geant_options();
        result.optical.emplace();
        result.optical->mie_scattering = false;
        // Disable Rayleigh model due to PR #2038
        result.optical->rayleigh_scattering = false;
        result.optical->wavelength_shifting = std::nullopt;
        result.optical->wavelength_shifting2 = std::nullopt;
        return result;
    }

    GeantImportDataSelection build_import_data_selection() const override
    {
        auto result = GeantTestBase::build_import_data_selection();
        result.processes |= GeantImportDataSelection::optical;
        return result;
    }
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
