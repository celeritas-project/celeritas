//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantOpticalModelImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportOpticalModel.hh"

class G4VProcess;
class G4MaterialPropertiesTable;

namespace celeritas
{
struct ImportPhysMaterial;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Imports optical model MFP tables from Geant4 material property tables.
 */
class GeantOpticalModelImporter
{
  public:
    //!@{
    //! \name Type aliases
    using IMC = optical::ImportModelClass;
    //!@}

  public:
    // Construct model importer with given optical material mapping
    GeantOpticalModelImporter(std::vector<ImportPhysMaterial> const& materials);

    // Import model MFP table for given model class
    ImportOpticalModel operator()(IMC imc) const;

  private:
    // Import MFP table for the given property name
    std::vector<ImportPhysicsVector>
    import_mfps(std::string const& mfp_property_name) const;

    std::vector<G4MaterialPropertiesTable const*> opt_to_mat_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
