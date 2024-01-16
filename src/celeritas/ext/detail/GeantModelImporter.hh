//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantModelImporter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/Types.hh"
#include "celeritas/io/ImportMaterial.hh"
#include "celeritas/io/ImportModel.hh"
#include "celeritas/phys/PDGNumber.hh"

class G4VEmModel;
class G4Material;
class G4ParticleDefinition;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Convert models for a single process.
 */
class GeantModelImporter
{
  public:
    //!@{
    //! \name Type aliases
    using VecMaterial = std::vector<ImportMaterial>;
    //!@}

  public:
    // Construct with materials, primary, and secondary
    GeantModelImporter(VecMaterial const& materials,
                       PDGNumber particle,
                       PDGNumber secondary);

    ImportModel operator()(G4VEmModel const& model) const;

  private:
    //// DATA ////
    VecMaterial const& materials_;
    PDGNumber particle_{};
    PDGNumber secondary_{};
    G4ParticleDefinition const* g4particle_{nullptr};

    //// FUNCTIONS ////
    double get_cutoff(size_type mat_idx) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
