//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/ImportModelConverter.hh
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
class ImportModelConverter
{
  public:
    //!@{
    //! \name Type aliases
    using VecMaterial = std::vector<ImportMaterial>;
    //!@}

  public:
    // Construct with materials, primary, and secondary
    ImportModelConverter(VecMaterial const& materials,
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
    void calc_micro_xs(G4VEmModel& model,
                       G4Material const& g4mat,
                       double secondary_cutoff,
                       ImportModelMaterial* result) const;
    double get_cutoff(size_type mat_idx) const;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
