//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantMicroXsCalculator.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "celeritas/Quantities.hh"
#include "celeritas/io/ImportModel.hh"

class G4Element;
class G4Material;
class G4ParticleDefinition;
class G4VEmModel;

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Calculate microscopic cross sections for an EM model.
 *
 * The input energy is in MeV, and the output units are in Celeritas native
 * area units.
 */
class GeantMicroXsCalculator
{
  public:
    //!@{
    //! \name Type aliases
    using EnergyUnits = units::Mev;
    using XsUnits = units::Native;  // len^2
    using VecDouble = std::vector<double>;
    using VecGrid = ImportModelMaterial::VecGrid;
    //!@}

  public:
    GeantMicroXsCalculator(G4VEmModel const& model,
                           G4ParticleDefinition const& particle,
                           G4Material const& material,
                           double secondary_production_cut);

    // Calculate micro cross sections for all elements in the material
    void operator()(VecDouble const& energy, VecGrid* xs) const;

  private:
    G4VEmModel& model_;
    G4ParticleDefinition const& particle_;
    G4Material const& material_;
    double secondary_cut_;
};

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
