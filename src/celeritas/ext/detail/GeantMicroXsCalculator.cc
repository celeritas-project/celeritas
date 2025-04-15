//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantMicroXsCalculator.cc
//---------------------------------------------------------------------------//
#include "GeantMicroXsCalculator.hh"

#include <G4Material.hh>
#include <G4VEmModel.hh>

#include "corecel/cont/Range.hh"
#include "corecel/math/Algorithms.hh"
#include "celeritas/io/ImportUnits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with Geant4 classes.
 *
 * The secondary production cut should be nonzero if a secondary is produced.
 */
GeantMicroXsCalculator::GeantMicroXsCalculator(
    G4VEmModel const& model,
    G4ParticleDefinition const& particle,
    G4Material const& material,
    double secondary_production_cut)
    : model_{const_cast<G4VEmModel&>(model)}
    , particle_{particle}
    , material_{material}
    , secondary_cut_{secondary_production_cut}
{
    CELER_EXPECT(secondary_cut_ >= 0);
}

//---------------------------------------------------------------------------//
/*!
 * Calculate cross sections for all elements in the material.
 */
void GeantMicroXsCalculator::operator()(VecDouble const& energy_grid,
                                        VecGrid* result_xs) const
{
    CELER_EXPECT(result_xs);

    auto const& elements = *material_.GetElementVector();

    // Resize microscopic cross sections for all elements
    result_xs->resize(elements.size());
    for (auto& grid : *result_xs)
    {
        grid.x = {std::log(energy_grid.front()), std::log(energy_grid.back())};
        grid.y.resize(energy_grid.size());
    }

    auto calc_element_xs
        = [this, &elements](std::size_t elcomp_idx, double energy) {
              // Calculate microscopic cross section
              double xs = model_.ComputeCrossSectionPerAtom(
                  &particle_,
                  elements[elcomp_idx],
                  energy,
                  secondary_cut_,
                  /* max_energy = */ std::numeric_limits<double>::max());
              return clamp_to_nonneg(xs);
          };
    double const xs_scaling = native_value_from_clhep(ImportUnits::len_sq);

    // Outer loop over energy to reduce material setup calls
    for (auto energy_idx : range(energy_grid.size()))
    {
        double const energy = energy_grid[energy_idx];
        CELER_ASSERT(energy > 0);
        model_.SetupForMaterial(&particle_, &material_, energy);

        // Inner loop over elements
        for (auto elcomp_idx : range(elements.size()))
        {
            (*result_xs)[elcomp_idx].y[energy_idx]
                = calc_element_xs(elcomp_idx, energy) * xs_scaling;
        }
    }

    for (auto& grid : *result_xs)
    {
        // Avoid cross-section vectors starting or ending with zero values.
        // Geant4 simply uses the next/previous bin value when the vector's
        // front/back are zero. This probably isn't correct but it replicates
        // Geant4's behavior.
        if (grid.y[0] == 0)
        {
            grid.y[0] = grid.y[1];
        }

        auto const last_idx = grid.y.size() - 1;
        if (grid.y[last_idx] == 0)
        {
            // Cross-section ends with zero, use previous bin value
            grid.y[last_idx] = grid.y[last_idx - 1];
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
