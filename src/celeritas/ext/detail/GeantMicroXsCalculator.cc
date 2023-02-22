//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantMicroXsCalculator.cc
//---------------------------------------------------------------------------//
#include "GeantMicroXsCalculator.hh"

#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Material.hh>
#include <G4VEmModel.hh>

#include "corecel/cont/Range.hh"
#include "celeritas/ext/GeantConfig.hh"

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
                                        VecVecDouble* result_xs) const
{
    CELER_EXPECT(result_xs);
#if CELERITAS_G4_V10
    std::vector<G4Element*> const& elements
#else
    std::vector<G4Element const*> const& elements
#endif
        = *material_.GetElementVector();

    // Resize microscopic cross sections for all elements
    result_xs->resize(elements.size());
    for (auto& mxs_vec : *result_xs)
    {
        mxs_vec.resize(energy_grid.size());
    }

    // Outer loop over energy to reduce material setup calls
    for (auto energy_idx : range(energy_grid.size()))
    {
        double const energy = energy_grid[energy_idx];
        model_.SetupForMaterial(&particle_, &material_, energy);

        // Inner loop over elements
        for (auto elcomp_idx : range(elements.size()))
        {
            (*result_xs)[elcomp_idx][energy_idx]
                = this->calc_element_xs(*elements[elcomp_idx], energy);
        }
    }

    for (std::vector<double>& xs : *result_xs)
    {
        // Avoid cross-section vectors starting or ending with zero values.
        // Geant4 simply uses the next/previous bin value when the vector's
        // front/back are zero. This probably isn't correct but it replicates
        // Geant4's behavior.
        if (xs[0] == 0.0)
        {
            xs[0] = xs[1];
        }

        auto const last_idx = xs.size() - 1;
        if (xs[last_idx] == 0.0)
        {
            // Cross-section ends with zero, use previous bin value
            xs[last_idx] = xs[last_idx - 1];
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Calculate after setting up for material and energy.
 */
double GeantMicroXsCalculator::calc_element_xs(G4Element const& el,
                                               double energy) const
{
    CELER_EXPECT(energy >= 0);
    // Calculate microscopic cross-section
    double xs = model_.ComputeCrossSectionPerAtom(
        &particle_,
        &el,
        energy,
        secondary_cut_,
        /* max_energy = */ std::numeric_limits<double>::max());

    // Convert to celeritas units and clamp to zero
    return std::max<double>(0, xs / CLHEP::cm2);
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
