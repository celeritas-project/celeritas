//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantModelImporter.cc
//---------------------------------------------------------------------------//
#include "GeantModelImporter.hh"

#include <unordered_map>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4Element.hh>
#include <G4EmParameters.hh>
#include <G4Material.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ParticleDefinition.hh>
#include <G4ParticleTable.hh>
#include <G4ProductionCutsTable.hh>
#include <G4VEmModel.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "celeritas/grid/VectorUtils.hh"

using CLHEP::cm2;
using CLHEP::MeV;

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Safely retrieve the correct model enum from a given string.
 */
ImportModelClass to_import_model(G4VEmModel const& model)
{
    static const std::unordered_map<std::string, ImportModelClass> model_map = {
        // clang-format off
        {"BraggIon",            ImportModelClass::bragg_ion},
        {"BetheBloch",          ImportModelClass::bethe_bloch},
        {"UrbanMsc",            ImportModelClass::urban_msc},
        {"ICRU73QO",            ImportModelClass::icru_73_qo},
        {"WentzelVIUni",        ImportModelClass::wentzel_VI_uni},
        {"hBrem",               ImportModelClass::h_brems},
        {"hPairProd",           ImportModelClass::h_pair_prod},
        {"eCoulombScattering",  ImportModelClass::e_coulomb_scattering},
        {"Bragg",               ImportModelClass::bragg},
        {"MollerBhabha",        ImportModelClass::moller_bhabha},
        {"eBremSB",             ImportModelClass::e_brems_sb},
        {"eBremLPM",            ImportModelClass::e_brems_lpm},
        {"eplus2gg",            ImportModelClass::e_plus_to_gg},
        {"LivermorePhElectric", ImportModelClass::livermore_photoelectric},
        {"Klein-Nishina",       ImportModelClass::klein_nishina},
        {"BetheHeitler",        ImportModelClass::bethe_heitler},
        {"BetheHeitlerLPM",     ImportModelClass::bethe_heitler_lpm},
        {"LivermoreRayleigh",   ImportModelClass::livermore_rayleigh},
        {"MuBetheBloch",        ImportModelClass::mu_bethe_bloch},
        {"MuBrem",              ImportModelClass::mu_brems},
        {"muPairProd",          ImportModelClass::mu_pair_prod},
        // clang-format on
    };
    std::string const& name = model.GetName();
    auto iter = model_map.find(name);
    if (iter == model_map.end())
    {
        static celeritas::TypeDemangler<G4VEmModel> demangle_model;
        CELER_LOG(warning) << "Encountered unknown model '" << name
                           << "' (RTTI: " << demangle_model(model) << ")";
        return ImportModelClass::other;
    }
    return iter->second;
}

//---------------------------------------------------------------------------//
/*!
 * Get a G4Material from a material index.
 */
G4Material const& get_g4material(unsigned int mat_idx)
{
    auto const* g4_cuts_table = G4ProductionCutsTable::GetProductionCutsTable();
    CELER_EXPECT(mat_idx < g4_cuts_table->GetTableSize());
    auto const* g4_material
        = g4_cuts_table->GetMaterialCutsCouple(mat_idx)->GetMaterial();
    CELER_ENSURE(g4_material);
    return *g4_material;
}

//---------------------------------------------------------------------------//
/*!
 * Get a G4ParticleDefinition reference from a particle identifier.
 */
G4ParticleDefinition const& get_g4particle(PDGNumber pdg)
{
    CELER_EXPECT(pdg);
    auto* particle
        = G4ParticleTable::GetParticleTable()->FindParticle(pdg.get());
    CELER_ENSURE(particle);
    return *particle;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Construct with materials, primary, and secondary.
 */
GeantModelImporter::GeantModelImporter(VecMaterial const& materials,
                                       PDGNumber particle,
                                       PDGNumber secondary)
    : materials_(materials), particle_(particle), secondary_(secondary)
{
    CELER_EXPECT(particle_);
    g4particle_ = &get_g4particle(particle);
    CELER_ENSURE(g4particle_);
}

//---------------------------------------------------------------------------//
/*!
 * Convert a model.
 *
 * - Model energy limits are set based on model, particle, and cutoff values
 * - If the model class requires elemental cross sections, a reduced-resolution
 *   energy grid is calculated and cross sections are evaluated for those grid
 *   points.
 */
ImportModel GeantModelImporter::operator()(G4VEmModel const& model) const
{
    ImportModel result;
    result.model_class = to_import_model(model);

    // Calculate lower cutoff energy for the model in each material
    result.materials.resize(materials_.size());

    for (auto mat_idx : celeritas::range(materials_.size()))
    {
        G4Material const& g4mat = get_g4material(mat_idx);

        // Calculate lower and upper energy bounds
        double min_energy
            = std::max(model.LowEnergyLimit(),
                       const_cast<G4VEmModel&>(model).MinPrimaryEnergy(
                           &g4mat, g4particle_, this->get_cutoff(mat_idx)))
              / MeV;
        double max_energy = model.HighEnergyLimit() / MeV;
        CELER_ASSERT(0 <= min_energy);
        CELER_ASSERT(min_energy < max_energy);

        auto& model_mat = result.materials[mat_idx];
        if (needs_micro_xs(result.model_class))
        {
            // Calculate microscopic cross section grid with a reduced number
            // of bins compared to to regular cross sections
            // (See G4VEmModel::InitialiseElementSelectors)
            static double const bins_per_decade
                = G4EmParameters::Instance()->NumberOfBinsPerDecade() / 6.0;
            double const num_bins = std::round(
                bins_per_decade * std::log10(max_energy / min_energy));

            // Interpolate energy in log space with at least 3 bins
            model_mat.energy = logspace(
                min_energy, max_energy, std::max<size_type>(3, num_bins));

            this->calc_micro_xs(const_cast<G4VEmModel&>(model),
                                g4mat,
                                this->get_cutoff(mat_idx),
                                &model_mat);
        }
        else
        {
            // Just save the energy bounds for the model
            model_mat.energy = {min_energy, max_energy};
        }
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Fill microscopic cross sections for elements in a material.
 *
 * \c G4VEmModel::InitialiseElementSelectors reduces the number of grid points
 * by a factor of 6 compared to the regular cross section (lambda) grid.
 */
void GeantModelImporter::calc_micro_xs(G4VEmModel& model,
                                       G4Material const& g4mat,
                                       double secondary_cutoff,
                                       ImportModelMaterial* result) const
{
    CELER_EXPECT(g4mat.GetElementVector());
    CELER_EXPECT(secondary_cutoff >= 0);
    CELER_EXPECT(result && result->energy.size() >= 2);

    std::vector<G4Element const*> const& elements = *g4mat.GetElementVector();
    std::vector<double> const& energy_grid = result->energy;

    // Resize microscopic cross sections for all elements
    result->micro_xs.resize(elements.size());
    for (auto& mxs_vec : result->micro_xs)
    {
        mxs_vec.resize(energy_grid.size());
    }

    // Outer loop over energy to reduce material setup calls
    for (auto energy_idx : range(energy_grid.size()))
    {
        double const energy = energy_grid[energy_idx];
        model.SetupForMaterial(g4particle_, &g4mat, energy);

        // Inner loop over elements
        for (auto elcomp_idx : range(elements.size()))
        {
            G4Element const& g4el = *elements[elcomp_idx];

            // Calculate microscopic cross-section
            double xs = model.ComputeCrossSectionPerAtom(
                g4particle_, &g4el, energy, secondary_cutoff, energy);

            // Convert to celeritas units and clamp to zero
            result->micro_xs[elcomp_idx][energy_idx] = std::max(0.0, xs / cm2);
        }
    }

    for (std::vector<double>& xs : result->micro_xs)
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
 * Get the energy cutoff for secondary production (in ImportMaterial units!).
 */
double GeantModelImporter::get_cutoff(size_type mat_idx) const
{
    CELER_EXPECT(mat_idx < materials_.size());
    if (!secondary_)
    {
        return 0;
    }

    auto const& cutoffs = materials_[mat_idx].pdg_cutoffs;
    auto iter = cutoffs.find(secondary_.get());
    if (iter == cutoffs.end())
    {
        // Particle unavailable: use infinite cutoff
        return std::numeric_limits<double>::infinity();
    }
    return iter->second.energy;
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
