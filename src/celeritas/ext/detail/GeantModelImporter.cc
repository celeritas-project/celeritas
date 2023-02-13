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
#include <G4EmParameters.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ParticleTable.hh>
#include <G4ProductionCutsTable.hh>
#include <G4VEmModel.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "celeritas/grid/VectorUtils.hh"

#include "GeantMicroXsCalculator.hh"

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
        double cutoff = this->get_cutoff(mat_idx);
        double min_energy
            = std::max(model.LowEnergyLimit(),
                       const_cast<G4VEmModel&>(model).MinPrimaryEnergy(
                           &g4mat, g4particle_, cutoff))
              / CLHEP::MeV;
        double max_energy = model.HighEnergyLimit() / CLHEP::MeV;
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

            // Calculate cross sections
            GeantMicroXsCalculator calc_xs(model, *g4particle_, g4mat, cutoff);
            calc_xs(model_mat.energy, &model_mat.micro_xs);
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
