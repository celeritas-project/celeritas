//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/GeantModelImporter.cc
//---------------------------------------------------------------------------//
#include "GeantModelImporter.hh"

#include <algorithm>
#include <initializer_list>
#include <unordered_map>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4EmParameters.hh>
#include <G4MaterialCutsCouple.hh>
#include <G4ParticleTable.hh>
#include <G4ProductionCutsTable.hh>
#include <G4VEmModel.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/cont/Range.hh"
#include "corecel/grid/VectorUtils.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "GeantMicroXsCalculator.hh"

namespace celeritas
{
namespace detail
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Get a G4Material from a *physics material* index.
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
/*!
 * Whether the model requires microscopic xs data for sampling.
 *
 * TODO: this could be an implementation detail of ImportModelConverter
 */
bool needs_micro_xs(ImportModelClass value)
{
    CELER_EXPECT(value < ImportModelClass::size_);

    using ModelBoolArray = EnumArray<ImportModelClass, bool>;

    static ModelBoolArray const needs_xs = [] {
        ModelBoolArray result;
        std::fill(result.begin(), result.end(), false);
        for (ImportModelClass c : {
                 ImportModelClass::e_brems_sb,
                 ImportModelClass::e_brems_lpm,
                 ImportModelClass::mu_brems,
                 ImportModelClass::mu_pair_prod,
                 ImportModelClass::bethe_heitler_lpm,
                 ImportModelClass::livermore_rayleigh,
                 ImportModelClass::e_coulomb_scattering,
             })
        {
            result[c] = true;
        }
        return result;
    }();

    return needs_xs[value];
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
    try
    {
        result.model_class = geant_name_to_import_model_class(model.GetName());
    }
    catch (celeritas::RuntimeError const&)
    {
        CELER_LOG(warning) << "Encountered unknown model '" << model.GetName()
                           << "'";
        result.model_class = ImportModelClass::other;
    }

    // Get the model energy limits
    result.low_energy_limit = model.LowEnergyLimit();
    result.high_energy_limit = model.HighEnergyLimit();

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

        auto& model_mat = result.materials[mat_idx];
        bool export_micros = needs_micro_xs(result.model_class);
        if (CELER_UNLIKELY(!(min_energy < max_energy)))
        {
            // It's possible for the lower energy bound to be higher than the
            // upper energy bound if the production cuts are very high.  Don't
            // build the microscopic cross section grid if this is the case.
            CELER_LOG(debug)
                << "Skipping microscopic cross section grid for model '"
                << model.GetName() << "' in material '" << g4mat.GetName()
                << ": lower energy bound " << min_energy
                << " [MeV] is not less than upper energy bound " << max_energy
                << " [MeV]";
            export_micros = false;
        }
        if (export_micros)
        {
            // Calculate microscopic cross section grid with a reduced number
            // of bins compared to to regular cross sections
            // (See G4VEmModel::InitialiseElementSelectors)
            static double const bins_per_decade
                = G4EmParameters::Instance()->NumberOfBinsPerDecade() / 6.0;
            double const num_bins = std::round(
                bins_per_decade * std::log10(max_energy / min_energy));

            // Interpolate energy in log space with at least 3 bins
            auto energy = logspace(
                min_energy, max_energy, std::max<size_type>(3, num_bins));

            // Calculate cross sections
            GeantMicroXsCalculator calc_xs(model, *g4particle_, g4mat, cutoff);
            calc_xs(energy, &model_mat.micro_xs);
        }
        model_mat.energy = {min_energy, max_energy};
    }

    CELER_ENSURE(result);
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy cutoff for secondary production (in ImportPhysMaterial
 * units!).
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
