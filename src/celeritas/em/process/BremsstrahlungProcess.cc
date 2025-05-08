//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/BremsstrahlungProcess.cc
//---------------------------------------------------------------------------//
#include "BremsstrahlungProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/CombinedBremModel.hh"
#include "celeritas/em/model/RelativisticBremModel.hh"
#include "celeritas/em/model/SeltzerBergerModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from host data.
 */
BremsstrahlungProcess::BremsstrahlungProcess(SPConstParticles particles,
                                             SPConstMaterials materials,
                                             SPConstImported process_data,
                                             ReadData load_sb,
                                             Options options)
    : particles_(std::move(particles))
    , materials_(std::move(materials))
    , imported_(process_data,
                particles_,
                ImportProcessClass::e_brems,
                {pdg::electron(), pdg::positron()})
    , load_sb_(std::move(load_sb))
    , options_(options)
{
    CELER_EXPECT(particles_);
    CELER_EXPECT(materials_);
    CELER_EXPECT(load_sb_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto BremsstrahlungProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    using IMC = ImportModelClass;

    if (options_.combined_model)
    {
        return {std::make_shared<CombinedBremModel>(*start_id++,
                                                    *particles_,
                                                    *materials_,
                                                    imported_.processes(),
                                                    load_sb_,
                                                    options_.enable_lpm)};
    }

    VecModel result;
    if (imported_.has_model(pdg::electron(), IMC::e_brems_sb))
    {
        CELER_ASSERT(imported_.has_model(pdg::positron(), IMC::e_brems_sb));
        result.push_back(
            std::make_shared<SeltzerBergerModel>(*start_id++,
                                                 *particles_,
                                                 *materials_,
                                                 imported_.processes(),
                                                 load_sb_));
    }
    if (imported_.has_model(pdg::electron(), IMC::e_brems_lpm))
    {
        CELER_ASSERT(imported_.has_model(pdg::positron(), IMC::e_brems_lpm));
        result.push_back(
            std::make_shared<RelativisticBremModel>(*start_id++,
                                                    *particles_,
                                                    *materials_,
                                                    imported_.processes(),
                                                    options_.enable_lpm));
    }
    CELER_VALIDATE(!result.empty(),
                   << "No models found for bremsstrahlung process");
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto BremsstrahlungProcess::macro_xs(Applicability applic) const -> XsGrid
{
    return imported_.macro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto BremsstrahlungProcess::energy_loss(Applicability applic) const
    -> EnergyLossGrid
{
    return imported_.energy_loss(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view BremsstrahlungProcess::label() const
{
    return "Bremsstrahlung";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
