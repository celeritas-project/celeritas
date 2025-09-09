//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/MuIonizationProcess.cc
//---------------------------------------------------------------------------//
#include "MuIonizationProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/interactor/detail/PhysicsConstants.hh"
#include "celeritas/em/model/BetheBlochModel.hh"
#include "celeritas/em/model/BraggModel.hh"
#include "celeritas/em/model/ICRU73QOModel.hh"
#include "celeritas/em/model/MuBetheBlochModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct process from host data.
 */
MuIonizationProcess::MuIonizationProcess(SPConstParticles particles,
                                         SPConstImported process_data,
                                         Options options)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::mu_ioni,
                {pdg::mu_minus(), pdg::mu_plus()})
    , options_(options)
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 *
 * \todo Possibly get model limits from imported data?
 */
auto MuIonizationProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    using IMC = ImportModelClass;
    using SetApplicability = Model::SetApplicability;

    VecModel result;

    // Construct ICRU73QO mu- model
    CELER_ASSERT(imported_.has_model(pdg::mu_minus(), IMC::icru_73_qo));
    Applicability mm;
    mm.particle = particles_->find(pdg::mu_minus());
    mm.lower = zero_quantity();
    mm.upper = options_.bragg_icru73qo_upper_limit;
    result.push_back(std::make_shared<ICRU73QOModel>(
        *start_id++, *particles_, SetApplicability{mm}));

    // Construct Bragg mu+ model
    CELER_ASSERT(imported_.has_model(pdg::mu_plus(), IMC::bragg));
    Applicability mp = mm;
    mp.particle = particles_->find(pdg::mu_plus());
    result.push_back(std::make_shared<BraggModel>(
        *start_id++, *particles_, SetApplicability{mp}));

    if (imported_.has_model(pdg::mu_minus(), IMC::bethe_bloch))
    {
        // Older Geant4 versions use Bethe-Bloch at intermediate energies
        CELER_ASSERT(imported_.has_model(pdg::mu_plus(), IMC::bethe_bloch));
        mm.lower = mm.upper;
        mm.upper = options_.bethe_bloch_upper_limit;
        mp.lower = mm.lower;
        mp.upper = mm.upper;
        result.push_back(std::make_shared<BetheBlochModel>(
            *start_id++, *particles_, SetApplicability{mm, mp}));
    }

    // Construct muon Bethe-Bloch model
    CELER_ASSERT(imported_.has_model(pdg::mu_minus(), IMC::mu_bethe_bloch));
    CELER_ASSERT(imported_.has_model(pdg::mu_plus(), IMC::mu_bethe_bloch));
    mm.lower = mm.upper;
    mm.upper = detail::high_energy_limit();
    mp.lower = mm.lower;
    mp.upper = mm.upper;
    result.push_back(std::make_shared<MuBetheBlochModel>(
        *start_id++, *particles_, SetApplicability{mm, mp}));

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto MuIonizationProcess::macro_xs(Applicability applic) const -> XsGrid
{
    return imported_.macro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto MuIonizationProcess::energy_loss(Applicability applic) const
    -> EnergyLossGrid
{
    return imported_.energy_loss(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view MuIonizationProcess::label() const
{
    return "Muon ionization";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
