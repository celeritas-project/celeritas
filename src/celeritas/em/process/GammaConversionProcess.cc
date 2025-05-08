//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/process/GammaConversionProcess.cc
//---------------------------------------------------------------------------//
#include "GammaConversionProcess.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "celeritas/em/model/BetheHeitlerModel.hh"
#include "celeritas/io/ImportProcess.hh"
#include "celeritas/phys/PDGNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct from particles and imported Geant data.
 */
GammaConversionProcess::GammaConversionProcess(SPConstParticles particles,
                                               SPConstImported process_data,
                                               Options options)
    : particles_(std::move(particles))
    , imported_(process_data,
                particles_,
                ImportProcessClass::conversion,
                {pdg::gamma()})
    , options_(options)
{
    CELER_EXPECT(particles_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct the models associated with this process.
 */
auto GammaConversionProcess::build_models(ActionIdIter start_id) const -> VecModel
{
    return {std::make_shared<BetheHeitlerModel>(
        *start_id++, *particles_, imported_.processes(), options_.enable_lpm)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the interaction cross sections for the given energy range.
 */
auto GammaConversionProcess::macro_xs(Applicability applic) const -> XsGrid
{
    return imported_.macro_xs(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Get the energy loss for the given energy range.
 */
auto GammaConversionProcess::energy_loss(Applicability applic) const
    -> EnergyLossGrid
{
    return imported_.energy_loss(std::move(applic));
}

//---------------------------------------------------------------------------//
/*!
 * Name of the process.
 */
std::string_view GammaConversionProcess::label() const
{
    return "Photon annihiliation";
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
