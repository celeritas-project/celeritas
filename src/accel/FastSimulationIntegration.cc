//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/FastSimulationIntegration.cc
//---------------------------------------------------------------------------//
#include "FastSimulationIntegration.hh"

#include <set>
#include <G4FastSimulationManagerProcess.hh>
#include <G4ParticleDefinition.hh>
#include <G4ProcessManager.hh>
#include <G4ProcessVector.hh>
#include <G4Threading.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/cont/Span.hh"
#include "corecel/io/Join.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantUtils.hh"

#include "ExceptionConverter.hh"

#include "detail/IntegrationSingleton.hh"

using G4PD = G4ParticleDefinition;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Check actual versus expected offloading for particles/processes.
 *
 * - All particles used by Celeritas should probably be offloaded
 * - All particles used by the FSM constructor should be in Celeritas
 * - All particles in the TM constructor should use the Celeritas TM and have
 *   the correct local transporter/shared
 */

void verify_fast_processes(Span<G4PD const* const> expected,
                           Span<G4PD const* const> actual)
{
    std::set<G4PD const*> not_offloaded{actual.begin(), actual.end()};
    std::vector<G4PD const*> missing;

    bool all_attached_correctly{true};
    auto log_fs_failure = [&all_attached_correctly](G4PD const* p) {
        all_attached_correctly = false;
        auto msg = CELER_LOG(error);
        msg << "Particle " << StreamablePD{p} << ": ";
        return msg;
    };

    auto contains_fs_process = [](G4ProcessManager const* pm) {
        G4ProcessVector* pv = pm->GetProcessList();
        CELER_ASSERT(pv);
        for (auto j : range(pv->size()))
        {
            if (dynamic_cast<G4FastSimulationManagerProcess*>((*pv)[j]))
            {
                return true;
            }
        }
        return false;
    };

    for (auto* p : expected)
    {
        CELER_ASSERT(p);

        auto iter = not_offloaded.find(p);
        if (iter == not_offloaded.end())
        {
            missing.push_back(p);
        }
        else
        {
            not_offloaded.erase(iter);
        }

#if G4VERSION_NUMBER >= 1110
        // Step 1. Check that particle has G4FastSimulationManagerProcess
        // attached at all
        if (auto* pm = p->GetProcessManager())
        {
            if (!contains_fs_process(pm))
            {
                log_fs_failure(p)
                    << R"(does not have G4FastSimulationManagerProcess attached)";
            }
        }
        else
        {
            log_fs_failure(p) << "does not have G4ProcessManager attached";
        }
#else
        CELER_DISCARD(expected_shared);
        CELER_DISCARD(expected_local);
        CELER_DISCARD(log_fs_failure);
        CELER_DISCARD(contains_fs_process);
        CELER_ASSERT_UNREACHABLE();
#endif
    }

    auto printable_pd = [](G4PD const* p) { return StreamablePD{p}; };

    if (!not_offloaded.empty())
    {
        CELER_LOG(warning) << "Some particles known to Celeritas are not "
                              "offloaded by FastSimulationModel: "
                           << join(not_offloaded.begin(),
                                   not_offloaded.end(),
                                   ", ",
                                   printable_pd)
                           << " (perhaps SetupOptions::offload_particles has "
                              "not been updated?)";
    }
    CELER_VALIDATE(missing.empty(),
                   << "not all particles from FastSimulationModel are "
                      "active in Celeritas: missing "
                   << join(missing.begin(), missing.end(), ", ", printable_pd));

    CELER_VALIDATE(all_attached_correctly,
                   << "fast simulation process(es) are not attached correctly "
                      "(maybe add G4FastSimulationPhysics to your physics "
                      "list?)");
}

//---------------------------------------------------------------------------//
/*!
 * Check actual versus expected offloading for regions.
 *
 * - All regions handled by Celeritas should exist
 * - All of these regions should have a G4FastSimulationManagerProcess attached
 * - All of these managers should hold a FastSimulationModel instance
 * - All of these FastSimulationModels should have the correct local
 *   transporter/shared params
 *
 * - Setup is a bit different to tracking manager
 *   - Models are added to regions as part of ConstructSDandField (nominal
 *     location, but guarantees geometry/physics state at this point).
 *   - Means we have to check two separate things:
 *     1. G4ProcessManager for offload particle p must have
 *        G4FastSimulationManagerProcess attached.
 *     2. Each offload region r must have:
 *        a. A non-null G4FastSimulationManager instance:
 *             auto* fsm = region->GetFastSimulationManager();
 *        b. That G4FastSimulationManager must contain an instance of the
 *           Celeritas FastSimulationModel:
 *             auto fsm_list = fsm->GetFastSimulationModelList();
 *             ... iterate/dynamic_cast ...
 *        c. Probably... check the model has the correct shared_params and
 *           local transport as we do in TrackingManager offload.
 */

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Access the public-facing integration singleton.
 */
FastSimulationIntegration& FastSimulationIntegration::Instance()
{
    static FastSimulationIntegration tmi;
    return tmi;
}

//---------------------------------------------------------------------------//
/*!
 * Verify fast simulation setup.
 */
void FastSimulationIntegration::verify_local_setup()
{
    CELER_VALIDATE(G4VERSION_NUMBER >= 1110,
                   << "the current version of Geant4 (" << G4VERSION_NUMBER
                   << ") is too old to support the fast simulation offload "
                      "interface (11.1 or higher is required)");

    auto& singleton = detail::IntegrationSingleton::instance();

    // Check particle/processes are consistent
    auto const& user_offload = singleton.setup_options().offload_particles;
    auto const& offload_particles
        = user_offload ? *user_offload
                       : SharedParams::default_offload_particles();

    CELER_LOG(debug) << "Verifying fast simulation particles and processes";
    verify_fast_processes(
        make_span(singleton.shared_params().OffloadParticles()),
        make_span(offload_particles));

    // TODO: Check requested regions have our FastSimulationModel attached
}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
FastSimulationIntegration::FastSimulationIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
