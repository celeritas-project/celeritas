//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/TrackingManagerIntegration.cc
//---------------------------------------------------------------------------//
#include "TrackingManagerIntegration.hh"

#include <memory>
#include <set>
#include <G4ParticleDefinition.hh>
#include <G4Threading.hh>
#include <G4Version.hh>

#if G4VERSION_NUMBER >= 1100
#    include <G4VTrackingManager.hh>

#    include "TrackingManager.hh"
#endif

#include "corecel/io/Join.hh"
#include "corecel/sys/Stopwatch.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/GeantUtils.hh"

#include "ExceptionConverter.hh"
#include "TimeOutput.hh"
#include "TrackingManagerConstructor.hh"

#include "detail/IntegrationSingleton.hh"

using G4PD = G4ParticleDefinition;

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Check actual versus expected offloading.
 *
 * - All particles used by Celeritas should probably be offloaded
 * - All particles used by the TM constructor should be in Celeritas
 * - All particles in the TM constructor should use the Celeritas TM and have
 *   the correct local transporter/shared
 */
void verify_tracking_managers(Span<G4PD const* const> expected,
                              Span<G4PD const* const> actual,
                              SharedParams const& expected_shared,
                              LocalTransporter const& expected_local)
{
    std::set<G4PD const*> not_offloaded{actual.begin(), actual.end()};
    std::vector<G4PD const*> missing;

    bool all_attached_correctly{true};
    auto log_tm_failure = [&all_attached_correctly](G4PD const* p) {
        all_attached_correctly = false;
        auto msg = CELER_LOG(error);
        msg << "Particle " << StreamablePD{p} << ": tracking manager ";
        return msg;
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

#if G4VERSION_NUMBER >= 1100
        static TypeDemangler<G4VTrackingManager> demangle_tm;

        // Check tracking manager setup: note that this is *thread-local*
        // whereas the offloaded track list is *global*
        if (auto* tm = p->GetTrackingManager())
        {
            if (auto* celer_tm = dynamic_cast<TrackingManager*>(tm))
            {
                if (celer_tm->shared_params() != &expected_shared)
                {
                    log_tm_failure(p)
                        << R"(does not have the expected shared params)";
                }
                if (celer_tm->local_transporter() != &expected_local)
                {
                    log_tm_failure(p)
                        << R"(does not have the expected local transporter)";
                }
            }
            else
            {
                log_tm_failure(p)
                    << "is the wrong type (actual: " << demangle_tm(*tm)
                    << ')';
            }
        }
        else
        {
            log_tm_failure(p) << "is not attached";
        }
#else
        CELER_DISCARD(expected_shared);
        CELER_DISCARD(expected_local);
        CELER_DISCARD(log_tm_failure);
        CELER_ASSERT_UNREACHABLE();
#endif
    }

    auto printable_pd = [](G4PD const* p) { return StreamablePD{p}; };

    if (!not_offloaded.empty())
    {
        CELER_LOG(warning) << "Some particles known to Celeritas are not "
                              "offloaded by TrackingManagerConstructor: "
                           << join(not_offloaded.begin(),
                                   not_offloaded.end(),
                                   ", ",
                                   printable_pd)
                           << " (perhaps SetupOptions::offload_particles has "
                              "not been updated?)";
    }
    CELER_VALIDATE(missing.empty(),
                   << "not all particles from TrackingManagerConstructor are "
                      "active in Celeritas: missing "
                   << join(missing.begin(), missing.end(), ", ", printable_pd));
    CELER_VALIDATE(all_attached_correctly,
                   << "tracking manager(s) are not attached correctly "
                      "(maybe add TrackingManagerConstructor to your physics "
                      "list?)");
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Access the public-facing integration singleton.
 */
TrackingManagerIntegration& TrackingManagerIntegration::Instance()
{
    static TrackingManagerIntegration tmi;
    return tmi;
}

//---------------------------------------------------------------------------//
/*!
 * Start the run, initializing Celeritas options.
 */
void TrackingManagerIntegration::BeginOfRunAction(G4Run const*)
{
    CELER_VALIDATE(G4VERSION_NUMBER >= 1100,
                   << "the current version of Geant4 (" << G4VERSION_NUMBER
                   << ") is too old to support the tracking manager offload "
                      "interface (11.0 or higher is required)");

    Stopwatch get_setup_time;

    auto& singleton = detail::IntegrationSingleton::instance();

    if (G4Threading::IsMasterThread())
    {
        singleton.initialize_shared_params();
    }

    bool enable_offload = singleton.initialize_local_transporter();

    if (enable_offload)
    {
        // Set particle offloading based on user options
        auto const& user_offload = singleton.setup_options().offload_particles;
        auto const& offload_particles
            = user_offload.empty() ? SharedParams::default_offload_particles()
                                   : user_offload;

        // Set tracking manager on workers when Celeritas is not fully disabled
        CELER_LOG(debug) << "Verifying tracking manager";
        CELER_TRY_HANDLE(
            verify_tracking_managers(
                make_span(singleton.shared_params().OffloadParticles()),
                make_span(offload_particles),
                singleton.shared_params(),
                singleton.local_transporter()),
            ExceptionConverter{"celer.init.verify"});
    }

    if (G4Threading::IsMasterThread())
    {
        singleton.shared_params().timer()->RecordSetupTime(get_setup_time());
        singleton.start_timer();
    }
}

//---------------------------------------------------------------------------//
/*!
 * Only allow the singleton to construct.
 */
TrackingManagerIntegration::TrackingManagerIntegration() = default;

//---------------------------------------------------------------------------//
}  // namespace celeritas
