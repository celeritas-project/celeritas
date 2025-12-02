//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/IntegrationTestBase.cc
//---------------------------------------------------------------------------//
#include "IntegrationTestBase.hh"

#include <exception>
#include <limits>
#include <G4Threading.hh>
#include <G4UserEventAction.hh>
#include <G4UserRunAction.hh>
#include <G4UserSteppingAction.hh>
#include <G4UserTrackingAction.hh>
#include <G4VSensitiveDetector.hh>
#include <G4VUserActionInitialization.hh>
#include <G4VUserDetectorConstruction.hh>
#include <G4VUserPrimaryGeneratorAction.hh>
#include <G4Version.hh>

#include "corecel/Assert.hh"

#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#else
#    include <G4MTRunManager.hh>
#endif

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"
#include "corecel/io/StringUtils.hh"
#include "corecel/math/ArrayUtils.hh"
#include "corecel/sys/Environment.hh"
#include "corecel/sys/ScopedProfiling.hh"
#include "corecel/sys/TracingSession.hh"
#include "corecel/sys/TypeDemangler.hh"
#include "geocel/GeantUtils.hh"
#include "geocel/ScopedGeantExceptionHandler.hh"
#include "geocel/ScopedGeantLogger.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/Units.hh"
#include "celeritas/ext/EmPhysicsList.hh"
#include "celeritas/ext/SimpleSensitiveDetector.hh"
#include "celeritas/g4/DetectorConstruction.hh"
#include "celeritas/inp/Events.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "accel/AlongStepFactory.hh"
#include "accel/PGPrimaryGeneratorAction.hh"
#include "accel/SetupOptions.hh"

#include "PersistentSP.hh"
#include "ShimSensitiveDetector.hh"

using SPTracing = std::shared_ptr<celeritas::TracingSession>;

namespace celeritas
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
using namespace std::string_literals;

std::string thread_label()
{
    return G4Threading::IsMasterThread()
               ? "M"s
               : std::to_string(G4Threading::G4GetThreadId());
}

std::string thread_description()
{
    return G4Threading::IsMasterThread()
               ? "main thread"s
               : "worker thread "s
                     + std::to_string(G4Threading::G4GetThreadId());
}

//---------------------------------------------------------------------------//
class RunAction final : public G4UserRunAction
{
  public:
    RunAction(IntegrationTestBase* test, SPTracing tracing)
        : test_{test}
        , tracing_{std::move(tracing)}
        , exceptions_(
              [this](std::exception_ptr ep) { this->handle_exception(ep); })
    {
    }

    void BeginOfRunAction(G4Run const* run) final
    {
        CELER_LOG_LOCAL(debug) << "RunAction::BeginOfRunAction";
        CELER_TRY_HANDLE(test_->BeginOfRunAction(run), this->handle_exception);
    }
    void EndOfRunAction(G4Run const* run) final
    {
        CELER_LOG_LOCAL(debug) << "RunAction::EndOfRunAction";
        CELER_TRY_HANDLE(test_->EndOfRunAction(run), this->handle_exception);
        if (tracing_)
        {
            CELER_LOG_LOCAL(debug) << "Flushing Perfetto trace";
            tracing_->flush();
        }
    }

    // TODO: push exception onto a vector so we can do validation testing
    void handle_exception(std::exception_ptr ep)
    {
        try
        {
            std::rethrow_exception(ep);
        }
        catch (RuntimeError const& e)
        {
            auto const& d = e.details();
            if (cstring_equal(d.which, "Geant4"))
            {
                // GeantExceptionHandler wrapped this error
                FAIL() << "GeantExceptionHandler caught runtime error ("
                       << thread_label() << ',' << d.condition << "): from "
                       << d.file << ": " << d.what;
            }
            else
            {
                // Some other error
                FAIL() << "Caught runtime error from " << thread_description()
                       << ": " << e.what();
            }
        }
        catch (std::exception const& e)
        {
            FAIL() << "From " << thread_description() << ": " << e.what();
        }
    }

  private:
    IntegrationTestBase* test_;
    SPTracing tracing_;
    ScopedGeantExceptionHandler exceptions_;
};

//---------------------------------------------------------------------------//
class EventAction final : public G4UserEventAction
{
  public:
    explicit EventAction(IntegrationTestBase* test) : test_{test} {}

    void BeginOfEventAction(G4Event const* event) final
    {
        if (test_->HasFatalFailure())
        {
            CELER_LOG_LOCAL(critical)
                << "Cancelling event " << event->GetEventID()
                << " due to fatal test failure";
            if (auto* event_mgr = G4EventManager::GetEventManager())
            {
                event_mgr->AbortCurrentEvent();
            }
            return;
        }
        CELER_LOG_LOCAL(debug) << "EventAction::BeginOfEventAction";
        test_->BeginOfEventAction(event);
    }
    void EndOfEventAction(G4Event const* event) final
    {
        CELER_LOG_LOCAL(debug) << "EventAction::EndOfEventAction";
        test_->EndOfEventAction(event);
    }

  private:
    IntegrationTestBase* test_;
};

//---------------------------------------------------------------------------//
class ActionInitialization final : public G4VUserActionInitialization
{
  public:
    explicit ActionInitialization(IntegrationTestBase* test) : test_{test}
    {
        if (CELERITAS_USE_PERFETTO && ScopedProfiling::enabled())
        {
            tracing_ = std::make_unique<TracingSession>(
                test_->make_unique_filename(".perf.proto"));
        }
    }

    ~ActionInitialization()
    {
        CELER_TRY_HANDLE((CELER_LOG_LOCAL(debug)
                          << R"(Tearing down action initialization)"),
                         static_cast<void>);
    }

    void BuildForMaster() const final
    {
        CELER_LOG_LOCAL(debug) << "ActionInitialization::BuildForMaster";
        this->SetUserAction(new RunAction{test_, tracing_});
    }
    void Build() const final
    {
        CELER_LOG_LOCAL(debug) << "ActionInitialization::Build";

        // Run and event actions
        this->SetUserAction(new RunAction{test_, tracing_});
        this->SetUserAction(new EventAction{test_});

        // Primary generator
        auto pg_inp = test_->make_primary_input();
        CELER_VALIDATE(pg_inp, << "incomplete primary input");
        this->SetUserAction(new PGPrimaryGeneratorAction{std::move(pg_inp)});

        // User actions
        if (auto track_action = test_->make_tracking_action())
        {
            TypeDemangler<G4UserTrackingAction> demangle_type;
            CELER_LOG_LOCAL(debug) << "Setting track action of type "
                                   << demangle_type(*track_action);
            this->SetUserAction(track_action.release());
        }
        if (auto stepping_action = test_->make_stepping_action())
        {
            TypeDemangler<G4UserSteppingAction> demangle_type;
            CELER_LOG_LOCAL(debug) << "Setting step action of type "
                                   << demangle_type(*stepping_action);
            this->SetUserAction(stepping_action.release());
        }
    }

  private:
    IntegrationTestBase* test_;
    SPTracing tracing_;
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// Default destructor to enable base class deletion and anchor vtable
IntegrationTestBase::~IntegrationTestBase() = default;

std::string IntegrationTestBase::make_unique_filename(std::string_view ext)
{
    std::string new_ext = "-";
    new_ext += celeritas::getenv("CELER_OFFLOAD");
    new_ext += "-";
    new_ext += celeritas::tolower(celeritas::getenv("G4RUN_MANAGER_TYPE"));
    new_ext += ext;
    return ::celeritas::test::Test::make_unique_filename(new_ext);
}

//---------------------------------------------------------------------------//
/*!
 * Create or access the run manager (created once per execution).
 *
 * A PersistentSP is used to tear down the run manager at the end of the
 * test app execution.
 */
G4RunManager& IntegrationTestBase::run_manager()
{
    static PersistentSP<G4RunManager> rm{"run manager"};

    std::string basename{this->gdml_basename()};

    if (rm)
    {
        CELER_VALIDATE(basename == rm.key(),
                       << "cannot create a run manager for two problems in "
                          "one execution: use '--gtest_filter'");
        return *rm.value();
    }

    rm.set(basename, [&] {
        CELER_LOG(status) << "Creating run manager";
        // Run manager writes output that cannot be redirected with
        // GeantLoggerAdapter: capture all output from this section
        ScopedGeantExceptionHandler scoped_exceptions;

        // Access the particle table before creating the run manager, so that
        // missing environment variables like G4ENSDFSTATEDATA get caught
        // cleanly rather than segfaulting
        G4ParticleTable::GetParticleTable();

        std::shared_ptr<G4RunManager> rm{
#if G4VERSION_NUMBER >= 1100
            G4RunManagerFactory::CreateRunManager()
#else
            std::make_shared<G4RunManager>()
#endif
        };
        CELER_ASSERT(rm);

        // Disable signal handling
        disable_geant_signal_handler();

        // Set up detector
        rm->SetUserInitialization(new DetectorConstruction{
            this->test_data_path("geocel", basename + ".gdml"),
            [this](std::string const& sd_name) {
                return this->make_sens_det(sd_name);
            }});

        // Set up physics
        auto phys = this->make_physics_list();
        CELER_ASSERT(phys);
        rm->SetUserInitialization(phys.release());

        // Set up runtime initialization
        rm->SetUserInitialization(new ActionInitialization{this});
        return rm;
    }());

    return *rm.value();
}

//---------------------------------------------------------------------------//
/*!
 * Create physics input: default is EM only.
 */
auto IntegrationTestBase::make_physics_input() const -> PhysicsInput
{
    PhysicsInput result;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create physics list: default is EM only using make_physics_input.
 */
auto IntegrationTestBase::make_physics_list() const -> UPPhysicsList
{
    CELER_LOG_LOCAL(debug) << "Creating default EM-only physics list";
    return std::make_unique<EmPhysicsList>(this->make_physics_input());
}

//---------------------------------------------------------------------------//
/*!
 * Create optional tracking action (local, default null).
 */
auto IntegrationTestBase::make_tracking_action() -> UPTrackAction
{
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Create optional stepping action (local, default null).
 */
auto IntegrationTestBase::make_stepping_action() -> UPStepAction
{
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Create Celeritas setup options.
 */
SetupOptions IntegrationTestBase::make_setup_options()
{
    celeritas::SetupOptions opts;

    // NOTE: these numbers are appropriate for CPU execution and can be set
    // through the UI using `/celer/`
    opts.max_num_tracks = 1024;
    opts.initializer_capacity = 1024 * 128;

    // Use a uniform (zero) magnetic field
    opts.make_along_step = celeritas::UniformAlongStepFactory();

    // Save diagnostic file to a unique name
    opts.output_file = this->make_unique_filename(".out.json");
    return opts;
}

//---------------------------------------------------------------------------//
/*!
 * Create an optional thread-local sensitive detector.
 */
auto IntegrationTestBase::make_sens_det(std::string const&) -> UPSensDet
{
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Create physics list: default is EM only using make_physics_input.
 */
auto LarSphereIntegrationMixin::make_physics_input() const -> PhysicsInput
{
    PhysicsInput result = Base::make_physics_input();
    result.em_bins_per_decade = 5;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a 10 MeV electron primary.
 */
auto LarSphereIntegrationMixin::make_primary_input() const -> PrimaryInput
{
    PrimaryInput result;
    result.pdg = {pdg::electron()};
    result.energy = inp::MonoenergeticDistribution{10};  // [MeV]
    result.shape
        = inp::PointDistribution{array_cast<double>(from_cm({99, 0.1, 0}))};
    result.angle = inp::IsotropicDistribution{};
    result.num_events = 4;  // Overridden with BeamOn
    result.primaries_per_event = 10;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create THREAD-LOCAL sensitive detectors.
 */
auto LarSphereIntegrationMixin::make_sens_det(std::string const& sd_name)
    -> UPSensDet
{
    EXPECT_EQ("detshell", sd_name);
    return std::make_unique<ShimSensitiveDetector>(
        sd_name,
        [this](G4Step const* step) { return this->process_hit(step); });
}

//---------------------------------------------------------------------------//
/*!
 * Process a hit locally.
 */
void LarSphereIntegrationMixin::process_hit(G4Step const* step)
{
    if (CELER_UNLIKELY(!step || !step->GetTrack()))
    {
        // Reduce testing overhead: google assertions allocate memory
        ASSERT_TRUE(step);
        ASSERT_TRUE(step->GetTrack());
        return;
    }

    auto& track = *step->GetTrack();
    if (CELER_UNLIKELY(!(track.GetWeight() > 0) || !track.GetVolume()
                       || !track.GetNextVolume()))
    {
        EXPECT_GT(track.GetWeight(), 0);
        EXPECT_TRUE(track.GetVolume());
        // Since we don't have any detectors on the boundary of the problem:
        EXPECT_TRUE(track.GetNextVolume());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create physics list: default is EM only using make_physics_input.
 */
auto TestEm3IntegrationMixin::make_physics_input() const -> PhysicsInput
{
    using MevEnergy = Quantity<units::Mev, double>;

    PhysicsInput result = Base::make_physics_input();
    result.em_bins_per_decade = 14;
    // Increase the lower energy limit of the physics tables
    result.min_energy = MevEnergy{0.1};
    result.default_cutoff = 0.1 * units::centimeter;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a 100 MeV electron primary.
 */
auto TestEm3IntegrationMixin::make_primary_input() const -> PrimaryInput
{
    PrimaryInput result;
    result.pdg = {pdg::electron()};
    result.energy = inp::MonoenergeticDistribution{100};  // [MeV]
    result.shape
        = inp::PointDistribution{array_cast<double>(from_cm({-22, 0, 0}))};
    result.angle = inp::MonodirectionalDistribution{{1, 0, 0}};
    result.num_events = 2;
    result.primaries_per_event = 1;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create THREAD-LOCAL sensitive detectors for an SD name in the GDML file.
 */
auto TestEm3IntegrationMixin::make_sens_det(std::string const& sd_name)
    -> UPSensDet
{
    EXPECT_EQ("lAr", sd_name);

    return std::make_unique<SimpleSensitiveDetector>(sd_name);
}

//---------------------------------------------------------------------------//
/*!
 * Create physics list
 */
auto OpNoviceIntegrationMixin::make_physics_input() const -> PhysicsInput
{
    auto result = Base::make_physics_input();

    // Enable optical physics (scintillation + Cherenkov)
    auto& optical = result.optical;
    optical = {};
    EXPECT_TRUE(optical);
    EXPECT_TRUE(optical.scintillation);
    EXPECT_TRUE(optical.cherenkov);
    EXPECT_TRUE(optical.mie_scattering);
    EXPECT_TRUE(optical.rayleigh_scattering);

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Create a 0.5 MeV positron primary.
 */
auto OpNoviceIntegrationMixin::make_primary_input() const -> PrimaryInput
{
    PrimaryInput result;
    result.pdg = {pdg::positron()};
    result.energy = inp::MonoenergeticDistribution{0.5};  // [MeV]
    result.shape
        = inp::PointDistribution{array_cast<double>(from_cm({0., 0., 0.}))};
    result.angle = inp::MonodirectionalDistribution{{1., 0., 0.}};
    result.num_events = 12;  // Overridden with BeamOn
    result.primaries_per_event = 10;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Return null pointer for the sensitive detector
 */
auto OpNoviceIntegrationMixin::make_sens_det(std::string const&) -> UPSensDet
{
    return nullptr;
}

//---------------------------------------------------------------------------//
/*!
 * Enable optical physics options
 */
SetupOptions OpNoviceIntegrationMixin::make_setup_options()
{
    auto result = Base::make_setup_options();
    result.sd.enabled = false;
    result.optical = [] {
        OpticalSetupOptions opt;
        opt.capacity.tracks = 32768;
        opt.capacity.generators = 32768 * 8;
        opt.capacity.primaries = opt.capacity.generators;
        return opt;
    }();
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
