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

#if G4VERSION_NUMBER >= 1100
#    include <G4RunManagerFactory.hh>
#else
#    include <G4MTRunManager.hh>
#endif

#include "corecel/Config.hh"

#include "corecel/io/Logger.hh"
#include "corecel/io/ScopedTimeAndRedirect.hh"
#include "corecel/io/StringUtils.hh"
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
    explicit RunAction(IntegrationTestBase* test)
        : test_{test}, exceptions_([this](std::exception_ptr ep) {
            this->handle_exception(ep);
        })
    {
    }

    void BeginOfRunAction(G4Run const* run) final
    {
        CELER_LOG_LOCAL(debug) << "RunAction::BeginOfRunAction";
        test_->BeginOfRunAction(run);
    }
    void EndOfRunAction(G4Run const* run) final
    {
        CELER_LOG_LOCAL(debug) << "RunAction::EndOfRunAction";
        test_->EndOfRunAction(run);
    }

    // TODO: push exception onto a vector that can be checked
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
                FAIL() << '(' << thread_label() << ',' << d.condition
                       << "): from " << d.file << ": " << d.what;
            }
            else
            {
                // Some other error
                FAIL() << "From " << thread_description() << ": " << e.what();
            }
        }
        catch (std::exception const& e)
        {
            FAIL() << "From " << thread_description() << ": " << e.what();
        }
    }

  private:
    IntegrationTestBase* test_;
    ScopedGeantExceptionHandler exceptions_;
};

//---------------------------------------------------------------------------//
class EventAction final : public G4UserEventAction
{
  public:
    explicit EventAction(IntegrationTestBase* test) : test_{test} {}

    void BeginOfEventAction(G4Event const* event) final
    {
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
    explicit ActionInitialization(IntegrationTestBase* test) : test_{test} {}

    void BuildForMaster() const final
    {
        CELER_LOG_LOCAL(debug) << "ActionInitialization::BuildForMaster";
        this->SetUserAction(new RunAction{test_});
    }
    void Build() const final
    {
        CELER_LOG_LOCAL(debug) << "ActionInitialization::Build";

        // Run and event actions
        this->SetUserAction(new RunAction{test_});
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
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
// Default destructor to enable base class deletion and anchor vtable
IntegrationTestBase::~IntegrationTestBase() = default;

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
        ScopedTimeAndRedirect scoped_time{"G4RunManager"};
        ScopedGeantExceptionHandler scoped_exceptions;

        // Access the particle table before creating the run manager, so that
        // missing environment variables like G4ENSDFSTATEDATA get caught
        // cleanly rather than segfaulting
        G4ParticleTable::GetParticleTable();

        std::shared_ptr<G4RunManager> rm{
#if G4VERSION_NUMBER >= 1100
            G4RunManagerFactory::CreateRunManager()
#else
            std::shared_ptr<G4RunManager>()
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

    // FIXME: weight flag isn't being set in GeantSd
    opts.sd.track = false;

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
    using MevEnergy = Quantity<units::Mev, double>;

    PrimaryInput result;
    result.pdg = {pdg::electron()};
    result.energy = inp::MonoenergeticDistribution{MevEnergy{10}};
    result.shape = inp::PointDistribution{from_cm({99, 0.1, 0})};
    result.angle = inp::IsotropicDistribution{};
    result.num_events = 4;
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
    EXPECT_EQ("detector", sd_name);

    return std::make_unique<SimpleSensitiveDetector>(sd_name);
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
    using MevEnergy = Quantity<units::Mev, double>;

    PrimaryInput result;
    result.pdg = {pdg::electron()};
    result.energy = inp::MonoenergeticDistribution{MevEnergy{100}};
    result.shape = inp::PointDistribution{from_cm({-22, 0, 0})};
    result.angle = inp::MonodirectionalDistribution{Real3{1, 0, 0}};
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
}  // namespace test
}  // namespace celeritas
