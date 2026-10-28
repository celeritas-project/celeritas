//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/detail/HitProcessor.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/detail/HitProcessor.hh"

#include <chrono>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <G4DynamicParticle.hh>
#include <G4LogicalVolume.hh>
#include <G4ParticleTable.hh>
#include <G4Threading.hh>
#include <G4ThreeVector.hh>
#include <G4Track.hh>
#include <G4WorkerThread.hh>

#include "corecel/data/Ref.hh"
#include "corecel/sys/Device.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"  // IWYU pragma: keep
#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantTrackReconstruction.hh"
#include "celeritas/geo/CoreGeoParams.hh"  // IWYU pragma: keep
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "celeritas_test.hh"
#include "../SensDetTestBase.hh"
#include "../SimpleSensitiveDetector.hh"

using celeritas::test::from_cm;
using celeritas::test::SimpleHitsResult;
using celeritas::test::SimpleSensitiveDetector;
using celeritas::units::MevEnergy;

namespace celeritas
{
namespace detail
{
namespace test
{
namespace
{
//---------------------------------------------------------------------------//
//! Copy and process pending device hits synchronously
void process_pending(HitProcessor& process_hits)
{
    process_hits.copy_pending_steps();
    process_hits.process_copied_steps();
}

//---------------------------------------------------------------------------//
/*!
 * Set up Geant4 thread-local geometry data as in a worker thread.
 *
 * Geant4 stores per-thread data (including each logical volume's sensitive
 * detector) in "split" classes that a worker thread copies from the master.
 */
class ScopedWorkerGeometry
{
  public:
    explicit ScopedWorkerGeometry(int thread_id)
    {
        G4Threading::G4SetThreadId(thread_id);
        G4WorkerThread::BuildGeometryAndPhysicsVector();
    }
    ~ScopedWorkerGeometry()
    {
        G4WorkerThread::DestroyGeometryAndPhysicsVector();
    }
    CELER_DELETE_COPY_MOVE(ScopedWorkerGeometry);
};

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
class SimpleCmsTest : public ::celeritas::test::SensDetTestBase,
                      public ::celeritas::test::SimpleCmsTestBase
{
  protected:
    using VecLV = std::vector<G4LogicalVolume const*>;
    using VecParticle = HitProcessor::VecParticle;
    using SPConstVecLV = HitProcessor::SPConstVecLV;

    void SetUp() override;
    SetStr detector_volumes() const final;

    SPConstVecLV make_detector_volumes();
    VecParticle make_particles();
    HitProcessor make_hit_processor();

    DetectorStepOutput make_dso() const;

    SimpleHitsResult const& get_hits(std::string const& name) const;

    //! Hits recorded by one mock worker thread
    struct ThreadResult
    {
        std::vector<double> em_edep;
        size_type num_hits{0};
        std::exception_ptr error;
    };

    // Process steps on independent threads with random delays
    template<MemSpace M>
    std::vector<ThreadResult> run_threads(int num_threads, int num_steps);

    // Check that each thread processed every step exactly once, in order
    void check_thread_results(std::vector<ThreadResult> const& results,
                              int num_steps) const;

    // Event ID mock for GeantTrackReconstruction (used in CELERITAS_DEBUG)
    static int test_cur_event;
    static int get_test_current_event_id() { return test_cur_event; }

    static void SetUpTestCase()
    {
        GeantTrackReconstruction::get_current_event_id
            = get_test_current_event_id;
    }

    static void TearDownTestCase()
    {
        GeantTrackReconstruction::get_current_event_id = nullptr;
    }

  protected:
    StepSelection selection_;
    HitProcessor::StepPointBool locate_touchable_{{false, false}};
};

int SimpleCmsTest::test_cur_event{0};

//---------------------------------------------------------------------------//
void SimpleCmsTest::SetUp()
{
    // Create default step selection (see GeantSd)
    selection_.energy_deposition = true;
    selection_.step_length = true;
    selection_.weight = true;
    selection_.points[StepPoint::pre].time = true;
    selection_.points[StepPoint::pre].energy = true;
    selection_.points[StepPoint::pre].pos = true;
    selection_.points[StepPoint::pre].dir = true;
    selection_.points[StepPoint::post].time = true;
    selection_.points[StepPoint::post].pos = true;
    selection_.points[StepPoint::post].dir = true;
    selection_.points[StepPoint::post].energy = true;
    selection_.particle_id = true;
    selection_.primary_id = true;
    selection_.parent_id = true;
    selection_.generation = true;
}

auto SimpleCmsTest::detector_volumes() const -> SetStr
{
    return {
        "em_calorimeter",
        "had_calorimeter",
        "si_tracker",
        "world",
    };
}

auto SimpleCmsTest::make_detector_volumes() -> SPConstVecLV
{
    // Make sure geometry is built
    this->geometry();

    // Loop over detectors (sorted by the LV name!), and add them
    VecLV lv;
    for (auto const& kv : this->detectors())
    {
        CELER_ASSERT(kv.second);
        lv.push_back(kv.second->lv());
        CELER_ASSERT(lv.back());
    }
    return std::make_shared<VecLV>(std::move(lv));
}

auto SimpleCmsTest::make_particles() -> VecParticle
{
    VecParticle result;
    if (!selection_.particle_id)
    {
        return result;
    }

    auto& g4particles = *G4ParticleTable::GetParticleTable();
    for (auto p : {pdg::gamma(), pdg::electron(), pdg::positron()})
    {
        result.push_back(g4particles.FindParticle(p.get()));
    }
    return result;
}

auto SimpleCmsTest::make_hit_processor() -> HitProcessor
{
    for (auto sp : range(StepPoint::size_))
    {
        if (locate_touchable_[sp])
        {
            selection_.points[sp].volume_instance_ids = true;
        }
    }
    return HitProcessor{this->make_detector_volumes(),
                        this->make_particles(),
                        selection_,
                        locate_touchable_};
}

auto SimpleCmsTest::get_hits(std::string const& name) const
    -> SimpleHitsResult const&
{
    auto iter = this->detectors().find(name);
    CELER_ASSERT(iter != detectors().end());
    CELER_ASSERT(iter->second);
    return iter->second->hits();
}

DetectorStepOutput SimpleCmsTest::make_dso() const
{
    DetectorStepOutput dso;
    dso.detector_id = {
        DetectorId{2},  // si_tracker
        DetectorId{0},  // em_calorimeter
        DetectorId{1},  // had_calorimeter
    };
    dso.track_id = {
        TrackId{0},
        TrackId{2},
        TrackId{4},
    };
    dso.weight = {
        1.0_r,  // si_tracker
        0.5_r,  // em_calorimeter
        0.8_r,  // had_calorimeter
    };
    if (selection_.primary_id)
    {
        dso.primary_id = {
            PrimaryId{0},
            PrimaryId{1},
            PrimaryId{1},
        };
    }
    if (selection_.parent_id)
    {
        // The last step is from a secondary of the second primary's track
        dso.parent_id = {
            TrackId{},
            TrackId{},
            TrackId{2},
        };
    }
    if (selection_.generation)
    {
        dso.generation = {0, 0, 1};
    }
    if (selection_.energy_deposition)
    {
        dso.energy_deposition = {
            MevEnergy{0.1},
            MevEnergy{0.2},
            MevEnergy{0.3},
        };
    }
    if (selection_.step_length)
    {
        dso.step_length = {
            from_cm(0.1),
            from_cm(2.0),
            from_cm(3.0),
        };
    }
    if (selection_.points[StepPoint::post].time)
    {
        using celeritas::units::second;

        dso.points[StepPoint::post].time = {
            1e-9 * second,
            2e-10 * second,
            3e-8 * second,
        };
    }
    if (selection_.points[StepPoint::pre].time)
    {
        using celeritas::units::second;

        dso.points[StepPoint::pre].time = {
            0.5e-9 * second,
            1.5e-10 * second,
            2.5e-8 * second,
        };
    }
    if (selection_.points[StepPoint::pre].pos)
    {
        // note: points must correspond to detector volumes!
        dso.points[StepPoint::pre].pos = {
            from_cm(Real3{100, 0, 0}),
            from_cm(Real3{0, 150, 10}),
            from_cm(Real3{0, 200, -20}),
        };
    }
    if (selection_.points[StepPoint::pre].dir)
    {
        dso.points[StepPoint::pre].dir = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, -1},
        };
    }
    if (selection_.points[StepPoint::post].pos)
    {
        // note: points must correspond to detector volumes!
        dso.points[StepPoint::post].pos = {
            from_cm(Real3{105, 0, 0}),
            from_cm(Real3{0, 155, 10}),
            from_cm(Real3{0, 205, -20}),
        };
    }
    if (selection_.points[StepPoint::post].dir)
    {
        dso.points[StepPoint::post].dir = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, -1},
        };
    }
    if (selection_.points[StepPoint::post].energy)
    {
        dso.points[StepPoint::post].energy = {
            MevEnergy{0.9},
            MevEnergy{1.8},
            MevEnergy{2.7},
        };
    }
    if (selection_.points[StepPoint::pre].energy)
    {
        dso.points[StepPoint::pre].energy = {
            MevEnergy{0.0},
            MevEnergy{0.0},
            MevEnergy{0.0},
        };
    }
    if (selection_.particle_id)
    {
        dso.particle_id = {
            ParticleId{2},
            ParticleId{1},
            ParticleId{0},
        };
    }
    if (selection_.points[StepPoint::pre].volume_instance_ids)
    {
        // Note: the volumes correspond to simple-cms and the detector IDs
        // above
        dso.num_volume_levels = 2;
        auto const& vi_names = this->volumes()->volume_instance_labels();
        auto wovi = vi_names.find_unique("world_PV");
        auto emvi = vi_names.find_unique("em_calorimeter_pv");
        auto havi = vi_names.find_unique("had_calorimeter_pv");
        auto sivi = vi_names.find_unique("si_tracker_pv");
        dso.points[StepPoint::pre].volume_instance_ids
            = {wovi, sivi, wovi, emvi, wovi, havi};
    }
    return dso;
}

//---------------------------------------------------------------------------//
/*!
 * Process steps on independent threads with random delays.
 *
 * This mocks up a multithreaded job: each thread owns Geant4 worker geometry,
 * thread-local sensitive detectors, a hit processor, and step state on its own
 * stream. Each step scales the energy deposition by the step number so that
 * lost, duplicated, or mixed-up batches are detected. Random waits between
 * the asynchronous phases vary how device work overlaps with host work.
 */
template<MemSpace M>
auto SimpleCmsTest::run_threads(int num_threads, int num_steps)
    -> std::vector<ThreadResult>
{
    CELER_EXPECT(num_threads > 0 && num_steps > 0);

    selection_ = {};
    selection_.energy_deposition = true;
    selection_.step_length = true;
    selection_.points[StepPoint::pre].time = true;

    // Construct shared, read-only data on the master thread
    auto const detector_volumes = this->make_detector_volumes();
    auto const dso = this->make_dso();
    auto const selection = selection_;
    HostVal<StepParamsData> params;
    params.selection = selection;
    std::vector<DetectorId> detector_map{DetectorId{0}};
    make_builder(&params.detector)
        .insert_back(detector_map.begin(), detector_map.end());
    if constexpr (M == MemSpace::device)
    {
        device().create_streams(num_threads);
    }

    std::vector<ThreadResult> results(num_threads);
    auto run_thread = [&](int thread_idx) {
        auto& result = results[thread_idx];
        try
        {
            if constexpr (M == MemSpace::device)
            {
                activate_device_local();
            }
            ScopedWorkerGeometry worker_geo{thread_idx};

            // Attach thread-local detectors, as each Geant4 worker owns
            std::vector<std::unique_ptr<SimpleSensitiveDetector>> detectors;
            for (G4LogicalVolume const* lv : *detector_volumes)
            {
                detectors.push_back(
                    std::make_unique<SimpleSensitiveDetector>(lv));
                const_cast<G4LogicalVolume*>(lv)->SetSensitiveDetector(
                    detectors.back().get());
            }

            HitProcessor process_hits{
                detector_volumes, {}, selection, {false, false}};

            StreamId const stream_id(thread_idx);
            HostVal<StepStateData> host_states;
            resize(&host_states, make_const_ref(params), stream_id, dso.size());
            StepStateData<Ownership::value, MemSpace::device> device_states;
            if constexpr (M == MemSpace::device)
            {
                resize(&device_states,
                       make_const_ref(params),
                       stream_id,
                       dso.size());
                process_hits.reserve(make_ref(device_states));
            }

            std::mt19937 rng(thread_idx);
            std::uniform_int_distribution<int> sample_wait(0, 500);
            auto random_wait = [&] {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(sample_wait(rng)));
            };

            for (int step : range(num_steps))
            {
                for (auto i : range(dso.size()))
                {
                    TrackSlotId const tid{i};
                    auto& data = host_states.data;
                    data.track_id[tid] = dso.track_id[i];
                    data.detector_id[tid] = dso.detector_id[i];
                    data.energy_deposition[tid]
                        = static_cast<real_type>(step + 1)
                          * dso.energy_deposition[i];
                    data.step_length[tid] = dso.step_length[i];
                    data.points[StepPoint::pre].time[tid]
                        = dso.points[StepPoint::pre].time[i];
                }

                if constexpr (M == MemSpace::device)
                {
                    // "Launch" the step, which enqueues the compaction
                    device_states.data = host_states.data;
                    process_hits(make_ref(device_states));
                    // Geant4 work while the step runs
                    random_wait();
                    // Wait for the compaction, then copy without waiting
                    process_hits.copy_pending_steps();
                    // Geant4 work while the copy runs
                    random_wait();
                    process_hits.process_copied_steps_if_ready();
                }
                else
                {
                    process_hits(make_ref(host_states));
                    random_wait();
                }
            }
            process_hits.process_copied_steps();

            result.num_hits = process_hits.exchange_hits();
            // Detectors are sorted by name: em_calorimeter is first
            result.em_edep = detectors.front()->hits().energy_deposition;
        }
        catch (...)
        {
            result.error = std::current_exception();
        }
    };

    std::vector<std::thread> threads;
    for (int i : range(num_threads))
    {
        threads.emplace_back(run_thread, i);
    }
    for (auto& t : threads)
    {
        t.join();
    }
    return results;
}

//---------------------------------------------------------------------------//
/*!
 * Check that each thread processed every step exactly once, in order.
 *
 * Each step has three hits, and the em_calorimeter energy deposition of step
 * \em s is the base value (0.2 MeV) scaled by <code>s + 1</code>.
 */
void SimpleCmsTest::check_thread_results(
    std::vector<ThreadResult> const& results, int num_steps) const
{
    // Use real_type for single-precision tolerance in float builds
    std::vector<real_type> expected_em_edep;
    for (int step : range(num_steps))
    {
        expected_em_edep.push_back(
            static_cast<real_type>(step + 1) * real_type{0.2});
    }
    for (auto const& result : results)
    {
        if (result.error)
        {
            std::rethrow_exception(result.error);
        }
        EXPECT_EQ(3 * num_steps, result.num_hits);
        EXPECT_VEC_SOFT_EQ(expected_em_edep, result.em_edep);
    }

    // Threads must use their own detectors, not the master thread's
    for (auto const& name : this->detector_volumes())
    {
        EXPECT_TRUE(this->get_hits(name).energy_deposition.empty()) << name;
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, no_touchable)
{
    HitProcessor process_hits = this->make_hit_processor();
    auto& recon = *process_hits.track_reconstruction();

    // Initialize a mock event and primaries for reconstruction.
    SimpleCmsTest::test_cur_event = 1;
    recon.init_event();
    auto* gamma_def = G4ParticleTable::GetParticleTable()->FindParticle(
        pdg::gamma().get());
    ASSERT_NE(nullptr, gamma_def);
    {
        auto primary0 = std::make_unique<G4Track>(
            new G4DynamicParticle(gamma_def, G4ThreeVector(1, 0, 0)),
            0.0,
            G4ThreeVector());
        primary0->SetTrackID(10);
        primary0->SetParentID(0);
        auto const pid0 = recon.acquire(*primary0);
        EXPECT_EQ(0, pid0.unchecked_get());
    }
    {
        auto primary1 = std::make_unique<G4Track>(
            new G4DynamicParticle(gamma_def, G4ThreeVector(0, 1, 0)),
            0.0,
            G4ThreeVector());
        primary1->SetTrackID(20);
        primary1->SetParentID(5);
        auto const pid1 = recon.acquire(*primary1);
        EXPECT_EQ(1, pid1.unchecked_get());
    }

    auto dso_hits = this->make_dso();
    process_hits(dso_hits);

    // Second hit
    dso_hits.energy_deposition = {
        MevEnergy{0.4},
        MevEnergy{0.5},
        MevEnergy{0.6},
    };
    dso_hits.step_length = {
        from_cm(1.0),
        from_cm(2.1),
        from_cm(3.1),
    };

    process_hits(dso_hits);
    recon.clear();

    {
        auto& result = this->get_hits("si_tracker");
        SimpleHitsResult ref = result;
        ref.energy_deposition = {1.0 * 0.1, 1.0 * 0.4};
        ref.step_length = {0.1, 1.0};
        ref.weight = {1.0, 1.0};
        ref.pre_energy = {0, 0};
        ref.pre_time = {0.5, 0.5};
        ref.pre_pos = {100, 0, 0, 100, 0, 0};
        ref.pre_dir = {1, 0, 0, 1, 0, 0};
        ref.post_time = {1, 1};
        ref.post_energy = {0.9, 0.9};
        ref.post_pos = {105, 0, 0, 105, 0, 0};
        ref.post_dir = {1, 0, 0, 1, 0, 0};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto& result = this->get_hits("em_calorimeter");
        SimpleHitsResult ref = result;
        ref.energy_deposition = {0.5 * 0.2, 0.5 * 0.5};
        ref.particle = {"e-", "e-"};
        ref.weight = {0.5, 0.5};
        ref.pre_energy = {0, 0};
        ref.pre_time = {0.15, 0.15};
        ref.pre_pos = {0, 150, 10, 0, 150, 10};
        ref.pre_dir = {0, 1, 0, 0, 1, 0};
        ref.post_time = {0.2, 0.2};
        ref.post_energy = {1.8, 1.8};
        ref.post_pos = {0, 155, 10, 0, 155, 10};
        ref.post_dir = {0, 1, 0, 0, 1, 0};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto& result = this->get_hits("had_calorimeter");
        SimpleHitsResult ref = result;
        ref.energy_deposition = {0.8 * 0.3, 0.8 * 0.6};
        ref.particle = {"gamma", "gamma"};
        ref.weight = {0.8, 0.8};
        ref.pre_energy = {0, 0};
        ref.pre_time = {25, 25};
        ref.pre_pos = {0, 200, -20, 0, 200, -20};
        ref.pre_dir = {0, 0, -1, 0, 0, -1};
        ref.post_time = {30, 30};
        ref.post_energy = {2.7, 2.7};
        ref.post_pos = {0, 205, -20, 0, 205, -20};
        ref.post_dir = {0, 0, -1, 0, 0, -1};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, pre_step_only)
{
    // An unselected post-step point can be null even when a hit is valid.
    selection_ = {};
    selection_.energy_deposition = true;
    selection_.points[StepPoint::pre].time = true;

    HitProcessor process_hits = this->make_hit_processor();
    process_hits(this->make_dso());

    auto const& result = this->get_hits("si_tracker");
    ASSERT_EQ(1, result.energy_deposition.size());
    EXPECT_SOFT_EQ(0.1_r, result.energy_deposition.front());
    ASSERT_EQ(1, result.pre_time.size());
    EXPECT_SOFT_EQ(0.5_r, result.pre_time.front());
    EXPECT_EQ(result.post_energy.size(), result.post_time.size());
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, TEST_IF_CELER_DEVICE(deferred_device))
{
    selection_ = {};
    selection_.energy_deposition = true;
    selection_.step_length = true;
    selection_.points[StepPoint::pre].time = true;

    HitProcessor process_hits = this->make_hit_processor();
    EXPECT_FALSE(process_hits.has_pending_steps());
    process_pending(process_hits);
    EXPECT_EQ(0, process_hits.exchange_hits());
    auto const dso = this->make_dso();

    HostVal<StepParamsData> params;
    params.selection = selection_;
    std::vector<DetectorId> detector_map{DetectorId{0}};
    make_builder(&params.detector)
        .insert_back(detector_map.begin(), detector_map.end());

    StepStateData<Ownership::value, MemSpace::host> host_states;
    resize(&host_states, make_const_ref(params), StreamId{0}, dso.size());
    for (auto i : range(dso.size()))
    {
        TrackSlotId const tid{i};
        host_states.data.track_id[tid] = dso.track_id[i];
        host_states.data.detector_id[tid] = dso.detector_id[i];
        host_states.data.energy_deposition[tid] = dso.energy_deposition[i];
        host_states.data.step_length[tid] = dso.step_length[i];
        host_states.data.points[StepPoint::pre].time[tid]
            = dso.points[StepPoint::pre].time[i];
    }

    celeritas::device().create_streams(1);
    StepStateData<Ownership::value, MemSpace::device> device_states;
    resize(&device_states, make_const_ref(params), StreamId{0}, dso.size());
    device_states.data = host_states.data;

    // Allocate the pinned output at setup rather than on the first step
    process_hits.reserve(make_ref(device_states));
    EXPECT_FALSE(process_hits.has_pending_steps());

    process_hits(make_ref(device_states));
    EXPECT_TRUE(process_hits.has_pending_steps());
    EXPECT_TRUE(this->get_hits("si_tracker").energy_deposition.empty());
    EXPECT_TRUE(this->get_hits("em_calorimeter").energy_deposition.empty());
    EXPECT_TRUE(this->get_hits("had_calorimeter").energy_deposition.empty());

    if (CELERITAS_DEBUG)
    {
        EXPECT_THROW(process_hits(make_ref(device_states)), DebugError);
        EXPECT_THROW(process_hits.exchange_hits(), DebugError);
        // Reallocating could free memory that pending copies target
        EXPECT_THROW(process_hits.reserve(make_ref(device_states)), DebugError);
        EXPECT_TRUE(process_hits.has_pending_steps());
    }

    process_pending(process_hits);
    EXPECT_FALSE(process_hits.has_pending_steps());
    EXPECT_EQ(3, process_hits.exchange_hits());
    EXPECT_EQ(1, this->get_hits("si_tracker").energy_deposition.size());
    EXPECT_EQ(1, this->get_hits("em_calorimeter").energy_deposition.size());
    EXPECT_EQ(1, this->get_hits("had_calorimeter").energy_deposition.size());

    // Reuse the same state with different data to catch replay of stale hits.
    for (auto i : range(dso.size()))
    {
        host_states.data.energy_deposition[TrackSlotId{i}]
            = 2 * dso.energy_deposition[i];
    }
    device_states.data = host_states.data;
    process_hits(make_ref(device_states));
    EXPECT_TRUE(process_hits.has_pending_steps());
    EXPECT_EQ(1, this->get_hits("em_calorimeter").energy_deposition.size());
    process_pending(process_hits);
    EXPECT_FALSE(process_hits.has_pending_steps());
    EXPECT_EQ(3, process_hits.exchange_hits());
    {
        auto const& result = this->get_hits("si_tracker");
        SimpleHitsResult ref = result;
        ref.energy_deposition = {real_type{0.1}, real_type{0.2}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto const& result = this->get_hits("em_calorimeter");
        SimpleHitsResult ref = result;
        ref.energy_deposition = {real_type{0.2}, real_type{0.4}};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto const& result = this->get_hits("had_calorimeter");
        SimpleHitsResult ref = result;
        ref.energy_deposition = {real_type{0.3}, real_type{0.6}};
        EXPECT_REF_EQ(ref, result);
    }
    EXPECT_TRUE(this->get_hits("world").energy_deposition.empty());

    // An empty batch must clear the previous output, not deliver it again.
    for (auto i : range(dso.size()))
    {
        host_states.data.detector_id[TrackSlotId{i}] = {};
    }
    device_states.data = host_states.data;
    process_hits(make_ref(device_states));
    EXPECT_TRUE(process_hits.has_pending_steps());
    process_pending(process_hits);
    EXPECT_FALSE(process_hits.has_pending_steps());
    EXPECT_EQ(0, process_hits.exchange_hits());
    process_pending(process_hits);
    EXPECT_EQ(0, process_hits.exchange_hits());
    EXPECT_EQ(2, this->get_hits("si_tracker").energy_deposition.size());
    EXPECT_EQ(2, this->get_hits("em_calorimeter").energy_deposition.size());
    EXPECT_EQ(2, this->get_hits("had_calorimeter").energy_deposition.size());

    // Compaction is enqueued by the call operator, so the gathered step data
    // can be overwritten before the pending steps are processed.
    for (auto i : range(dso.size()))
    {
        host_states.data.detector_id[TrackSlotId{i}] = dso.detector_id[i];
    }
    device_states.data = host_states.data;
    process_hits(make_ref(device_states));
    celeritas::device().stream(StreamId{0}).sync();
    for (auto i : range(dso.size()))
    {
        host_states.data.detector_id[TrackSlotId{i}] = {};
    }
    device_states.data = host_states.data;
    process_pending(process_hits);
    EXPECT_EQ(3, process_hits.exchange_hits());
    EXPECT_EQ(3, this->get_hits("si_tracker").energy_deposition.size());
    EXPECT_EQ(3, this->get_hits("em_calorimeter").energy_deposition.size());
    EXPECT_EQ(3, this->get_hits("had_calorimeter").energy_deposition.size());

    // Copied steps can be processed after the next compaction is enqueued,
    // which waits on the device for the copy to finish reading the scratch
    // space. Copying the next step processes the previous one first.
    auto set_device_steps = [&](real_type edep_scale) {
        for (auto i : range(dso.size()))
        {
            TrackSlotId const tid{i};
            host_states.data.detector_id[tid] = dso.detector_id[i];
            host_states.data.energy_deposition[tid]
                = edep_scale * dso.energy_deposition[i];
        }
        device_states.data = host_states.data;
    };
    set_device_steps(3);
    process_hits(make_ref(device_states));
    celeritas::device().stream(StreamId{0}).sync();
    process_hits.copy_pending_steps();
    EXPECT_TRUE(process_hits.has_pending_steps());
    EXPECT_EQ(3, this->get_hits("em_calorimeter").energy_deposition.size());

    set_device_steps(4);
    process_hits(make_ref(device_states));
    EXPECT_TRUE(process_hits.has_pending_steps());
    EXPECT_EQ(3, this->get_hits("em_calorimeter").energy_deposition.size());
    celeritas::device().stream(StreamId{0}).sync();
    process_hits.copy_pending_steps();
    EXPECT_TRUE(process_hits.has_pending_steps());
    EXPECT_EQ(4, this->get_hits("em_calorimeter").energy_deposition.size());

    celeritas::device().memcpy_stream(StreamId{0}).sync();
    process_hits.process_copied_steps_if_ready();
    EXPECT_FALSE(process_hits.has_pending_steps());
    EXPECT_EQ(6, process_hits.exchange_hits());
    {
        auto const& result = this->get_hits("em_calorimeter");
        SimpleHitsResult ref = result;
        ref.energy_deposition = {real_type{0.2},
                                 real_type{0.4},
                                 real_type{0.4},
                                 real_type{0.6},
                                 real_type{0.8}};
        EXPECT_REF_EQ(ref, result);
    }

    // Processing without a pending copy is a null-op
    process_hits.process_copied_steps();
    process_hits.process_copied_steps_if_ready();
    EXPECT_EQ(0, process_hits.exchange_hits());
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, multithread_host)
{
    constexpr int num_steps = 20;
    this->check_thread_results(this->run_threads<MemSpace::host>(4, num_steps),
                               num_steps);
}

TEST_F(SimpleCmsTest, TEST_IF_CELER_DEVICE(multithread_device))
{
    constexpr int num_steps = 20;
    this->check_thread_results(
        this->run_threads<MemSpace::device>(4, num_steps), num_steps);
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, touchable_midvol)
{
    selection_.particle_id = false;
    selection_.primary_id = false;
    selection_.parent_id = false;
    selection_.generation = false;
    locate_touchable_ = {true, false};
    HitProcessor process_hits = this->make_hit_processor();
    auto dso_hits = this->make_dso();
    process_hits(dso_hits);
    process_hits(dso_hits);

    {
        auto& result = this->get_hits("si_tracker");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"si_tracker_pv", "si_tracker_pv"};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto& result = this->get_hits("em_calorimeter");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"em_calorimeter_pv", "em_calorimeter_pv"};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto& result = this->get_hits("had_calorimeter");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"had_calorimeter_pv", "had_calorimeter_pv"};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, touchable_edgecase)
{
    selection_.particle_id = false;
    selection_.primary_id = false;
    selection_.parent_id = false;
    selection_.generation = false;
    locate_touchable_ = {true, false};
    HitProcessor process_hits = this->make_hit_processor();

    auto dso_hits = this->make_dso();
    auto& pos = dso_hits.points[StepPoint::pre].pos;
    auto& dir = dso_hits.points[StepPoint::pre].dir;
    pos = {
        from_cm(Real3{30, 0, 0}),
        from_cm(Real3{0, 125, 10}),
        from_cm(Real3{0, 175, -20}),
    };
    process_hits(dso_hits);

    pos = {
        from_cm(Real3{-120.20472398905, 34.290294993135, -58.348475076307}),
        from_cm(Real3{-58.042349740868, -165.09417202481, -315.41125902053}),
        from_cm(Real3{0, 275, -20}),
    };
    EXPECT_SOFT_EQ(from_cm(125.0), std::hypot(pos[0][0], pos[0][1]));
    EXPECT_SOFT_EQ(from_cm(175.0), std::hypot(pos[1][0], pos[1][1]));
    dir = {
        {0.39117837162751, -0.78376148752334, -0.48238720157779},
        {0.031769215780742, 0.6378450322959, -0.76950921482729},
        {0, -1, 0},
    };
    process_hits(dso_hits);

    {
        auto& result = this->get_hits("si_tracker");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"si_tracker_pv", "si_tracker_pv"};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto& result = this->get_hits("em_calorimeter");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"em_calorimeter_pv", "em_calorimeter_pv"};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto& result = this->get_hits("had_calorimeter");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"had_calorimeter_pv", "had_calorimeter_pv"};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, touchable_exiting)
{
    locate_touchable_ = {true, true};
    selection_.particle_id = false;
    selection_.primary_id = false;
    selection_.parent_id = false;
    selection_.generation = false;
    selection_.points[StepPoint::pre].time = false;
    selection_.points[StepPoint::post].time = false;
    selection_.points[StepPoint::pre].energy = false;
    selection_.points[StepPoint::post].energy = false;

    HitProcessor process_hits = this->make_hit_processor();
    DetectorStepOutput dso;
    dso.detector_id = {DetectorId{3}, DetectorId{2}};
    dso.weight = {1.0, 1.0};
    dso.energy_deposition = {MevEnergy{1.0}, MevEnergy{10.0}};
    dso.step_length = {
        from_cm(300),
        from_cm(10),
    };
    dso.points[StepPoint::pre].pos = {
        from_cm(Real3{0, 0, 1700}),
        from_cm(Real3{50.0, 0, 690}),
    };
    dso.points[StepPoint::post].pos = {
        from_cm(Real3{0, 0, 2000}),
        from_cm(Real3{50.0, 0, 700}),
    };
    dso.points[StepPoint::pre].dir = dso.points[StepPoint::post].dir
        = {Real3{0, 0, 1}, Real3{0, 0, 1}};

    dso.num_volume_levels = 2;
    auto const& vol_inst = this->volumes()->volume_instance_labels();
    auto wovi = vol_inst.find_unique("world_PV");
    auto sivi = vol_inst.find_unique("si_tracker_pv");
    dso.points[StepPoint::pre].volume_instance_ids = {wovi, {}, wovi, sivi};
    dso.points[StepPoint::post].volume_instance_ids = {{}, {}, wovi, {}};
    process_hits(dso);

    {
        auto& result = this->get_hits("si_tracker");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"si_tracker_pv"};
        ref.post_physvol = {"world_PV"};
        ref.post_status = {"geo"};
        EXPECT_REF_EQ(ref, result);
    }
    {
        auto& result = this->get_hits("world");
        SimpleHitsResult ref = result;
        ref.pre_physvol = {"world_PV"};
        ref.post_physvol = {"<nullptr>"};
        ref.post_status = {"world"};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, with_primary_id)
{
    selection_.primary_id = true;
    HitProcessor process_hits = this->make_hit_processor();
    auto& recon = *process_hits.track_reconstruction();

    // Initialize a mock event so debug builds don't fail the event-ID check
    SimpleCmsTest::test_cur_event = 1;
    recon.init_event();

    // Acquire two primaries: PrimaryId{0} -> track 10, PrimaryId{1} -> track
    // 20
    auto* gamma_def = G4ParticleTable::GetParticleTable()->FindParticle(
        pdg::gamma().get());
    ASSERT_NE(nullptr, gamma_def);

    auto primary0 = std::make_unique<G4Track>(
        new G4DynamicParticle(gamma_def, G4ThreeVector(1, 0, 0)),
        0.0,
        G4ThreeVector());
    primary0->SetTrackID(10);
    primary0->SetParentID(0);
    auto pid0 = recon.acquire(*primary0);
    EXPECT_EQ(0, pid0.unchecked_get());

    auto primary1 = std::make_unique<G4Track>(
        new G4DynamicParticle(gamma_def, G4ThreeVector(0, 1, 0)),
        0.0,
        G4ThreeVector());
    primary1->SetTrackID(20);
    primary1->SetParentID(5);
    auto pid1 = recon.acquire(*primary1);
    EXPECT_EQ(1, pid1.unchecked_get());

    // Run a single batch of hits (dso.primary_id = {PrimaryId{0}, {1}, {1}})
    auto dso = this->make_dso();
    process_hits(dso);
    recon.clear();

    {
        // si_tracker -> PrimaryId{0} -> track_id=10, parent_id=0
        auto& result = this->get_hits("si_tracker");
        SimpleHitsResult ref = result;
        ref.track_id = {10};
        ref.parent_id = {0};
        EXPECT_REF_EQ(ref, result);
    }
    {
        // em_calorimeter -> PrimaryId{1} -> track_id=20, parent_id=5
        auto& result = this->get_hits("em_calorimeter");
        SimpleHitsResult ref = result;
        ref.track_id = {20};
        ref.parent_id = {5};
        EXPECT_REF_EQ(ref, result);
    }
    {
        // had_calorimeter -> secondary (Celeritas track 4) of PrimaryId{1}:
        // its own mapped ID, with the offloaded track as parent
        auto& result = this->get_hits("had_calorimeter");
        SimpleHitsResult ref = result;
        ref.track_id = {std::numeric_limits<int>::max() - 4};
        ref.parent_id = {20};
        EXPECT_REF_EQ(ref, result);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
