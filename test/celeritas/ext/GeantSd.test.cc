//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantSd.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantSd.hh"

#include <memory>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4NistManager.hh>
#include <G4Orb.hh>
#include <G4ParticleDefinition.hh>

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/sys/Device.hh"
#include "geocel/GeantGeoUtils.hh"
#include "geocel/UnitUtils.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/ext/GeantSdOutput.hh"
#include "celeritas/ext/detail/HitProcessor.hh"
#include "celeritas/geo/CoreGeoParams.hh"
#include "celeritas/global/Stepper.hh"
#include "celeritas/inp/Scoring.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/phys/ParticleParams.hh"
#include "celeritas/phys/Primary.hh"
#include "celeritas/user/StepCollector.hh"

#include "SensDetTestBase.hh"
#include "SimpleSensitiveDetector.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class SimpleCmsTest : public SensDetTestBase, public SimpleCmsTestBase
{
  protected:
    void SetUp() override
    {
        sd_setup_.ignore_zero_deposition = false;
        sd_setup_.track = false;

        this->geometry();
        scoped_log_.clear();
    }

    SPConstGeantGeo build_geant_geo(std::string const& filename) const override
    {
        auto result = SensDetTestBase::build_geant_geo(filename);

        // Create unused volume after building geometry
        G4Material* mat
            = G4NistManager::Instance()->FindOrBuildMaterial("G4_AIR");
        SimpleCmsTest::detached_lv = new G4LogicalVolume(
            new G4Orb("unused_solid", 10.0), mat, "unused");

        return result;
    }

    SetStr detector_volumes() const final
    {
        // *Don't* add SD for si_tracker
        return {"em_calorimeter", "had_calorimeter"};
    }

    std::vector<std::string> volume_names(
        std::vector<VolumeId> const& vols) const
    {
        auto const& labels = this->volumes()->volume_labels();

        std::vector<std::string> result;
        for (VolumeId vid : vols)
        {
            result.push_back(labels.at(vid).name);
        }
        return result;
    }

    std::vector<std::string> particle_names(
        GeantSd::VecParticle const& particles) const
    {
        std::vector<std::string> result;
        for (auto* par : particles)
        {
            CELER_ASSERT(par);
            result.push_back(par->GetParticleName());
        }
        return result;
    }

    GeantSd make_hit_manager(bool make_hit_proc = true)
    {
        CELER_EXPECT(!processor_);
        GeantSd result(*this->particle(), sd_setup_, 1);

        if (make_hit_proc)
        {
            processor_ = result.make_local_processor(StreamId{0});
            EXPECT_TRUE(processor_);
        }

        return result;
    }

    std::string get_diagnostics(GeantSd const& hm)
    {
        GeantSdOutput out(
            std::shared_ptr<GeantSd const>(&hm, [](GeantSd const*) {}));
        return to_string(out);
    }

    size_type num_hits() const
    {
        size_type result{};
        for (auto const& [name, detector] : this->detectors())
        {
            result += detector->hits().energy_deposition.size();
        }
        return result;
    }

    template<MemSpace M>
    void test_step_lifecycle();

  protected:
    inp::GeantSd sd_setup_;
    ::celeritas::test::ScopedLogStorer scoped_log_{&celeritas::world_logger()};
    static G4LogicalVolume const* detached_lv;
    GeantSd::SPProcessor processor_;
};

G4LogicalVolume const* SimpleCmsTest::detached_lv{nullptr};

//---------------------------------------------------------------------------//
// Exercise the Stepper, gather actions, and Geant4 sensitive detectors together.
template<MemSpace M>
void SimpleCmsTest::test_step_lifecycle()
{
    auto manager = std::make_shared<GeantSd>(this->make_hit_manager());
    auto collector = StepCollector::make_and_insert(*this->core(), {manager});
    StepperInput input;
    input.params = this->core();
    input.stream_id = StreamId{0};
    input.num_track_slots = 16;
    input.actions = std::make_shared<ActionSequence>(
        *this->action_reg(), ActionSequence::Options{});
    if constexpr (M == MemSpace::device)
    {
        // Core state construction already uses the assigned device stream.
        device().create_streams(1);
    }
    Stepper<M> step(input);

    for (int i = 0; i < 2; ++i)
    {
        step.warm_up();
        EXPECT_FALSE(step.valid());
        ASSERT_FALSE(processor_->has_pending_steps());
        EXPECT_EQ(0, this->num_hits());
        EXPECT_EQ(0, processor_->exchange_hits());
    }

    size_type consumed_hits{};
    auto complete_step = [&] {
        step.wait();
        EXPECT_TRUE(step.ready());
        EXPECT_EQ(M == MemSpace::device, processor_->has_pending_steps());
        if constexpr (M == MemSpace::device)
        {
            // Neither executing the actions nor waiting delivers device hits.
            EXPECT_EQ(consumed_hits, this->num_hits());
        }

        auto const hits_before_get = this->num_hits();
        auto result = step.get();
        EXPECT_FALSE(step.valid());
        EXPECT_EQ(M == MemSpace::device, processor_->has_pending_steps());
        EXPECT_EQ(hits_before_get, this->num_hits());

        processor_->process_pending_steps();
        EXPECT_FALSE(processor_->has_pending_steps());
        EXPECT_EQ(this->num_hits() - consumed_hits,
                  processor_->exchange_hits());
        consumed_hits = this->num_hits();

        // Processing again must not deliver the same batch twice.
        processor_->process_pending_steps();
        EXPECT_EQ(consumed_hits, this->num_hits());
        EXPECT_EQ(0, processor_->exchange_hits());
        return result;
    };

    step.async();
    EXPECT_FALSE(complete_step());
    EXPECT_EQ(0, consumed_hits);

    Primary primary;
    primary.particle_id = this->particle()->find(pdg::electron());
    ASSERT_TRUE(primary.particle_id);
    primary.position = from_cm(Real3{0, 150, 10});
    primary.direction = {0, 0, 1};
    primary.event_id = EventId{0};
    PrimaryId primary_id{0};
    for (real_type energy : {1.0_r, 2.0_r})
    {
        // Drain two showers through the same step state and hit buffers.
        auto const previous_hits = consumed_hits;
        primary.energy = units::MevEnergy{energy};
        primary.primary_id = primary_id++;
        step.async({&primary, 1});
        auto result = complete_step();
        EXPECT_EQ(1, result.generated);
        EXPECT_EQ(1, result.active);
        ASSERT_EQ(previous_hits + 1, consumed_hits);
        auto const& energies
            = this->detectors().at("em_calorimeter")->hits().pre_energy;
        ASSERT_FALSE(energies.empty());
        EXPECT_SOFT_EQ(energy, energies.back());
        for (int num_steps = 0; result; ++num_steps)
        {
            ASSERT_LT(num_steps, 256);
            step.async();
            result = complete_step();
        }
        EXPECT_GT(consumed_hits, previous_hits);

        // A terminal result still has a device callback to consume. The next
        // empty step must not replay hits left in the previous host buffer.
        auto const final_hits = consumed_hits;
        step.async();
        EXPECT_FALSE(complete_step());
        EXPECT_EQ(final_hits, consumed_hits);
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, step_lifecycle_host)
{
    this->test_step_lifecycle<MemSpace::host>();
}

TEST_F(SimpleCmsTest, TEST_IF_CELER_DEVICE(step_lifecycle_device))
{
    this->test_step_lifecycle<MemSpace::device>();
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, no_change)
{
    GeantSd man = this->make_hit_manager();

    EXPECT_EQ(0, man.geant_particles().size());
    EXPECT_EQ(2, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());
    static char const* const expected_vnames[]
        = {"em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;

    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"hit-manager","locate_touchable":[true,true],"lv_name":["em_calorimeter","had_calorimeter"],"sd_name":["em_calorimeter","had_calorimeter"],"sd_type":["celeritas::test::SimpleSensitiveDetector","celeritas::test::SimpleSensitiveDetector"],"vol_id":[2,3]})json",
        this->get_diagnostics(man));
}

TEST_F(SimpleCmsTest, delete_one)
{
    // Create tracks for each particle type
    sd_setup_.track = true;

    sd_setup_.skip_volumes = find_geant_volumes({"had_calorimeter"});
    GeantSd man = this->make_hit_manager();

    // Check volumes
    EXPECT_EQ(1, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());
    static char const* const expected_vnames[] = {"em_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);

    // Check particles
    auto pnames = this->particle_names(man.geant_particles());
    static std::string const expected_pnames[] = {"gamma", "e-", "e+"};
    EXPECT_VEC_EQ(expected_pnames, pnames);

    // Check log
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;

    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"hit-manager","locate_touchable":[true,true],"lv_name":["em_calorimeter"],"sd_name":["em_calorimeter"],"sd_type":["celeritas::test::SimpleSensitiveDetector"],"vol_id":[2]})json",
        this->get_diagnostics(man));
}

TEST_F(SimpleCmsTest, add_duplicate)
{
    sd_setup_.force_volumes = find_geant_volumes({"em_calorimeter"});
    scoped_log_.level(LogLevel::debug);
    GeantSd man = this->make_hit_manager();
    scoped_log_.level(Logger::default_level());

    EXPECT_EQ(2, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());

    static char const* const expected_vnames[]
        = {"em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
    if (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_VECGEOM)
    {
        static char const* const expected_log_messages[] = {
            R"(Mapped sensitive detector "em_calorimeter" on logical volume "em_calorimeter"@0x0 (ID=2) to volume ID 2)",
            R"(Mapped sensitive detector "had_calorimeter" on logical volume "had_calorimeter"@0x0 (ID=3) to volume ID 3)",
            R"(Ignored duplicate logical volume "em_calorimeter"@0x0 (ID=2))",
            "Setting up thread-local hit processor for 2 sensitive detectors",
        };
        EXPECT_VEC_EQ(expected_log_messages, scoped_log_.messages());
        static char const* const expected_log_levels[]
            = {"debug", "debug", "debug", "debug"};
        EXPECT_VEC_EQ(expected_log_levels, scoped_log_.levels());
    }

    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"hit-manager","locate_touchable":[true,true],"lv_name":["em_calorimeter","had_calorimeter"],"sd_name":["em_calorimeter","had_calorimeter"],"sd_type":["celeritas::test::SimpleSensitiveDetector","celeritas::test::SimpleSensitiveDetector"],"vol_id":[2,3]})json",
        this->get_diagnostics(man));
}

TEST_F(SimpleCmsTest, add_one)
{
    sd_setup_.force_volumes = find_geant_volumes({"si_tracker"});
    // Since we're asking for a volume that doesn't currently have an
    // SD attached, we can't make the hit processor
    GeantSd man = this->make_hit_manager(/* make_hit_proc = */ false);

    EXPECT_EQ(3, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());

    static char const* const expected_vnames[]
        = {"si_tracker", "em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
    EXPECT_JSON_EQ(
        R"json({"_category":"internal","_label":"hit-manager","locate_touchable":[true,true],"lv_name":["si_tracker","em_calorimeter","had_calorimeter"],"sd_name":[null,"em_calorimeter","had_calorimeter"],"sd_type":[null,"celeritas::test::SimpleSensitiveDetector","celeritas::test::SimpleSensitiveDetector"],"vol_id":[1,2,3]})json",
        this->get_diagnostics(man));
}

TEST_F(SimpleCmsTest, no_detector)
{
    // No detectors
    sd_setup_.skip_volumes
        = find_geant_volumes({"em_calorimeter", "had_calorimeter"});
    EXPECT_THROW(this->make_hit_manager(), celeritas::RuntimeError);
    EXPECT_TRUE(scoped_log_.empty()) << scoped_log_;
}

TEST_F(SimpleCmsTest, detached_detector)
{
    // Detector for LV that isn't in the world tree
    sd_setup_.skip_volumes = {};
    sd_setup_.force_volumes = std::unordered_set<G4LogicalVolume const*>{
        SimpleCmsTest::detached_lv};
    EXPECT_THROW(
        try {
            this->make_hit_manager();
        } catch (celeritas::RuntimeError const& e) {
            EXPECT_EQ(
                R"(failed to find Geant4 volume(s) "unused" while mapping sensitive detectors)",
                e.details().what);
            throw;
        },
        celeritas::RuntimeError);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
