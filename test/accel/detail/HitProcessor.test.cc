//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.test.cc
//---------------------------------------------------------------------------//
#include "accel/detail/HitProcessor.hh"

#include <G4ParticleTable.hh>

#include "geocel/UnitUtils.hh"
#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/geo/GeoParams.hh"
#include "celeritas/phys/PDGNumber.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"
#include "accel/SDTestBase.hh"
#include "accel/SimpleSensitiveDetector.hh"

#include "celeritas_test.hh"

using celeritas::test::from_cm;
using celeritas::test::SimpleHitsResult;
using celeritas::units::MevEnergy;

namespace celeritas
{
namespace detail
{
namespace test
{

//---------------------------------------------------------------------------//
class SimpleCmsTest : public ::celeritas::test::SDTestBase,
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

  protected:
    StepSelection selection_;
    bool locate_touchable_{false};
};

//---------------------------------------------------------------------------//
void SimpleCmsTest::SetUp()
{
    // Create default step selection (see HitManager)
    selection_.energy_deposition = true;
    selection_.step_length = true;
    selection_.points[StepPoint::pre].energy = true;
    selection_.points[StepPoint::pre].pos = true;
    selection_.points[StepPoint::post].time = true;
    selection_.particle = true;
}

auto SimpleCmsTest::detector_volumes() const -> SetStr
{
    return {
        "em_calorimeter",
        "had_calorimeter",
        "si_tracker",
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
    if (!selection_.particle)
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
    if (locate_touchable_)
    {
        if constexpr (CELERITAS_CORE_GEO == CELERITAS_CORE_GEO_ORANGE)
        {
            selection_.points[StepPoint::pre].dir = true;
            selection_.points[StepPoint::pre].pos = true;
        }
        else
        {
            selection_.points[StepPoint::pre].volume_instance_ids = true;
        }
    }
    return HitProcessor{this->make_detector_volumes(),
                        this->geometry(),
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
    dso.detector = {
        DetectorId{2},  // si_tracker
        DetectorId{0},  // em_calorimeter
        DetectorId{1},  // had_calorimeter
    };
    dso.track_id = {
        TrackId{0},
        TrackId{2},
        TrackId{4},
    };
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
    if (selection_.particle)
    {
        dso.particle = {
            ParticleId{2},
            ParticleId{1},
            ParticleId{0},
        };
    }
    if (selection_.points[StepPoint::pre].volume_instance_ids)
    {
        // Note: the volumes correspond to simple-cms and the detector IDs
        // above
        dso.volume_instance_depth = 2;
        auto const& vi_names = this->geometry()->volume_instances();
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
TEST_F(SimpleCmsTest, no_touchable)
{
    HitProcessor process_hits = this->make_hit_processor();
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

    {
        auto& result = this->get_hits("si_tracker");
        static real_type const expected_energy_deposition[] = {0.1, 0.4};
        EXPECT_VEC_SOFT_EQ(expected_energy_deposition,
                           result.energy_deposition);
        static real_type const expected_step_length[] = {0.1, 1.0};
        EXPECT_VEC_SOFT_EQ(expected_step_length, result.step_length);
        static real_type const expected_pre_energy[] = {0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);
        static real_type const expected_pre_pos[] = {100, 0, 0, 100, 0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);
        static real_type const expected_post_time[] = {1, 1};
        EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);
    }
    {
        auto& result = this->get_hits("em_calorimeter");
        static real_type const expected_energy_deposition[] = {0.2, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_energy_deposition,
                           result.energy_deposition);
        static char const* const expected_particle[] = {"e-", "e-"};
        EXPECT_VEC_EQ(expected_particle, result.particle);
        static real_type const expected_pre_energy[] = {0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);
        static real_type const expected_pre_pos[] = {0, 150, 10, 0, 150, 10};
        EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);
        static real_type const expected_post_time[] = {0.2, 0.2};
        EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);
    }
    {
        auto& result = this->get_hits("had_calorimeter");
        static real_type const expected_energy_deposition[] = {0.3, 0.6};
        EXPECT_VEC_SOFT_EQ(expected_energy_deposition,
                           result.energy_deposition);
        static char const* const expected_particle[] = {"gamma", "gamma"};
        EXPECT_VEC_EQ(expected_particle, result.particle);
        static real_type const expected_pre_energy[] = {0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);
        static real_type const expected_pre_pos[] = {0, 200, -20, 0, 200, -20};
        EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);
        static real_type const expected_post_time[] = {30, 30};
        EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, touchable_midvol)
{
    selection_.particle = false;
    locate_touchable_ = true;
    HitProcessor process_hits = this->make_hit_processor();
    auto dso_hits = this->make_dso();
    process_hits(dso_hits);
    process_hits(dso_hits);

    {
        auto& result = this->get_hits("si_tracker");
        static char const* const expected_pre_physvol[]
            = {"si_tracker_pv", "si_tracker_pv"};
        EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);
    }
    {
        auto& result = this->get_hits("em_calorimeter");
        static char const* const expected_pre_physvol[]
            = {"em_calorimeter_pv", "em_calorimeter_pv"};
        EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);
    }
    {
        auto& result = this->get_hits("had_calorimeter");
        static char const* const expected_pre_physvol[]
            = {"had_calorimeter_pv", "had_calorimeter_pv"};
        EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);
    }
}

//---------------------------------------------------------------------------//
TEST_F(SimpleCmsTest, touchable_edgecase)
{
    locate_touchable_ = true;
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
        static char const* const expected_pre_physvol[]
            = {"si_tracker_pv", "si_tracker_pv"};
        EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);
    }
    {
        auto& result = this->get_hits("em_calorimeter");
        static char const* const expected_pre_physvol[]
            = {"em_calorimeter_pv", "em_calorimeter_pv"};
        EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);
    }
    {
        auto& result = this->get_hits("had_calorimeter");
        static char const* const expected_pre_physvol[]
            = {"had_calorimeter_pv", "had_calorimeter_pv"};
        EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
