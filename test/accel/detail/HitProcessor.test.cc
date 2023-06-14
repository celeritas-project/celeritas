//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitProcessor.test.cc
//---------------------------------------------------------------------------//
#include "accel/detail/HitProcessor.hh"

#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <CLHEP/Units/SystemOfUnits.h>
#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>
#include <G4SDManager.hh>
#include <G4VSensitiveDetector.hh>

#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepData.hh"

#include "celeritas_test.hh"

using namespace celeritas::units;

namespace celeritas
{
namespace detail
{
namespace test
{
//---------------------------------------------------------------------------//
struct HitsResult
{
    std::vector<double> energy_deposition;  // [MeV]
    std::vector<double> pre_energy;  // [MeV]
    std::vector<double> pre_pos;  // [cm]
    std::vector<std::string> pre_physvol;
    std::vector<double> post_time;  // [ns]

    void print_expected() const;
};

void HitsResult::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static double const expected_energy_deposition[] = "
         << repr(this->energy_deposition)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_energy_deposition, "
            "result.energy_deposition);\n"

            "static double const expected_pre_energy[] = "
         << repr(this->pre_energy)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);\n"

            "static double const expected_pre_pos[] = "
         << repr(this->pre_pos)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);\n"

            "static char const * const expected_pre_physvol[] = "
         << repr(this->pre_physvol)
         << ";\n"
            "EXPECT_VEC_EQ(expected_pre_physvol, result.pre_physvol);\n"

            "static double const expected_post_time[] = "
         << repr(this->post_time)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);\n"

            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
class SensitiveDetector final : public G4VSensitiveDetector
{
  public:
    explicit SensitiveDetector(std::string const& name)
        : G4VSensitiveDetector(name)
    {
    }

    //! Access hit data
    HitsResult const& hits() const { return hits_; }

    //! Reset hits between tests
    void clear() { hits_ = {}; }

  protected:
    void Initialize(G4HCofThisEvent*) final { this->clear(); }
    bool ProcessHits(G4Step*, G4TouchableHistory*) final;

    HitsResult hits_;
};

//---------------------------------------------------------------------------//
bool SensitiveDetector::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    CELER_EXPECT(step);

    auto* pre_step = step->GetPreStepPoint();
    CELER_ASSERT(pre_step);

    hits_.energy_deposition.push_back(step->GetTotalEnergyDeposit()
                                      / CLHEP::MeV);
    hits_.pre_energy.push_back(pre_step->GetKineticEnergy() / CLHEP::MeV);

    for (int i : range(3))
    {
        hits_.pre_pos.push_back(pre_step->GetPosition()[i] / CLHEP::cm);
    }

    if (auto* touchable = pre_step->GetTouchable())
    {
        auto* vol = touchable->GetVolume();
        hits_.pre_physvol.push_back(vol ? vol->GetName() : "<nullptr>");
    }
    hits_.post_time.push_back(step->GetPostStepPoint()->GetGlobalTime()
                              / CLHEP::ns);
    return true;
}

//---------------------------------------------------------------------------//

class HitProcessorTest : public ::celeritas::test::SimpleCmsTestBase
{
  protected:
    using MapStrSD = std::map<std::string, SensitiveDetector*>;
    using SPConstVecLV
        = std::shared_ptr<const std::vector<G4LogicalVolume const*>>;

    void SetUp() override
    {
        // Make sure Geant4 is loaded and detctors are set up
        ASSERT_TRUE(!this->imported_data().volumes.empty());
        this->setup_detectors();

        // Reset detectors between simulations
        for (auto&& kv : detectors())
        {
            CELER_ASSERT(kv.second);
            kv.second->clear();
        }

        // Create default step selection (see HitManager)
        selection_.energy_deposition = true;
        selection_.points[StepPoint::pre].energy = true;
        selection_.points[StepPoint::pre].pos = true;
        selection_.points[StepPoint::post].time = true;
    }

    static MapStrSD& detectors();
    static SPConstVecLV& detector_volumes();
    static void setup_detectors();

    static HitsResult const& get_hits(std::string const& name)
    {
        auto iter = detectors().find(name);
        CELER_ASSERT(iter != detectors().end());
        CELER_ASSERT(iter->second);
        return iter->second->hits();
    }

    DetectorStepOutput make_dso() const;

  protected:
    StepSelection selection_;
};

//---------------------------------------------------------------------------//
auto HitProcessorTest::detectors() -> MapStrSD&
{
    // Non-owning pointers
    static MapStrSD det{
        {"em_calorimeter", nullptr},
        {"had_calorimeter", nullptr},
        {"si_tracker", nullptr},
    };

    return det;
}

//---------------------------------------------------------------------------//
auto HitProcessorTest::detector_volumes() -> SPConstVecLV&
{
    static SPConstVecLV dv;
    return dv;
}

//---------------------------------------------------------------------------//
void HitProcessorTest::setup_detectors()
{
    auto& sp_dv = HitProcessorTest::detector_volumes();
    if (sp_dv)
    {
        // We've already set them up
        return;
    }

    std::vector<G4LogicalVolume const*> dv;

    // Find and set up sensitive detectors
    G4SDManager* sd_manager = G4SDManager::GetSDMpointer();
    G4LogicalVolumeStore* lv_store = G4LogicalVolumeStore::GetInstance();
    CELER_ASSERT(lv_store);

    auto& det = HitProcessorTest::detectors();
    for (G4LogicalVolume* lv : *lv_store)
    {
        CELER_ASSERT(lv);
        auto iter = det.find(lv->GetName());
        if (iter == det.end())
            continue;
        CELER_ASSERT(iter->second == nullptr);

        // Found a volume we want for a detector
        dv.push_back(lv);

        // Create SD, attach to volume, and save a reference to it
        auto sd = std::make_unique<SensitiveDetector>(iter->first);
        lv->SetSensitiveDetector(sd.get());
        iter->second = sd.get();
        sd_manager->AddNewDetector(sd.release());
    }

    for (auto&& [name, sdptr] : det)
    {
        EXPECT_TRUE(sdptr != nullptr) << "for detector " << name;
    }

    // Sort detector volumes by name since these will correspond to the
    // DetectorId
    std::sort(dv.begin(),
              dv.end(),
              [](G4LogicalVolume const* lhs, G4LogicalVolume const* rhs) {
                  return lhs->GetName() < rhs->GetName();
              });
    sp_dv
        = std::make_shared<std::vector<G4LogicalVolume const*>>(std::move(dv));
    CELER_ENSURE(HitProcessorTest::detector_volumes());
}

//---------------------------------------------------------------------------//
DetectorStepOutput HitProcessorTest::make_dso() const
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
    if (selection_.points[StepPoint::post].time)
    {
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
            {100, 0, 0},
            {0, 150, 10},
            {0, 200, -20},
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
    return dso;
}

//---------------------------------------------------------------------------//
TEST_F(HitProcessorTest, no_touchable)
{
    HitProcessor process_hits{detector_volumes(), selection_, false};
    auto dso_hits = this->make_dso();
    process_hits(dso_hits);
    dso_hits.energy_deposition = {
        MevEnergy{0.4},
        MevEnergy{0.5},
        MevEnergy{0.6},
    };
    process_hits(dso_hits);

    {
        auto& result = this->get_hits("si_tracker");
        static double const expected_energy_deposition[] = {0.1, 0.4};
        EXPECT_VEC_SOFT_EQ(expected_energy_deposition,
                           result.energy_deposition);
        static double const expected_pre_energy[] = {0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);
        static double const expected_pre_pos[] = {100, 0, 0, 100, 0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);
        static double const expected_post_time[] = {1, 1};
        EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);
    }
    {
        auto& result = this->get_hits("em_calorimeter");
        static double const expected_energy_deposition[] = {0.2, 0.5};
        EXPECT_VEC_SOFT_EQ(expected_energy_deposition,
                           result.energy_deposition);
        static double const expected_pre_energy[] = {0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);
        static double const expected_pre_pos[] = {0, 150, 10, 0, 150, 10};
        EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);
        static double const expected_post_time[] = {0.2, 0.2};
        EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);
    }
    {
        auto& result = this->get_hits("had_calorimeter");
        static double const expected_energy_deposition[] = {0.3, 0.6};
        EXPECT_VEC_SOFT_EQ(expected_energy_deposition,
                           result.energy_deposition);
        static double const expected_pre_energy[] = {0, 0};
        EXPECT_VEC_SOFT_EQ(expected_pre_energy, result.pre_energy);
        static double const expected_pre_pos[] = {0, 200, -20, 0, 200, -20};
        EXPECT_VEC_SOFT_EQ(expected_pre_pos, result.pre_pos);
        static double const expected_post_time[] = {30, 30};
        EXPECT_VEC_SOFT_EQ(expected_post_time, result.post_time);
    }
}

//---------------------------------------------------------------------------//
TEST_F(HitProcessorTest, touchable_midvol)
{
    selection_.points[StepPoint::pre].dir = true;
    HitProcessor process_hits{detector_volumes(), selection_, true};
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
TEST_F(HitProcessorTest, touchable_edgecase)
{
    selection_.points[StepPoint::pre].dir = true;
    HitProcessor process_hits{detector_volumes(), selection_, true};
    auto dso_hits = this->make_dso();
    auto& pos = dso_hits.points[StepPoint::pre].pos;
    auto& dir = dso_hits.points[StepPoint::pre].dir;
    pos = {
        {30, 0, 0},
        {0, 125, 10},
        {0, 175, -20},
    };
    process_hits(dso_hits);

    pos = {
        {-120.20472398905, 34.290294993135, -58.348475076307},
        {-58.042349740868, -165.09417202481, -315.41125902053},
        {0, 275, -20},
    };
    EXPECT_SOFT_EQ(125.0, std::hypot(pos[0][0], pos[0][1]));
    EXPECT_SOFT_EQ(175.0, std::hypot(pos[1][0], pos[1][1]));
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
