//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/HitManager.test.cc
//---------------------------------------------------------------------------//
#include "accel/detail/HitManager.hh"

#include <G4LogicalVolume.hh>
#include <G4LogicalVolumeStore.hh>

#include "celeritas/SimpleCmsTestBase.hh"
#include "celeritas/geo/GeoParams.hh"
#include "accel/SDTestBase.hh"
#include "accel/SetupOptions.hh"

#include "celeritas_test.hh"

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
    void SetUp() override
    {
        sd_setup_.enabled = true;
        sd_setup_.ignore_zero_deposition = false;
    }

    SetStr detector_volumes() const final
    {
        // *Don't* add SD for si_tracker
        return {"em_calorimeter", "had_calorimeter"};
    }

    std::vector<std::string>
    volume_names(std::vector<VolumeId> const& vols) const
    {
        auto const& geo = *this->geometry();

        std::vector<std::string> result;
        for (VolumeId vid : vols)
        {
            result.push_back(geo.id_to_label(vid).name);
        }
        return result;
    }

    std::unordered_set<G4LogicalVolume const*>
    find_lvs(std::unordered_set<std::string> const& inp)
    {
        EXPECT_TRUE(this->geometry());

        std::unordered_set<G4LogicalVolume const*> result;
        for (G4LogicalVolume* lv : *G4LogicalVolumeStore::GetInstance())
        {
            if (inp.find(lv->GetName()) != inp.end())
            {
                result.insert(lv);
            }
        }
        return result;
    }

    SDSetupOptions sd_setup_;
};

TEST_F(SimpleCmsTest, no_change)
{
    HitManager man(*this->geometry(), sd_setup_);

    EXPECT_EQ(2, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());
    static char const* const expected_vnames[]
        = {"em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
}

TEST_F(SimpleCmsTest, delete_one)
{
    sd_setup_.skip_volumes = this->find_lvs({"had_calorimeter"});
    HitManager man(*this->geometry(), sd_setup_);

    EXPECT_EQ(1, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());

    static char const* const expected_vnames[] = {"em_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
}

TEST_F(SimpleCmsTest, add_one)
{
    sd_setup_.force_volumes = this->find_lvs({"si_tracker"});
    HitManager man(*this->geometry(), sd_setup_);

    EXPECT_EQ(3, man.geant_vols()->size());
    auto vnames = this->volume_names(man.celer_vols());

    static char const* const expected_vnames[]
        = {"si_tracker", "em_calorimeter", "had_calorimeter"};
    EXPECT_VEC_EQ(expected_vnames, vnames);
}

TEST_F(SimpleCmsTest, errors)
{
    {
        // No detectors
        sd_setup_.skip_volumes
            = this->find_lvs({"em_calorimeter", "had_calorimeter"});
        EXPECT_THROW(HitManager(*this->geometry(), sd_setup_),
                     celeritas::RuntimeError);
    }
    {
        // Nonexistent forced detector (nullptr in this case)
        sd_setup_.skip_volumes = {};
        sd_setup_.force_volumes = {nullptr};
        EXPECT_THROW(HitManager(*this->geometry(), sd_setup_),
                     celeritas::RuntimeError);
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace detail
}  // namespace celeritas
