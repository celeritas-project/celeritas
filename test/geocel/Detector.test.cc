//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Detector.test.cc
//---------------------------------------------------------------------------//

#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/Assert.hh"
#include "corecel/OpaqueIdIO.hh"
#include "geocel/DetectorParams.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"

#include "celeritas_test.hh"
#include "g4/GeantGeoTestBase.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class DetectorParamsTest : public GeantGeoTestBase
{
  public:
    using VecStr = std::vector<std::string>;

    std::string_view gdml_basename() const override
    {
        return "testem3-flat"sv;
    }

    // This assigns a unique detector to each volume label passed in
    inp::Detectors find_volumes(VecStr const& labels)
    {
        auto const& vols = this->volumes();
        CELER_VALIDATE(vols, << "volumes were not set up");

        inp::Detectors result;
        auto const& all_vol_labels = vols->volume_labels();
        for (auto const& name : labels)
        {
            VolumeId vol_id = all_vol_labels.find_unique(name);
            CELER_VALIDATE(vol_id, << "invalid detector volume " << name);
            inp::Detector detector;
            detector.label = name;
            detector.volumes.push_back(vol_id);
            result.detectors.push_back(detector);
        }
        return result;
    }
};

TEST_F(DetectorParamsTest, no_label_test)
{
    auto const& vols = *this->volumes();
    inp::Detectors detectors;
    DetectorParams params(vols, detectors);
    EXPECT_EQ(0, params.size());
}

TEST_F(DetectorParamsTest, detector_test)
{
    VecStr detector_labels = {"gap_10", "absorber_40", "absorber_31"};

    auto const& vols = *this->volumes();

    DetectorParams params(vols, this->find_volumes(detector_labels));
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(3, params.size());

    for (auto vol_id : range(VolumeId{vols.num_volumes()}))
    {
        auto det_id = params.detector_id(vol_id);
        if (det_id)
        {
            auto const& det_vols = params.volume_id(det_id);
            EXPECT_TRUE(std::find(det_vols.begin(), det_vols.end(), vol_id)
                        != det_vols.end())
                << "did not find volume " << vol_id
                << " in list of volumes for detector " << det_id;
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
