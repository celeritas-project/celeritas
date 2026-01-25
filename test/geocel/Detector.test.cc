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
#include "geocel/DetectorParams.hh"
#include "geocel/VolumeParams.hh"

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
    using VecVolId = std::vector<VolumeId>;

    std::string_view gdml_basename() const override
    {
        return "testem3-flat"sv;
    }

    VecVolId find_volumes(VecStr const& labels)
    {
        auto const& vols = this->volumes();
        CELER_VALIDATE(vols, << "volumes were not set up");

        VecVolId result;
        auto const& all_vol_labels = vols->volume_labels();
        for (auto const& name : labels)
        {
            result.push_back(all_vol_labels.find_unique(name));
            CELER_VALIDATE(result.back(),
                           << "invalid detector volume " << name);
        }
        return result;
    }
};

TEST_F(DetectorParamsTest, empty_constructor_test)
{
    DetectorParams params;
    EXPECT_TRUE(params.empty());

    if (CELERITAS_DEBUG)
    {
        auto det_id = DetectorId{0};
        auto vol_id = ImplVolumeId{0};
        EXPECT_THROW(params.volume_to_detector_id(vol_id),
                     celeritas::DebugError);
        EXPECT_THROW(params.detector_to_volume_id(det_id),
                     celeritas::DebugError);
    }
}

TEST_F(DetectorParamsTest, TEST_IF_CELERITAS_DEBUG(invalid_label_test))
{
    auto const& geo = *this->geometry();
    EXPECT_THROW(DetectorParams(geo, {VolumeId{}}), celeritas::RuntimeError);
}

TEST_F(DetectorParamsTest, detector_test)
{
    VecStr detector_labels = {"gap_10", "absorber_40", "absorber_31"};

    auto const& geo = *this->geometry();
    auto const& impl_volumes = this->geometry()->impl_volumes();

    DetectorParams params(geo, this->find_volumes(detector_labels));
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(3, params.size());

    for (auto iv_id : range(ImplVolumeId{impl_volumes.size()}))
    {
        auto det_id = params.volume_to_detector_id(iv_id);
        if (det_id)
        {
            EXPECT_EQ(detector_labels[det_id.get()],
                      this->geometry()->impl_volumes().at(iv_id).name);
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
