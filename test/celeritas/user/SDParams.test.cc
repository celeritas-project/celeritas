//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.test.cc
//---------------------------------------------------------------------------//

#include "celeritas/user/SDParams.hh"

#include <memory>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>

#include "corecel/Assert.hh"
#include "geocel/VolumeParams.hh"
#include "celeritas/GlobalTestBase.hh"
#include "celeritas/OnlyCoreTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"
#include "celeritas/geo/CoreGeoParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class SDParamsTest : public OnlyGeoTestBase
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

TEST_F(SDParamsTest, empty_constructor_test)
{
    SDParams params;
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

TEST_F(SDParamsTest, TEST_IF_CELERITAS_DEBUG(invalid_label_test))
{
    auto const& geo = *this->geometry();
    EXPECT_THROW(SDParams(geo, {VolumeId{}}), celeritas::RuntimeError);
}

TEST_F(SDParamsTest, detector_test)
{
    VecStr detector_labels = {"gap_10", "absorber_40", "absorber_31"};

    auto const& geo = *this->geometry();
    auto const& impl_volumes = this->geometry()->impl_volumes();

    SDParams params(geo, this->find_volumes(detector_labels));
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

}  // namespace test
}  // namespace celeritas
