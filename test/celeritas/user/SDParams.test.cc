//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.test.cc
//------------

#include "celeritas/user/SDParams.hh"

#include <memory>
#include <string>
#include <string_view>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Types.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/OnlyCoreTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"
#include "celeritas/geo/GeoParams.hh"

#include "TestMacros.hh"
#include "celeritas_test.hh"
#include "gtest/gtest.h"

namespace celeritas
{

namespace test
{
class SDParamsTest : public GlobalGeoTestBase,
                     public OnlyGeoTestBase,
                     public OnlyCoreTestBase
{
  public:
    using VecLabel = std::vector<Label>;
    std::string_view geometry_basename() const override
    {
        return "testem3-flat"sv;
    }

  protected:
};

TEST_F(SDParamsTest, empty_constructor_test)
{
    auto test_detector_params = std::make_shared<SDParams>();

    auto det_id = DetectorId{0};
    auto vol_id = VolumeId{0};
    EXPECT_THROW(test_detector_params->volume_to_detector_id(vol_id),
                 celeritas::DebugError);
    EXPECT_THROW(test_detector_params->detector_to_volume_id(det_id),
                 celeritas::DebugError);
}

TEST_F(SDParamsTest, invalid_label_test)
{
    VecLabel detector_labels = {"invalid_label"};

    EXPECT_THROW(auto test_detector_params = std::make_shared<SDParams>(
                     detector_labels, *(this->build_geometry())),
                 celeritas::RuntimeError);
}

TEST_F(SDParamsTest, detector_test)
{
    VecLabel detector_labels = {"gap_10", "absorber_40", "absorber_31"};

    auto test_detector_params = std::make_shared<SDParams>(
        detector_labels, *(this->build_geometry()));

    auto v_id0 = test_detector_params->detector_to_volume_id(DetectorId{0});
    auto v_id1 = test_detector_params->detector_to_volume_id(DetectorId{1});
    auto v_id2 = test_detector_params->detector_to_volume_id(DetectorId{2});

    auto d_id0 = test_detector_params->volume_to_detector_id(VolumeId{v_id0});
    auto d_id1 = test_detector_params->volume_to_detector_id(VolumeId{v_id1});
    auto d_id2 = test_detector_params->volume_to_detector_id(VolumeId{v_id2});

    EXPECT_EQ(0, d_id0.get());
    EXPECT_EQ(1, d_id1.get());
    EXPECT_EQ(2, d_id2.get());
}
}  // namespace test
}  // namespace celeritas