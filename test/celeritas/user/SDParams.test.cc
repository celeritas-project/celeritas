//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/SDParams.test.cc
//------------

#include "celeritas/user/SDParams.hh"

#include <memory>
#include <string_view>
#include <vector>

#include "corecel/Assert.hh"
#include "celeritas/GlobalGeoTestBase.hh"
#include "celeritas/OnlyCoreTestBase.hh"
#include "celeritas/OnlyGeoTestBase.hh"
#include "celeritas/geo/GeoParams.hh"

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

    auto params = std::make_shared<SDParams>(detector_labels,
                                             *(this->build_geometry()));
    for (auto d_id : range(DetectorId{params->size()}))
    {
        auto v_id = params->detector_to_volume_id(d_id);
        EXPECT_EQ(detector_labels[d_id.get()],
                  this->geometry()->volumes().at(v_id).name);
        EXPECT_EQ(d_id, params->volume_to_detector_id(VolumeId{v_id}));
    }
}
}  // namespace test
}  // namespace celeritas
