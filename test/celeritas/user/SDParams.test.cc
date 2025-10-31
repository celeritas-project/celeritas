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
#include "geocel/inp/Model.hh"
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

// TODO:: reformat this for empty detectors in model
TEST_F(SDParamsTest, TEST_IF_CELERITAS_DEBUG(invalid_label_test))
{
    auto const& geo = *this->geometry();
    inp::Detectors detectors;
    EXPECT_THROW(SDParams(geo, detectors), celeritas::RuntimeError);
}

TEST_F(SDParamsTest, detector_test)
{
    VecStr detector_labels = {"gap_10", "absorber_40", "absorber_31"};

    auto const& geo = *this->geometry();
    auto const& impl_volumes = this->geometry()->impl_volumes();

    SDParams params(geo, this->find_volumes(detector_labels));
    EXPECT_FALSE(params.empty());
    EXPECT_EQ(3, params.size());

    for (auto iv_id :
         range(ImplVolumeId{static_cast<size_type>(impl_volumes.size())}))
    {
        auto det_id = params.volume_to_detector_id(iv_id);
        if (det_id)
        {
            EXPECT_EQ(detector_labels[det_id.get()],
                      this->geometry()->impl_volumes().at(iv_id).name);

            std::vector<VolumeId> vol_ids
                = params.detector_to_volume_id(det_id);
            auto vol_id = id_cast<VolumeId>(iv_id.get());
            EXPECT_NE(std::find_if(vol_ids.begin(),
                                   vol_ids.end(),
                                   [vol_id](VolumeId const& v_id) {
                                       return v_id.get() == vol_id.get();
                                   }),
                      vol_ids.end());
        }
    }
}

}  // namespace test
}  // namespace celeritas
