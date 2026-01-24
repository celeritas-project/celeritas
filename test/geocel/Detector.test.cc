//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/Detector.test.cc
//---------------------------------------------------------------------------//

#include "geocel/DetectorParams.hh"
#include "geocel/DetectorView.hh"
#include "geocel/VolumeParams.hh"
#include "geocel/inp/Model.hh"  // IWYU pragma: keep

#include "VolumeTestBase.hh"
#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
class DetectorTest : public ComplexVolumeTestBase
{
  public:
    using VecStr = std::vector<std::string>;

    // Get volume ID by label
    VolumeId vol_id(std::string const& label) const
    {
        return this->volumes().volume_labels().find_unique(label);
    }

    // Build detector input from volume labels
    inp::Detectors
    make_detectors(std::vector<std::pair<std::string, VecStr>> const& det_vols)
    {
        inp::Detectors result;
        for (auto const& [det_label, vol_names] : det_vols)
        {
            inp::Detector detector;
            detector.label = det_label;
            for (auto const& vol_name : vol_names)
            {
                VolumeId v_id = this->vol_id(vol_name);
                CELER_VALIDATE(v_id, << "invalid detector volume " << vol_name);
                detector.volumes.push_back(v_id);
            }
            result.detectors.push_back(std::move(detector));
        }
        return result;
    }
};

TEST_F(DetectorTest, empty)
{
    auto const& vols = this->volumes();
    DetectorParams params(vols, {});
    EXPECT_TRUE(params.empty());
    EXPECT_EQ(0, params.size());
}

TEST_F(DetectorTest, errors)
{
    auto const& vols = this->volumes();

    // Test out-of-range volume ID
    {
        inp::Detectors dets;
        inp::Detector detector;
        detector.label = "bad_detector";
        detector.volumes.push_back(VolumeId{999});
        dets.detectors.push_back(std::move(detector));

        EXPECT_THROW(DetectorParams(vols, std::move(dets)), RuntimeError);
    }

    // Test duplicate volume assignment
    {
        inp::Detectors dets;
        inp::Detector det1;
        det1.label = "det1";
        det1.volumes.push_back(VolumeId{0});
        dets.detectors.push_back(std::move(det1));

        inp::Detector det2;
        det2.label = "det2";
        det2.volumes.push_back(VolumeId{0});  // Same volume as det1
        dets.detectors.push_back(std::move(det2));

        EXPECT_THROW(DetectorParams(vols, std::move(dets)), RuntimeError);
    }
}

TEST_F(DetectorTest, multi_vol)
{
    // Single detector covering multiple volumes
    auto dets = this->make_detectors({{"tracker", {"B", "C", "D"}}});

    auto const& vols = this->volumes();
    DetectorParams params(vols, std::move(dets));

    EXPECT_FALSE(params.empty());
    EXPECT_EQ(1, params.size());

    // Check detector ID
    DetectorId tracker_id{0};

    EXPECT_EQ(tracker_id, params.detector_id(this->vol_id("B")));
    EXPECT_EQ(tracker_id, params.detector_id(this->vol_id("C")));
    EXPECT_EQ(tracker_id, params.detector_id(this->vol_id("D")));

    // Volumes not in detector
    EXPECT_FALSE(params.detector_id(this->vol_id("A")));
    EXPECT_FALSE(params.detector_id(this->vol_id("E")));

    // Check reverse mapping by iterating over all volumes
    auto const& tracker_vols = params.volume_id(tracker_id);
    EXPECT_EQ(3, tracker_vols.size());
    for (VolumeId vol_id : range(VolumeId{vols.num_volumes()}))
    {
        bool is_in_tracker = params.detector_id(vol_id) == tracker_id;
        bool is_in_list
            = std::find(tracker_vols.begin(), tracker_vols.end(), vol_id)
              != tracker_vols.end();
        EXPECT_EQ(is_in_tracker, is_in_list)
            << "volume " << vol_id << " tracker membership mismatch";
    }

    // Test device-compatible view
    DetectorView view(params.host_ref());
    EXPECT_EQ(tracker_id, view.detector_id(this->vol_id("B")));
    EXPECT_EQ(tracker_id, view.detector_id(this->vol_id("C")));
    EXPECT_EQ(tracker_id, view.detector_id(this->vol_id("D")));
    EXPECT_FALSE(view.detector_id(this->vol_id("A")));
    EXPECT_FALSE(view.detector_id(this->vol_id("E")));
}

TEST_F(DetectorTest, multi_det)
{
    // Multiple detectors with various volume configurations
    auto dets = this->make_detectors({
        {"calorimeter", {"A", "B"}},  // Two volumes
        {"tracker", {"C"}},  // Single volume
        {"muon", {"D", "E"}},  // Two volumes
    });

    auto const& vols = this->volumes();
    DetectorParams params(vols, std::move(dets));

    EXPECT_FALSE(params.empty());
    EXPECT_EQ(3, params.size());

    // Check detector IDs
    DetectorId calo_id{0};
    DetectorId tracker_id{1};
    DetectorId muon_id{2};

    EXPECT_EQ(calo_id, params.detector_id(this->vol_id("A")));
    EXPECT_EQ(calo_id, params.detector_id(this->vol_id("B")));
    EXPECT_EQ(tracker_id, params.detector_id(this->vol_id("C")));
    EXPECT_EQ(muon_id, params.detector_id(this->vol_id("D")));
    EXPECT_EQ(muon_id, params.detector_id(this->vol_id("E")));

    // Check reverse mappings by iteration
    for (DetectorId det_id : range(DetectorId{params.size()}))
    {
        auto const& det_vols = params.volume_id(det_id);
        for (VolumeId vol_id : range(VolumeId{vols.num_volumes()}))
        {
            bool is_in_detector = params.detector_id(vol_id) == det_id;
            bool is_in_list
                = std::find(det_vols.begin(), det_vols.end(), vol_id)
                  != det_vols.end();
            EXPECT_EQ(is_in_detector, is_in_list)
                << "volume " << vol_id << " detector " << det_id
                << " membership mismatch";
        }
    }

    // Test device view for all detectors
    DetectorView view(params.host_ref());
    EXPECT_EQ(calo_id, view.detector_id(this->vol_id("A")));
    EXPECT_EQ(calo_id, view.detector_id(this->vol_id("B")));
    EXPECT_EQ(tracker_id, view.detector_id(this->vol_id("C")));
    EXPECT_EQ(muon_id, view.detector_id(this->vol_id("D")));
    EXPECT_EQ(muon_id, view.detector_id(this->vol_id("E")));
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
