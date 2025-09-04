//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ExampleInstanceCalo.cc
//---------------------------------------------------------------------------//
#include "ExampleInstanceCalo.hh"

#include <iomanip>
#include <iostream>

#include "corecel/Config.hh"
#if CELERITAS_USE_GEANT4
#    include <G4VPhysicalVolume.hh>
#endif

#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "geocel/VolumeIdBuilder.hh"
#include "geocel/VolumeParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Print key/value pairs.
 */
void ExampleInstanceCalo::Result::print_expected() const
{
    using std::cout;
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static char const* const expected_instance[] = "
         << repr(this->instance)
         << ";\n"
            "EXPECT_VEC_EQ(expected_instance, "
            "result.instance);\n"
            "static real_type const expected_edep[] = "
         << repr(this->edep)
         << ";\n"
            "EXPECT_VEC_SOFT_EQ(expected_edep, "
            "result.edep);\n"
            "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
/*!
 * Construct with detector labels.
 */
ExampleInstanceCalo::ExampleInstanceCalo(VecLabel vol_labels)
    : det_labels_{std::move(vol_labels)}
{
    CELER_VALIDATE(!celeritas::global_volumes().expired(),
                   << "geometry volumes were not constructed before "
                      "instantiating ExampleInstanceCalo");

    // Map labels to volume IDs
    volume_ids_.resize(det_labels_.size());
    VolumeIdBuilder label_to_vol_id;
    for (auto i : range(det_labels_.size()))
    {
        volume_ids_[i] = label_to_vol_id(det_labels_[i]);
    }
    CELER_VALIDATE(
        std::all_of(volume_ids_.begin(), volume_ids_.end(), Identity{}),
        << "failed to find one or more volumes while "
           "constructing SimpleCalo");
}

//---------------------------------------------------------------------------//
/*!
 * Map volume names to detector IDs and exclude tracks with no deposition.
 */
auto ExampleInstanceCalo::filters() const -> Filters
{
    Filters result;

    for (auto didx : range<DetectorId::size_type>(volume_ids_.size()))
    {
        result.detectors[volume_ids_[didx]] = DetectorId{didx};
    }
    result.nonzero_energy_deposition = true;

    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Only save energy deposition and pre-step volume.
 */
auto ExampleInstanceCalo::selection() const -> StepSelection
{
    StepSelection result;
    result.energy_deposition = true;
    result.points[StepPoint::pre].volume_id = true;
    result.points[StepPoint::pre].volume_instance_ids = true;
    return result;
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (CPU).
 */
void ExampleInstanceCalo::process_steps(HostStepState state)
{
    copy_steps(&steps_, state.steps);
    if (steps_)
    {
        this->process_steps(steps_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Process detector tallies (GPU).
 */
void ExampleInstanceCalo::process_steps(DeviceStepState state)
{
    copy_steps(&steps_, state.steps);
    if (steps_)
    {
        this->process_steps(steps_);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Accumulate energy into the volume instance name.
 */
void ExampleInstanceCalo::process_steps(DetectorStepOutput const& out)
{
    CELER_EXPECT(!out.detector.empty());
    CELER_EXPECT(out.energy_deposition.size() == out.detector.size());
    auto const& vi_ids = out.points[StepPoint::pre].volume_instance_ids;
    auto const vi_depth = out.volume_instance_depth;
    CELER_EXPECT(vi_ids.size() == out.size() * vi_depth);

    CELER_LOG_LOCAL(debug) << "Processing " << out.size() << " hits";

    auto volumes = celeritas::global_volumes().lock();
    CELER_ASSERT(volumes);
    auto const& vi_labels = volumes->volume_instance_labels();

    for (auto hit : range(out.size()))
    {
        std::ostringstream os;
        CELER_ASSERT(out.detector[hit] < det_labels_.size());
        os << det_labels_[out.detector[hit].get()].name;
        for (auto id_index : range(vi_depth))
        {
            VolumeInstanceId vi_id = vi_ids[hit * vi_depth + id_index];
            if (!vi_id)
            {
                break;
            }
            os << (id_index == 0 ? ':' : '/') << vi_labels.at(vi_id);
        }
        edep_[std::move(os).str()]
            += value_as<units::MevEnergy>(out.energy_deposition[hit]);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Get unfolded name/energy.
 */
auto ExampleInstanceCalo::result() const -> Result
{
    Result result;
    for (auto&& [key, value] : edep_)
    {
        result.instance.push_back(key);
        result.edep.push_back(value);
    }
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
