//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/ExampleInstanceCalo.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "corecel/io/Label.hh"
#include "celeritas/geo/GeoFwd.hh"
#include "celeritas/user/DetectorSteps.hh"
#include "celeritas/user/StepInterface.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Store calorimeter results based on pre-step volume instance names.
 *
 * \note This class isn't thread safe, but none of our unit tests use threads.
 */
class ExampleInstanceCalo final : public StepInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstCoreGeo = std::shared_ptr<CoreGeoParams const>;
    using VecLabel = std::vector<Label>;
    //!@}

    struct Result
    {
        std::vector<std::string> instance;
        std::vector<double> edep;

        void print_expected() const;
    };

  public:
    // Construct with geometry
    ExampleInstanceCalo(SPConstCoreGeo geo, VecLabel vol_labels);

    // Selection of data required for this interface
    Filters filters() const final;

    // Return flags corresponding to the "Step" above
    StepSelection selection() const final;

    // Tally host data for a step iteration
    void process_steps(HostStepState) final;

    // Tally device data for a step iteration
    void process_steps(DeviceStepState) final;

    // Tally device data for a step iteration
    void process_steps(DetectorStepOutput const&);

    // Get unfolded name/energy
    Result result() const;

  private:
    SPConstCoreGeo geo_;
    VecLabel det_labels_;
    std::vector<VolumeId> volume_ids_;

    std::map<std::string, real_type> edep_;

    //! Temporary CPU hit information
    DetectorStepOutput steps_;
};

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
