//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/track/SimParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "corecel/math/NumericLimits.hh"
#include "celeritas/inp/Tracking.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "SimData.hh"

namespace celeritas
{
class ParticleParams;
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Manage persistent simulation data.
 */
class SimParams final : public ParamsDataInterface<SimParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

    //! Input data to construct this class
    struct Input
    {
        // Construct with imported data and default max field substeps
        static Input from_import(ImportData const&, SPConstParticles);

        // Construct with imported data and max field substeps
        static Input from_import(ImportData const&,
                                 SPConstParticles,
                                 inp::TrackingLimits const& limits);

        //// DATA ////

        SPConstParticles particles;
        std::unordered_map<PDGNumber, LoopingThreshold> looping;
        size_type max_steps = numeric_limits<size_type>::max();
    };

  public:
    // Construct with simulation input data
    explicit SimParams(Input const&);

    //! Access data on host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access data on device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<SimParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
