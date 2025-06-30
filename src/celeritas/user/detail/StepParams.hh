//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/user/detail/StepParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/AuxInterface.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "../StepData.hh"

namespace celeritas
{
class AuxStateVec;
class StepInterface;

namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Manage params and state data for step collector.
 *
 * \todo Move out of detail, take core params/state to copy detector steps? Not
 * currently possible right now because the step interface doesn't take params.
 */
class StepParams : public ParamsDataInterface<StepParamsData>,
                   public AuxParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    using SPStepInterface = std::shared_ptr<StepInterface>;
    using VecInterface = std::vector<SPStepInterface>;
    //!@}

  public:
    // Construct from data IDs and interfaces
    StepParams(AuxId aux_id,
               CoreGeoParams const& geo,
               VecInterface const& interfaces);

    //!@{
    //! \name Aux interface

    //! Short name for the aux data
    std::string_view label() const final { return "detector-step"; }
    //! Index of this class instance in its registry
    AuxId aux_id() const final { return aux_id_; }
    // Build core state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name Data interface

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }
    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }
    //!@}

    // Access state
    template<MemSpace M>
    StepStateData<Ownership::reference, M>& state_ref(AuxStateVec&) const;

    // Access data selection
    inline StepSelection const& selection() const;

    // Whether detectors are defined (false to gather *all* data)
    inline bool has_detectors() const;

  private:
    AuxId aux_id_;
    CollectionMirror<StepParamsData> mirror_;
};

//---------------------------------------------------------------------------//
/*!
 * See which data are being gathered.
 */
StepSelection const& StepParams::selection() const
{
    return this->host_ref().selection;
}
//---------------------------------------------------------------------------//
/*!
 * Whether detectors are defined (false to gather *all* data).
 */
bool StepParams::has_detectors() const
{
    return !this->host_ref().detector.empty();
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
