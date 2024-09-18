//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/OpticalPhysicsParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/Types.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/Types.hh"
#include "celeritas/global/ActionRegistry.hh"

#include "OpticalPhysicsData.hh"

namespace celeritas
{
class OpticalPropertyParams;
class OpticalModel;
class OpticalModelBuilder;
class ConcreteAction;

//---------------------------------------------------------------------------//
/*!
 * Options to construct the optical physics data.
 */
struct OpticalPhysicsParamsOptions
{
};

//---------------------------------------------------------------------------//
/*!
 * Parameter interface for optical physics data.
 *
 * Performs the construction of optical physics data and optical models
 * from imported data, and provides an interface for accessing this data
 * on the host and on the device.
 */
class OpticalPhysicsParams : public ParamsDataInterface<OpticalPhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstModel = std::shared_ptr<OpticalModel const>;
    using SPConstProperties = std::shared_ptr<OpticalPropertyParams const>;

    using ActionIdRange = Range<ActionId>;
    using Options = OpticalPhysicsParamsOptions;
    //!@}


    //! Optical physics parameter construction arguments
    struct Input
    {
        SPConstProperties properties;
        ActionRegistry* action_registry = nullptr;
        Options options;
    };

  public:
    //! Construct from input
    explicit OpticalPhysicsParams(OpticalModelBuilder const&, Input);

    //! Number of optical models
    OpticalModelId::size_type num_models() const { return models_.size(); }

    //! Get an optical model
    inline SPConstModel const& model(OpticalModelId) const;

    //! Get the action IDs for all models
    inline ActionIdRange model_actions() const;

    //! Access optical physics parameters on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical physics parameters on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    using SPAction = std::shared_ptr<ConcreteAction>;
    using VecModel = std::vector<SPConstModel>;
    using HostValue = HostVal<OpticalPhysicsParamsData>;

    // Kernels / actions
    SPAction discrete_action_;

    // Host metadata / access
    VecModel models_;

    // Host/device storage and reference
    CollectionMirror<OpticalPhysicsParamsData> data_;

  private:
    void build_options(Options const&, HostValue*) const;
    void build_mfp(SPConstProperties, HostValue*) const;

    VecModel build_models(OpticalModelBuilder const&, ActionRegistry&) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Get an optical model.
 */
auto OpticalPhysicsParams::model(OpticalModelId id) const -> SPConstModel const&
{
    CELER_EXPECT(id < this->num_models());
    return models_[id.get()];
}

//---------------------------------------------------------------------------//
/*!
 * Get the action IDs for all optical models.
 */
auto OpticalPhysicsParams::model_actions() const -> ActionIdRange
{
    auto offset = host_ref().scalars.model_to_action;
    return {ActionId{offset}, ActionId{offset + this->num_models()}};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
