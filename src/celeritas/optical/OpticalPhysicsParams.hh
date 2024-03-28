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

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/global/ActionInterface.hh"

#include "OpticalPhysics.hh"

namespace celeritas
{
class ActionRegistry;
class MaterialParams;
class OpticalProcess;
class OpticalModel;

//---------------------------------------------------------------------------//
/*!
 */
struct OpticalPhysicsParamsOptions
{
};


//---------------------------------------------------------------------------//
class OpticalPhysicsParams final : public ParamsDataInterface<OpticalPhysicsParamsData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstProcess = std::shared_ptr<OpticalProcess const>;
    using SPConstModel = std::shared_ptr<OpticalModel const>;

    using SpanConstOpticalProcessId = Span<OpticalProcessId const>;
    using ActionIdRange = Range<ActionId>;
    using VecProcess = std::vector<SPConstProcess>;
    using Options = OpticalPhysicsParamsOptions;
    //!@}

    //! Optical phyiscs parameter construction arguments
    struct Input
    {
        SPConstMaterials materials;
        VecProcess processes;
        ActionRegistry* action_registry = nullptr;

        Options options;
    };

  public:
    // Construct with processes and helper classes
    explicit OpticalPhysicsParams(Input);

    //// HOST ACCESSORS ////

    //! Number of opticla processes
    OpticalProcessId::size_type num_processes() const { return processes_.size(); }

    //! Get process associated with a particular optical process id
    inline SPConstProcess const& process(OpticalProcessId) const;

    //! Get all optical actions
    inline ActionIdRange actions() const;

    //! Get all optical processes
    SpanConstOpticalProcessId processes() const;

    //! Access optical physics properties on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access optical physics properties on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    using SPAction = std::shared_ptr<ConcreteAction>;
    using VecModel = std::vector<std::pair<SPConstModel, OpticalProcessId> >;
    using HostValue = HostVal<OpticalPhysicsParamsData>;

    // Kernels/actions
    SPAction discrete_action_;
    SPAction failure_action_;

    // Host metadata/access
    VecProcess processes_;
    VecModel models_;

    // Host/device storage and reference
    CollectionMirror<OpticalPhysicsParamsData> data_;

  private:
    VecModel build_models(ActionRegistry& action_registry) const;
    void build_options(Options const& opts, HostValue* data) const;
    void build_lambda(MaterialParams const& mats, HostValue* data) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
}  // namespace celeritas
