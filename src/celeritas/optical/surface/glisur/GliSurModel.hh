//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/glisur/GliSurModel.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/optical/surface/SurfaceModel.hh"

#include "GliSurData.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 */
class GliSurModel : public SurfaceModel,
                    public ParamsDataInterface<GliSurData>,
                    public GliSurPolishedNormalParamsInterface
{
  public:
    //!@{
    //! \name Type aliases
    //!@}

    struct SurfaceInput
    {
        real_type polish{-1};
        GliSurFinishType finish{GliSurFinishType::size_};
        GliSurInterfaceType interface_type{GliSurInterfaceType::size_};

        explicit operator bool() const
        {
            return 0 <= polish && polish <= 1
                   && finish != GliSurFinishType::size_
                   && interface_type != GliSurInterfaceType::size_;
        }
    };

    struct Input
    {
        ActionId trivial_normal_action;
        ActionId glisur_normal_action;

        ActionId grid_reflectivity_action;

        ActionId glisur_dielectric_interaction;
        ActionId glisur_metal_interaction;

        std::vector<SurfaceInput> surfaces;
    };

  public:
    GliSurModel(ActionId, Input const&);

    void step(CoreParams const&, CoreStateHost&) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;

    //! Access model data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access model data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    HostCRef<GliSurPolishedNormalData> const&
    glisur_polished_normal_host_ref() const final
    {
        return glisur_polished_normal_data_.host_ref();
    }
    DeviceCRef<GliSurPolishedNormalData> const&
    glisur_polished_normal_device_ref() const final
    {
        return glisur_polished_normal_data_.device_ref();
    }

  private:
    CollectionMirror<GliSurData> data_;
    CollectionMirror<GliSurPolishedNormalData> glisur_polished_normal_data_;
};

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
