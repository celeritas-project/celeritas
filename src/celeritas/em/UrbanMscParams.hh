//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMscParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"

#include "MscParams.hh"
#include "data/UrbanMscData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
class MaterialParams;
class MaterialView;
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Construct and store data for Urban multiple scattering.
 *
 * Multiple scattering is used by the along-step kernel(s).
 */
class UrbanMscParams final : public MscParams,
                             public ParamsDataInterface<UrbanMscData>
{
  public:
    // Construct if MSC process data is present, else return nullptr
    static std::shared_ptr<UrbanMscParams>
    from_import(ParticleParams const& particles,
                MaterialParams const& materials,
                ImportData const& data);

    // Construct from process data
    inline UrbanMscParams(ParticleParams const& particles,
                          MaterialParams const& materials,
                          VecImportMscModel const& mdata);

    // TODO: possible "applicability" interface used for constructing
    // along-step kernels?

    //! Access UrbanMsc data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access UrbanMsc data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<UrbanMscData> data_;

    static UrbanMscMaterialData
    calc_material_data(MaterialView const& material_view);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
