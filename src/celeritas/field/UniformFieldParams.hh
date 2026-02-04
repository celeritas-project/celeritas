//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/UniformFieldParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/geo/GeoFwd.hh"

#include "UniformFieldData.hh"

namespace celeritas
{
namespace inp
{
struct UniformField;
}
//---------------------------------------------------------------------------//
/*!
 * Construct and store data for a uniform magnetic field.
 */
class UniformFieldParams final
    : public ParamsDataInterface<UniformFieldParamsData>
{
  public:
    //@{
    //! \name Type aliases
    using Input = inp::UniformField;
    //@}

  public:
    // Construct with a uniform magnetic field
    UniformFieldParams(CoreGeoParams const& geo, Input const& inp);

    // Construct with a uniform magnetic field with no volume dependency
    explicit UniformFieldParams(Input const& inp);

    //! Access field data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access field data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

    //! Whether the field is present everywhere
    bool in_all_volumes() const { return data_.host_ref().has_field.empty(); }

  private:
    CollectionMirror<UniformFieldParamsData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
