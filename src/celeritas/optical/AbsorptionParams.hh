//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionParams.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
    AbsorptionParams ...;
   \endcode
 */
class AbsorptionParams final : public ParamsDataInterface<AbsorptionData>
{
  public:
    //!@{
    //! \name Type aliases
    using AbsorptionDataCRef = HostCRef<AbsorptionData>;
    //!@}

    struct Input
    {
        std::vector<ImportOpticalAbsorption> data;
    };

  public:
    // Construct with imported data
    static std::shared_ptr<AbsorptionParams>
    from_import(ImportData const& data);

    // Construct with absorption data
    explicit AbsorptionParams(Input const& input);

    //! Access physics properties on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access physics properties on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<AbsorptionData> mirror_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
