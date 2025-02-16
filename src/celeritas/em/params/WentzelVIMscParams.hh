//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/WentzelVIMscParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/em/data/WentzelVIMscData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class ParticleParams;
struct ImportData;
struct ImportMscModel;

//---------------------------------------------------------------------------//
/*!
 * Construct and store data for Wentzel VI multiple scattering.
 *
 * Multiple scattering is used by the along-step kernel(s).
 */
class WentzelVIMscParams final : public ParamsDataInterface<WentzelVIMscData>
{
  public:
    //!@{
    //! \name Type aliases
    using VecImportMscModel = std::vector<ImportMscModel>;
    //!@}

    //! Wentzel VI multiple scattering options
    struct Options
    {
        //! Use cubic spline interpolation for cross sections
        bool spline{false};
    };

  public:
    // Construct if MSC process data is present, else return nullptr
    static std::shared_ptr<WentzelVIMscParams>
    from_import(ParticleParams const& particles,
                ImportData const& data,
                Options options);

    // Construct from process data
    WentzelVIMscParams(ParticleParams const& particles,
                       VecImportMscModel const& mdata,
                       Options options);

    //! Access Wentzel VI data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access Wentzel VI data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<WentzelVIMscData> data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
