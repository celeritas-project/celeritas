//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
class MaterialParams;
class MaterialView;
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

    struct Options
    {
        //! Use combined single and multiple scattering
        bool is_combined{true};
        //! Polar angle limit between single and multiple scattering
        real_type polar_angle_limit{constants::pi};
        //! Factor for dynamic computation of angular limit between SS and MSC
        real_type angle_limit_factor{1};
        //! User defined screening factor
        real_type screening_factor{1};
    };

  public:
    // Construct if MSC process data is present, else return nullptr
    static std::shared_ptr<WentzelVIMscParams>
    from_import(ParticleParams const& particles,
                MaterialParams const& materials,
                ImportData const& data);

    // Construct from process data
    WentzelVIMscParams(ParticleParams const& particles,
                       MaterialParams const& materials,
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
