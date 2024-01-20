//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/UrbanMscParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <vector>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/em/data/UrbanMscData.hh"

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
 * Construct and store data for Urban multiple scattering.
 *
 * Multiple scattering is used by the along-step kernel(s).
 *
 * TODO: UrbanMsc is the only MSC model presently in Celeritas, but we should
 * extend this to be an interface since it's an interchangeable component of
 * the along-step kernel(s).
 */
class UrbanMscParams final : public ParamsDataInterface<UrbanMscData>
{
  public:
    //!@{
    //! \name Type aliases
    using VecImportMscModel = std::vector<ImportMscModel>;
    //!@}

    //! MSC configuration options
    struct Options
    {
        real_type lambda_limit{1 * units::millimeter};  //!< Lambda limit
        real_type range_fact{0.04};  //!< Range factor for e-/e+
        real_type safety_fact{0.6};  //!< Safety factor
    };

  public:
    // Construct if MSC process data is present, else return nullptr
    static std::shared_ptr<UrbanMscParams>
    from_import(ParticleParams const& particles,
                MaterialParams const& materials,
                ImportData const& import);

    // Construct from process data
    inline UrbanMscParams(ParticleParams const& particles,
                          MaterialParams const& materials,
                          VecImportMscModel const& mdata,
                          Options options);

    // TODO: possible "applicability" interface used for constructing
    // along-step kernels?

    //! Access UrbanMsc data on the host
    HostRef const& host_ref() const final { return mirror_.host_ref(); }

    //! Access UrbanMsc data on the device
    DeviceRef const& device_ref() const final { return mirror_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<UrbanMscData> mirror_;

    static UrbanMscMaterialData
    calc_material_data(MaterialView const& material_view);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
