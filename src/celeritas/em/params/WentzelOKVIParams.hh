//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/WentzelOKVIParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"
#include "celeritas/mat/IsotopeView.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class MaterialParams;
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Construct and store shared Coulomb scattering data.
 *
 * This data is used by both the single Coulomb scattering and Wentzel VI
 * multiple scattering models.
 */
class WentzelOKVIParams final : public ParamsDataInterface<WentzelOKVIData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    //!@}

  public:
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
        //! Nuclear form factor model
        NuclearFormFactorType form_factor{NuclearFormFactorType::exponential};
    };

  public:
    // Construct if Wentzel VI or Coulomb is present, else return nullptr
    static std::shared_ptr<WentzelOKVIParams>
    from_import(ImportData const& data, SPConstMaterials materials);

    // Construct from material data and options
    WentzelOKVIParams(SPConstMaterials materials, Options options);

    //! Access Wentzel OK&VI data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access Wentzel OK&VI data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    // Host/device storage and reference
    CollectionMirror<WentzelOKVIData> data_;

    // Construct per-element data (loads Mott coefficients)
    void build_data(HostVal<WentzelOKVIData>& host_data,
                    MaterialParams const& materials);

    // Retrieve matrix of interpolated Mott coefficients
    static MottElementData::MottCoeffMatrix
    get_mott_coeff_matrix(AtomicNumber z);

    // Calculate the nuclear form prefactor
    static real_type calc_nuclear_form_prefactor(IsotopeView const& iso);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
